# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import shutil
import tempfile
import os
import importlib
import sys

import dm_control
tmp_dir = tempfile.mkdtemp()
src_file = os.path.dirname(dm_control.__file__)
dst_dir = os.path.join(tmp_dir, 'dm_control')
shutil.copytree(src_file, dst_dir)
sys.path.insert(0, tmp_dir)
importlib.reload(dm_control)
from dmc2gymnasium import DMCGym

import xml.etree.ElementTree as ET
# import gymnasium as gym

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from tqdm import trange

from synther.corl.shared.buffer import prepare_replay_buffer, StateNormalizer, RewardNormalizer, DiffusionConfig
from synther.corl.shared.logger import Logger
from synther.diffusion.utils import make_inputs

def set_pole_length(
    file_path = os.path.join(tmp_dir, 'dm_control/suite/cartpole.xml'), 
    new_size = 0.045
    ):
    tree = ET.parse(file_path)
    root = tree.getroot()
    geom = root.find('.//geom')

    if geom is not None and 'size' in geom.attrib:
        geom.attrib['size'] = str(new_size)

    tree.write(file_path)
    print(f'Pole length set to {new_size}')
    return float(geom.attrib['size'])


TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"


@dataclass
class TrainConfig:
    # wandb params
    dataset: str = "dm-cartpole-test-length0.025-0.35-v0"
    context_aware: int = 0
    diffuser: bool = True
    pole_length: Optional[float] = None  #half pole length. 0.045 defalt
    cond: list = None  
    cond_dim: int = None
    project: str = "DiffusionRL-edac"
    group: str = "EDAC"
    name: str = "EDAC"
    # model params
    hidden_dim: int = 128
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    # training params
    buffer_size: int = 10_000_000
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 100
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    checkpoints_path: Optional[str] = None
    save_checkpoints: bool = False  # Save model checkpoints
    log_every: int = 1000
    deterministic_torch: bool = False
    seed: int = 10
    eval_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Diffusion config
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def __post_init__(self):
        if self.diffusion.path == "None":
            self.diffusion.path = None
        self.diffuser = True if self.diffusion.path is not None else False
        
        if self.pole_length is not None:
            self.cond = [set_pole_length(new_size = self.pole_length)]
        if self.cond_dim is None:
            self.cond_dim = len(self.cond) if self.cond is not None else None
            
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    wandb.run.save()


def set_seed(
        seed: int, env: Optional[DMCGym] = None, deterministic_torch: bool = False
):
    if env is not None:
        env._observation_space.seed(1)
        env._action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    

# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
            need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
        # print("tanh_action: ", tanh_action)
        # print("tanh_action * self.max_action: ", tanh_action * self.max_action)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class EDAC:
    def __init__(
            self,
            actor: Actor,
            actor_optimizer: torch.optim.Optimizer,
            critic: VectorizedCritic,
            critic_optimizer: torch.optim.Optimizer,
            gamma: float = 0.99,
            tau: float = 0.005,
            eta: float = 1.0,
            alpha_learning_rate: float = 1e-4,
            device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _critic_diversity_loss(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        num_critics = self.critic.num_critics
        # almost exact copy from the original implementation, only style changes:
        # https://github.com/snu-mllab/EDAC/blob/198d5708701b531fd97a918a33152e1914ea14d7/lifelong_rl/trainers/q_learning/sac.py#L192

        # [num_critics, batch_size, *_dim]
        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = (
            action.unsqueeze(0)
            .repeat_interleave(num_critics, dim=0)
            .requires_grad_(True)
        )
        # [num_critics, batch_size]
        q_ensemble = self.critic(state, action)

        q_action_grad = torch.autograd.grad(
            q_ensemble.sum(), action, retain_graph=True, create_graph=True
        )[0]
        q_action_grad = q_action_grad / (
                torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10
        )
        # [batch_size, num_critics, action_dim]
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = (
            torch.eye(num_critics, device=self.device)
            .unsqueeze(0)
            .repeat(q_action_grad.shape[0], 1, 1)
        )
        # removed einsum as it is usually slower than just torch.bmm
        # [batch_size, num_critics, num_critics]
        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss

    def _critic_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        diversity_loss = self._critic_diversity_loss(state, action)

        loss = critic_loss + self.eta * diversity_loss

        return loss, q_values

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss, q_values = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "q_values": q_values.mean().item(),
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
        env: DMCGym, actor: Actor, device: str, n_episodes: int, seed: int, contexts: Optional[np.ndarray] = None, state_normalizer: Optional[StateNormalizer] = None
) -> np.ndarray:
    state_normalizer = state_normalizer.to_numpy()
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, info = env.reset(seed=seed, options={})
        done = False
        episode_reward = 0.0
        while not done:
            state = (np.concatenate([state, contexts])[None, :]) if contexts is not None else state
            state = state_normalizer(state)
            action = actor.act(state, device)
            # print("state: ", state)
            # print("action: ", action)
            state, reward, terminated, truncated, info = env.step(action.squeeze(-1))
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    eval_env = DMCGym("cartpole", "swingup")
    state_dim = eval_env.observation_space.shape[0] + 1 if config.context_aware else eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    dataset = make_inputs(config.dataset, original=True, context=False)
    state_normalizer, replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        dataset=dataset,
        env_name=config.env_name,
        device=config.device,
        reward_normalizer=RewardNormalizer(dataset, config.env_name) if config.normalize_reward else None,
        diffusion_config=config.diffusion,
        cond_dim=config.cond_dim,
        context_aware=config.context_aware,
        context=config.pole_length,
    )

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, float(eval_env.action_space.high[0]))
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = EDAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
        logger = Logger(config.checkpoints_path, seed=config.seed)
    else:
        logger = Logger('/tmp', seed=config.seed)

    total_updates = 0

    # for epoch in trange(config.num_epochs, desc="Training"):
    for epoch in range(config.num_epochs):
        # training
        # for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
        for _ in range(config.num_updates_on_epoch):
            batch = replay_buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})
                # logger.log({"step": total_updates, **update_info}, mode='train')
            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
                state_normalizer=state_normalizer,
                contexts=(np.array(config.cond, dtype=np.float32)) if config.context_aware and config.cond is not None else None,
            )
            eval_log = {
                "reward_mean": np.mean(eval_returns),
                "reward_std": np.std(eval_returns),
                "step": total_updates,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["d4rl_normalized_score"] = np.mean(normalized_score)
                eval_log["d4rl_normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)
            # logger.log(eval_log, mode='eval')

            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
