# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

# import d4rl
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from synther.corl.shared.buffer import prepare_replay_buffer, RewardNormalizer, StateNormalizer, DiffusionConfig, DiffusionGenerator
from synther.corl.shared.logger import Logger
from synther.diffusion.utils import make_inputs
from dm_control import suite


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
    # Experiment
    context_aware: int = 0
    diffuser: bool = True
    segment: Optional[str] = None
    pole_length: Optional[float] = None  #half pole length. 0.045 defalt
    percentile: Optional[int] = None
    cond: list = None  
    cond_dim: int = None
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(3e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    save_checkpoints: bool = False  # Save model checkpoints
    log_every: int = 1000
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "DiffusionRL-td3_bc_non_filtered_1"
    group: str = "TD3_BC"
    name: str = "Dataset0.025-0.35"
    # Diffusion config
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    # Network size
    network_width: int = 64
    network_depth: int = 5
    dataset: str = "dm-cartpole-test-length0.025-0.35-v0"

    def __post_init__(self):
        if self.diffusion.path == "None":
            self.diffusion.path = None
        self.diffuser = True if self.diffusion.path is not None else False
        if self.context_aware:
            print("context_aware")
        elif self.diffuser:
            print("diffuser")
        print(tmp_dir)
        print("seed: ", self.seed)
            
        if self.pole_length is not None:
            self.cond = [set_pole_length(new_size = self.pole_length)]
        if self.cond_dim is None:
            self.cond_dim = len(self.cond) if self.cond is not None else None
        
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std
    
# def wrap_env(
#         env: gym.Env,
#         state_mean: Union[np.ndarray, float] = 0.0,
#         state_std: Union[np.ndarray, float] = 1.0,
#         reward_scale: float = 1.0,
# ) -> gym.Env:
#     # PEP 8: E731 do not assign a lambda expression, use a def
#     def normalize_state(state):
#         return (
#                 state - state_mean
#         ) / state_std  # epsilon should be already added in std.

#     def scale_reward(reward):
#         # Please be careful, here reward is multiplied by scale!
#         return reward_scale * reward
#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)



def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, env_name: str, contexts: Optional[np.ndarray] = None, state_normalizer: Optional[StateNormalizer] = None
) -> np.ndarray:
    state_normalizer = state_normalizer.to_numpy()
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, info = env.reset(seed=seed, options={})
        done = False
        # state, done = env.reset(seed=seed), False
        episode_reward = 0.0
        while not done:
            state = (np.concatenate([state, contexts])[None, :]) if contexts is not None else state
            state = state_normalizer(state)
            action = actor.act(state, device)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

from torch.autograd import Function

class StepModule(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if not hasattr(ctx, 'value'):
            ctx.value = torch.tensor(0.0, device=input.device)
        ctx.value = torch.where(input >= 0, torch.tensor(1.0, device=input.device), ctx.value*0.9)
        return torch.where(input >= 0, torch.tensor(1.0, device=input.device), ctx.value)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone() * ctx.value
        grad_input[input != 0] = 0
        return grad_input

class Step_n_feedback(nn.Module):
    def forward(self, input):
        return StepModule.apply(input)

class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))
    
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256, n_hidden: int = 2):
        super(Actor, self).__init__()

        # layers = [nn.Linear(state_dim, hidden_dim), Step_n_feedback()]
        # for _ in range(n_hidden - 1):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     layers.append(Step_n_feedback())
        # layers.append(nn.Linear(hidden_dim, action_dim))
        # layers.append(nn.Tanh())
        
        layers = [nn.Linear(state_dim, hidden_dim)]
        for _ in range(n_hidden):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, activation='relu'))
            
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super(Critic, self).__init__()

        layers = [nn.Linear(state_dim + action_dim, hidden_dim)]
        for _ in range(n_hidden):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, activation='relu'))

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:  # noqa
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            alpha: float = 2.5,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        # print(batch[0])
        # print(batch[0].shape)

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        log_dict["q1"] = current_q1.mean().item()
        log_dict["q2"] = current_q2.mean().item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed(config.seed)
    if config.env == "cartpole":
        env = DMCGym("cartpole", "swingup", task_kwargs={'random':config.seed})
        inputs = make_inputs(config.dataset, original=True, context=False, segment=config.segment)
        dataset = inputs
        import re
        match = re.search(r'length(.+?)-v0', config.dataset)
        if match:
            result = match.group(1)
            config.group = f'Dataset{result}'
    else:
        env = gym.make(config.env)
        dataset = d4rl.qlearning_dataset(env)

    if config.context_aware:
        state_dim = env.observation_space.shape[0] + 1
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.context_aware:
            state_mean_buffer, state_std_buffer = compute_mean_std(np.concatenate([dataset["observations"], dataset["contexts"]], axis=1), eps=1e-3)
    else:
        state_mean_buffer, state_std_buffer = compute_mean_std(dataset["observations"], eps=1e-3)

    state_normalizer, replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        dataset=dataset,
        env_name=config.env,
        device=config.device,
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean_buffer, state_std_buffer),
        diffusion_config=config.diffusion,
        cond_dim=config.cond_dim,
        context_aware=config.context_aware,
        context=config.pole_length,
        percentile=config.percentile,
        env=suite.load(domain_name="cartpole", task_name="swingup"),
    )

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
        logger = Logger(config.checkpoints_path, seed=config.seed)
    else:
        logger = Logger('/tmp', seed=config.seed)

    # Set seeds
    seed = config.seed
    if config.env != "cartpole":
        set_seed(seed, env)

    actor = Actor(
        state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth).to(
        config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(
        state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(
        state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        # policy_file = Path(config.load_model) / "checkpoint_" + str(config.max_timesteps - 1) + ".pt"
        policy_file = os.path.join(str(Path(config.load_model)), "checkpoint_" + str(config.max_timesteps - 1) + ".pt")
        trainer.load_state_dict(torch.load(policy_file))
        print(f"Loaded model from {policy_file}")
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        if isinstance(replay_buffer, DiffusionGenerator):
            batch = replay_buffer.sample(
                config.batch_size, 
                cond=torch.tensor(config.cond, dtype=torch.float32)[:, None] if config.cond is not None else None,
                )
        else:
            batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if t % config.log_every == 0:
            wandb.log(log_dict, step=trainer.total_it)
            # logger.log({'step': trainer.total_it, **log_dict}, mode='train')

        # Evaluate episode
        if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                env_name = config.env,
                state_normalizer=state_normalizer,
                contexts=(np.array(config.cond, dtype=np.float32)) if config.context_aware and config.cond is not None else None,
            )
            eval_score = eval_scores.mean()
            if config.env == "cartpole":
                normalized_eval_score = 100.0
            else:
                normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            # log_dict = {"d4rl_normalized_score": normalized_eval_score}
            log_dict = {"evaluate_return": eval_score}
            wandb.log(log_dict, step=trainer.total_it)
            # logger.log({'step': trainer.total_it, **log_dict}, mode='eval')


if __name__ == "__main__":
    train()
