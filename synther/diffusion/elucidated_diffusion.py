"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
import pathlib
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple, Union

import gin
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from einops import reduce
from ema_pytorch import EMA
from redq.algos.core import ReplayBuffer
from torch import nn
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm, trange

from synther.diffusion.norm import BaseNormalizer
from synther.online.utils import make_inputs_from_replay_buffer
from synther.diffusion.norm import MinMaxNormalizer
from synther.early_stopper import EarlyStopper
import gymnasium as gym
from dmc2gymnasium import DMCGym

import os
import glob

def get_latest_model_file(folder_path, pattern="model-*.pt"):
    files = glob.glob(os.path.join(folder_path, pattern))
    
    if not files:
        return None
    
    latest_file = max(files, key=os.path.getmtime)
    
    return latest_file

# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
        num_transition: int = 1,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs, actions, rewards, next_obs = [], [], [], []
    for i in range(num_transition):
        anchor = (obs_dim + action_dim + 1) * i
        obs.append(samples[:, anchor:anchor + obs_dim])
        actions.append(samples[:, anchor + obs_dim:anchor +  obs_dim + action_dim])
        rewards.append(samples[:, anchor + obs_dim + action_dim])
        next_obs.append(samples[:, anchor + obs_dim + action_dim + 1: anchor + obs_dim + action_dim + 1 + obs_dim])

        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_obs = np.concatenate(next_obs, axis=0)
            
        if modelled_terminals:
            terminals = samples[:, -1]
            if terminal_threshold is not None:
                if isinstance(terminals, torch.Tensor):
                    terminals = (terminals > terminal_threshold).float()
                else:
                    terminals = (terminals > terminal_threshold).astype(np.float32)
            return obs, actions, rewards, next_obs, terminals
        else:
            return obs, actions, rewards, next_obs

@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            ema_model,
            env: Optional[gym.Env] = None,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
            cond: torch.Tensor = None,
            num_transition: int = 1,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size // num_transition
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, DMCGym("cartpole", "swingup", task_kwargs={'random':1}), num_transition=num_transition)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals

# helpers

def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)

def reset_to_state(env, state):
    env._reset_next_step = False
    env._step_count = 0
    with env._physics.reset_context():
        env._physics.named.data.qpos[:] = np.array([state[0], state[1]])
        env._physics.named.data.qvel[:] = np.array([state[2], state[3]])
    env._task.after_step(env._physics)
    
def calculate_diffusion_loss(diffusion_dataset, env, range = None):
    if range is None:
        range = diffusion_dataset["observations"].shape[0]
    reward_loss = np.zeros((range, 1))
    observation_loss = np.zeros((range, diffusion_dataset["observations"].shape[1]))

    # for transition in trange(diffusion_dataset["observations"].shape[0]):
    for transition in trange(range):
        index = np.ones(diffusion_dataset["observations"].shape[1], dtype=bool)
        index[2] = False
        observation = diffusion_dataset["observations"][transition]
        state = observation[index]
        # state[1] = np.arccos(state[1].clip(-1, 1))
        state[1] = np.arctan2(observation[2], observation[1])
        action = diffusion_dataset["actions"][transition]
        reward = diffusion_dataset["rewards"][transition]
        next_observation = diffusion_dataset["next_observations"][transition]

        # if transition == 0:
        #     print(observation, action, reward, next_observation)
        reset_to_state(env, state)
        timestamp = env.step(action)
        reward_true = timestamp.reward
        next_observation_true = _flatten_obs(timestamp.observation)
        reward_loss[transition, :] = np.abs(reward - reward_true)
        observation_loss[transition, :] = np.abs(next_observation - next_observation_true)
    return observation_loss, reward_loss


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# main class
@gin.configurable
class ElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            normalizer: BaseNormalizer,
            event_shape: Sequence[int],  # shape of the input and output
            num_sample_steps: int = 32,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.normalizer = normalizer

        # input dimensions
        self.event_shape = event_shape

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, clamp=False, cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        net_out = self.net(
            self.c_in(padded_sigma) * noised_inputs,
            self.c_noise(sigma),
            cond=cond,
        )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
            self,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            disable_tqdm: bool = False,
    ):
        cond = cond.to(self.device) if exists(cond) else None
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            clamp: bool = False,
            cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    @torch.no_grad()
    def log_likelihood(self, x, atol=1e-4, rtol=1e-4, clamp=False, normalize=True, cond=None):
        # Input to the ODE solver must be in normalized space.
        if normalize:
            x = self.normalizer.normalize(x)
        v = torch.randint_like(x, 2) * 2 - 1
        s_in = x.new_ones([x.shape[0]])
        fevals = 0

        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                sigma = sigma * s_in
                padded_sigma = sigma.view(x.shape[0], *([1] * len(self.event_shape)))
                denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
                denoised_over_sigma = (x - denoised) / padded_sigma
                fevals += 1
                grad = torch.autograd.grad((denoised_over_sigma * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return denoised_over_sigma.detach(), d_ll

        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([self.sigma_min, self.sigma_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, self.sigma_max).log_prob(latent).flatten(1).sum(1)

        return ll_prior + delta_ll, {'fevals': fevals}

    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, inputs, cond=None):
        inputs = self.normalizer.normalize(inputs)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, cond=cond)
        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


@gin.configurable
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_dataset: Optional[torch.utils.data.Dataset] = None,
            test_dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 32,
            small_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            env = None,
            eval_interval = 1000,
            step = 0,
    ):
        super().__init__()
        self.earlystopper = EarlyStopper(patience=4, delta=0.002)
        self.eval_interval = eval_interval
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        if train_dataset is not None:
            # If dataset size is less than 800K use the small batch size
            if len(train_dataset) < int(8e5):
                self.batch_size = small_batch_size
            else:
                self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            # dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count()//2)
            dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None
            
        if test_dataset is not None:
            # self.eval_dl = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count()//2)
            self.eval_dl = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
            self.eval_dl = self.accelerator.prepare(self.eval_dl)
            self.eval_dl = cycle(self.eval_dl)

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = step

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == 'linear':
            print('using linear learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using cosine learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)
        
        self.generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=self.ema.ema_model,
            sample_batch_size = 1000,
        )
        self.env = env

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: Optional[int] = None):
        accelerator = self.accelerator
        device = accelerator.device

        if milestone is not None:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            latest_file = get_latest_model_file(self.results_folder)
            data = torch.load(latest_file, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def evaluate(self, accumulate_every = 1000):
        accelerator = self.accelerator
        device = accelerator.device
        self.model.eval()
        
        eval_loss = 0.
        for i in range(accumulate_every):
            with torch.no_grad():
                if type(next(self.eval_dl)) != torch.Tensor:
                    data, context = next(self.eval_dl)
                    data, context = data.to(device), context.to(device)
                else:
                    data, context = (next(self.eval_dl)[0]).to(device), None

                with self.accelerator.autocast():
                    loss = self.model(data, cond=context)

            eval_loss += loss.item()
        eval_loss /= accumulate_every
        
        print(f'Evaluation loss: {eval_loss:.4f}')
        wandb.log({
            'eval_loss': eval_loss,
            'step': self.step
        })
        accelerator.wait_for_everyone()
        # accelerator.free_memory()
        
        self.model.train()
        return eval_loss
    
    
    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if type(next(self.dl)) != torch.Tensor:
                        data, context = next(self.dl)
                        data, context = data.to(device), context.to(device)
                        # print(context)
                    else:
                        data, context = (next(self.dl)[0]).to(device), None
                    # print(type(next(self.dl)[0]))
                    # print(next(self.dl)[0])
                    with self.accelerator.autocast():
                        loss = self.model(data, cond=context)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        
                    # breakpoint()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({
                    'step': self.step,
                    'loss': total_loss,
                    'lr': self.opt.param_groups[0]['lr']
                })

                if self.step % self.eval_interval == 0:
                    eval_loss = self.evaluate()
                    self.earlystopper(eval_loss)
                    if self.earlystopper.early_stop:
                        print("Early stopping")
                        self.save(self.step)
                        # break
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Early stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                    
                    for cond in (0.2, 0.4 ,0.6):
                        observations, actions, rewards, next_observations, terminals = self.generator.sample(
                            num_samples=self.generator.sample_batch_size,
                            cond=torch.tensor([cond], dtype=torch.float32)[:, None],
                            num_transition=1,
                        )
                        observation_err, reward_err = calculate_diffusion_loss(
                            {
                                "observations": observations,
                                "actions": actions,
                                "rewards": rewards,
                                "next_observations": next_observations,
                                "terminals": terminals,
                            },
                            self.env,
                        )
                        wandb.log({
                            'pos_1_mse_eval_len= ' + str(cond): np.mean(observation_err[0]),
                            'pos_2_mse_eval_len= ' + str(cond): np.mean(observation_err[1]),
                            'pos_3_mse_eval_len= ' + str(cond): np.mean(observation_err[2]),
                            'vel_1_mse_eval_len= ' + str(cond): np.mean(observation_err[3]),
                            'vel_2_mse_eval_len= ' + str(cond): np.mean(observation_err[4]),
                            'reward_mse_eval_len= ' + str(cond): np.mean(reward_err),
                        })
                
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                

        accelerator.print('training complete')

    # Allow user to pass in external data.
    def train_on_batch(
            self,
            data: torch.Tensor,
            use_wandb=True,
            splits=1,  # number of splits to split the batch into
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)

        total_loss = 0.
        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(data, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and data.shape[0] % splits == 0
            split_data = torch.split(data, data.shape[0] // splits)

            for idx, d in enumerate(split_data):
                with self.accelerator.autocast():
                    # Split condition as well
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            new_kwargs[k] = torch.split(v, v.shape[0] // splits)[idx]
                        else:
                            new_kwargs[k] = v

                    loss = self.model(d, **new_kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log({
                'step': self.step,
                'loss': total_loss,
                'lr': self.opt.param_groups[0]['lr'],
            })

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss


@gin.configurable
class REDQTrainer(Trainer):
    def __init__(
            self,
            diffusion_model,
            train_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            model_terminals: bool = False,
    ):
        super().__init__(
            diffusion_model,
            train_dataset=None,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            lr_scheduler=lr_scheduler,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            weight_decay=weight_decay,
            results_folder=results_folder,
            amp=amp,
            fp16=fp16,
            split_batches=split_batches,
        )

        self.model_terminals = model_terminals

    def train_from_redq_buffer(self, buffer: ReplayBuffer, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            b = buffer.sample_batch(self.batch_size)
            obs = b['obs1']
            next_obs = b['obs2']
            actions = b['acts']
            rewards = b['rews'][:, None]
            done = b['done'][:, None]
            data = [obs, actions, rewards, next_obs]
            if self.model_terminals:
                data.append(done)
            data = np.concatenate(data, axis=1)
            data = torch.from_numpy(data).float()
            loss = self.train_on_batch(data, use_wandb=False)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def update_normalizer(self, buffer: ReplayBuffer, device=None):
        data = make_inputs_from_replay_buffer(buffer, self.model_terminals)
        data = torch.from_numpy(data).float()
        self.model.normalizer.reset(data)
        self.ema.ema_model.normalizer.reset(data)
        if device:
            self.model.normalizer.to(device)
            self.ema.ema_model.normalizer.to(device)
