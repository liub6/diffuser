# Shared functions for the CORL algorithms.
from typing import Union
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import gin
import gym
import numpy as np
import torch
from typing import Tuple

from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import make_inputs, construct_diffusion_model
from synther.diffusion.elucidated_diffusion import split_diffusion_samples

from dmc2gymnasium import DMCGym

TensorBatch = List[torch.Tensor]


@dataclass
class DiffusionConfig:
    path: Optional[str] = None  # Path to model checkpoints or .npz file with diffusion samples
    num_steps: int = 128  # Number of diffusion steps
    sample_limit: int = -1  # If not -1, limit the number of diffusion samples to this number

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0, keepdims=True)
    std = states.std(0, keepdims=True) + eps
    return mean, std

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


class RewardNormalizer:
    def __init__(self, dataset, env_name, max_episode_steps=1000):
        self.env_name = env_name
        self.scale = 1.
        self.shift = 0.
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
            self.scale = max_episode_steps / (max_ret - min_ret)
        elif "antmaze" in env_name:
            self.shift = -1.

    def __call__(self, reward):
        return (reward + self.shift) * self.scale


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = mean[None, :] if len(mean.shape) == 1 else mean
        self.std = std[None, :] if len(std.shape) == 1 else std

    def to_torch(self, device: str):
        self.mean = torch.tensor(self.mean, device=device)
        self.std = torch.tensor(self.std, device=device)
    
    def to_numpy(self):
        return StateNormalizer(self.mean.cpu().numpy(), self.std.cpu().numpy())

    def __call__(self, state):
        return (state - self.mean) / self.std


class ReplayBufferBase:
    def __init__(
            self,
            device: str = "cpu",
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        self.reward_normalizer = reward_normalizer
        self.state_normalizer = state_normalizer
        if self.state_normalizer is not None:
            self.state_normalizer.to_torch(device)
        self._device = device

    # Un-normalized samples.
    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        raise NotImplementedError

    def sample(self, batch_size: int, **kwargs) -> TensorBatch:
        states, actions, rewards, next_states, dones = self._sample(batch_size, **kwargs)
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer(rewards)
        if self.state_normalizer is not None:
            # print(states.shape)
            # print(self.state_normalizer.mean.shape)
            # print(self.state_normalizer.std.shape)
            states = self.state_normalizer(states)
            next_states = self.state_normalizer(next_states)

        return [states, actions, rewards, next_states, dones]


class ReplayBuffer(ReplayBufferBase):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        super().__init__(
            device, reward_normalizer, state_normalizer,
        )
        self._buffer_size = buffer_size
        self._pointer = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._contexts = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        
    @property
    def empty(self):
        return self._pointer == 0

    @property
    def full(self):
        return self._pointer == self._buffer_size

    def __len__(self):
        return self._pointer

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray], context_aware=False):
        if not self.empty:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        if context_aware:
            self._states[:n_transitions] = self._to_tensor(np.concatenate([data["observations"], data["contexts"]], axis=1))
        else:
            self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        if context_aware:
            self._next_states[:n_transitions] = self._to_tensor(np.concatenate([data["next_observations"], data["contexts"]], axis=1))
        else:
            self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        # self._contexts[:n_transitions] = self._to_tensor(data["contexts"])
        self._pointer = n_transitions

        print(f"Dataset size: {n_transitions}")

    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        indices = np.random.randint(0, self._pointer, size=batch_size)
        # indices = np.random.randint(0, self._pointer-batch_size, size=1)
        # indices = np.arange(indices[0], indices[0]+batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition_batch(self, batch: TensorBatch):
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]

        # If the buffer is full, do nothing.
        if self.full:
            return
        if self._pointer + batch_size > self._buffer_size:
            # Trim the samples to fit the buffer size.
            states = states[: self._buffer_size - self._pointer]
            actions = actions[: self._buffer_size - self._pointer]
            rewards = rewards[: self._buffer_size - self._pointer]
            next_states = next_states[: self._buffer_size - self._pointer]
            dones = dones[: self._buffer_size - self._pointer]
            batch_size = states.shape[0]

        self._states[self._pointer: self._pointer + batch_size] = states
        self._actions[self._pointer: self._pointer + batch_size] = actions
        self._rewards[self._pointer: self._pointer + batch_size] = rewards
        self._next_states[self._pointer: self._pointer + batch_size] = next_states
        self._dones[self._pointer: self._pointer + batch_size] = dones
        self._pointer += batch_size


class DiffusionGenerator(ReplayBufferBase):
    def __init__(
            self,
            env_name: str,
            dataset: Dict[str, np.ndarray],
            diffusion_path: str,
            use_ema: bool = True,
            num_steps: int = 32,
            batch_parallelism: int = 100,  # How many batches to generate each diffusion sample.
            device: str = "cpu",
            max_samples: int = -1,
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
            cond_dim: Optional[int] = None,
    ):
        super().__init__(
            device, reward_normalizer, state_normalizer,
        )
        # Create the environment
        if env_name == 'cartpole':
            self.env = DMCGym("cartpole", "swingup")
            # self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
            # inputs = make_inputs("dm-cartpole-test-length-all-v0")
            inputs = dataset
        else:
            self.env = gym.make(env_name)
            inputs = make_inputs(self.env)
        inputs = torch.from_numpy(inputs).float()
        self.diffusion = construct_diffusion_model(inputs=inputs, cond_dim=cond_dim).to(device)

        data = torch.load(diffusion_path)
        if use_ema:
            ema_dict = data['ema']
            ema_dict = {k: v for k, v in ema_dict.items() if k.startswith('ema_model')}
            ema_dict = {k.replace('ema_model.', ''): v for k, v in ema_dict.items()}
            self.diffusion.load_state_dict(ema_dict)
        else:
            self.diffusion.load_state_dict(data['model'])
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_steps = num_steps

        # Batching of diffusion samples
        self.batch_parallelism = batch_parallelism
        self.cache = []
        self.cache_pointer = 0

        # If max samples is not -1, then we will limit to that many unique samples.
        if max_samples != -1:
            print(f"Limiting to {max_samples} samples.")
            self.replay_buffer = ReplayBuffer(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.shape[0],
                buffer_size=max_samples,
                device=device,
                reward_normalizer=reward_normalizer,
                state_normalizer=state_normalizer,
            )
        else:
            self.replay_buffer = None

    def _sample_from_diffusion(self, batch_size: int, **kwargs) -> TensorBatch:
        sampled_outputs = self.diffusion.sample(
            batch_size=batch_size,
            num_sample_steps=self.num_steps,
            clamp=self.clamp_samples,
            **kwargs,
        )
        x = split_diffusion_samples(sampled_outputs, self.env)

        # Use the ground-truth done function if the diffusion model doesn't model it.
        if len(x) == 4:
            observations, actions, rewards, next_observations = x
            terminals = torch.zeros_like(next_observations[..., 0]).float()
        else:
            observations, actions, rewards, next_observations, terminals = x

        if self.replay_buffer is not None:
            self.replay_buffer.add_transition_batch(
                [observations, actions, rewards[..., None], next_observations, terminals[..., None]])
            print(f'Samples collected: {self.replay_buffer._pointer}.')
        return [observations, actions, rewards, next_observations, terminals]

    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        # If max samples reached, sample from replay buffer.
        if self.replay_buffer is not None and self.replay_buffer.full:
            return self.replay_buffer._sample(batch_size)

        # Otherwise, sample from diffusion.
        if self.batch_parallelism == 1:
            return self._sample_from_diffusion(batch_size, **kwargs)
        else:
            diffusion_sample_size = batch_size * self.batch_parallelism
            if len(self.cache) == 0 or self.cache_pointer == diffusion_sample_size:
                self.cache = self._sample_from_diffusion(diffusion_sample_size, **kwargs)
                self.cache_pointer = 0
            batch = [x[self.cache_pointer: self.cache_pointer + batch_size] for x in self.cache]
            self.cache_pointer += batch_size
            return batch

from tqdm import trange
from dm_control import suite

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

def filter_data(loss, percentile):
    pos1_percentile = np.percentile(loss[:, 0], percentile)

    pos2_percentile = np.percentile(loss[:, 1], percentile)

    pos3_percentile = np.percentile(loss[:, 2], percentile)

    vel1_percentile = np.percentile(loss[:, 3], percentile)

    vel2_percentile = np.percentile(loss[:, 4], percentile)

    reward_percentile = np.percentile(loss[:, 5], percentile)

    filtered_out_indices = np.where((loss[:, 0] >= pos1_percentile) & 
                                     (loss[:, 1] >= pos2_percentile) &
                                     (loss[:, 2] >= pos3_percentile) &
                                        (loss[:, 3] >= vel1_percentile) &
                                        (loss[:, 4] >= vel2_percentile) &
                                        (loss[:, 5] >= reward_percentile))[0]
    indices = np.where((loss[:, 0] <= pos1_percentile) &
                                     (loss[:, 1] <= pos2_percentile) &
                                     (loss[:, 2] <= pos3_percentile) &
                                        (loss[:, 3] <= vel1_percentile) &
                                        (loss[:, 4] <= vel2_percentile) &
                                        (loss[:, 5] <= reward_percentile))[0]
    # indices = np.where((loss[:, 0] <= 0.005) &
    #                                  (loss[:, 1] <= 0.005) &
    #                                  (loss[:, 2] <= 0.005) &
    #                                     (loss[:, 3] <= 0.01) &
    #                                     (loss[:, 4] <= 0.01) &
    #                                     (loss[:, 5] <= 0.001))[0]
    return filtered_out_indices, indices

def filter_by_boundary(training_dataset, diffusion_dataset):
    indices = np.arange(0,diffusion_dataset["observations"].shape[0])
    for key in diffusion_dataset.keys():
        if (key == "observations") | (key == "next_observations"):
            boundary_Pos1 = (training_dataset[key][:,0].min(), training_dataset[key][:,0].max())
            boundary_Pos2 = (training_dataset[key][:,1].min(), training_dataset[key][:,1].max())
            boundary_Pos3 = (training_dataset[key][:,2].min(), training_dataset[key][:,2].max())
            boundary_Vel1 = (training_dataset[key][:,3].min(), training_dataset[key][:,3].max())
            boundary_Vel2 = (training_dataset[key][:,4].min(), training_dataset[key][:,4].max())
            
            filtered_indices = np.where(((diffusion_dataset[key][:,0] >= boundary_Pos1[0]) & (diffusion_dataset[key][:,0] <= boundary_Pos1[1])) &
                                ((diffusion_dataset[key][:,1] >= boundary_Pos2[0]) & (diffusion_dataset[key][:,1] <= boundary_Pos2[1])) &
                                ((diffusion_dataset[key][:,2] >= boundary_Pos3[0]) & (diffusion_dataset[key][:,2] <= boundary_Pos3[1])))[0]
                                # ((diffusion_dataset[key][:,3] >= boundary_Vel1[0]) & (diffusion_dataset[key][:,3] <= boundary_Vel1[1])) &
                                # ((diffusion_dataset[key][:,4] >= boundary_Vel2[0]) & (diffusion_dataset[key][:,4] <= boundary_Vel2[1]))
            
        else:
            boundary = (training_dataset[key].min(), training_dataset[key].max())
            filtered_indices = np.where((diffusion_dataset[key] >= boundary[0]) & (diffusion_dataset[key] <= boundary[1]))[0]
        
        indices = np.intersect1d(indices, filtered_indices)
    
    return {key: diffusion_dataset[key][indices] for key in diffusion_dataset.keys()}

def prepare_replay_buffer(
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        dataset: Dict[str, np.ndarray],
        env_name: str,
        diffusion_config: DiffusionConfig,
        device: str = "cuda",
        reward_normalizer: Optional[RewardNormalizer] = None,
        state_normalizer: Optional[StateNormalizer] = None,
        cond_dim: Optional[int] = None,
        context_aware:bool = False,
        context: float = 1.0,
        percentile: Optional[int] = None,
        env = None,
        
):
    buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': state_normalizer,
        'device': device,
        }
    if diffusion_config.path is not None and context_aware:
        print(f'Loading diffusion dataset from {diffusion_config.path}.')
        diffusion_dataset = np.load(diffusion_config.path)
        diffusion_dataset = {key: diffusion_dataset[key] for key in diffusion_dataset.files}
        


        for key in diffusion_dataset.keys():
            diffusion_dataset[key] = diffusion_dataset[key][:dataset['rewards'].shape[0]]
        diffusion_length = diffusion_dataset['rewards'].shape[0]
            
        if percentile is not None:
            range = diffusion_dataset['rewards'].shape[0]
            # **********************************************************
            # if env is None:
            #     env = suite.load(domain_name="cartpole", task_name="swingup")
            # observation_erro, reward_erro = calculate_diffusion_loss(diffusion_dataset, env, range)
            # erro = np.concatenate([observation_erro, reward_erro], axis=1)
            # filtered_out_indices, indices = filter_data(erro, percentile)
            # diffusion_dataset = {key: diffusion_dataset[key][indices] for key in diffusion_dataset.keys()}
            # **********************************************************
            diffusion_dataset = filter_by_boundary(dataset, diffusion_dataset)
            print('Limited diffusion dataset to {} samples'.format(diffusion_dataset['rewards'].shape[0] / diffusion_length))
            
        # for key in dataset.keys():
        #     dataset[key] = dataset[key][:1]

        if 'contexts' not in diffusion_dataset:
            diffusion_dataset['contexts'] = np.ones((diffusion_dataset["rewards"].shape[0], dataset["contexts"].shape[1]), dtype=np.float32) * context
        if 'terminals' not in diffusion_dataset:
            diffusion_dataset['terminals'] = np.zeros((diffusion_dataset["rewards"].shape[0], dataset["contexts"].shape[1]), dtype=np.float32)
        # print(dataset['contexts'].shape, diffusion_dataset['contexts'].shape)
        for key in diffusion_dataset.keys():
            dataset[key] = np.concatenate([dataset[key], diffusion_dataset[key]], axis=0)
        # print(dataset['contexts'].shape, diffusion_dataset['contexts'].shape)
        state_mean_buffer, state_std_buffer = compute_mean_std(np.concatenate([dataset["observations"], dataset["contexts"]], axis=1), eps=1e-3)
        buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': StateNormalizer(state_mean_buffer, state_std_buffer),
        'device': device,
        }
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=dataset['rewards'].shape[0],
            **buffer_args,
        )
        replay_buffer.load_dataset(dataset, context_aware=context_aware)
        # replay_buffer.load_d4rl_dataset(diffusion_dataset, context_aware=context_aware)
    elif diffusion_config.path is None:
        print('Loading true dataset.')
        if context_aware:
            state_mean_buffer, state_std_buffer = compute_mean_std(np.concatenate([dataset["observations"], dataset["contexts"]], axis=1), eps=1e-3)
        else:
            state_mean_buffer, state_std_buffer = compute_mean_std(dataset["observations"], eps=1e-3)
            print("no context no diffusion")
        buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': StateNormalizer(state_mean_buffer, state_std_buffer),
        'device': device,
        }
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=dataset['rewards'].shape[0],
            **buffer_args,
        )
        replay_buffer.load_dataset(dataset, context_aware=context_aware)
    elif diffusion_config.path.endswith(".npz"):
        print(f'Loading diffusion dataset from {diffusion_config.path}.')
        diffusion_dataset = np.load(diffusion_config.path)
        diffusion_dataset = {key: diffusion_dataset[key] for key in diffusion_dataset.files}
        
        diffusion_length = diffusion_dataset['rewards'].shape[0]
        if percentile is not None:
            range = diffusion_dataset['rewards'].shape[0]
            if env is None:
                env = suite.load(domain_name="cartpole", task_name="swingup")
            observation_erro, reward_erro = calculate_diffusion_loss(diffusion_dataset, env, range)
            erro = np.concatenate([observation_erro, reward_erro], axis=1)
            filtered_data, indices = filter_data(erro, percentile)
            diffusion_dataset = {key: diffusion_dataset[key][indices] for key in diffusion_dataset.keys()}
            print('Limited diffusion dataset to {} samples'.format(diffusion_dataset['rewards'].shape[0] / diffusion_length))
            
        if 'contexts' not in diffusion_dataset:
            diffusion_dataset['contexts'] = np.ones((diffusion_dataset["rewards"].shape[0], dataset["contexts"].shape[1]), dtype=np.float32) * context
            
        if diffusion_config.sample_limit != -1:
            # Limit the number of samples
            for key in diffusion_dataset.keys():
                diffusion_dataset[key] = diffusion_dataset[key][:diffusion_config.sample_limit]
            print('Limited diffusion dataset to {} samples'.format(diffusion_config.sample_limit))

        dataset = diffusion_dataset
        state_mean_buffer, state_std_buffer = compute_mean_std(dataset["observations"], eps=1e-3)
        
        buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': StateNormalizer(state_mean_buffer, state_std_buffer),
        'device': device,
        }
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=diffusion_dataset['rewards'].shape[0],
            **buffer_args,
        )
        replay_buffer.load_dataset(dataset, context_aware=context_aware)
    elif diffusion_config.path.endswith(".pt"):
        print('Loading diffusion model.')
        # Load gin config from the same directory.
        gin_path = os.path.join(os.path.dirname(diffusion_config.path), 'config.gin')
        gin.parse_config_file(gin_path, skip_unknown=True)

        replay_buffer = DiffusionGenerator(
            env_name=env_name,
            dataset = dataset,
            diffusion_path=diffusion_config.path,
            use_ema=True,
            num_steps=diffusion_config.num_steps,
            max_samples=diffusion_config.sample_limit,
            cond_dim=cond_dim,
            **buffer_args,
        )
    else:
        raise ValueError("Unknown diffusion_path format")
    
    return buffer_args['state_normalizer'], replay_buffer
