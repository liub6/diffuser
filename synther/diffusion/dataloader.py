import random
import minari
import numpy as np
from tqdm.auto import trange
from torch.utils.data import IterableDataset
from typing import Any, DefaultDict, Dict, List, Tuple

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


# def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
#     cumsum = np.zeros_like(x)
#     cumsum[-1] = x[-1]
#     for t in reversed(range(x.shape[0] - 1)):
#         cumsum[t] = x[t] + gamma * cumsum[t + 1]
#     return cumsum


def load_trajectories(dataset_name: str) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    
    dataset = minari.load_dataset(dataset_name, download=False)
    # dataset.set_seed(seed=TrainConfig().train_seed)
    states, actions, rewards, contexts, episode_lens = [], [], [], [], []
    episodes_generator = dataset.iterate_episodes(episode_indices=dataset.episode_indices)
    for episode in episodes_generator:
        states.append(episode.observations[np.newaxis, :, :])
        actions.append(episode.actions[np.newaxis, :, :])
        if episode.rewards.ndim < 2:
            rewards.append(episode.rewards[np.newaxis, :, np.newaxis])
        elif episode.rewards.ndim == 2:
            rewards.append(episode.rewards[np.newaxis, :, :])
            
        if episode.infos['length'].ndim < 2:
            contexts.append(episode.infos['length'][np.newaxis, :, np.newaxis])
        elif episode.infos['length'].ndim == 2:
            contexts.append(episode.infos['length'][np.newaxis, :, :])
        episode_lens.append(episode.total_timesteps)
    states = np.concatenate(states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    rewards = np.concatenate(rewards).astype(np.float32)
    contexts = np.concatenate(contexts).astype(np.float32)
    
    # infos = {
    #     "states_min": states.min((0, 1), keepdims=True),
    #     "states_scale": states.max((0, 1), keepdims=True) - states.min((0, 1), keepdims=True),
    #     "actions_min": actions.min((0, 1), keepdims=True),
    #     "actions_scale": actions.max((0, 1), keepdims=True) - actions.min((0, 1), keepdims=True),
    #     "rewards_min": rewards.min((0, 1), keepdims=True),
    #     "rewards_scale": rewards.max((0, 1), keepdims=True) - rewards.min((0, 1), keepdims=True),
    #     "contexts_min": contexts.min((0, 1), keepdims=True),
    #     "contexts_scale": contexts.max((0, 1), keepdims=True) - contexts.min((0, 1), keepdims=True),
    #     "total_episodes": dataset.total_episodes,
    #     "episode_len": np.array(episode_lens),
    # }
    # infos = {
    #     "states_mean": states.mean((0, 1), keepdims=True) / 2,
    #     "states_std": states.max((0, 1), keepdims=True)  - states.min((0, 1), keepdims=True) + 1e-7,
    #     "actions_mean": actions.mean((0, 1), keepdims=True) / 2,
    #     "actions_std": actions.max((0, 1), keepdims=True)  - actions.min((0, 1), keepdims=True) + 1e-7,
    #     "rewards_mean": rewards.mean((0, 1), keepdims=True) / 2,
    #     "rewards_std": rewards.max((0, 1), keepdims=True)  - rewards.min((0, 1), keepdims=True) + 1e-7,
    #     "contexts_mean": contexts.mean((0, 1), keepdims=True) / 2,
    #     # "contexts_std": contexts.max((0, 1), keepdims=True)  - contexts.min((0, 1), keepdims=True) + 1e-7,
    #     "contexts_std": np.ones_like(contexts).mean((0, 1), keepdims=True),
    #     "total_episodes": dataset.total_episodes,
    #     "episode_len": np.array(episode_lens),
    # }
    infos = {
        # "states_mean": states.mean((0, 1), keepdims=True),
        # "states_std": states.std((0, 1), keepdims=True) + 1e-6,
        # "actions_mean": actions.mean((0, 1), keepdims=True),
        # "actions_std": actions.std((0, 1), keepdims=True) + 1e-6,
        # "rewards_mean": rewards.mean((0, 1), keepdims=True),
        # "rewards_std": rewards.std((0, 1), keepdims=True) + 1e-6,
        # "contexts_mean": contexts.mean((0, 1), keepdims=True),
        # "contexts_std": contexts.std((0, 1), keepdims=True) + 1e-6,
        "total_episodes": dataset.total_episodes,
        "episode_len": np.array(episode_lens),
    }
    # states = (states - infos["states_mean"]) / infos["states_std"]
    # actions = (actions - infos["actions_mean"]) / infos["actions_std"]
    # rewards = (rewards - infos["rewards_mean"]) / infos["rewards_std"]
    # contexts = (contexts - infos["contexts_mean"]) / infos["contexts_std"]

    # states = (states - infos["states_min"]) / (infos["states_scale"]) #if infos["states_scale"] != 0 else np.zeros_like(states)
    # actions = (actions - infos["actions_min"]) / (infos["actions_scale"]) #if infos["actions_scale"] != 0 else np.zeros_like(actions)
    # rewards = (rewards - infos["rewards_min"]) / (infos["rewards_scale"]) #if infos["rewards_scale"] != 0 else np.zeros_like(rewards)
    # contexts = (contexts - infos["contexts_min"]) / (infos["contexts_scale"]) if infos["contexts_scale"] != 0 else np.zeros_like(contexts)
    # contexts = (contexts - infos["contexts_min"]) / (1)
    
    return states, actions, rewards, contexts, infos

class SequenceDataset(IterableDataset):
    def __init__(self, dataset_name: str, seq_len: int = 5, reward_scale: float = 1.0):
        self.states, self.actions, self.rewards, self.contexts, self.infos = load_trajectories(dataset_name)
        
        # self.states_scale = self.infos["states_scale"]
        # self.actions_min = self.infos["actions_min"]
        # self.actions_scale = self.infos["actions_scale"]
        # self.rewards_min = self.infos["rewards_min"]
        # self.rewards_scale = self.infos["rewards_scale"]
        # self.contexts_min = self.infos["contexts_min"]
        # self.contexts_scale = self.infos["contexts_scale"]
        
        # self.reward_scale = reward_scale
        dim = max(self.states.shape[-1], self.actions.shape[-1], self.rewards.shape[-1])
        self.input_dim = dim
        # self.input_dim = (dim // 3 + 1) * 3 if dim % 3 != 0 else dim
        self.states = pad_along_axis(self.states, pad_to=self.input_dim, axis=2)
        self.actions = pad_along_axis(self.actions, pad_to=self.input_dim, axis=2)
        self.rewards = pad_along_axis(self.rewards, pad_to=self.input_dim, axis=2)
        
        self.states_min = self.states.min((0, 1), keepdims=True)
        self.states_scale = self.states.max((0, 1), keepdims=True) - self.states.min((0, 1), keepdims=True)
        self.actions_min = self.actions.min((0, 1), keepdims=True)
        self.actions_scale = self.actions.max((0, 1), keepdims=True) - self.actions.min((0, 1), keepdims=True)
        self.rewards_min = self.rewards.min((0, 1), keepdims=True)
        self.rewards_scale = self.rewards.max((0, 1), keepdims=True) - self.rewards.min((0, 1), keepdims=True)
        self.contexts_min = self.contexts.min((0, 1), keepdims=True)
        self.contexts_scale = self.contexts.max((0, 1), keepdims=True) - self.contexts.min((0, 1), keepdims=True)
        self.seq_len = seq_len
        self.reward_scale = reward_scale
        self.sample_prob = self.infos["episode_len"] / self.infos["episode_len"].sum()
        self.state_dim = self.states.shape[-1]
        self.action_dim = self.actions.shape[-1]
        self.reward_dim = self.rewards.shape[-1]
        self.context_dim = self.contexts.shape[-1]
        
        self.states = np.where(self.states_scale != 0, (self.states - self.states_min) / (self.states_scale), np.zeros_like(self.states))
        self.actions = np.where(self.actions_scale != 0, (self.actions - self.actions_min) / (self.actions_scale), np.zeros_like(self.actions))
        self.rewards = np.where(self.rewards_scale != 0, (self.rewards - self.rewards_min) / (self.rewards_scale), np.zeros_like(self.rewards))
        self.contexts = np.where(self.contexts_scale != 0, (self.contexts - self.contexts_min) / (self.contexts_scale), np.zeros_like(self.contexts))
        # self.states = (self.states - self.states_min) / (self.states_scale)
        # self.actions = (self.actions - self.actions_min) / (self.actions_scale)
        # self.rewards = (self.rewards - self.rewards_min) / (self.rewards_scale)
        # self.contexts = (self.contexts - self.contexts_min) / (self.contexts_scale)
        # self.state = (self.states - self.states.min((0, 1), keepdims=True)) / (self.states.max((0, 1), keepdims=True) - self.states.min((0, 1), keepdims=True))
        # self.action = (self.actions - self.actions.min((0, 1), keepdims=True)) / (self.actions.max((0, 1), keepdims=True) - self.actions.min((0, 1), keepdims=True))
        # self.reward = (self.rewards - self.rewards.min((0, 1), keepdims=True)) / (self.rewards.max((0, 1), keepdims=True) - self.rewards.min((0, 1), keepdims=True))
        # self.context = (self.contexts - self.contexts.min((0, 1), keepdims=True)) / (self.contexts.max((0, 1), keepdims=True) - self.contexts.min((0, 1), keepdims=True))

        
        # self.states_mean = self.infos["states_mean"]
        # self.states_std = self.infos["states_std"]
        # self.actions_mean = self.infos["actions_mean"]
        # self.actions_std = self.infos["actions_std"]
        # self.rewards_mean = self.infos["rewards_mean"]
        # self.rewards_std = self.infos["rewards_std"]
        # self.contexts_mean = self.infos["contexts_mean"]
        # self.contexts_std = self.infos["contexts_std"]
        
        
        self.total_episodes = self.infos["total_episodes"]
        self.episode_len = self.infos["episode_len"]

        
    def recover_data(self, sequence):
        sequence = sequence.transpose(0, 2, 1).reshape(-1, self.seq_len, 3, self.input_dim)
        states = sequence[:, :, 0, :]
        actions = sequence[:, :, 1, :]
        rewards = sequence[:, :, 2, :] * self.reward_scale

        states = states * self.states_scale + self.states_min
        actions = actions * self.actions_scale + self.actions_min
        rewards = rewards * self.rewards_scale + self.rewards_min
        return states, actions, rewards
        # states = states * self.states_std + self.states_mean
        # actions = actions * self.actions_std + self.actions_mean
        # rewards = rewards * self.rewards_std + self.rewards_mean
        # # contexts = contexts * self.contexts_std + self.contexts_mean
        # return states, actions, rewards
    
    def recover_states(self, states):
        return states * self.states_scale + self.states_min
        # return states * self.states_std + self.states_mean
    def recover_actions(self, actions):
        return (actions + 1) / 2 * self.actions_scale + self.actions_min
        # return actions * self.actions_std + self.actions_mean
    def recover_rewards(self, rewards):
        return (rewards + 1) / 2 * self.rewards_scale + self.rewards_min
        # return rewards * self.rewards_std + self.rewards_mean
    
    def calculate_contexts(self, contexts):
        return (contexts) / 2 * 1 + self.contexts_min
        # return contexts - self.contexts_mean / self.contexts_std
        
    def __prepare_sample(self, episode_idx, start_idx):
        states = self.states[episode_idx, start_idx : start_idx + self.seq_len, :]
        actions = self.actions[episode_idx, start_idx : start_idx + self.seq_len, :]
        rewards = self.rewards[episode_idx, start_idx : start_idx + self.seq_len, :] * self.reward_scale
        # contexts = self.contexts[episode_idx, start_idx : start_idx + self.seq_len, :]
        contexts = self.contexts[episode_idx, start_idx, :][np.newaxis, :]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        
        # pad up to seq_len if needed, padding is masked during training
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            rewards = pad_along_axis(rewards, pad_to=self.seq_len)
        
        # print(np.stack([states, actions, rewards], axis=1).shape)
        sequence = (
            np.stack([states, actions, rewards], axis=0)
            .transpose(1, 0, 2)
            .reshape(3 * self.seq_len, self.input_dim)
        ).transpose(1, 0)
        # sequence = (
        #     np.stack([states, actions, rewards], axis=0)
        #     .reshape(states.shape[0] * 3,  states.shape[1]).transpose(1, 0)
        # )
        return sequence , mask
        # return states, actions, rewards, contexts, time_steps, mask

    def __iter__(self):
        while True:
            episodes_idx = np.random.choice(self.infos["total_episodes"], p=self.sample_prob)
            start_idx = random.randint(0, self.infos["episode_len"][episodes_idx] - self.seq_len)
            yield self.__prepare_sample(episodes_idx, start_idx)