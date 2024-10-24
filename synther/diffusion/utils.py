# Utilities for diffusion.
from typing import Optional, List, Union

# import d4rl
import gin
import gym
import numpy as np
import torch
from torch import nn

# GIN-required Imports.
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory

import minari
from collections import Counter

# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs

def make_inputs(
        dataset_name: str,
        minari: bool = False,
        modelled_terminals: bool = False,
        original: bool = False,
        context: bool = False,
        top: bool = False,
        uniform: bool = False,
        segment: Optional[str] = None

) -> np.ndarray:
    
    if minari:
        dataset = minari.load_dataset(dataset_name, download=False)
        obs, actions, rewards, next_obs, contexts, episode_lens, dones = [], [], [], [], [], [], []

        # ********************************************************************************************************************
        if top:
            num_ep_per_split = 250
            num_top = 50
            top_episodes = []
            n = dataset.total_episodes // num_ep_per_split
            for i in range(n):
                indices = np.arange(0, num_ep_per_split) + i * num_ep_per_split
                episodes_generator = dataset.iterate_episodes(episode_indices=indices)
                episodes_with_rewards = []
                for episode in episodes_generator:
                    total_reward = sum(episode.rewards)
                    episodes_with_rewards.append((episode, total_reward))
                episodes_with_rewards.sort(key=lambda x: x[1], reverse=True)
                top_episodes_with_rewards = episodes_with_rewards[:num_top]
                top_episodes += [episode for episode, _ in top_episodes_with_rewards]
            for episode in top_episodes:
                obs.append(episode.observations[:-1])
                actions.append(episode.actions)
                rewards.append(episode.rewards)
                next_obs.append(episode.observations[1:])
                dones.append(episode.terminations)
                contexts.append(episode.infos['length'][:-1])
                # obs.append(episode.observations[:300])
                # actions.append(episode.actions[:300])
                # rewards.append(episode.rewards[:300])
                # next_obs.append(episode.observations[1:301])
                # dones.append(episode.terminations[:300])
                # contexts.append(episode.infos['length'][:300])
        # ********************************************************************************************************************
        else:
            episodes_generator = dataset.iterate_episodes(episode_indices=dataset.episode_indices)
            for i, episode in enumerate(episodes_generator):
                if i % 500 >= 250:
                    obs.append(episode.observations[:-1])
                    actions.append(episode.actions)
                    rewards.append(episode.rewards)
                    next_obs.append(episode.observations[1:])
                    dones.append(episode.terminations)
                    contexts.append(episode.infos['length'][:-1])
                # obs.append(episode.observations[:450])
                # actions.append(episode.actions[:450])
                # rewards.append(episode.rewards[:450])
                # next_obs.append(episode.observations[1:451])
                # dones.append(episode.terminations[:450])
                # contexts.append(episode.infos['length'][:450])
        # ******************************************************************************************************************** 
        obs = np.concatenate(obs).astype(np.float32)
        actions = np.concatenate(actions).astype(np.float32)
        rewards = np.concatenate(rewards).astype(np.float32)
        next_obs = np.concatenate(next_obs).astype(np.float32)
        terminals = np.concatenate(dones).astype(np.float32)
        contexts = np.concatenate(contexts).astype(np.float32)[:, None]
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        
    else:
        dataset = np.load("/scratch/work/liub6/diffusionRL/diffuser/synther/dataset/50*998episodes.npz")
        dataset = {key: dataset[key] for key in dataset.files}
        obs = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        next_obs = dataset['next_observations']
        terminals = dataset['terminals']
        contexts = dataset['contexts']
    

    
    if uniform:
        cat = np.concatenate([obs, actions, rewards[..., None], next_obs, terminals[..., None], contexts], axis=1)
        original_data = rewards
        num_bins =50
        bins = np.linspace(0, 1, num_bins + 1)
        digitized = np.digitize(original_data, bins) - 1  # 将数据分配到桶中
        data_counter = Counter(digitized)
        target_frequency = max(data_counter.values()) // 10
        
        new_data = cat[0:1, :]
        for bin_index, count in data_counter.items():
            index = np.where(digitized == bin_index)
            if count < target_frequency:
                new_data = np.concatenate([new_data, cat[np.random.choice(index[0], target_frequency, replace=True), :]], axis=0)
            else:
                new_data = np.concatenate([new_data, cat[index[0]]], axis=0)

        new_data = np.array(new_data)

        obs = new_data[:, 0:5]
        actions =  new_data[:, 5:6]
        rewards =  new_data[:, 6]
        next_obs =  new_data[:, 7:12]
        terminals =  new_data[:, 12]
        contexts =  new_data[:, 13:14]
    
    # print("#" + segment + "#")
    if segment is not None:
        if segment == 'front':
            index = np.where((contexts > 0.15) & (contexts < 0.45))
        elif segment == 'middle':
            index = np.where((contexts > 0.25) & (contexts < 0.55))
        elif segment == 'rear':
            index = np.where((contexts > 0.35) & (contexts < 0.65))
        elif segment == 'extremes':
            index = np.where((contexts < 0.25) | (contexts > 0.55))
        
        obs = obs[index[0]]
        actions = actions[index[0]]
        rewards = rewards[index[0]]
        next_obs = next_obs[index[0]]
        terminals = terminals[index[0]]
        contexts = contexts[index[0]]
        
    
    if original:
        return {'observations': obs, 'actions': actions, 'rewards': rewards, 'next_observations': next_obs, 'terminals': terminals, 'contexts': contexts}   
    else:
        inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
        # inputs = np.concatenate([obs[0:-1], actions[0:-1], rewards[:, None][0:-1], next_obs[0:-1]], axis=1)
        # inputs_next = np.concatenate([actions[1:], rewards[:, None][1:], next_obs[1:]], axis=1)
        # inputs = np.concatenate([inputs, inputs_next], axis=1)
        
        # inputs = np.delete(inputs, np.arange(0, inputs.shape[0], 1000), axis=0)
        # contexts = np.delete(contexts, -1, axis=0)
        # contexts = np.delete(contexts, np.arange(0, contexts.shape[0], 1000), axis=0)
        
        if modelled_terminals:
            inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
        if context:
            return inputs, contexts
        else:
            return inputs


@gin.configurable
def construct_diffusion_model(
        inputs: Union[torch.Tensor, list],
        normalizer_type: str,
        denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    if type(inputs) == list:
        inputs = inputs[0]
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )
