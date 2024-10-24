# Train diffusion model on D4RL transitions.
import argparse
import pathlib
import re

# import d4rl
import gin
import gymnasium as gym
import numpy as np
import torch
import wandb

from synther.diffusion.elucidated_diffusion import Trainer, SimpleDiffusionGenerator

from synther.diffusion.utils import make_inputs, construct_diffusion_model

from dmc2gymnasium import DMCGym

from torch.utils.data import random_split

from dm_control import suite



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dm-cartpole-test-length0.025-0.35-v0")
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['../config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    # wandb config
    parser.add_argument('--wandb-project', type=str, default="Diffusion")
    parser.add_argument('--wandb-entity', type=str, default="")
    parser.add_argument('--wandb-group', type=str, default="Dataset")
    #
    parser.add_argument('--segment', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_samples', type=int, default=int(0))
    parser.add_argument('--num_transition', type=int, default=int(1))
    parser.add_argument('--train_num_steps', type=int, default=int(3e5))
    parser.add_argument('--save_num_samples', type=int, default=int(5e6))
    parser.add_argument('--save_file_name', type=str, default='5m_samples.npz')
    parser.add_argument('--load_checkpoint', type=int, default=int(0))
    parser.add_argument('--minari', type=int, default=int(1))
    parser.add_argument('--cond', type=float, nargs='+', default=None)
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    match = re.search(r'length(.+?)-v0', args.dataset)
    if match:
        result = match.group(1)
        args.wandb_group = f'Dataset{result}'
    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    if args.minari:
        env = DMCGym("cartpole", "swingup", task_kwargs={'random':args.seed})
        env = gym.wrappers.RecordEpisodeStatistics(env)
        inputs = make_inputs(args.dataset, context=True, segment=args.segment)
        inputs = torch.from_numpy(inputs[0]).float(), torch.from_numpy(inputs[1]).float()
        # print(inputs[0].shape, inputs[1].shape)
        dataset = torch.utils.data.TensorDataset(*inputs)
        print("save_samples: ", args.save_samples)
        print("load_checkpoint: ", args.load_checkpoint)
        train_size = int(0.8 * len(dataset))  # 80% 用于训练
        test_size = len(dataset) - train_size  # 20% 用于测试
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        env = gym.make(args.dataset)
        inputs = make_inputs(env)
        inputs = torch.from_numpy(inputs).float()
        dataset = torch.utils.data.TensorDataset(inputs)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(inputs=inputs[0] if args.minari else inputs, cond_dim=len(args.cond) if args.cond is not None else None)
    trainer = Trainer(
        diffusion,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        results_folder=args.results_folder,
        train_num_steps=args.train_num_steps,
        env = suite.load(domain_name="cartpole", task_name="swingup")
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=args.results_folder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        trainer.ema.to(trainer.accelerator.device)
        # Load the last checkpoint.
        trainer.load(milestone=trainer.train_num_steps)

    # Generate samples and save them.
    if args.save_samples:
        generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=trainer.ema.ema_model,
        )
        observations, actions, rewards, next_observations, terminals = generator.sample(
            num_samples=args.save_num_samples,
            cond=torch.tensor(args.cond, dtype=torch.float32)[:, None] if args.cond is not None else None,
            num_transition=args.num_transition,
        )
        np.savez_compressed(
            results_folder / (args.save_file_name + "_" + "_".join(map(str, args.cond)) if args.cond else ""),
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )
