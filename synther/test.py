import pyrallis
from dataclasses import asdict, dataclass, field
from synther.corl.shared.buffer import DiffusionConfig
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
from typing import Any, Dict, List, Optional, Tuple, Union

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


@dataclass
class TrainConfig:
    # Experiment
    context_aware: int = 0
    diffuser: bool = True
    context_and_diffuser: bool = False
    save_checkpoints: bool = False  # Save model checkpoints
    log_every: int = 1000
    load_model: str = ""  # Model load file name, "" doesn't load
    pole_length: float = None  #half pole length. 0.045 defalt
    # TD3
    checkpoints_path: Optional[str] = None  # Save path
    seed: int = 0  # Random seed
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
    project: str = "DiffusionRL"
    group: str = "TD3_BC"
    name: str = "Dataset0.025-0.35"
    # Diffusion config
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    # Network size
    network_width: int = 256
    network_depth: int = 2
    dataset: str = "dm-cartpole-test-length0.025-0.35-v0"

    def __post_init__(self):
        env = DMCGym("cartpole", "swingup", task_kwargs={'random':self.seed})
        print(type(self.diffusion.path))
        if self.diffusion.path == "None":
            self.diffusion.path = None
        self.diffuser = True if self.diffusion.path is not None else False
        self.context_and_diffuser = self.context_aware and self.diffuser
        if self.context_and_diffuser:
            print("context_and_diffuser") 
        elif self.context_aware:
            print("context_aware")
        elif self.diffuser:
            print("diffuser")
            
        if self.pole_length is not None:
            self.cond = [set_pole_length(new_size = self.pole_length)]
            
@pyrallis.wrap()
def train(config: TrainConfig):
    print("hello")
    print(os.path.join(tmp_dir, 'dm_control/suite/cartpole.xml'))
    print(config.diffusion.path)
    print(type(config.diffusion.path))
    
if __name__ == "__main__":
    train()