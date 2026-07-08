import os
from pathlib import Path

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from amp_env import AMPHumanoidEnv
from amp_discriminator import AMPDiscriminator
from motion_lib import MotionLib
from amp_callback import AMPDiscriminatorCallback


TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8
CHECKPOINT_EVERY = 500_000

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
TENSORBOARD_DIR = BASE_DIR / "tensorboard_real_amp_filtered"

MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
TENSORBOARD_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

motion_lib = MotionLib(str(BASE_DIR / "retargeted_pkl"))
amp_mean, amp_std = motion_lib.compute_amp_stats(num_samples=10000)

disc = AMPDiscriminator(input_dim=90).to(device)
disc_optimizer = torch.optim.Adam(
    disc.parameters(),
    lr=1e-5,
    weight_decay=1e-4,
)


def make_env(rank):
    def _init():
        env = AMPHumanoidEnv(
            discriminator=disc,
            amp_weight=0.2,
            device=device,
            amp_mean=amp_mean,
            amp_std=amp_std,
        )
        env = Monitor(env)
        return env

    return _init


env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

amp_callback = AMPDiscriminatorCallback(
    motion_lib=motion_lib,
    discriminator=disc,
    optimizer=disc_optimizer,
    batch_size=256,
    updates_per_call=1,
    train_freq=8192,
    save_freq=CHECKPOINT_EVERY,
    save_path=str(CHECKPOINT_DIR),
    device=device,
    amp_mean=amp_mean,
    amp_std=amp_std,
    fake_replay_size=50000,
    gradient_penalty_weight=10.0,
    real_label=0.9,
    fake_label=0.1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_real_amp_filtered",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

callbacks = CallbackList([
    amp_callback,
    checkpoint_callback,
])

model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(TENSORBOARD_DIR),
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
)

model.save(str(MODEL_DIR / "ppo_humanoid_direction_real_amp_filtered"))
torch.save(disc.state_dict(), MODEL_DIR / "amp_discriminator_real_amp_final.pt")

env.close()

print("Saved final PPO model and discriminator.")