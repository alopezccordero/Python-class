import os
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

os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

motion_lib = MotionLib("retargeted_pkl")

disc = AMPDiscriminator(input_dim=90).to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=5e-5)


def make_env(rank):
    def _init():
        env = AMPHumanoidEnv(
            discriminator=disc,
            amp_weight=0.02,
            device=device,
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
    updates_per_call=2,
    train_freq=2048,
    save_freq=CHECKPOINT_EVERY,
    save_path="./models/checkpoints",
    device=device,
)

checkpoint_callback = CheckpointCallback(
    save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
    save_path="./models/checkpoints",
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
    tensorboard_log="./tensorboard_real_amp_filtered/",
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
)

model.save("models/ppo_humanoid_direction_real_amp_filtered")
torch.save(disc.state_dict(), "models/amp_discriminator_real_amp_final.pt")

env.close()

print("Saved final PPO model and discriminator.")