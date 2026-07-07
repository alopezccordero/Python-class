import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from amp_env import AMPHumanoidEnv
from amp_discriminator import AMPDiscriminator
from motion_lib import MotionLib
from amp_callback import AMPDiscriminatorCallback


os.makedirs("models", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

motion_lib = MotionLib("retargeted_pkl")

disc = AMPDiscriminator(input_dim=90).to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)

env = AMPHumanoidEnv(
    discriminator=disc,
    amp_weight=0.2,
    device=device,
)

env = Monitor(env)

callback = AMPDiscriminatorCallback(
    motion_lib=motion_lib,
    discriminator=disc,
    optimizer=disc_optimizer,
    batch_size=256,
    updates_per_call=4,
    train_freq=2048,
    device=device,
)

model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard_real_amp/",
)

model.learn(
    total_timesteps=20_000_000,
    callback=callback,
)

model.save("models/ppo_humanoid_direction_real_amp")
torch.save(disc.state_dict(), "models/amp_discriminator_real_amp.pt")

env.close()

print("Saved PPO model and discriminator.")