from pathlib import Path

import gymnasium as gym
import torch

import register_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from amp_callback import AMPDiscriminatorCallback
from amp_discriminator import AMPDiscriminator
from amp_env import AMPHumanoidEnv
from motion_lib import MotionLib


TOTAL_TIMESTEPS = 20_000_000
N_ENVS = 8
CHECKPOINT_EVERY = 500_000

# Paper-closer AMP settings.
AMP_WEIGHT = 0.2
REFERENCE_STATE_INIT_PROB = 0.3
DISCRIMINATOR_LR = 3e-5
DISCRIMINATOR_HIDDEN_DIM = 512

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

# Match expert AMP transition spacing to the environment control timestep.
_tmp_env = gym.make("HumanoidDirection-v0")
ENV_DT = float(_tmp_env.unwrapped.dt)
_tmp_env.close()
print("Environment dt:", ENV_DT)

motion_lib = MotionLib(
    str(BASE_DIR / "retargeted_pkl"),
    transition_dt=ENV_DT,
)
amp_mean, amp_std = motion_lib.compute_amp_stats(num_samples=10000)

disc = AMPDiscriminator(
    input_dim=90,
    hidden_dim=DISCRIMINATOR_HIDDEN_DIM,
).to(device)
disc_optimizer = torch.optim.Adam(
    disc.parameters(),
    lr=DISCRIMINATOR_LR,
    weight_decay=1e-4,
)


def make_env(rank):
    def _init():
        env = AMPHumanoidEnv(
            discriminator=disc,
            motion_lib=motion_lib,
            amp_weight=AMP_WEIGHT,
            device=device,
            amp_mean=amp_mean,
            amp_std=amp_std,
            reference_state_init_prob=REFERENCE_STATE_INIT_PROB,
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
    train_freq=16384,
    save_freq=CHECKPOINT_EVERY,
    save_path=str(CHECKPOINT_DIR),
    device=device,
    amp_mean=amp_mean,
    amp_std=amp_std,
    fake_replay_size=100000,
    gradient_penalty_weight=1.0,
    score_reg_weight=1e-4,
    max_grad_norm=1.0,
    real_label=1.0,
    fake_label=-1.0,
)

checkpoint_callback = CheckpointCallback(
    save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_real_amp_filtered_softer",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

callbacks = CallbackList([
    amp_callback,
    checkpoint_callback,
])

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(
        pi=[1024, 512],
        vf=[1024, 512],
    ),
)

model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=5e-5,
    n_steps=2048,
    batch_size=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    target_kl=0.03,
    clip_range=0.1,
    n_epochs=5,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=str(TENSORBOARD_DIR),
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
)

model.save(str(MODEL_DIR / "ppo_humanoid_direction_real_amp_filtered_softer"))
torch.save(disc.state_dict(), MODEL_DIR / "amp_discriminator_real_amp_softer.pt")

env.close()

print("Saved final PPO model and discriminator.")
