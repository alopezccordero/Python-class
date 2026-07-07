import register_env
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("HumanoidDirection-v0")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    verbose=1
)
checkpoint_callback = CheckpointCallback(
    save_freq=500_00,
    save_path="models/checkpoints_amp_fixed",
    name_prefix="ppo_amp_fixed_humanoiddir"
)
model.learn(total_timesteps=20_000_000)

model.save("models/ppo_amp_fixed_humanoiddir_20m")

