import os
os.environ["MUJOCO_GL"]= "egl"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = gym.make("Reacher-v5")
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/reacher_ppo",
    device="cpu"
)

model.learn(total_timesteps=500_000)
model.save("models/ppo_reacher_v5")

env.close()
print("saved model to models/ppo_reacher_v5.zip")

