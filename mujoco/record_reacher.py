import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

model = PPO.load("models/ppo_reacher_v5")

env = gym.make(
    "Reacher-v5",
    render_mode="rgb_array"
)

obs, info = env.reset()

frames = []
episode_reward = 0

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        break

env.close()

imageio.mimsave(
    "reacher.mp4",
    frames,
    fps=30
)

print(f"Episode reward: {episode_reward:.2f}")
print("Saved reacher.mp4")
