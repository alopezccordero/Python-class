import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("models/ppo_reacher_v5")
env = gym.make("Reacher-v5")

obs, info = env.reset()

episode_reward = 0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward +=reward

    if terminated or truncated:
        break

print(f"Episode reward: {episode_reward:.2f}")

env.close()
