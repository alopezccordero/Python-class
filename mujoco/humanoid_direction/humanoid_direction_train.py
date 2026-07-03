import register_env
from stable_baselines3 import PPO
import gymnasium as gym


env = gym.make("HumanoidDirection-v0")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    verbose=1
)

model.learn(total_timesteps=5_000_000)

model.save("models/ppo_humanoid_direction_5m")

