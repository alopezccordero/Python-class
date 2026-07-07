from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from amp_env import AMPHumanoidEnv

env = AMPHumanoidEnv(
    discriminator_path="amp_discriminator.pt",
    amp_weight=0.2,
    device="cpu"
)

env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard_amp/"
)

model.learn(total_timesteps=20_000_000)
model.save("models/ppo_humanoid_direction_amp_fixed_disc")

env.close()