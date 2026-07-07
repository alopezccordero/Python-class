import register_env
import gymnasium as gym
import numpy as np
import torch

from amp_discriminator import AMPDiscriminator


class AMPHumanoidEnv(gym.Wrapper):
    def __init__(
        self,
        env_id="HumanoidDirection-v0",
        render_mode=None,
        discriminator_path="amp_discriminator.pt",
        amp_weight=0.2,
        device="cpu",
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.prev_amp_obs = None
        self.amp_weight = amp_weight
        self.device = device

        self.disc = AMPDiscriminator(input_dim=90).to(self.device)
        self.disc.load_state_dict(torch.load(discriminator_path, map_location=self.device))
        self.disc.eval()

    def get_amp_obs(self):
        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        return np.concatenate([qpos[2:], qvel]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_amp_obs = self.get_amp_obs()
        return obs, info

    def step(self, action):
        obs, task_reward, terminated, truncated, info = self.env.step(action)

        current_amp_obs = self.get_amp_obs()

        amp_transition = np.concatenate([
            self.prev_amp_obs,
            current_amp_obs,
        ]).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(amp_transition, dtype=torch.float32, device=self.device).unsqueeze(0)
            amp_reward = self.disc.amp_reward(x).item()

        total_reward = task_reward + self.amp_weight * amp_reward

        info["task_reward"] = task_reward
        info["amp_reward"] = amp_reward
        info["total_reward"] = total_reward
        info["amp_transition"] = amp_transition

        self.prev_amp_obs = current_amp_obs

        return obs, total_reward, terminated, truncated, info