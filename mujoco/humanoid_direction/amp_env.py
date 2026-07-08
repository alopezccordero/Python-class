import register_env
import gymnasium as gym
import numpy as np
import torch


class AMPHumanoidEnv(gym.Wrapper):
    def __init__(
        self,
        discriminator,
        env_id="HumanoidDirection-v0",
        render_mode=None,
        amp_weight=0.2,
        device="cpu",
        amp_mean=None,
        amp_std=None,
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.disc = discriminator
        self.amp_weight = amp_weight
        self.device = device
        self.prev_amp_obs = None
        self.fake_amp_transitions = []
        self.amp_mean = amp_mean
        self.amp_std = amp_std

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

        self.fake_amp_transitions.append(amp_transition)

        amp_transition_for_reward = amp_transition
        if self.amp_mean is not None and self.amp_std is not None:
            amp_transition_for_reward = (amp_transition_for_reward - self.amp_mean) / self.amp_std

        with torch.no_grad():
            x = torch.tensor(amp_transition_for_reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            amp_reward = self.disc.amp_reward(x).item()

        total_reward = task_reward + self.amp_weight * amp_reward

        info["task_reward"] = task_reward
        info["amp_reward"] = amp_reward
        info["total_reward"] = total_reward
        info["amp_transition"] = amp_transition

        self.prev_amp_obs = current_amp_obs

        return obs, total_reward, terminated, truncated, info

    def pop_fake_transitions(self):
        transitions = self.fake_amp_transitions
        self.fake_amp_transitions = []
        return transitions