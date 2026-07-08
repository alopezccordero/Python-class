import register_env
import gymnasium as gym
import numpy as np
import torch


class AMPHumanoidEnv(gym.Wrapper):
    def __init__(
        self,
        discriminator,
        motion_lib=None,
        env_id="HumanoidDirection-v0",
        render_mode=None,
        amp_weight=0.5,
        device="cpu",
        amp_mean=None,
        amp_std=None,
        reference_state_init_prob=0.3,
    ):
        env = gym.make(env_id, render_mode=render_mode)
        super().__init__(env)

        self.disc = discriminator
        self.motion_lib = motion_lib
        self.amp_weight = amp_weight
        self.device = device
        self.prev_amp_obs = None
        self.fake_amp_transitions = []
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.reference_state_init_prob = reference_state_init_prob

    def get_amp_obs(self):
        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        return np.concatenate([qpos[2:], qvel]).astype(np.float32)

    def _current_policy_obs(self):
        """Return policy observation after manually setting a reference state."""
        unwrapped = self.env.unwrapped
        humanoid_obs = unwrapped._get_obs()
        if hasattr(unwrapped, "_get_obs_with_direction"):
            return unwrapped._get_obs_with_direction(humanoid_obs)
        return humanoid_obs

    def maybe_reference_state_init(self):
        if self.motion_lib is None or self.reference_state_init_prob <= 0.0:
            return False

        rng = self.env.unwrapped.np_random
        if rng.random() >= self.reference_state_init_prob:
            return False

        qpos, qvel = self.motion_lib.sample_reference_state()
        self.env.unwrapped.set_state(qpos, qvel)
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        used_reference_state = self.maybe_reference_state_init()
        if used_reference_state:
            obs = self._current_policy_obs()

        self.prev_amp_obs = self.get_amp_obs()
        info["reference_state_init"] = used_reference_state
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

        was_training = self.disc.training
        self.disc.eval()
        with torch.no_grad():
            x = torch.tensor(amp_transition_for_reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            amp_reward = self.disc.amp_reward(x).item()
        if was_training:
            self.disc.train()

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
