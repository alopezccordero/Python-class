import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.utils import EzPickle

class HumanoidDirectionEnv(HumanoidEnv, EzPickle):
    def __init__(self, direction=(1.0, 0.0), **kwargs):
        self.target_dir = np.array(direction, dtype=np.float32)
        self.target_dir /= np.linalg.norm(self.target_dir)

        EzPickle.__init__(self, direction, **kwargs)
        super().__init__(**kwargs)

    def reset_model(self):
        obs = super().reset_model()

        #random direction every episode
        angle = self.np_random.uniform(-np.pi, np.pi)

        self.target_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        return self._get_obs_with_direction(obs)
    
    def step(self, action):
        xy_before = self.data.xpos[self.model.body("torso").id][:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_after = self.data.xpos[self.model.body("torso").id][:2].copy()
        xy_velocity = (xy_after - xy_before) / self.dt

        direction_reward = np.dot(xy_velocity, self.target_dir)
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0
        ctrl_cost = self.control_cost(action)

        reward = 5.0 * direction_reward + healthy_reward - ctrl_cost

        terminated = self.terminated if not self.is_healthy else False
        obs = self._get_obs_with_direction(self._get_obs())

        info = {
            "xy_velocity": xy_velocity,
            "target_dir": self.target_dir,
            "direction_reward": direction_reward,
        }

        return obs, reward, terminated, False, info
    
    def _get_obs_with_direction(self, humanoid_obs):
        return np.concatenate([humanoid_obs, self.target_dir]).astype(np.float64)