import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.utils import EzPickle
from gymnasium import spaces

class HumanoidDirectionEnv(HumanoidEnv, EzPickle): # my class inherits from humanoid prebuilt env
    def __init__(self, direction=(1.0, 0.0), **kwargs):
        #direction in x. since 1, 0. 0, 1 is y.
        #1, 0 -> right
        #-1, 0 -> left
        #0, 1 -> forward
        #0, -1 backward
        #1, 0 is just initialization
        self.target_dir = np.array(direction, dtype=np.float32)
        #convert tuple into a numpy vector
        #dtype=np.float32 is store every number as a 2 bit floating point number
        #libraries like pytorch use float32 instead of float64
        self.target_dir /= np.linalg.norm(self.target_dir)
        #normalization of the vector. linalg.norm computes
        #the vector length. [3,4]-> 5. normalized is 0.6, 0.8
        # #initialize ezpickle so gymnasium can reconstruct this object
        #np.linalg.norm computes distance which is sqr root of each number
        #squared. 1, 0 would be 1. 3, 4 would be 5
        #python divides every element of the vector by its length
        #to normalize values. 
        #this is since we want only direction, not magnitude.
        #10, 0 would be the direction of 1.0, 0.
        
        EzPickle.__init__(self, direction, **kwargs)

        #initialzie the original humanoid env
        #with model, data, observation space, action space and renderer
        super().__init__(**kwargs)

        old_space = self.observation_space
        #modify the observation space

        self.observation_space = spaces.Box(
            low=-np.inf, # lowest obs val can be neg infinity
            high=np.inf,#highest obs val can be positive infinity
            shape=(old_space.shape[0]+2,), #348 + 2 = 350 obs space
            dtype=np.float64 #convert to float64
        )

    def reset_model(self):
        obs = super().reset_model()
        #when reseting model we reset the observation

        #random direction every episode
        angle = self.np_random.uniform(-np.pi, np.pi)
        #we obtain random angles
        #one random angle between -pi to pi.
        self.target_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        #we update target dir to the cosine of the angle as first value
        # and sin of angle as second value
        # we use type float32
        return self._get_obs_with_direction(obs)
        #concatenates observation with direciton


    def step(self, action):
        xy_before = self.data.xpos[self.model.body("torso").id][:2].copy()
        #gets the world position of the torso
        #saves x and y coordinate only (not z) in xy_before.
        #xy before is coordinate of torso before action
        self.do_simulation(action, self.frame_skip)
        #applies the action given by policy to the robot and lets
        #mujoco simulate the physics for several time steps

        xy_after = self.data.xpos[self.model.body("torso").id][:2].copy()
        #coordinate of torso after action.
        #coordinate in x and y. world position.

        #xy velocity is a vector. it stores velocity in x and velocity in y
        xy_velocity = (xy_after - xy_before) / self.dt
        #calculates torso velocity by substaction xy_after - xy_before divvided by
        #simulation time. self.dt = simulation time. distance over time = velocity

        #this direction_reward computes how much of humanoids vel is in the targets direction
        #the dot prduct is positive moving toward the target
        #zero if is moviing perpendicular to it
        #negative if is moving away from it.
        direction_reward = np.dot(xy_velocity, self.target_dir)
        #direction reward is the dot product of velocity and target direction
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0
        #healthy_reward is if humanoid is standing.
        ctrl_cost = self.control_cost(action)
        #the larger the values the larger the torques applied.
        #penalize larger motor commands
        #energy efficient movements.
        reward = 10.0 * direction_reward + healthy_reward - ctrl_cost
        #without 5. healthy reward would dominate. would make the robot just stand
        #5.0 is used to balance direction reward too.
        if direction_reward < 0.3:
            reward -= 1.0
        #terminate if robot is not standing
        terminated = not self.is_healthy

        obs = self._get_obs_with_direction(self._get_obs())

        info = {
            "xy_velocity": xy_velocity,
            "target_dir": self.target_dir,
            "direction_reward": direction_reward,
        }
        #info contains xyvelocity, target_dir, direction_Reward

        return obs, reward, terminated, False, info
    
    def _get_obs_with_direction(self, humanoid_obs):
        return np.concatenate([humanoid_obs, self.target_dir]).astype(np.float64)
    #gets obswith direction
    #humanoid obs is data of all the bodies in humanoid (its angles)
    #get_obs_with_direction concatenates obs and target_dir
    #this is used to add the targets position to the observations and give it to the NN

#self. model 
#static mujoco model
#contains info that doesnt change during simulation
#bodies, joints, actuators, masses, geometry.
#it is the robot blue print
#self.model.body("torso")
#this searches the model named torso
#  returns a body vview object
#
#.id 
#every body inside mujoco has an integer ID
#example torso is 1.
#
#data
#dynamic simulation data
#   updates every physic step
# like body positions
#body velocities
#contact forces
#joint positions
#joint velocities
#
#   self.data.xpos or position in x
#world position of every body
#
#shape 
#number of bodies, 3
#example
#self.data.xpos[id]
#
#   [1.25, -0.40, 1.37]
#1.25 meters in x
#-0.40 meters in y
#1.37 meters in z
#
#[:2]
#only keep x and y. result [1.25, -0.40]
#
#copy()
#create independent copy
#otherwhise mujoco would later modify the same memory
#
#
#