
import numpy as np
from utils.env_core import EnvCore

class DummyVecEnv(object):
    def __init__(self, env_fns):
        self.env = EnvCore()
        self.agent_types = np.ones(self.env.agent_num).astype(int)  
        self.actions = None
        self.action_space = (np.ones(self.env.agent_num) * self.env.action_dim).astype(int)
        self.observation_space = (np.ones(self.env.agent_num) * self.env.obs_dim).astype(int)

    def step(self, actions):
        obs, reward, dones, infos = self.env.step(actions)
        return np.array(obs), np.array(reward), np.array(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.array(obs)

    def close(self):
        return
