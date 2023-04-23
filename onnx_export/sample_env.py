import numpy as np
from gym import Env, spaces


class SampleEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
                                            dtype=np.float32)
        
        machine_count = 5
        self.action_space = spaces.MultiBinary(machine_count)
    
    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass
