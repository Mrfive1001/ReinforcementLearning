# 太阳帆运动模型搭建
import numpy as np


class Env:

    def __init__(self):
        self.t = None
        self.state = None
        self.reset()
        self.state_dim = len(self.state)
        self.action_dim = 1
        self.abound = np.array([-np.pi, np.pi])

    def render(self):
        pass

    def reset(self):
        self.t = 0
        self.state = np.array([1])  # [r theta v_r v_theta]
        return self.state.copy()

    def step(self, action):
        pass
