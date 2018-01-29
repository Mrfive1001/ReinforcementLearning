import time
import numpy as np
import requests
import pickle
import random

class MissileAI:
    def __init__(self):
        near,mid,long,moon,blood = 6,4,3,1,0
        self.init_state = np.array([near,mid,long,moon,blood]*2) # 双方仓库导弹数目、卫星个数，血量
        self.state = self.init_state
        self.state_dim = len(self.state)  #状态的维度是10
        self.action_dim = 2  # 动作的维度是2
        self.first = True   # 是否是先手
        self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0], [0, 0]],
                             [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 0], [0.7, 60]],
                             [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0, 0], [0.55, 100]]])
        # hit[i,j]第i个导弹命中j个地方的概率[命中率，损毁率]


    def step(self,action):



        if sum(self.state[:3]) + sum(self.state[5:8]) == 0:
            done = True
        else:
            done = False

    def reset(self):
        self.state = self.init_state
        return self.state

    def render(self):
        pass
