import time
import numpy as np
import requests
import pickle
import random


class MissileAI:
    def __init__(self):
        near, mid, long, moon, blood = 6, 4, 3, 1, 0
        self.init_state = np.array([near, mid, long, moon, blood] * 2)  # 双方仓库导弹数目、卫星个数，血量
        self.state = self.init_state
        self.state_dim = len(self.state)  # 状态的维度是10
        self.action_dim = 2  # 动作的维度是2
        self.first = True  # 是否是先手
        self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0], [0, 0]],
                             [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 0], [0.7, 60]],
                             [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0, 0], [0.55, 100]]])
        # hit[i,j]第i个导弹命中j个地方的概率[命中率，损毁率]
        self.jump = int(self.state_dim)  # 先后手区别的位数
        self.moon_help = 1.2

    def step(self, action):
        missile = action[0] if self.first else action[0] + self.jump
        store = action[1] + self.jump if self.first else action[1]
        moon_add = self.moon_help if self.state[3 if self.first else 8] > 0 else 1
        hit_rate, damage_rate = self.hit[action[0], action[1]]
        if sum(self.state[:3]) + sum(self.state[self.jump:8]) == 0:
            done = True
            return self.state, 0, done, None
        else:
            done = False
            ran1, ran2 = np.random.rand(), np.random.rand()
            if self.state[missile] > 0:  # 如果有弹
                self.state[missile] -= 1  # 减少弹
                hit_rate *= moon_add
                if ran1 < hit_rate:  # 命中
                    if action[1] < (self.jump - 1):  # 命中非基地
                        damage_rate *= moon_add
                        if ran2 < damage_rate:  # 损伤了
                            self.state[store] = 0
                    else:
                        self.state[store] -= damage_rate
            reward = 0
            return self.state, reward, done, None

    def reset(self):
        self.state = self.init_state
        return self.state

    def render(self):
        pass
