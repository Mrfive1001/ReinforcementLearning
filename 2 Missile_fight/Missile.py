import time
import numpy as np
import requests
np.random.seed(2)

class MissileAI:
    def __init__(self):
        near, mid, long, moon, blood = 6, 4, 3, 1, 0
        self.init_state = np.array([near, mid, long, moon, blood] * 2)  # 双方仓库导弹数目、卫星个数，血量
        self.state = self.init_state
        self.state_dim = len(self.state)  # 状态的维度是10
        self.action_dim = 2  # 动作的维度是2
        self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0], [0, 0]],
                             [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 0], [0.7, 60]],
                             [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0, 0], [0.55, 100]]])
        # hit[i,j]第i个导弹命中j个地方的概率[命中率，损毁率]
        self.jump = int(self.state_dim / 2)  # 先后手区别的位数
        self.moon_help = 1.2

    def step(self, action):
        a1 = action[0]
        t1 = action[1] + self.jump
        a2 = action[2] + self.jump
        t2 = action[3]
        moon_add1 = self.moon_help if self.state[3] > 0 else 1
        moon_add2 = self.moon_help if self.state[8] > 0 else 1
        hit_rate1, damage_rate1 = self.hit[action[0], action[1]]
        hit_rate2, damage_rate2 = self.hit[action[2], action[3]]

        ran = [np.random.rand() for _ in range(4)]
        state1 = self.state.copy()
        state2 = self.state.copy()
        reward1, reward2 = 0, 0
        for missile, store, hit_rate, damage_rate, moon_add, state, reward, ran1, ran2 in zip(
                [a1, a2], [t1, t2], [hit_rate1, hit_rate2], [damage_rate1, damage_rate2]
                , [moon_add1, moon_add2], [state1, state2], [reward1, reward2], ran[:2], ran[2:]):
            if state[missile] > 0:  # 如果有弹
                state[missile] -= 1  # 减少弹
                if ran1 < hit_rate * moon_add:  # 命中
                    if store != 4 or store != 9:  # 命中非基地
                        if ran2 < damage_rate * moon_add:  # 损伤了
                            state[store] = 0
                    else:
                        state[store] -= damage_rate
            reward = 0
        self.state = np.array([min(x, y) for x, y in zip(state1, state2)])
        if sum(self.state[:3]) + sum(self.state[self.jump:8]) == 0:
            done = True
        else:
            done = False
        return self.state, [reward1, reward2], done, None

    def reset(self):
        self.state = self.init_state
        return self.state

    def render(self):
        pass
