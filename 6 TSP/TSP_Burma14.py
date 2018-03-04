'''
TSP 缅甸 Burma 14 城市
'''

import matplotlib.pyplot as plt
import numpy as np


class ENV(object):
    def __init__(self):
        self.city_location = np.array([[16.47, 96.10],
                                       [16.47, 94.44],
                                       [20.09, 92.54],
                                       [22.39, 93.37],
                                       [25.23, 97.24],
                                       [22.00, 96.05],
                                       [20.47, 97.02],
                                       [17.20, 96.29],
                                       [16.30, 97.38],
                                       [14.05, 98.12],
                                       [16.53, 97.38],
                                       [21.52, 95.59],
                                       [19.42, 97.13],
                                       [20.09, 94.55],
                                       ])  # 地图上的点坐标
        self.action_dim = len(self.city_location)  # 动作维度
        self.state_dim = self.action_dim  # 状态维度
        self.action_old = int(0)
        self.state = self.reset()

    def location_display(self):  # 显示位置
        plt.figure(1)
        for i in range(len(self.city_location)):
            plt.scatter(self.city_location[i][0], self.city_location[i][1])
        plt.show()

    def reset(self):  # 初始化
        state_ini = np.ones([self.state_dim])
        state_ini[0] = -1
        self.action_old = int(0)
        self.state = state_ini
        return self.state.copy()

    def render(self):
        pass

    def step(self, action):

        action = int(action)  # 动作
        reward_penalty = 0
        distance = 0
        delta_location = np.array(self.city_location[action]) - np.array(self.city_location[self.action_old])
        distance = np.sqrt(np.sum(np.square(delta_location)))

        self.state[action] = -1
        if np.sum(self.state) == -1 * len(self.state):  # 判断是否结束
            delta_location = np.array(self.city_location[action]) - np.array(self.city_location[0])
            distance += np.sqrt(np.sum(np.square(delta_location)))
            done = True
        else:
            done = False

        reward = -(reward_penalty + distance)
        info = dict()
        info["distance"] = distance
        self.action_old = action
        return self.state.copy(), reward, done, info


if __name__ == '__main__':
    env = ENV()
    env.location_display()
