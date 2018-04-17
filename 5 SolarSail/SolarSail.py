# 太阳帆运动模型搭建
import numpy as np


class Env:

    def __init__(self):
        self.t = None
        self.state = None
        self._state = None
        self.info = None
        # 归一化参数长度除以AU,时间除以TU
        self.AU = 1.4959787 * (10 ** 11)
        self.mu = 1.32712348 * (10 ** 20)
        self.TU = np.sqrt(self.AU ** 3 / self.mu)
        # 特征加速度ac和光压因子beta或者说k的转换关系ac = 5.93beta
        self.delta_d = 1  # 仿真步长，未归一化，单位天
        self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位
        self.constant = {'k': 0.5 / 5.93, 'r0': 1, 'u0': 0, 'phi0': 0,
                         'r_f': 1.524, 'u_f': 0, 'phi_f': 0}
        self.constant['v0'] = 1 / np.sqrt(self.constant['r0'])
        self.constant['v_f'] = 1 / np.sqrt(self.constant['r_f'])
        self.reset()
        self.state_dim = len(self.state)
        self.action_dim = 1
        self.abound = np.array([-1, 1])
        self.times = 40

    def render(self):
        pass

    def reset(self):
        self.t = 0
        self._state = np.array([self.constant['r0'], self.constant['phi0'],
                                self.constant['u0'], self.constant['v0']])  # [r phi u v]
        self.state = np.array([self.constant['r0'],
                               self.constant['u0'], self.constant['v0']])  # [r u v]
        self.info = self.constant.copy()
        self.info['states'] = np.array(self._state.copy())
        self.info['target'] = np.array([self.constant['r_f'], self.constant['phi_f'],
                                        self.constant['u_f'], self.constant['v_f']])
        self.info['angle'] = None
        return self.state.copy()

    def step(self, action):
        theta = (action / 2) * np.pi
        reward = 0
        for i in range(self.times):
            self.info['angle'] = np.vstack((self.info['angle'], (theta) * 180 / np.pi)) \
                if self.info['angle'] is not None else np.array([(theta) * 180 / np.pi])
            _r, _phi, _u, _v = self._state  # 当前状态的参数值
            if _r > 0.01:
                # 求微分
                r_dot = _u
                phi_dot = _v / _r
                u_dot = self.constant['k'] * ((np.cos(theta)) ** 3) / (_r ** 2) + \
                        (_v ** 2) / _r - 1 / (_r ** 2)
                v_dot = self.constant['k'] * np.sin(theta) * (np.cos(theta) ** 2) / (_r ** 2) - _u * _v / _r
                # 下一个状态
                self._state += self.delta_t * np.array([r_dot, phi_dot, u_dot, v_dot])  # [r,phi,u,v]
                self.state += self.delta_t * np.array([r_dot, u_dot, v_dot])  # [r,u,v]
                self.info['states'] = np.vstack((self.info['states'], self._state))
                # 判断是否结束
                self.t += self.delta_d  # 单位是天
                reward -= (np.abs(self._state[0] - self.constant['r_f'])) / 5  # 考虑时间和距离
                dif = np.abs(self.constant['r_f'] - self._state[0]) + \
                      np.abs(self.constant['u_f'] - self._state[2]) + \
                      np.abs(self.constant['v_f'] - self._state[3])
                if self.t >= 555 or dif < 1e-4:  # 超过一定距离和一定天数就结束
                    done = True
                    c1 = -100
                    c2 = -50
                    c3 = -50
                    reward = 70 + c1 * np.abs(self.constant['r_f'] - self._state[0]) + \
                             c2 * np.abs(self.constant['u_f'] - self._state[2]) + \
                             c3 * np.abs(self.constant['v_f'] - self._state[3])
                    break
                else:
                    done = False
            else:
                done = True
                reward = -1000
        return self.state.copy(), reward, done, self.info.copy()


if __name__ == '__main__':
    env = Env()
    print(env.step(0))
