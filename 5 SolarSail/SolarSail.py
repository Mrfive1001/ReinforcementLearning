# 太阳帆运动模型搭建
import numpy as np
import matplotlib.pyplot as plt


class Env:

    def __init__(self):
        self.t = None
        self.state = None
        # 归一化参数长度除以AU,时间除以TU
        self.AU = 1.4959787 * (10 ** 11)
        self.mu = 1.32712348 * (10 ** 20)
        self.TU = np.sqrt(self.AU ** 3 / self.mu)
        # 特征加速度ac和光压因子beta或者说k的转换关系ac = 5.93beta
        self.delta_d = 0.5  # 仿真步长，未归一化，单位天
        self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位
        self.constant = {'k': 2 / 5.93, 'r0': 1, 'u0': 0, 'phi0': 0,
                         'r_f': 1.524, 'u_f': 0, 'phi_f': 0}
        self.constant['v0'] = 1 / np.sqrt(self.constant['r0'])
        self.constant['v_f'] = 1 / np.sqrt(self.constant['r_f'])
        self.reset()
        self.state_dim = len(self.state)
        self.theta_dim = 1
        self.abound = np.array([-90, 90])

    def render(self):
        pass

    def reset(self):
        self.t = 0
        self.state = np.array([self.constant['r0'], self.constant['phi0'],
                               self.constant['u0'], self.constant['v0']])  # [r phi u v]
        return self.state.copy()

    def step(self, action):
        # 传入单位是度
        theta = (action / 180) * np.pi
        _r, _phi, _u, _v = self.state  # 当前状态的参数值
        # 求微分
        r_dot = _u
        phi_dot = _v / _r
        u_dot = self.constant['k'] * ((np.cos(theta)) ** 3) / (_r ** 2) + \
                (_v ** 2) / _r - 1 / (_r ** 2)
        v_dot = self.constant['k'] * np.sin(theta) * (np.cos(theta) ** 2) / (_r ** 2) - _u * _v / _r
        # 下一个状态
        self.state += self.delta_t * np.array([r_dot, phi_dot, u_dot, v_dot])  # [r,phi,u,v]
        # 判断是否结束
        self.t += self.delta_d
        if self.t >= 200 or self.state[0] >= self.constant['r_f']:
            done = True
        else:
            done = False
        info = None
        # 设计reward函数
        reward = None
        return self.state.copy(), reward, done, info

    # def display(self):
    #     phi = []
    #     r = []
    #     # for i in range(int(66 / self.delta_d)):
    #     #     state = env.step(0)[0]
    #     #     r0, phi0, _, _ = state
    #     #     phi.append(phi0)
    #     #     r.append(r0)
    #     # for i in range(int(66 / self.delta_d)):
    #     #     state = env.step(70)[0]
    #     #     r0, phi0, _, _ = state
    #     #     phi.append(phi0)
    #     #     r.append(r0)
    #     # for i in range(int(66 / self.delta_d)):
    #     #     state = env.step(60)[0]
    #     #     r0, phi0, _, _ = state
    #     #     phi.append(phi0)
    #     #     r.append(r0)
    #     # for i in range(int(66 / self.delta_d)):
    #     #     state = env.step(50)[0]
    #     #     r0, phi0, _, _ = state
    #     #     phi.append(phi0)
    #     #     r.append(r0)
    #     # for i in range(int(66 / self.delta_d)):
    #     #     state = env.step(45)[0]
    #     #     r0, phi0, _, _ = state
    #     #     phi.append(phi0)
    #     #     r.append(r0)
    #     for i in range(int(32 / self.delta_d)):
    #         state = env.step(-48)[0]
    #         r0, phi0, _, _ = state
    #         phi.append(phi0)
    #         r.append(r0)
    #     for i in range(int(32 / self.delta_d)):
    #         state = env.step(-55)[0]
    #         r0, phi0, _, _ = state
    #         phi.append(phi0)
    #         r.append(r0)
    #     for i in range(int(32 / self.delta_d)):
    #         state = env.step(-67)[0]
    #         r0, phi0, _, _ = state
    #         phi.append(phi0)
    #         r.append(r0)
    #     for i in range(int(32 / self.delta_d)):
    #         state = env.step(-79)[0]
    #         r0, phi0, _, _ = state
    #         phi.append(phi0)
    #         r.append(r0)
    #     for i in range(int(32 / self.delta_d)):
    #         state = env.step(0)[0]
    #         r0, phi0, _, _ = state
    #         phi.append(phi0)
    #         r.append(r0)
    #     print(r[-1])
    #     plt.subplot(111, polar=True)
    #     theta = np.arange(0, 2 * np.pi, 0.02)
    #     plt.plot(theta, 1 * np.ones_like(theta))
    #     plt.plot(theta, 1.547 * np.ones_like(theta))
    #     plt.plot(theta, 0.7233 * np.ones_like(theta))
    #     plt.plot(phi, r)
    #     plt.show()


if __name__ == '__main__':
    env = Env()
    print(env.step(0))
