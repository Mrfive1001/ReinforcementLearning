
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root, minimize


class SolarSail:

    def __init__(self, random=False):
        self.t = None
        self.state = None
        self.random = random
        # 归一化参数长度除以AU,时间除以TU
        self.AU = 1.4959787 * (10 ** 11)
        self.mu = 1.32712348 * (10 ** 20)
        self.VU = np.sqrt(self.mu / self.AU)
        self.TU = np.sqrt(self.AU ** 3 / self.mu)
        self.constant = {'beta': 0.16892, 'u0': 0, 'phi0': 0, 'r_f': 1.524, 'u_f': 0, 'phi_f': 0}
        self.constant['v_f'] = 1.0 / np.sqrt(self.constant['r_f'])
        # 特征加速度ac和光压因子beta或者说k的转换关系ac = 5.93beta
        self.delta_d = 0.1  # 仿真步长，未归一化，单位天
        self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位
        self.reset()
        self.ob_dim = len(self.observation)
        self.lambda_dim = 4
        self.u_dim = 1

    def render(self):
        pass

    def reset(self, state0=None):
        self.t = 0
        self.td = 0
        if state0 is None:
            if self.random == True:
                self.constant['r0'] = (1.0 + 0.2 * (np.random.rand(1) - 0.5) * 2)[0]
                self.constant['u0'] = (0.1 * (np.random.rand(1) - 0.5) * 2)[0]
                self.constant['v0'] = (1.0 / np.sqrt(self.constant['r0']) + 0.1 * (np.random.rand(1) - 0.5) * 2)[0]
                self.state = np.array([self.constant['r0'], self.constant['phi0'],
                                       self.constant['u0'], self.constant['v0']])  # [r phi u v]
            else:
                self.constant['r0'] = 1.0
                self.constant['v0'] = 1.0 / np.sqrt(self.constant['r0'])
                self.state = np.array([self.constant['r0'], self.constant['phi0'],
                                       self.constant['u0'], self.constant['v0']])  # [r phi u v]
        else:
            self.constant['r0'] = state0[0]
            self.constant['u0'] = state0[1]
            self.constant['v0'] = state0[2]
            self.state = np.array([self.constant['r0'], self.constant['phi0'],
                                   self.constant['u0'], self.constant['v0']])  # [r phi u v]
        self.observation = np.array([self.constant['r0'],
                                     self.constant['u0'],
                                     self.constant['v0']])  # [r phi u v]
        return self.observation

    def get_optimize(self, action_ini):
        res = root(self.get_reward, action_ini)
        return res

    def get_reward(self, action):
        observation, ceq, done, info = self.get_lam_tra(action)
        return ceq

    def get_sample(self, action):

        lambda_n = action[0:3]
        t_f = action[3]

        # 微分方程
        X0 = np.hstack([self.state, lambda_n])
        t = np.linspace(0, t_f, 101)
        X = odeint(self.motionequation, X0, t, rtol=1e-12, atol=1e-12)

        aa_sample = t_f - t.reshape([len(t), 1])

        u_sample = np.ones_like(X[:, 0])
        for i in range(len(u_sample)):

            lambda3 = X[i, 5]
            lambda4 = X[i, 6]

            if np.abs(lambda4) < 0.00001:
                if lambda3 <= 0:
                    alpha = 0
                else:
                    alpha = np.pi / 2
            else:
                aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
                alpha = np.arctan(aaa)

            u_sample[i] = alpha
        u_sample = u_sample.reshape([len(t), 1])
        r_sample = X[:, 0].reshape([len(t), 1])

        sample = np.hstack((r_sample, X[:, 2:7], aa_sample, u_sample))
        return sample


    def get_lam_tra(self, action):

        lambda_n = action[0:3]
        t_f = action[3]


        # 微分方程
        X0 = np.hstack([self.state, lambda_n])
        t = np.linspace(0, t_f, 101)
        X = odeint(self.motionequation, X0, t, rtol=1e-12, atol=1e-12)

        # 末端Hamilton函数值
        # 末端Hamilton函数值
        X_end = X[-1, :]
        r = X_end[0]
        u = X_end[2]
        v = X_end[3]
        lambda1 = X_end[4]
        lambda3 = X_end[5]
        lambda4 = X_end[6]

        if np.abs(lambda4) < 0.00001:
            if lambda3 <= 0:
                alpha = 0
            else:
                alpha = np.pi / 2
        else:
            aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
            alpha = np.arctan(aaa)

        r_dot = u
        u_dot = self.constant['beta'] * ((np.cos(alpha)) ** 3) / (r ** 2) + \
                (v ** 2) / r - 1 / (r ** 2)
        v_dot = self.constant['beta'] * np.sin(alpha) * (np.cos(alpha) ** 2) / (r ** 2) - u * v / r

        H_end = 1 + lambda1 * r_dot + lambda3 * u_dot + lambda4 * v_dot

        r_error = np.abs(self.constant['r_f'] - X_end[0])
        u_error = np.abs(self.constant['u_f'] - X_end[2])
        v_error = np.abs(self.constant['v_f'] - X_end[3])

        if t_f < 0:
            r_error = 100

        ceq = [r_error, u_error, v_error, H_end]

        done = True
        info = {}
        info['X'] = X
        info['t'] = t

        return self.state.copy(), ceq, done, info


    def motionequation(self, input, t):

        r = input[0]
        u = input[2]
        v = input[3]
        lambda1 = input[4]
        lambda2 = 0
        lambda3 = input[5]
        lambda4 = input[6]

        if np.abs(lambda4) < 0.00001:
            if lambda3 <= 0:
                alpha = 0
            else:
                alpha = np.pi / 2
        else:
            aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
            alpha = np.arctan(aaa)

        r_dot = u
        phi_dot = v / r
        u_dot = self.constant['beta'] * ((np.cos(alpha)) ** 3) / (r ** 2) + \
                (v ** 2) / r - 1 / (r ** 2)
        v_dot = self.constant['beta'] * np.sin(alpha) * (np.cos(alpha) ** 2) / (r ** 2) - u * v / r

        lambda1_dot = lambda2 * v / (r ** 2) + \
                      lambda3 * (2 * self.constant['beta'] * np.cos(alpha) ** 3 / (r ** 3) + \
                      v ** 2 / (r ** 2) - 2 / (r ** 3)) + \
                      lambda4 * (2 * self.constant['beta'] * np.sin(alpha) * np.cos(alpha) ** 2 / (r ** 3) - \
                      u * v / r ** 2)
        lambda3_dot = - lambda1 + lambda4 * v / r
        lambda4_dot = - lambda2 / r - 2 * lambda3 * v / r + lambda4 * u / r

        state_dot = [r_dot, phi_dot, u_dot, v_dot, lambda1_dot, lambda3_dot, lambda4_dot]
        return state_dot


    def step_lambda(self, action):

        lambda1 = action[0]
        lambda3 = action[1]
        lambda4 = action[2]
        t_f = action[3]

        r, phi, u, v = self.state

        if np.abs(lambda4) < 0.00001:
            if lambda3 <= 0:
                alpha = 0
            else:
                alpha = np.pi / 2
        else:
            aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
            alpha = np.arctan(aaa)

        r_dot = u
        phi_dot = v / r
        u_dot = self.constant['beta'] * ((np.cos(alpha)) ** 3) / (r ** 2) + \
                (v ** 2) / r - 1 / (r ** 2)
        v_dot = self.constant['beta'] * np.sin(alpha) * (np.cos(alpha) ** 2) / (r ** 2) - u * v / r

        # 判断下一个状态的距离
        self.state += self.delta_t * np.array([r_dot, phi_dot, u_dot, v_dot])
        self.t += self.delta_t

        r_error = np.abs(self.constant['r_f'] - r)


        if self.t > self.t_f1:
            r, phi, u, v = self.state
            r_error = np.abs(self.constant['r_f'] - r)
            u_error = np.abs(self.constant['u_f'] - u)
            v_error = np.abs(self.constant['v_f'] - v)
            ceq = [r_error, u_error, v_error]
            done = True
            info = {}
        else:
            ceq = []
            done = False
            info = {}

        self.observation  = np.array([r, u, v])
        return self.observation.copy(), ceq, done, info

    def step_u(self, action):
        alpha = action
        if alpha > np.pi/2:
            alpha = np.pi / 2
        elif alpha < -np.pi/2:
            alpha = -np.pi / 2
        r, phi, u, v = self.state
        r_dot = u
        phi_dot = v / r
        u_dot = self.constant['beta'] * ((np.cos(alpha)) ** 3) / (r ** 2) + \
                (v ** 2) / r - 1 / (r ** 2)
        v_dot = self.constant['beta'] * np.sin(alpha) * (np.cos(alpha) ** 2) / (r ** 2) - u * v / r

        # 判断下一个状态的距离
        self.state += self.delta_t * np.array([r_dot, phi_dot, u_dot, v_dot])
        self.t += self.delta_t



        if self.t > self.t_f1:
            r, phi, u, v = self.state
            r_error = np.abs(self.constant['r_f'] - r)
            u_error = np.abs(self.constant['u_f'] - u)
            v_error = np.abs(self.constant['v_f'] - v)
            ceq = [r_error, u_error, v_error]
            done = True
            info = {}
        else:
            r, phi, u, v = self.state
            r_error = np.abs(self.constant['r_f'] - r)
            u_error = np.abs(self.constant['u_f'] - u)
            v_error = np.abs(self.constant['v_f'] - v)
            ceq = [r_error, u_error, v_error]
            done = False
            info = {}

        self.observation = np.array([r, u, v])
        return self.observation.copy(), ceq, done, info











