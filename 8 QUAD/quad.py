# 四旋翼悬停
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt


class QUAD:
    # 四旋翼简单运动方程
    def __init__(self, random=False):
        self.t = None
        self.state = None  # 与外界交互的变量
        self.x = None  # 内部积分变量
        self.random = random  # 是否随机初始化
        self.constant = {'g': -9.81, 'c1': 20., 'c2': 2., 'alpha': 1., 'm': 1.,
                         'gamma1': 1, 'gamma2': 1.0,
                         'x_f': 0., 'z_f': 0., 'vx_f': 0., 'vz_f': 0., 'theta_f': 0.}
        self.reset()
        self.state_dim = len(self.state)
        self.action_dim = 7

    def render(self):
        pass

    def reset(self):
        self.t = 0
        if self.random:
            xz = np.random.rand(2) * 10 - 5  # xz初始位置-5~5
            v_xz = np.random.rand(2) * 2 - 1  # 速度初始-1~1
            theta = np.random.rand(1) * np.pi / 5 - np.pi / 10  # 初始角度
            self.state = np.hstack((xz, v_xz, theta))
        else:
            self.state = np.array([-5., -5., 0., 0., 0.])
        return self.state

    def _get_reward(self, action):
        observation, ceq, done, info = self.step(action)
        return ceq

    def get_result(self, action_ini):
        # 得到某个初始动作优化的结果
        res = root(self._get_reward, action_ini)
        return res

    def step(self, action):
        # 输入选择动作lambda_0(控制缩放的),lambda_n(需要积分变量),t_f(时间)
        # 输出是否平衡，附加信息
        lambda_0 = (action[0] + 1) / 2
        lambda_n = action[1:-1]
        t_f = (action[-1] + 1) * 5
        if t_f < 0 or lambda_0 < 0:
            lambda_0 = 10
        lambda_all = np.hstack((lambda_0, lambda_n))

        # 微分方程
        X0 = np.hstack([self.state, lambda_n / lambda_0])
        t = np.linspace(0, t_f, 101)
        X = odeint(self.motionequation, X0, t, args=(lambda_0,), rtol=1e-12, atol=1e-12)

        # 末端Hamilton函数值
        X_end = X[-1, :]

        X_dot = self.motionequation(X_end, 0, lambda_0)

        H_end = lambda_0 + np.sum(np.array(X_end[5:]) * lambda_0 * np.array(X_dot[:5]))

        lamnda_normal = np.linalg.norm(lambda_all) - 1
        ceq = X_end[:5].copy()
        ceq = np.hstack([ceq, H_end, lamnda_normal])
        # 判断是否平衡，满足终端条件

        done = True
        info = {}
        info['X'] = X.copy()
        info['t'] = t.copy()
        info['store'] = np.hstack([X, t[::-1].reshape((101, 1))])
        return self.state.copy(), ceq, done, info

    def motionequation(self, input, t, lambda_0):
        # 运动方程 输入[x] 输出x_dot
        x = input[0]
        z = input[1]
        xv = input[2]
        zv = input[3]
        theta = input[4]
        lambda_x = input[5]
        lambda_z = input[6]
        lambda_xv = input[7]
        lambda_zv = input[8]
        lambda_Theata = input[9]

        S1 = lambda_xv * np.sin(theta) + lambda_zv * np.cos(theta)
        S2 = lambda_Theata

        u1 = 1 if S1 < 0 else 0.05
        u2 = 1 if S2 < 0 else -1

        # dynamic equation
        x_dot = xv
        z_dot = zv
        vx_dot = self.constant['c1'] * u1 * np.sin(theta) / self.constant['m']
        vz_dot = self.constant['c1'] * u1 * np.cos(theta) / self.constant['m'] + self.constant['g']
        theta_dot = self.constant['c2'] * u2

        lambda_x_dot = 0.
        lambda_z_dot = 0.
        lambda_xv_dot = -lambda_x
        lambda_zv_dot = -lambda_z
        lambda_theata_dot = - self.constant['c1'] * u1 / self.constant['m'] * \
                            (lambda_xv * np.cos(theta) - lambda_zv * np.sin(theta))

        X_dot = [x_dot, z_dot, vx_dot, vz_dot, theta_dot, lambda_x_dot, lambda_z_dot, lambda_xv_dot,
                 lambda_zv_dot, lambda_theata_dot]
        return X_dot


if __name__ == '__main__':
    env = QUAD()
    for i in range(100):
        # lambda0 = np.random.rand(1)
        # lambda_n = np.random.randn(5)
        # t_f = np.random.rand(1) * 10
        # action = np.hstack([lambda0, lambda_n, t_f])
        action = np.random.rand(7) * 2 - 1
        res = env.get_result(action)
        print('step', i, 'fun', res.fun, '\n', 'action', res.x)
        if res.success:
            break
            print('sucess')

    # action = [ 0.88974051 ,-0.07939553 ,-0.04843978 ,-0.06693855 ,-0.03286776, -0.3049739 ,-0.63933857]
    action = res.x
    observation, ceq, done, info = env.step(action)
    plt.figure(1)
    plt.plot(info['X'][:, 0], info['X'][:, 1])
    plt.figure(2)
    plt.plot(info['t'], info['X'][:, 1])
    plt.show()
