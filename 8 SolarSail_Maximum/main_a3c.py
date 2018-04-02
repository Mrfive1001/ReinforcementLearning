import numpy as np
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SolarSail_Max import SolarSail_Max as Env
import A3C
import os
import sys
import pickle

if __name__ == '__main__':
    random = True
    train = 1
    name = 'random' if random else 'static'
    env = Env(random)
    para = A3C.Para(env,
                    a_constant=True,
                    units_a=512,
                    units_c=512,
                    MAX_GLOBAL_EP=30000,
                    UPDATE_GLOBAL_ITER=50,
                    gamma=0.90,
                    ENTROPY_BETA_init=0.01,
                    ENTROPY_BETA_end=0.01,
                    ENTROPY_BETA_times=20000,
                    LR_A=0.00001,
                    LR_C=0.0001,
                    train_mode=train,
                    name=name)
    number = 1  # 调试参数编号
    print(number)
    RL = A3C.A3C(para)
    RL.run()
    env = Env(random)
    s = env.reset()
    # action = np.array([(-1.609601 + 5) / 10, (0.042179 + 5) / 10, \
    #                    (-0.160488 + 5) / 10, (-1.597537 + 5) / 10, (568 - 100) / 500])
    observation, reward, done, info = env.step(RL.choose_best(s))
    print(number)
    print('total reward:', reward)
    print('total day:', info['total_day'])
    print('rf_error:', info['rf_error'])
    print('uf_error:', info['uf_error'])
    print('vf_error:', info['vf_error'])

    states_profile = info['states_profile']
    alpha_profile = info['alpha_profile']

    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), 'm')
    plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
    plt.plot(states_profile[:, 1], states_profile[:, 0], 'r')

    plt.figure(2)
    plt.plot(states_profile[:, 0], 'm')
    plt.plot(env.constant['r_f'] * np.ones(len(states_profile[:, 0])))
    plt.title('r')

    plt.figure(3)
    plt.plot(states_profile[:, 2], 'm')
    plt.plot(env.constant['u_f'] * np.ones(len(states_profile[:, 0])))
    plt.title('v')
    plt.title('u')

    plt.figure(4)
    plt.plot(states_profile[:, 3], 'm')
    plt.plot(env.constant['v_f'] * np.ones(len(states_profile[:, 0])))
    plt.title('v')

    plt.figure(5)
    plt.plot(alpha_profile * 57.3, 'm')
    plt.title('alpha')

    plt.show()

