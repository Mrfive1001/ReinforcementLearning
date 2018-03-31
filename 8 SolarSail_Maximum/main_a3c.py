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
    random = False
    env = Env(random)
    para = A3C.Para(env,
                    a_constant=True,
                    units_a=128,
                    units_c=256,
                    MAX_GLOBAL_EP=100,
                    UPDATE_GLOBAL_ITER=4,
                    gamma=0.95,
                    ENTROPY_BETA_init=0.001,
                    ENTROPY_BETA_end=0.01,
                    ENTROPY_BETA_times=20000,
                    LR_A=0.00002,
                    LR_C=0.0001,
                    train=True)
    number = 2  # 调试参数编号
    RL = A3C.A3C(para)
    RL.run()  # 训练或者载入数据
    env = Env(random)
    s = env.reset()
    # action = np.array([(-1.609601 + 5) / 10, (0.042179 + 5) / 10, \
    #                    (-0.160488 + 5) / 10, (-1.597537 + 5) / 10, (568 - 100) / 500])
    observation, reward, done, info = env.step(RL.choose_action(s))
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
    # plt.figure(6)
    # plt.plot(info['reward_profile'], 'm')
    # plt.title('reward')

    plt.show()

# # 画出最好的动作
# phi = []
# r = []
# env = Env()
# env.times = 1
# t = 0
# actions = []
# state_now = env.reset()
# r.append(state_now[0])
# phi.append(state_now[1])
# epr = 0
# while True:
#     action = RL.choose_action(state_now)
#     actions.append(action)
#     state_next, reward, done, info = env.step(RL.choose_action(state_now))
#     t = info['t']
#     epr += reward
#     state_now = state_next
#     r.append(state_now[0])
#     phi.append(state_now[1])
#     if done:
#         break
# # 显示
# print('测试轨道参数是：', state_now)
# print('测试轨道奖励是：', epr)
# print('测试轨道天数是：', t)
#
# phi_best = []
# r_best = []
# env = Env()
# state_now = env.reset()
# r_best.append(state_now[0])
# phi_best.append(state_now[1])
# actions_plot = []
# epr_best = 0
# times_old = env.times
# env.times = 1
# t_best = 0
# while True:
#     action = actions_best[int(t_best / times_old)]
#     actions_plot.append(action)
#     state_next, reward, done, info = env.step(action)
#     t_best = info['t']
#     state_now = state_next
#     r_best.append(state_now[0])
#     phi_best.append(state_now[1])
#     epr_best += reward
#     if done:
#         break
# print('最好轨道参数是：', state_now)
# print('最好轨道奖励是：', epr_best)
# print('最好轨道天数是：', t_best)
# print('目标轨道参数是：', info['target'])
# plt.figure(1)
# ax1 = plt.subplot(111, polar=True)
# plt.thetagrids(range(45, 360, 90))
# plt.plot(phi_best, r_best, 'k')
# theta = np.arange(0, 2 * np.pi, 0.02)
# plt.plot(theta, 1 * np.ones_like(theta), '--')
# plt.plot(theta, 0.7233* np.ones_like(theta), '--')
# plt.xlabel('x/(AU)')
# plt.ylabel('y/(AU)')
# plt.ylim((0, 1))
# plt.yticks(np.arange(2))
# plt.plot(phi, r, 'y')
# plt.savefig(os.path.join(path0, 'A3C_effect' + str(number) + '.png'))
# plt.figure(2)
# plt.subplot(111)
# # plt.plot(actions, 'y')
# plt.plot(actions_plot, 'k')
# plt.savefig(os.path.join(path0, 'A3C_action' + str(number) + '.png'))
# print('number:', number)
# print(para.best_action)
# # 画出测试
