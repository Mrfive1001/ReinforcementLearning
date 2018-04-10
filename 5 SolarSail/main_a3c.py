import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SolarSail import Env
import A3C
import os
import sys
import pickle

if __name__ == '__main__':
    train_mode = 0  # 0 测试 1 从头开始训练 2 从已有阶段开始训练
    choose_mode = 0  # 0 最好结果 1 测试选择随机动作 2 测试选择最好动作
    env = Env()
    para = A3C.Para(env,
                    a_constant=True,
                    units_a=256,
                    units_c=256,
                    MAX_GLOBAL_EP=20000,
                    UPDATE_GLOBAL_ITER=4,
                    gamma=0.95,
                    ENTROPY_BETA_init=0.2,  # 太大最后测试效果很差
                    ENTROPY_BETA_times=10000,
                    ENTROPY_BETA_end=0.01,
                    LR_A=0.00002,
                    LR_C=0.0001,
                    train_mode=train_mode)
    number = 3  # 调试参数编号
    RL = A3C.A3C(para)
    # RL.run()  # 训练或者载入数据
    actions_best = []
    path00 = os.path.join(sys.path[0], 'A3C_result')
    path0 = os.path.join(path00, str(number) + '_result')
    if not os.path.exists(path00):
        os.mkdir(path00)
    if not os.path.exists(path0):
        os.mkdir(path0)

    if para.train_mode:
        with open(os.path.join(path0, 'action' + '.txt'), 'wb') as file:
            pickle.dump(para.best_action, file)
            actions_best = para.best_action.copy()
    else:
        with open(os.path.join(path0, 'action' + '.txt'), 'rb') as file:
            actions_best = pickle.load(file)
    # 画出最好的动作
    env = Env()
    state_now = env.reset()
    epr_best = 0
    t_best = 0
    while True:
        if choose_mode == 0:
            action = actions_best[int(t_best)]
        elif choose_mode == 1:
            action = RL.choose_action(state_now)
        elif choose_mode == 2:
            action = RL.choose_best(state_now)
        state_next, reward, done, info = env.step(action)
        t_best += 1
        state_now = state_next
        epr_best += reward
        if done:
            break
    print('最好轨道参数是：', list(info['states'][-1]))
    print('最好轨道奖励是：', epr_best)
    print('本次测试序号是：', number)
    print('本次测试误差是：', list(info['states'][-1] - info['target']))
    plt.figure(1)
    ax1 = plt.subplot(111, polar=True)
    plt.thetagrids(range(45, 360, 90))
    plt.plot(info['states'][:, 1], info['states'][:, 0], 'k')
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), '--')
    plt.plot(theta, 1.547 * np.ones_like(theta), '--')
    plt.xlabel('x/(AU)')
    plt.ylabel('y/(AU)')
    plt.ylim((0, 2))
    plt.yticks(np.arange(2))
    plt.savefig(os.path.join(path0, 'r-phi' + '.png'))

    plt.figure(2)
    plt.plot(info['states'][:, 0], 'm')
    plt.plot(info['r_f'] * np.ones(len(info['states'][:, 0])))
    plt.ylim(0.5, 2)
    plt.title('r')
    plt.savefig(os.path.join(path0, 'r' + '.png'))

    plt.figure(3)
    plt.plot(info['states'][:, 2], 'm')
    plt.plot(info['u_f'] * np.ones(len(info['states'][:, 0])))
    plt.title('u')
    plt.savefig(os.path.join(path0, 'u' + '.png'))

    plt.figure(4)
    plt.plot(info['states'][:, 3], 'm')
    plt.plot(info['v_f'] * np.ones(len(info['states'][:, 0])))
    plt.title('v')
    plt.savefig(os.path.join(path0, 'v' + '.png'))

    plt.figure(5)
    plt.plot(info['angle'], 'm')
    plt.title('angle')
    plt.savefig(os.path.join(path0, 'angle' + '.png'))
