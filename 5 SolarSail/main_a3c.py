import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SolarSail import Env
import A3C

if __name__ == '__main__':
    env = Env()
    para = A3C.Para(env,
                    a_constant=True,
                    units_a=128,
                    units_c=256,
                    MAX_GLOBAL_EP=30000,
                    UPDATE_GLOBAL_ITER=4,
                    gamma=0.95,
                    ENTROPY_BETA_init=0.01, # 太大最后测试效果很差
                    ENTROPY_BETA_times=10000,
                    ENTROPY_BETA_end=0.05,
                    LR_A=0.00002,
                    LR_C=0.0001,
                    train=True)
    number = 6
    RL = A3C.A3C(para)
    RL.run()
    phi = []
    r = []
    env = Env()
    t = 0
    env.times = 1
    state_now = env.reset()
    r.append(state_now[0])
    phi.append(state_now[1])
    epr = 0
    while True:
        state_next, reward, done, info = env.step(RL.choose_action(state_now))
        t = info['t']
        epr += reward
        state_now = state_next
        r.append(state_now[0])
        phi.append(state_now[1])
        if done:
            break
    print('转移轨道时间%d天' % t)
    print('过程中最短天数是', para.best_day)
    print('测试轨道参数', state_now)
    print('最好轨道参数', para.best_state)
    print('目标参数', info['target'])
    print('测试总奖励是', epr)
    print('过程中最好奖励是', para.best_epr)
    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta))
    plt.plot(theta, 1.547 * np.ones_like(theta))
    plt.plot(para.best_phi, para.best_r, '--')
    plt.plot(phi, r)
    plt.savefig('A3C'+str(number)+'.png')
    print('number:',number)
