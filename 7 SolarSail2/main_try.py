import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SolarSail import Env

# actions = [0.12775242, 0.98694867, 1,  1, 0.5018742, 0.4078638, 0.6552601, 1]
actions = np.array([-48, -55, -66, -79, 0])/90
if __name__ == '__main__':
    phi = []
    r = []
    env = Env()
    old_times = 33
    # old_times = env.times
    env.times = 1
    t = 0
    state_now = env.reset()
    r.append(state_now[0])
    phi.append(state_now[1])
    while True:
        action = actions[min(int(t/old_times),len(actions)-1)]
        action = -0/90.0
        state_next, reward, done, info = env.step(action)
        t = info['t']
        state_now = state_next
        r.append(state_now[0])
        phi.append(state_now[1])
        if done:
            break
    print('转移轨道时间%d天' % t)
    print(state_now)
    print('目标参数', info['target'])
    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta))
    plt.plot(theta, 0.7233 * np.ones_like(theta))
    plt.plot(phi, r)
    plt.savefig('try.png')
