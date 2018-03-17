import numpy as np
import matplotlib.pyplot as plt
from SolarSail import Env


def choose_action(t):
    if t <= 66:
        return 0
    elif t <= 2 * 66:
        return 70
    elif t <= 3 * 66:
        return 60
    elif t <= 4 * 66:
        return 50
    else:
        return 45


if __name__ == '__main__':
    phi = []
    r = []
    env = Env()
    t = 0
    state_now = env.reset()
    r.append(state_now[0])
    phi.append(state_now[1])
    while True:
        action = choose_action(t)/90.0
        # action = -45/90.0
        state_next, reward, done, info = env.step(action)
        t = info['t']
        state_now = state_next
        r.append(state_now[0])
        phi.append(state_now[1])
        if done:
            break
    print('转移轨道时间%d天' % t)
    print(state_now)
    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta))
    plt.plot(theta, 1.547 * np.ones_like(theta))
    plt.plot(phi, r)
    plt.show()
