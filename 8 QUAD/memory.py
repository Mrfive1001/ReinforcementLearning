import numpy as np
from quad import QUAD
import matplotlib.pyplot as plt


def get_memory(groups_num):
    memory = None  # 存储的记忆数据
    env = QUAD(True)
    for i in range(groups_num):
        env.reset()  # 随机位置
        for j in range(100):
            action = np.random.rand(7) * 2 - 1
            res = env.get_result(action)  # 打一次靶
            if res.success:
                if np.linalg.norm(res.fun) < 1e-5:
                    # 打中了
                    print('epside', i, 'step', j, 'action', res.x)
                    action = res.x
                    observation, ceq, done, info = env.step(action)
                    info['t']
                    if memory is not None:
                        memory = np.vstack([memory, info['store'].copy()])
                    else:
                        memory = info['store'].copy()
                    break
    np.save('memory.npy', memory)


def test_memory():
    memory = np.load('memory.npy')
    env = QUAD(True)
    for i in range(100):
        X0 = memory[np.random.choice(len(memory))]
        env.state = X0[:5].copy()
        if np.linalg.norm(env.state) < 1e-6:
            continue
        state, ceq, done, info = env.step(X0[5:], False)
        print('steps', i + 1, 'terminal', ceq[:-1], 'action', X0[5:])
    plt.figure(1)
    plt.plot(info['X'][:, 0], info['X'][:, 1])
    plt.figure(2)
    plt.plot(info['t'], info['X'][:, 1])
    plt.show()


if __name__ == '__main__':
    # get_memory(100)
    test_memory()
