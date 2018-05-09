import numpy as np
from quad import QUAD
import matplotlib.pyplot as plt

memory = None

env = QUAD(True)
for i in range(25):
    env.reset()
    for j in range(100):
        action = np.random.rand(7) * 2 - 1
        res = env.get_result(action)
        if res.success:
            print('epside', i, 'step', j, 'action', res.x)
            action = res.x
            observation, ceq, done, info = env.step(action)
            if memory is not None:
                memory = np.vstack([memory, info['X'].copy()])
            else:
                memory = info['X'].copy()
            break
np.save('memory.npy',memory)