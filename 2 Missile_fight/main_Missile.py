import Missile
import numpy as np


env = Missile.MissileAI()

state = env.reset()
while True:

    a1 = env.rand_action()
    a2 = env.rand_action()
    state,reward,done,info = env.step(np.hstack((a1,a2)))
    print(state)
    if done:
        break
