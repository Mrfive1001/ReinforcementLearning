import Missile
import numpy as np


env = Missile.MissileAI()

state = env.reset()
while True:

    a1 = env.robot_action(mode='rand_smart',first=True)
    a2 = env.robot_action(mode='base_smart',first=False)
    state,reward,done,info = env.step(np.hstack((a1,a2)))
    print(state)
    if done:
        break
