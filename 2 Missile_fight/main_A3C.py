import Missile
import numpy as np
import matplotlib
import A3C_Discrete as A3C
import matplotlib
import matplotlib.pyplot as plt

env = Missile.MissileAI()
para = A3C.Para(env,
                units_a=100,
                units_c=100,
                MAX_GLOBAL_EP=10000,
                UPDATE_GLOBAL_ITER=20,
                gamma=0.95,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001,
                oppo='rand_smart')
RL = A3C.A3C(para)
RL.run()
step = 0
episodes = 2000
win_rate = []
win = 0
for episode in range(episodes):
    ep_reward = np.array([0, 0])
    state_now = env.reset()
    while True:
        a1 = env.robot_action(mode='rand_smart', first=True)
        a2 = RL.choose_action(state_now)
        state_next, reward, done, info = env.step(np.array([a1, a2]))
        step += 1
        ep_reward += reward
        state_now = state_next
        if done:
            if ep_reward[0] < ep_reward[1]:
                win += 1
            break
    if episode % 100 == 0:
        print("Big Episode: %d" % (episode // 100), "Win rate:%.2f" % (win / 100))
        win_rate.append(win / 100)
        win = 0
plt.plot(win_rate)
plt.xlabel('Episode')
plt.ylabel('Win_rate')
plt.savefig('Result.png')
plt.show()

