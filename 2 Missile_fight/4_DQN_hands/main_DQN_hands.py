import Missile
import numpy as np
import D3QN

import matplotlib.pyplot as plt

env = Missile.MissileAI()

RL_A = D3QN.DQN(env.action_dim, env.state_dim,
                memory_size=1000, batch_size=64,
                learning_rate=0.001, dueling=True, double=True,
                e_greedy_end=0.1, e_liner_times=20000, units=50,
                train=False, replace_target_iter=50, gamma=0.95)
RL_B = D3QN.DQN(env.action_dim, env.state_dim,
                memory_size=1000, batch_size=64,
                learning_rate=0.001, dueling=True, double=True,
                e_greedy_end=0.1, e_liner_times=20000, units=50,
                train=False, replace_target_iter=50, gamma=0.95)
step = 0
episodes = 100000
win_rate = []
win = 0
win_memory = []
for episode in range(1, episodes):
    ep_reward = np.array([0, 0])
    state_now = env.reset()
    while True:
        a1 = RL_A.choose_action(state_now, first=True)
        a2 = RL_B.choose_action(state_now, first=False)
        # a1 = env.robot_action('rand_fool',first=True)
        # a2 = env.robot_action('rand_fool', first=False)
        state_next, reward, done, info = env.step(np.array([a1, a2]))
        step += 1
        ep_reward += reward
        state_now = state_next
        if done:
            if info['winner'] == 1:
                win += 1
                win_memory.append(1)
            else:
                win_memory.append(0)
            break
    if episode % 100 == 0:
        print("Big Episode: %d" % (episode // 100), "Win rate:%.2f" % (win / 100))
        win_rate.append(win / 100)
        win = 0
print('Total Episodes : %d ,' % episodes, 'Average Rate : %.2f' % np.average(win_memory))