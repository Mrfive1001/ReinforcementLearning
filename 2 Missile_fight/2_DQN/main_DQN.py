import Missile
import numpy as np
import D3QN

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

env = Missile.MissileAI()
RL = D3QN.DQN(env.action_dim, env.state_dim,
              memory_size=1000, batch_size=64,
              learning_rate=0.001, dueling=True, double=True,
              e_greedy_end=0.1, e_liner_times=20000, units=50,
              train=True, replace_target_iter=50, gamma=0.95)
step = 0
episodes = 100000
win_rate = []
win = 0
for episode in range(1, episodes):
    ep_reward = np.array([0, 0])
    state_now = env.reset()
    while True:
        a1 = env.robot_action(mode='rand_smart', first=True)
        a2 = RL.choose_action(state_now)
        state_next, reward, done, info = env.step(np.array([a1, a2]))
        step += 1
        RL.store_transition(state_now, a2, reward[1], state_next)
        ep_reward += reward
        state_now = state_next
        if done:
            if info['winner'] == 1:
                win += 1
            break
        if step % 20 == 0:
            RL.learn()
    if episode % 100 == 0:
        print("Big Episode: %d" % (episode // 100), "Win rate:%.2f" % (win / 100), 'Epsilon:%.2f' % (RL.epsilon))
        win_rate.append(win / 100)
        win = 0
RL.model_save()
plt.plot(win_rate)
plt.xlabel('Episode')
plt.ylabel('Win_rate')
plt.savefig(r'Train_Result.png')
plt.show()
