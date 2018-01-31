import Missile
import numpy as np
import D3QN

env = Missile.MissileAI()
RL = D3QN.DQN(env.action_dim, env.state_dim,
              memory_size = 1000,batch_size=64,
              learning_rate=0.001, dueling=True,
              e_greedy_end = 0.1,e_liner_times=10000)
step = 0
episodes = 30000
for episode in range(episodes):
    ep_reward = np.array([0, 0])
    state_now = env.reset()
    while True:
        a1 = env.robot_action(mode='rand_fool', first=True)
        a2 = RL.choose_action(state_now)
        state_next, reward, done, info = env.step(np.array([a1, a2]))
        step += 1
        RL.store_transition(state_now,a2,reward[1],state_next)
        ep_reward += reward
        state_now = state_next
        if done:
            break
        if step % 20 == 0:
            RL.learn()
    print('Episode:', episode + 1, 'epsilon: %.3f' % RL.epsilon, 'Reward：%.f,%.f' % (reward[0], reward[1]))
