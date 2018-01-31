import Missile
import numpy as np
import D3QN

env = Missile.MissileAI()
RL = D3QN.DQN(env.action_dim, env.state_dim,
              learning_rate=0.001, dueling=True)
step = 0
episodes = 300
for episode in range(episodes):
    ep_reward = np.array([0, 0])
    state_now = env.reset()
    while True:
        a1 = env.robot_action(mode='rand_smart', first=True)
        a2 = Rl.choose_action(state_now)
        state_next, reward, done, info = env.step(np.hstack((a1, a2)))
        step += 1
        ep_reward += reward
        state_now = state_next
        if done:
            break
    print('Episode:', episode + 1, 'epsilon: %.3f' % RL.epsilon, 'Reward：%.f,%.f' % (reward[0], reward[1]))
