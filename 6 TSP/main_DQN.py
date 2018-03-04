import matplotlib.pyplot as plt
from D3QN import DQN
from TSP_Burma14 import ENV

if __name__ == "__main__":
    env = ENV()
    RL = DQN(n_actions=env.action_dim,
             n_features=env.state_dim,
             learning_rate=0.01,
             gamma=0.9,
             e_greedy_end=0.05,
             memory_size=3000,
             e_liner_times=10000,
             batch_size=256,
             double=True,
             dueling=True,
             train = True
             )
    step = 0
    ep_reward = 0
    episodes = 100000
    for episode in range(episodes):
        ep_reward = 0

        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            ep_reward += reward
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print('Episode:', episode + 1, ' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f' % RL.epsilon)
    # end of game
    print('game over')
    # print(ep_reward)