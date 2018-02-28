import matplotlib.pyplot as plt
from TSP_Burma14 import ENV

import numpy as np

env = ENV()

plt.figure(1)

trajectory_record = np.zeros([500, 2])
trajectory_record[0, 0] = env.city_location[0][0]
trajectory_record[0, 1] = env.city_location[0][1]
ep_reward = 0
observation = env.reset()  # initial observation
step = 0
for step in range(500):
    # action = RL.choose_action(observation)               # RL choose action based on observation
    action_index = [i for i in range(env.action_dim) if env.state[i] != -1]
    action = np.random.choice(action_index)
    trajectory_record[step+1, 0] = env.city_location[action][0]
    trajectory_record[step+1, 1] = env.city_location[action][1]
    observation_, reward, done, info = env.step(action)  # RL get next observation and reward
    ep_reward += info["distance"]
    print('reward', reward)

    # swap observation
    observation = observation_

    for i in range(len(env.city_location)):
        plt.scatter(env.city_location[i][0], env.city_location[i][1])
    plt.plot(trajectory_record[:step+2, 0], trajectory_record[:step+2, 1])
    plt.show()
    plt.pause(0.1)

    # break while loop when end of this episode
    if done:
        print(step)
        print(ep_reward)
        break
