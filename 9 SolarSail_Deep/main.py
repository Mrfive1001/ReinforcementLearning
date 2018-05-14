
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

########################### Package  Input  #################################

from Method import Method as Method
from SolarSail import SolarSail as Object_AI
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################
max_SL = 20000
max_Episodes = 1000
max_Ep_Steps = 2000
rendering = False
############################ Object and Method  ####################################

env = Object_AI(random=True)

ob_dim = env.ob_dim
print("环境状态空间维度为", ob_dim)
print('-----------------------------\t')
a_dim = env.lambda_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')


## method settting
method = 'r0=1,4'
# tensorboard --logdir="2 QUAD/6 DL_OP/1 DDPG_OP/logs"
train_flag = True
train_flag = False
RLmethod = Method(
            method,
            env.lambda_dim,  # 动作的维度
            env.ob_dim,  # 状态的维度
            e_greedy_end=0.000,  # 最后的探索值 0.1倍幅值
            e_liner_times=max_Episodes*0.2,  # 探索值经历多少次学习变成e_end
            epilon_init=1,  # 表示1倍的幅值作为初始值
            LR_A=0.0001,  # Actor的学习率
            LR_C=0.0001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=20000,  # 记忆池容量
            BATCH_SIZE=500,  # 批次数量
            units_a=300,  # Actor神经网络单元数
            units_c=300,  # Crtic神经网络单元数
            actor_learn_start=10000,  # Actor开始学习的代数
            tensorboard=True,  # 是否存储tensorboard
            train=train_flag  # 训练的时候有探索
            )

RLmethod.constant = env.constant
###############################  training  ####################################


if RLmethod.train:

    for i in range(max_SL):
        RLmethod.learn()
        print('step', i)

    # iiii = 0
    # for j in range(100):
    #     observation = env.reset()
    #     action = RLmethod.choose_action(observation)
    #     res = env.get_optimize(action)
    #     success = True if np.linalg.norm(res.fun) < 1e-5 else False
    #     if success:
    #         print(j)
    #         iiii += 1
    # print(iiii)  # 69次成功


    # for i in range(max_Episodes):
    #     observation = env.reset()
    #     action = RLmethod.choose_action(observation)
    #     res = env.get_optimize(action)
    #     success = True if np.linalg.norm(res.fun) < 1e-5 else False
    #     if success:
    #         sample = env.get_sample(res.x)
    #         RLmethod.store_transition(sample)
    #     RLmethod.learn()
    #     print('step', i, 'success', success)

    RLmethod.net_save()
    # np.save("memory.npy", RLmethod.memory)

else:
    sample_all = np.load("sample_all.npy")
    indices = np.random.choice(len(sample_all[:, 0]), size=1)
    indices = np.array([101])
    print(indices)
    sample = sample_all[indices, :]
    sample = sample.reshape([len(sample[0, :])])

    sample_state = sample[0:env.ob_dim]
    observation = env.reset(sample_state)
    print('observation', observation)
    sample_lambda = sample[env.ob_dim:env.ob_dim + env.lambda_dim]
    print('sample_lambda', sample_lambda)
    print('u', sample_all[indices[0]:indices[0]+100, 7])

    observation, ceq, done, info = env.get_lam_tra(sample_lambda)
    print(ceq)

    X = info['X']
    plt.figure(1)
    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), 'm')
    plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
    plt.plot(X[:, 1], X[:, 0], 'r')
    plt.title('sample')

    ############################### Comparison  ####################################

    test_flag = 3   # flag
    # 1 测试样本的学习情况
    # 2 测试lambda0作为轨迹初值的情况
    # 3 测试连续调用lambda控制轨迹的情况
    # 4 测试连续调用u控制轨迹的情况
    if test_flag == 1:  # 测试样本的学习情况
        for i in range(100):
            indices += 1
            sample = sample_all[indices, :]
            sample = sample.reshape([len(sample[0, :])])

            sample_state = sample[0:env.ob_dim]
            observation = env.reset(sample_state)
            print('observation', observation)
            sample_lambda = sample[env.ob_dim:env.ob_dim + env.lambda_dim]
            print('sample_lambda', sample_lambda)
            sample_u = sample[-1]
            print('sample_u', sample_u)

            net_u = RLmethod.choose_u(sample_state)
            print('net_u', net_u)

            net_lambda = RLmethod.choose_action(sample_state)
            print('net_lambda', net_lambda)

    elif test_flag == 2:  # 测试lambda0作为轨迹初值的情况
        net_action = RLmethod.choose_action(sample_state)
        print('action', net_action)
        observation, ceq, done, info = env.get_lam_tra(net_action)
        print(ceq)

        X = info['X']
        plt.figure(2)
        plt.subplot(111, polar=True)
        theta = np.arange(0, 2 * np.pi, 0.02)
        plt.plot(theta, 1 * np.ones_like(theta), 'm')
        plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
        plt.plot(X[:, 1], X[:, 0], 'r')
        plt.title('net')
        plt.show()

    elif test_flag == 3: # 测试连续调用lambda控制轨迹的情况

        observation = env.reset(sample_state)
        action = RLmethod.choose_action(observation)
        env.t_f1 = sample_lambda[3]
        # env.t_f1 = 10
        while True:
            # Add exploration noise
            action = RLmethod.choose_action(observation)
            observation_, ceq, done, info = env.step_lambda(action)

            plt.scatter(env.state[1], env.state[0], color='b')
            # plt.pause(0.000001)

            observation = observation_

            if done:
                print(ceq)
                plt.scatter(env.state[1], env.state[0], color='b')
                plt.pause(100000000)
                plt.show()
                break

    elif test_flag == 4: # 测试连续调用u控制轨迹的情况
        observation = env.reset(sample_state)
        env.t_f1 = sample_lambda[3]
        while True:

            # Add exploration noise
            lambda_n = RLmethod.choose_action(observation)
            lambda3 = lambda_n[1]
            lambda4 = lambda_n[2]

            if np.abs(lambda4) < 0.00001:
                if lambda3 <= 0:
                    alpha = 0
                else:
                    alpha = np.pi / 2
            else:
                aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
                alpha = np.arctan(aaa)
            print('lambda_u', alpha)
            action = RLmethod.choose_u(observation)
            if np.abs(lambda4) < 0.00001:
                if lambda3 <= 0:
                    action = 0
                else:
                    action = np.pi / 2
            else:
                aaa = (-3 * lambda3 - np.sqrt(9 * lambda3 ** 2 + 8 * lambda4 ** 2)) / 4 / lambda4
            action = np.sign(aaa)*action
            print('u', action)
            observation_, ceq, done, info = env.step_u(action)

            plt.scatter(env.state[1], env.state[0], color='b')
            # plt.pause(0.000001)

            observation = observation_
            print('ceq', ceq)

            if done:
                print(ceq)
                plt.scatter(env.state[1], env.state[0], color='b')
                plt.pause(100000000)
                plt.show()
                break

















