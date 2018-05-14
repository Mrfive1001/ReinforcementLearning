'''
Add API by MrFive
DDPG Method
'''

# 这里面的DDPG是双层网络
# 增加done的处理
# 可以保存网络
# 三层网络

import tensorflow as tf
import numpy as np
import os
import sys


# tf.set_random_seed(2)


class Method(object):
    def __init__(
            self,
            method,
            a_dim,  # 动作的维度
            ob_dim,  # 状态的维度
            e_greedy_end=0.1,  # 最后的探索值 0.1倍幅值
            e_liner_times=1000,  # 探索值经历多少次学习变成e_end
            epilon_init=1,  # 表示1倍的幅值作为初始值
            LR_A=0.0001,  # Actor的学习率
            LR_C=0.001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=10000,  # 记忆池容量
            BATCH_SIZE=256,  # 批次数量
            units_a=64,  # Actor神经网络单元数
            units_c=64,  # Crtic神经网络单元数
            actor_learn_start=10000,  # Actor开始学习的代数
            tensorboard=True,
            train=True  # 训练的时候有探索
    ):
        # DDPG网络参数
        self.method = method + '/' + 'LR_A' + str(LR_A) + '/' + 'LR_C' + str(LR_C) + '/' + 'units_a' + str(units_a) \
                      + '/' + 'units_c' + str(units_c)
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.MEMORY_CAPACITY = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.units_a = units_a
        self.units_c = units_c
        self.actor_learn_start = actor_learn_start
        self.epsilon_init = epilon_init  # 初始的探索值
        self.epsilon = self.epsilon_init
        self.epsilon_end = e_greedy_end
        self.e_liner_times = e_liner_times
        self.train = train
        self.tensorboard = tensorboard

        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.iteration = 0

        self.model_path0 = os.path.join(sys.path[0], 'DDPG_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')

        # DDPG构建
        self.memory = np.zeros((self.MEMORY_CAPACITY, ob_dim + a_dim + 1), dtype=np.float32)  # 存储s,a,r,s_,done
        sample_all_U = np.load("sample_all.npy")
        self.memory[0:len(sample_all_U[:, 0])] = sample_all_U
        self.pointer = len(sample_all_U[:, 0])


        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, ob_dim,
        self.S = tf.placeholder(tf.float32, [None, ob_dim], 'state')
        self.a = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.u = tf.placeholder(tf.float32, [None, 1], 'control')

        # 建立actor网络
        with tf.variable_scope('Actor'):
            self.a_pre = self._build_a(self.S, scope='eval', trainable=True)
            tf.summary.histogram('Actor/eval', self.a_pre)
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')

        with tf.variable_scope('control_u'):
            self.u_pre = self._build_u(self.S, scope='eval', trainable=True)
            tf.summary.histogram('control_u/eval', self.u_pre)
            self.ue_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='control_u/eval')

        # actor train


        # self.action_error = tf.reduce_mean(tf.square(td))
        self.action_error = tf.losses.mean_squared_error(labels=self.a, predictions=self.a_pre)
        tf.summary.scalar('action_error', self.action_error)
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.action_error, var_list=self.ae_params)

        # control train
        self.ud = tf.subtract(self.u, self.u_pre, name='action_error')
        self.control_error = tf.losses.mean_squared_error(labels=self.u, predictions=self.u_pre)
        tf.summary.scalar('control_error', self.control_error)
        self.utrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.control_error, var_list=self.ue_params)

        self.actor_saver = tf.train.Saver()
        if self.train:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.actor_saver.restore(self.sess, self.model_path)

        if self.train and self.tensorboard:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./logs/' + self.method, self.sess.graph)

    def choose_action(self, ob):
        s = ob.copy()
        s[0] = 10 * (s[0] - self.constant['r_f'])
        s[1] = 10 * (s[1])
        s[2] = 10 * (s[2] - self.constant['v_f'])
        if self.train:
            action = self.sess.run(self.a_pre, {self.S: s[np.newaxis, :]})[0]  # 直接根据网络得到动作的值
            action = np.random.normal(action, self.epsilon)
        else:
            action = self.sess.run(self.a_pre, {self.S: s[np.newaxis, :]})[0]  # 直接根据网络得到动作的值
        return action

    def choose_u(self, ob):
        s = ob.copy()
        s[0] = 10 * (s[0] - self.constant['r_f'])
        s[1] = 10 * (s[1])
        s[2] = 10 * (s[2] - self.constant['v_f'])
        if self.train:
            action = self.sess.run(self.u_pre, {self.S: s[np.newaxis, :]})[0]  # 直接根据网络得到动作的值
            action[0] = (action[0]) * np.pi / 2
        else:
            action = self.sess.run(self.u_pre, {self.S: s[np.newaxis, :]})[0]  # 直接根据网络得到动作的值
            action[0] = (action[0]) * np.pi / 2
        return action


    def learn(self):
        if self.pointer < self.BATCH_SIZE:
            # 未存储够足够的记忆池的容量
            print('store')
            td_error = 0
            pass
        else:
            # 更新目标网络，有可以改进的地方，可以更改更新目标网络的频率，不过减小tau会比较好
            indices = np.random.choice(min(self.MEMORY_CAPACITY, self.pointer), size=self.BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            bs[:, 0] = 10 * (bs[:, 0] - self.constant['r_f'])
            bs[:, 1] = 10 * (bs[:, 1])
            bs[:, 2] = 10 * (bs[:, 2] - self.constant['v_f'])
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            bu = bt[:, self.s_dim + self.a_dim:]
            bu = np.abs(bu / np.pi * 2)

            # 更新a和c，有可以改进的地方，可以适当更改一些更新a和c的频率
            #
            self.sess.run(self.atrain, {self.S: bs, self.a: ba})
            print('action_error', self.sess.run(self.action_error, {self.S: bs, self.a: ba}))
            # print('a_pre', self.sess.run(self.a_pre, {self.S: bs, self.a: ba}))
            # print('ba', ba)
            self.sess.run(self.utrain, {self.S: bs, self.u: bu})
            print('control_error', self.sess.run(self.control_error, {self.S: bs, self.u: bu}))
            print('u_error', np.max(self.sess.run(self.ud, {self.S: bs, self.u: bu})))


            if self.tensorboard:
                if self.iteration % 10 == 0:
                    result_merge = self.sess.run(self.merged, {self.S: bs, self.a: ba, self.u: bu})
                    self.writer.add_summary(result_merge, self.iteration)

            self.epsilon = max(self.epsilon - (self.epsilon_init - self.epsilon_end) / self.e_liner_times,
                               self.epsilon_end)

            self.iteration += 1



    def store_transition(self, sample):
        # 存储需要的信息到记忆池
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        sample_len = len(sample[:, 0])
        if index+sample_len < self.MEMORY_CAPACITY:
            self.memory[index:index+sample_len, :] = sample
        else:
            self.memory[index:self.MEMORY_CAPACITY, :] = sample[0:self.MEMORY_CAPACITY-index, :]
            self.memory[0:sample_len-(self.MEMORY_CAPACITY-index), :] = sample[self.MEMORY_CAPACITY-index:, :]
        self.pointer += sample_len

    def _build_a(self, s, scope, trainable):
        # 建立actor网络
        with tf.variable_scope(scope):
            n_l1 = self.units_a
            net0 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l0', trainable=trainable)
            net1 = tf.layers.dense(net0, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, n_l1, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net2, self.a_dim, name='a', trainable=trainable)
            return a

    def _build_u(self, s, scope, trainable):
        # 建立actor网络
        with tf.variable_scope(scope):
            n_l1 = self.units_c
            net0 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l0', trainable=trainable)
            net1 = tf.layers.dense(net0, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, n_l1, activation=tf.nn.relu, name='l2', trainable=trainable)
            net3 = tf.layers.dense(net2, n_l1, activation=tf.nn.relu, name='l3', trainable=trainable)
            net4 = tf.layers.dense(net3, n_l1, activation=tf.nn.relu, name='l4', trainable=trainable)
            u = tf.layers.dense(net4, 1, activation=tf.sigmoid, name='a', trainable=trainable)
            # u = tf.layers.dense(net2, 1, name='a', trainable=trainable)
            return u

    def net_save(self):
        self.actor_saver.save(self.sess, self.model_path)
