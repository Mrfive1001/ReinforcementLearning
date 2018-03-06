"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581
Add API by MrFive
Include DQN DuelingDQN DoubleDQN
"""
# 包含done的处理
# 可以保存网络
# 三层神经网络

import numpy as np
import tensorflow as tf
import os
import sys


class DQN:
    def __init__(
            self,
            n_actions,  # 动作空间个数（输出动作维度）
            n_features,  # 状态空间个数（输入状态量个数）
            learning_rate=0.001,  # 学习率
            replace_target_iter=200,  # 更新target网络代数
            memory_size=500,  # 记忆池数量
            batch_size=32,  # 每次样本更新数目
            dueling=False,  # 使用Dueling 使用优势函数网络改进
            double=False,  # 使用Double 使用两个网络
            gamma=0.9,  # 衰减系数
            e_greedy_init=1,  # 初始探索概率
            e_greedy_end=0.1,  # 最后的探索概率
            e_liner_times=1000,  # 探索概率经历多少次学习不再减少
            units=50,  # 每层网络的单元数
            train=True,  # 训练的时候有探索

    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_liner_times = e_liner_times
        self.epsilon_init = e_greedy_init
        self.epsilon = self.epsilon_init
        self.epsilon_end = e_greedy_end
        self.dueling = dueling
        self.double = double
        self.units = units
        self.train = train

        self.learn_step_counter = 0  # 学习轮数

        self.model_path0 = os.path.join(sys.path[0], 'DQN_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_net()
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            self.sess = tf.Session(graph=self.graph)
            self.actor_saver = tf.train.Saver()
            if self.train:
                self.sess.run(tf.global_variables_initializer())
            else:
                self.actor_saver.restore(self.sess, self.model_path)

    def _build_net(self):
        # 建立网络
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            # 建立一层
            with tf.variable_scope('l0'):
                w0 = tf.get_variable('w0', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b0 = tf.get_variable('b0', [1, n_l1], initializer=b_initializer, collections=c_names)
                l0 = tf.nn.relu(tf.matmul(s, w0) + b0)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(l0, w1) + b1)

            if self.dueling:  # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.units, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_, done):
        # 将历史记录存进记忆池
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_, done))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 根据状态选择动作,可以稍微进行一下筛选
        observation = observation[np.newaxis, :]
        if self.train and np.random.uniform() > self.epsilon:
            # 训练状态并且不是随机选择
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
            action = int(np.argmax(actions_value))
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 更新网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 更新target网络
            self.sess.run(self.replace_target_op)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = batch_memory[:, :self.n_features]
        batch_action = batch_memory[:, self.n_features].astype(int)
        batch_reward = batch_memory[:, self.n_features + 1]
        batch_state_next = batch_memory[:, -self.n_features - 1:-1]
        batch_done = batch_memory[:, -1].astype(int)
        if self.double == False:
            q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_state_next})  # next observation
            q_eval = self.sess.run(self.q_eval, {self.s: batch_state})
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, batch_action] = batch_reward + (1 - batch_done) * self.gamma * np.max(q_next, axis=1)
        else:
            # double DQN
            q_target = self.sess.run(self.q_eval, feed_dict={self.s: batch_state})
            q_next1 = self.sess.run(self.q_eval, feed_dict={self.s: batch_state_next})
            q_next2 = self.sess.run(self.q_next, feed_dict={self.s_: batch_state_next})
            batch_action_withMaxQ = np.argmax(q_next1, axis=1)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_next_Max = q_next2[batch_index, batch_action_withMaxQ]
            q_target[batch_index, batch_action] = batch_reward + (1 - batch_done) * self.gamma * q_next_Max
        _,  = self.sess.run([self._train_op, self.loss],
                                     feed_dictcost={self.s: batch_state,
                                                self.q_target: q_target})
        if self.train == True:
            self.epsilon = max(self.epsilon - (self.epsilon_init - self.epsilon_end) / self.e_liner_times,
                               self.epsilon_end)
            self.learn_step_counter += 1
        else:
            self.epsilon = 0

    def model_save(self):
        self.actor_saver.save(self.sess, os.path.join(self.model_path0, 'data.chkp'))


if __name__ == '__main__':
    env = ENV()
    RL = DQN(n_actions=env.action_dim,
             n_features=env.state_dim,
             learning_rate=0.001,
             gamma=0.90,
             e_greedy_end=0.1,
             e_greedy_init=0.8,
             memory_size=3000,
             e_liner_times=10000,
             units=10,
             batch_size=64,
             double=True,
             dueling=True,
             train=True
             )
    step = 0
    ep_reward = 0
    if RL.train:
        episodes = 20000
        for episode in range(episodes):
            ep_reward = 0
            # initial observation
            observation = env.reset()
            while True:
                env.render()
                action = RL.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                ep_reward += reward
                RL.store_transition(observation, action, reward, observation_, done)
                if (step > 200) and (step % 5 == 0):
                    RL.learn()
                observation = observation_
                if done:
                    break
                step += 1
            print('Episode:', episode + 1, ' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f' % RL.epsilon)
        RL.model_save()
    else:
        pass
    # Display
