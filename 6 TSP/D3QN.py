"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Add API by MrFive

Include DQN DuelingDQN DoubleDQN

双层网络
"""

import numpy as np
import tensorflow as tf

# np.random.seed(1)
# tf.set_random_seed(1)


class DQN:
    def __init__(
            self,
            n_actions,  # 动作空间动作数目
            n_features,  # 状态个数
            learning_rate=0.001,
            replace_target_iter=200,  # 更新目标网络代数
            memory_size=500,  # 记忆池数量
            batch_size=32,  # 每次更新数目
            output_graph=False,
            sess=None,
            dueling=False,  # 使用Dueling
            double=False,
            gamma=0.9,
            e_greedy_init = 1,
            e_greedy_end=0.1,  # 最后的探索值 e_greedy
            e_liner_times=1000,  # 探索值经历多少次学习变成e_end
            units = 50,
            train = True  # 训练的时候有探索

    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_liner_times = e_liner_times
        self.epsilon_init = e_greedy_init  # 初始的探索值
        self.epsilon = self.epsilon_init
        self.epsilon_end = e_greedy_end

        self.dueling = dueling  # decide to use dueling DQN or not
        self.double = double

        self.units = units
        self.train = train

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
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

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        state_tem = observation.copy()
        observation = observation[np.newaxis, :]
        if self.train:
            if np.random.uniform() > self.epsilon:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
                index_valid = [x for x in range(len(state_tem)) if state_tem[x] == 1]
                value_valid = [actions_value[x] for x in index_valid]
                action_valid = int(np.argmax(value_valid))
                action = index_valid[action_valid]
            else:
                # action = np.random.randint(0, self.n_actions)
                index_valid = [x for x in range(len(state_tem)) if state_tem[x] == 1]
                action = np.random.choice(index_valid)
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
            index_valid = [x for x in range(len(state_tem)) if state_tem[x] == 1]
            value_valid = [actions_value[x] for x in index_valid]
            action_valid = int(np.argmax(value_valid))
            action = index_valid[action_valid]
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_state = batch_memory[:, :self.n_features]
        batch_action = batch_memory[:, self.n_features].astype(int)
        batch_reward = batch_memory[:, self.n_features + 1]
        batch_state_next = batch_memory[:, -self.n_features:]

        if self.double == False:
            q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_state_next}) # next observation
            q_eval = self.sess.run(self.q_eval, {self.s: batch_state})

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, batch_action] = batch_reward + self.gamma * np.max(q_next, axis=1)
        else:
            # double DQN
            q_target = self.sess.run(self.q_eval, feed_dict={self.s: batch_state})
            q_next1 = self.sess.run(self.q_eval, feed_dict={self.s: batch_state_next})
            q_next2 = self.sess.run(self.q_next, feed_dict={self.s_: batch_state_next})
            batch_action_withMaxQ = np.argmax(q_next1, axis=1)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_next_Max = q_next2[batch_index, batch_action_withMaxQ]
            q_target[batch_index, batch_action] = batch_reward + self.gamma * q_next_Max
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_state,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        if self.train == True:
            self.epsilon = max(self.epsilon - (self.epsilon_init - self.epsilon_end) / self.e_liner_times, self.epsilon_end)
        else:
            self.epsilon = 0
        self.learn_step_counter += 1
