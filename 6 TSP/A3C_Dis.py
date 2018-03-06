"""
Asynchronous Advantage Actor Critic (A3C) with Discrete
action space, Reinforcement Learning by MoFan.
Add API by MrFive
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import sys
import copy
import matplotlib.pyplot as plt
from tensorflow.contrib import layers


# tf.set_random_seed(2)


class Para:
    def __init__(self,
                 env,  # 环境参数包括state_dim,action_dim,step,reset
                 units_a=30,  # 双层网络，第一层的大小
                 units_c=100,  # 双层网络，critic第一层的大小
                 MAX_GLOBAL_EP=2000,  # 全局需要跑多少轮数
                 UPDATE_GLOBAL_ITER=30,  # 多少代进行一次学习并且上传到全局网络
                 gamma=0.9,  # 奖励衰减率
                 ENTROPY_BETA=0.01,  # 表征探索大小的量
                 LR_A=0.0001,  # Actor的学习率
                 LR_C=0.001,  # Crtic的学习率
                 ):
        self.N_WORKERS = multiprocessing.cpu_count()
        self.MAX_EP_STEP = 510
        self.MAX_GLOBAL_EP = MAX_GLOBAL_EP
        self.GLOBAL_NET_SCOPE = 'Global_Net'
        self.UPDATE_GLOBAL_ITER = UPDATE_GLOBAL_ITER
        self.gamma = gamma
        self.units_a = units_a
        self.units_c = units_c

        self.ENTROPY_BETA = ENTROPY_BETA
        self.LR_A = LR_A  # learning rate for actor
        self.LR_C = LR_C  # learning rate for critic
        self.modelpath = sys.path[0] + '/my_net/data.chkp'
        self.env = env
        self.N_S = env.state_dim  # 状态的维度
        self.N_A = env.action_dim  # 动作的个数
        self.GLOBAL_RUNNING_R = []
        self.GLOBAL_EP = 0

        self.SESS = tf.Session()
        with tf.device("/cpu:0"):
            self.OPT_A = tf.train.RMSPropOptimizer(self.LR_A, name='RMSPropA')  # actor优化器定义
            self.OPT_C = tf.train.RMSPropOptimizer(self.LR_C, name='RMSPropC')  # critic优化器定义


class A3C:
    def __init__(self, para):
        self.para = para
        with tf.device("/cpu:0"):
            self.GLOBAL_AC = ACNet(para.GLOBAL_NET_SCOPE, para)  # 定义global ， 不过只需要它的参数空间
            self.workers = []
            for i in range(para.N_WORKERS):  # N_WORKERS 为cpu个数
                i_name = 'W_%i' % i  # worker name，形如W_1
                self.workers.append(Worker(i_name, self.GLOBAL_AC, para))  # 添加名字为W_i的worker
        self.actor_saver = tf.train.Saver()

    def run(self):
        self.para.SESS.run(tf.global_variables_initializer())
        COORD = tf.train.Coordinator()
        worker_threads = []
        for worker in self.workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        self.actor_saver.save(self.para.SESS, self.para.modelpath)

        self.display(train_flag=True)

    def display(self, train_flag=False):
        if train_flag:
            pass
        else:
            self.actor_saver.restore(self.para.SESS, self.para.modelpath)

        trajectory_record = np.zeros([500, 2])
        trajectory_record[0, 0] = self.para.env.city_location[0][0]
        trajectory_record[0, 1] = self.para.env.city_location[0][1]
        distance = 0
        observation = self.para.env.reset()  # initial observation
        step = 0
        step_his = []
        for step in range(500):
            action = self.GLOBAL_AC.choose_action_best(observation,
                                                       self.para.env)  # RL choose action based on observation
            # if step == 3:
            #     action = 4
            step_his.append(action)
            trajectory_record[step + 1, 0] = self.para.env.city_location[action][0]
            trajectory_record[step + 1, 1] = self.para.env.city_location[action][1]
            observation_, reward, done, info = self.para.env.step(action)  # RL get next observation and reward
            distance += info["distance"]
            print('reward', reward)

            # swap observation
            observation = observation_

            for i in range(len(self.para.env.city_location)):
                plt.scatter(self.para.env.city_location[i][0], self.para.env.city_location[i][1])
                plt.text(self.para.env.city_location[i][0], self.para.env.city_location[i][1], str(i), size=15,
                         alpha=0.2)
            plt.plot(trajectory_record[:step + 2, 0], trajectory_record[:step + 2, 1])
            plt.show()
            plt.pause(0.5)

            # break while loop when end of this episode
            if done:
                print(step)
                print(distance)
                print('action', step_his)
                break


class ACNet(object):
    def __init__(self, scope, para, globalAC=None):
        self.para = para
        if scope == self.para.GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.para.N_S], 'S')
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
        else:  # worker, local net, calculate losses
            with tf.variable_scope(scope):
                # 网络引入
                self.s = tf.placeholder(tf.float32, [None, self.para.N_S], 'S')  # 状态
                self.a_his = tf.placeholder(tf.int32, [None, 1], 'A')  # 动作
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # 目标价值

                # 网络构建
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                # 正则化项

                a_regu = layers.l2_regularizer(0.1)
                self.a_regu_loss = tf.contrib.layers.l2_regularizer(a_regu)

                # 价值网络优化
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob) * tf.one_hot(self.a_his, self.para.N_A, dtype=tf.float32),
                        axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = self.para.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v) + self.a_regu_loss

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)  # 计算梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)  # 计算梯度
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 把全局的pull到本地
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  # 根据本地的梯度，优化global的参数
                    self.update_a_op = self.para.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.para.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # 网络定义
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a1 = tf.layers.dense(self.s, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la1')
            l_a = tf.layers.dense(l_a1, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            a_prob = tf.layers.dense(l_a, self.para.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c1 = tf.layers.dense(self.s, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc1')
            l_c = tf.layers.dense(l_c1, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # 函数：执行push动作
        self.para.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # 函数：执行pull动作
        self.para.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, env):  # 函数：选择动作action
        prob_weights = self.para.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]}).reshape(self.para.N_S)
        state_tem = env.state.copy()
        index_valid = [x for x in range(env.action_dim) if state_tem[x] == 1]
        prob_valid = [prob_weights[x] for x in index_valid]  # 有效的prob
        prob_valid /= np.sum(prob_valid)  # 归一化
        num_valid = prob_valid.shape[0]
        action_valid = np.random.choice(range(num_valid),
                                        p=prob_valid.ravel())  # select action w.r.t the actions prob
        action = index_valid[action_valid]
        return action

    def choose_action_best(self, s, env):  # 函数：选择动作action
        prob_weights = self.para.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]}).reshape(self.para.N_S)
        state_tem = env.state.copy()
        index_valid = [x for x in range(env.action_dim) if state_tem[x] == 1]
        prob_valid = [prob_weights[x] for x in index_valid]  # 有效的prob
        prob_valid /= np.sum(prob_valid)  # 归一化       ]
        action_valid = np.argmax(prob_valid)
        action = index_valid[action_valid]
        return action


class Worker(object):
    def __init__(self, name, globalAC, para):
        self.name = name
        self.para = para
        self.env_l = copy.deepcopy(self.para.env)
        self.AC = ACNet(name, para, globalAC)

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []  # 类似于memory，存储运行轨迹
        while self.para.GLOBAL_EP < self.para.MAX_GLOBAL_EP:
            s = self.env_l.reset()
            ep_r = 0
            for ep_t in range(self.para.MAX_EP_STEP):  # MAX_EP_STEP每个片段的最大个数
                a = self.AC.choose_action(s, self.env_l)  # 选取动作
                s_, r, done, info = self.env_l.step(a)

                ep_r += info["distance"]
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)  # normalize

                if total_step % self.para.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.para.SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.para.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:  # 每个片段结束，输出一下结果
                    self.para.GLOBAL_RUNNING_R.append(ep_r)
                    print(
                        self.name,
                        "Ep:", self.para.GLOBAL_EP,
                        "| Ep_r: %.1f" % self.para.GLOBAL_RUNNING_R[-1],
                    )
                    self.para.GLOBAL_EP += 1
                    break
