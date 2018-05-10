import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from quad import QUAD


class DNN:
    def __init__(self, s_dim, a_dim, units, train=True):
        # 输入维度、输出维度、单元数、是否训练
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.units = units
        self.train = train
        # 保存网络位置
        self.model_path0 = os.path.join(sys.path[0], 'DNN_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')
        # 输入向量
        self.scale = None
        self.s = tf.placeholder(tf.float32, [None, s_dim], name='s')
        self.areal = tf.placeholder(tf.float32, [None, a_dim], 'areal')
        # 网络和输出向量
        net0 = tf.layers.dense(self.s, self.units, activation=tf.nn.relu, name='l0')
        net1 = tf.layers.dense(net0, self.units, activation=tf.nn.relu, name='l1')
        net2 = tf.layers.dense(net1, self.units, name='l2', activation=tf.nn.relu)
        self.apre = tf.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='apre')  # 输出线性

        self.loss = tf.reduce_mean(tf.squared_difference(self.areal, self.apre))  # loss函数
        self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)  # 训练函数
        # 保存或者读取网络
        self.sess = tf.Session()
        self.actor_saver = tf.train.Saver()
        if self.train:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.actor_saver.restore(self.sess, self.model_path)

    def learn(self, X, Y):
        # 使用网络对样本进行训练
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.s: X, self.areal: Y})
        return loss

    def store(self):
        # 存储网络
        self.actor_saver.save(self.sess, self.model_path)

    def predict(self, X):
        # 对X进行预测
        if X.shape == (self.s_dim,):
            x = (X.reshape(1, self.s_dim) / self.scale[:self.s_dim])
            return ((self.sess.run(self.apre, feed_dict={self.s: x})) * self.scale[self.s_dim:]).reshape(self.a_dim)
        else:
            return self.sess.run(self.apre, feed_dict={self.s: X})

    def norm(self, data):
        if self.scale is None:
            self.scale = np.fabs(data).max(axis=0)
        return data / self.scale

    def unorm(self, data):
        return data * self.scale


if __name__ == '__main__':
    train = False  # 是否进行网络训练
    # train = True  # 是否进行网络训练
    net = DNN(5, 6, 100, train=train)
    memory = np.load('memory.npy')
    memory_norm = net.norm(memory)
    if train:
        X = memory_norm[:, :5].copy()
        Y = memory_norm[:, 5:].copy()
        # 读取数据
        losses = []
        for i in range(20000):
            sample_index = np.random.choice(len(X), size=1000)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss = net.learn(batch_x, batch_y)
            losses.append(loss)
            print(i + 1, loss)
        net.store()
        plt.plot(losses)
        plt.show()
    else:
        # 验证集的打靶率
        X = memory[:, :5].copy()
        Y = memory[:, 5:].copy()
        sample_index = np.random.choice(len(X), size=100)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]

        env = QUAD(True)
        good = 0.0
        episodes = 1
        for i in range(episodes):
            for j in range(1):
                env.state = batch_x[i, :].copy()
                y = batch_y[i, :]
                if np.linalg.norm(env.state) < 1e-6:
                    continue
                res = env.get_result(y, store=False)  # 打一次靶
                success = True if np.linalg.norm(res.fun) < 1e-5 else False
                print(i + 1, 'state', env.state, success)
                if success:
                    good += 1
                    break
        print(good / episodes)
        # 验证数据的一次打靶率
        env = QUAD(True)
        good = 0.0
        episodes = 100
        for i in range(episodes):
            for j in range(2):
                x = env.reset()
                y = net.predict(x)
                res = env.get_result(y, store=False)  # 打一次靶
                success = True if np.linalg.norm(res.fun) < 1e-5 else False
                print(i + 1, 'state', x, success)
                if success:
                    good += 1
                    break
        print(good / episodes)
