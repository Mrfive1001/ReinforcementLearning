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
        self.s = tf.placeholder(tf.float32, [None, s_dim], name='s')
        self.areal = tf.placeholder(tf.float32, [None, a_dim], 'areal')
        # 网络和输出向量
        net0 = tf.layers.dense(self.s, self.units, activation=tf.nn.relu, name='l0')
        net1 = tf.layers.dense(net0, self.units, activation=tf.nn.relu, name='l1')
        net2 = tf.layers.dense(net1, self.units, name='l2', activation=tf.nn.relu)
        self.apre = tf.layers.dense(net2, self.a_dim, name='apre')  # 输出线性

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
            return (self.sess.run(self.apre, feed_dict={self.s: X.reshape(1, self.s_dim)})).reshape(self.a_dim)
        else:
            return self.sess.run(self.apre, feed_dict={self.s: X})


if __name__ == '__main__':
    train = False  # 是否进行网络训练
    train = True  # 是否进行网络训练
    net = Dnn(5, 6, 20, train=train)
    if train:
        memory = np.load('memory.npy')
        X = memory[:, :5].copy()
        Y = memory[:, 5:].copy()
        # 读取数据
        # TODO 增加数据预处理
        losses = []
        for i in range(2000):
            sample_index = np.random.choice(len(X), size=100)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss = net.learn(batch_x, batch_y)
            losses.append(loss)
        net.store()
        plt.plot(losses)
        plt.show()
    else:
        # 验证数据的一次打靶率
        env = QUAD(True)
        good = 0.0
        episodes = 200
        for i in range(episodes):
            x = env.reset()
            y = net.predict(x)
            res = env.get_result(y, store=False)  # 打一次靶
            print('action', res.x, res.success)
            if res.success:
                good += 1
        print(good / episodes)
