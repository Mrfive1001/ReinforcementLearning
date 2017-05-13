import numpy as np
import pickle

person_numbers = 2
store_numbers = 3


class State(object):
    def __init__(self):
        self.data = np.array([[3, 4, 4, 10], [3, 4, 4, 10]])
        self.hashVals = None
        self.end = None
        self.actions = None
        
        self.winner = None
        self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0]],
                             [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 60]],
                             [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0.55, 100]]])
        # self.hit = np.array([[[1, 0], [1, 0], [0, 0], [0, 0]],
        #                      [[1, 0], [1, 0], [1, 0], [0.7, 60]],
        #                      [[1, 0], [1, 0], [1, 0], [0.55, 100]]])
        # hit[i,j]第i个导弹命中j个地方的概率

    def get_hash_action(self):
        if self.hashVals:
            return self.hashVals
        self.hashVals = []
        for action in self.actions:
            hashVal = 0
            for i in action.reshape(person_numbers*(store_numbers+1)):
                hashVal = hashVal * 2 + i
            for i in self.data.reshape(person_numbers*store_numbers):
                hashVal = hashVal * 5 + i
            # 这里的5保证哈希值唯一
            self.hashVals.append(int(hashVal))
        return self.hashVals

    def get_actions(self, symbol):
        if self.actions:
            return self.actions
        for num, i in enumerate(self.data[symbol, :-1]):
            if i == 0:
                continue
            for j in range(store_numbers+1):
                a = np.zeros((person_numbers, store_numbers+1))
                a[symbol, num] = 1
                a[0 if symbol else 1, j] = 1
                self.actions.append(a)
        return self.actions

    def is_end(self):
        if self.end is not None:
            return self.end
        damages = self.data[:, 3]
        if (self.data.sum()-damages.sum()) and damages[0] and damages[1]:
            self.end = False
        else:
            self.end = True
            self.winner = 1 if damages[1] > damages[0] else (0 if damages[1] != damages[0] else -1)
            # -1是平局
        return self.end


class Player(object):
    def __init__(self, symbol):
        self.sas = []
        self.possible = dict()
        self.symbol = symbol

    def reset(self):
        self.sas = []

    def feedState(self, sa):
        self.sas.append(sa)

    def feedReward(self):
        pass

def train(epochs = 10):
    player1 = Player(0)
    player2 = Player(1)


