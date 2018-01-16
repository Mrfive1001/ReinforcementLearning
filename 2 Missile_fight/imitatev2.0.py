import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
url = 'http://sc.ftqq.com/SCU7896Td21f33141f474e9dfec81d7da26daacc5900906769c16.send'

# 策略梯度
person_numbers = 2
store_numbers = 3
near = 6
mid = 4
long = 3


learn = 0.05
# 学习率
train_times = 30000
# 训练次数
# 1000000次 2800s
ip = 0.3
# 拦截率
test_times = 5000
# 测试次数


class State(object):
    def __init__(self):
        self.data = np.array([[near, mid, long], [near, mid, long]])
        self.hashVal = None
        self.end = None
        self.actions = []
        self.damage = 0
        self.ip = ip
        # self.hit = np.array([[[1, 1], [1, 1], [0, 0], [0, 0]],
        #                      [[1, 1], [1, 1], [1, 1], [1, 60]],
        #                      [[1, 1], [1, 1], [1, 1], [1, 100]]])  # 90%     200000次数时间
        self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0]],
                             [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 60]],
                             [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0.55, 100]]])   # 69%胜率
        # self.hit = np.array([[[0.4, 0.4], [0.3, 0.4], [0, 0], [0, 0]],
        #                      [[0.5, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 60]],
        #                      [[0.7, 0.6], [0.6, 0.6], [0.6, 0.6], [0.6, 100]]])# 73%胜率   200000次数时间 578s
        # hit[i,j]第i个导弹命中j个地方的概率

    def get_hash(self):
        if self.hashVal is None:
            self.hashVal = 0
            for num in self.data.reshape(person_numbers*store_numbers):
                self.hashVal = self.hashVal * 10 + num
                # 这里的10保证哈希值唯一
        return int(self.hashVal)

    def is_end(self):
        if self.end is not None:
            return self.end
        if self.data.sum():
            self.end = False
        else:
            self.end = True
        return self.end

    # symbol是0或者1
    # Fool的话是可以打对面的空基地
    def get_actions(self, symbol, fool=True):
        if self.actions:
            return self.actions
        for num, val in enumerate(self.data[symbol, :]):
            if val == 0:
                continue
            for target in range(store_numbers+1):
                if fool or target == store_numbers or self.data[symbol^1, target] != 0:
                    a = np.zeros((person_numbers, store_numbers+1))
                    a[symbol, num] = 1
                    a[0 if symbol else 1, target] = 1
                    self.actions.append(a)
                else:
                    continue
        return self.actions

    # 一个动作分为，选动作和攻击动作
    def next_state_action(self, action, symbol):
        result = action.argmax(1)
        newState = State()
        newState.data = np.copy(self.data)
        newState.data[symbol, result[symbol]] -= 1
        return newState

    def next_state_hit(self, action, symbol):
        result = action.argmax(1)
        newState = State()
        newState.data = np.copy(self.data)
        target = result[symbol ^ 1]
        if np.random.binomial(1, self.ip):
            return newState
        else:
            if target < 3:
                rate = self.hit[result[symbol], target]
                if np.random.binomial(1, rate[0]):
                    num_left = newState.data[0 if symbol else 1, target]
                    sub = 0
                    for num in range(num_left):
                        if np.random.binomial(1, rate[1]):
                            sub += 1
                    num_left -= sub
                    newState.data[0 if symbol else 1, target] = num_left
            else:
                rate = self.hit[result[symbol], target]
                if np.random.binomial(1, rate[0]):
                    damage = rate[1]
                    newState.damage = -damage
            return newState
        # return newState


def get_all_states():
    allStates = dict()
    for i in range(near+1):
        for j in range(mid+1):
            for k in range(long+1):
                for a in range(near+1):
                    for b in range(mid+1):
                        for c in range(long+1):
                            state = State()
                            state.data = np.array([[i, j, k], [a, b, c]])
                            allStates[state.get_hash()] = (state, state.is_end())
    return allStates


class Player(object):
    # strategy 策略是0的时候随机打，是1的时候利用策略梯度
    # strategy 策略是2的时候
    # fool为1的时候可以打对面的空基地
    # 名字是为了后面的继承准备
    def __init__(self, symbol, strategy=1, fool=True, name='red'):
        self.symbol = symbol
        self.states_actions = {}
        self.states = []  # 是state不是哈希值
        self.actions = []
        self.strategy = strategy
        self.fool = fool
        self.name = name
        for state in allStates.values():
            state[0].actions = []
        for key, val in allStates.items():
            self.states_actions[key] = [val[0], val[1], val[0].get_actions(self.symbol, fool=self.fool)]
            length = len(self.states_actions[key][2])
            if length:
                rates = np.linspace(1/length, 1/length, length)
                values = np.zeros(length)
                self.states_actions[key].append(rates)
                self.states_actions[key].append(values)
        # self.states_actions[key][state,isend,actions,rate,values]

    def reset(self):
        self.states = []
        self.actions = []

    def feed_state(self, state):
        self.states.append(state)

    def feed_reward(self, reward):
        if self.strategy == 1:
            self.states = [state.get_hash() for state in self.states]
            for index0, state in enumerate(self.states):
                action = self.actions[index0]
                for index1, action1 in enumerate(self.states_actions[state][2]):
                    if (action == action1).all():
                        self.states_actions[state][3][index1] += reward
                        self.states_actions[state][3] = [x / sum(self.states_actions[state][3])
                                                         for x in self.states_actions[state][3]]
                        break
        self.reset()

    def select_action(self):
        old_state = self.states[-1]
        nextactions = self.states_actions[old_state.get_hash()][2]
        if nextactions:
            if self.strategy == 0 or self.strategy == 1:
                rates = self.states_actions[old_state.get_hash()][3]
                ran = np.random.random()
                temp = 0
                for index, rate in enumerate(rates):
                    temp += rate
                    if ran < temp:
                        action = nextactions[index]
                        break
            elif self.strategy == 2:
                action = np.zeros((person_numbers, store_numbers + 1))
                result = [0, 0]
                if old_state.data[self.symbol, 0] > 0:
                    if old_state.data[self.symbol ^ 1, 0] > 0:
                        result = [0, 0]
                    elif old_state.data[self.symbol ^ 1, 1] > 0:
                        result = [0, 1]
                elif old_state.data[self.symbol, 1] > 0:
                    if old_state.data[self.symbol ^ 1, 1] > 0:
                        result = [1, 1]
                    elif old_state.data[self.symbol ^ 1, 2] > 0:
                        result = [1, 2]
                    else:
                        result = [1, 3]
                else:
                    if old_state.data[self.symbol ^ 1, 1] > 0:
                        result = [2, 1]
                    elif old_state.data[self.symbol ^ 1, 2] > 0:
                        result = [2, 2]
                    else:
                        result = [2, 3]
                action[self.symbol, result[0]] = 1
                action[self.symbol ^ 1, result[1]] = 1
            self.actions.append(action)
            return action
        else:
            self.states.pop(-1)
            return None

    def next_state_action(self, state, action):
        return state.next_state_action(action, self.symbol)

    def next_state_hit(self, state, action):
        return state.next_state_hit(action, self.symbol)

    def save_policy(self):
        with open('policy_2.0' + self.name, 'wb') as f:
            pickle.dump(self.states_actions, f)

    def load_policy(self):
        with open('policy_2.0' + self.name, 'rb') as f:
            self.states_actions = pickle.load(f)


def war(player1, player2):
    player1.reset()
    player2.reset()
    state = State()
    flag = state.is_end()
    damages = [-np.random.random(), -np.random.random()]
    while flag == 0:
        player1.feed_state(state)
        player2.feed_state(state)
        actionA = player1.select_action()
        actionB = player2.select_action()
        if actionA is not None:
            state = player1.next_state_action(state, actionA)
        if actionB is not None:
            state = player2.next_state_action(state, actionB)
        if actionA is not None:
            state = player1.next_state_hit(state, actionA)
            damages[1] += state.damage
        if actionB is not None:
            state = player2.next_state_hit(state, actionB)
            damages[0] += state.damage
        flag = state.is_end()
    if damages[0] > damages[1]:
        player1.feed_reward(learn)
        return 1
    elif damages[1] > damages[0]:
        player2.feed_reward(learn)
        return -1
    else:
        player1.feed_reward(learn / 2)
        player2.feed_reward(learn / 2)
        return 0


def train(player1, player2):
    time_init = time.time()
    win = 0
    lose = 0
    equal = 0
    fig = []
    step = 1000
    x = np.linspace(1, train_times+1, int(train_times/step))
    for i in range(1, train_times+1):
        if i % step == 0:
            fig.append(win/step)
            win = 0
        result = war(player1, player2)
        if result == 1:
            win += 1
        elif result == -1:
            lose += 1
        else:
            equal += 1
    plt.figure('胜率')
    plt.ylim(0, 1)
    plt.plot(x, fig)
    win = train_times - lose - equal
    time_final = time.time()
    print('训练结束,用时%ds' % (int(time_final-time_init)))
    print('平局率为', equal / train_times)
    print('胜率为', win / train_times)
    print('输率为', lose / train_times)
    plt.savefig('胜率')


def testtest(player1, player2):
    time_init = time.time()
    win = 0
    lose = 0
    equal = 0
    for i in range(test_times):
        result = war(player1, player2)
        if result == 1:
            win += 1
        elif result == -1:
            lose += 1
        else:
            equal += 1
    time_final = time.time()
    print('测试结束,用时%ds' % (int(time_final - time_init)))
    print('平局率为', equal / test_times)
    print('胜率为', win / test_times)
    print('输率为', lose / test_times)
    return win/test_times


allStates = get_all_states()
A = Player(0, strategy=1, fool=False, name='red')
B = Player(1, strategy=0, fool=False, name='blue')
train(A, B)
B = Player(1, strategy=0, fool=False, name='blue')
win = testtest(A, B)


title = '程序员'
text = '您的程序跑完了，没有错误，恭喜恭喜！而且这一局的胜率是%f!!' % win
keyward = {'text': title, 'desp': text}
requests.get(url, params=keyward)
plt.show()