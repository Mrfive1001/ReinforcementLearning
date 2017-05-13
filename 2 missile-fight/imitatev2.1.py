import time
import numpy as np
import requests
import pickle
import random
# 策略梯度
person_numbers = 2
store_numbers = 3
near = 6
mid = 4
long = 3
moon = 1
debug = 1
load = 1
ip = 0.3

learn = 0.2


class State(object):
    def __init__(self):
        self.data = np.array([[near, mid, long, moon], [near, mid, long, moon]])
        self.hashVal = None
        self.end = None
        self.actions = []
        # self.set_hit_help = [0.6, 0.6, 0.6]
        self.set_hit_help = [2, 2, 2]
        self.set_stop_help = 1
        # self.hit = np.array([[[1, 1], [1, 1], [0, 0], [0, 0], [0, 0]],
        #                      [[1, 1], [1, 1], [1, 1], [0.7, 0], [1, 60]],
        #                      [[1, 1], [1, 1], [1, 1], [1, 100]]])  # 90%     200000次数时间
        # self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0], [0, 0]],
        #                      [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 0], [0.7, 60]],
        #                      [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0, 0], [0.55, 100]]])   # 69%胜率
        self.hit = np.array([[[0.4, 0.4], [0.3, 0.4], [0.1, 0.4], [0, 0], [0, 0]],
                             [[0.5, 0.5], [0.4, 0.5], [0.4, 0.5], [0.7, 0], [0.4, 60]],
                             [[0.7, 0.6], [0.6, 0.6], [0.6, 0.6], [0, 0], [0.6, 100]]])# 73%胜率   200000次数时间 578s
        # hit[i,j]第i个导弹命中j个地方的概率

    def new(self):
        new_state = State()
        new_state.data = np.copy(self.data)
        return new_state

    def get_hash(self):
        if self.hashVal is None:
            self.hashVal = 0
            for num in self.data.reshape(person_numbers*(store_numbers+1)):
                self.hashVal = self.hashVal * 8 + num
                # 这里的10保证哈希值唯一
        return int(self.hashVal)

    def is_end(self):
        if self.end is not None:
            return self.end
        if self.data[:, :store_numbers].sum():
            self.end = False
        else:
            self.end = True
        return self.end

    # symbol是0或者1
    # Fool的话是可以打对面的空基地
    def get_actions(self, symbol, fool):
        if self.actions:
            return self.actions
        for num, val in enumerate(self.data[symbol, :store_numbers]):
            if val == 0:
                continue
            for target in range(store_numbers+2):
                if fool or target == store_numbers+1 or self.data[symbol ^ 1, target] != 0:
                    a = np.zeros((person_numbers, store_numbers+2))
                    a[symbol, num] = 1
                    a[symbol ^ 1, target] = 1
                    self.actions.append(a)
                else:
                    continue
        return self.actions


def get_all_states():
    allStates = dict()
    for i in range(near+1):
        for j in range(mid+1):
            for k in range(long+1):
                for l in range(moon+1):
                    for a in range(near+1):
                        for b in range(mid+1):
                            for c in range(long+1):
                                for d in range(moon+1):
                                    state = State()
                                    state.data = np.array([[i, j, k, l], [a, b, c, d]])
                                    allStates[state.get_hash()] = (state, state.is_end())
    return allStates


class Player(object):
    def __init__(self, symbol, fool=False, train=True):
        self.symbol = symbol
        self.states_actions = {}  # 最重要的数据
        self.allstates = get_all_states()
        self.states = []  # 是state不是哈希值
        self.actions = []  # state与action是每一次比赛中的暂时过程
        self.fool = fool
        self.train = train
        self.ip = ip

        for key, val in self.allstates.items():
            self.states_actions[key] = [val[0], val[1], val[0].get_actions(self.symbol, fool=self.fool)]
            length = len(self.states_actions[key][2])
            if length:
                rates = np.linspace(1/length, 1/length, length)
                self.states_actions[key].append(rates)
        # self.states_actions[key][state,isend,actions,rate,values]

    def feed_state(self, state):
        self.states.append(state)

    def select_action(self):
        old_state = self.states[-1]
        nextactions = self.states_actions[old_state.get_hash()][2]
        if nextactions:
            if self.train:
                action = []
                rates = self.states_actions[old_state.get_hash()][3]
                ran = np.random.random()
                temp = 0
                for index, rate in enumerate(rates):
                    temp += rate
                    if ran < temp:
                        action = nextactions[index]
                        break
                self.actions.append(action)
                return action
            else:
                rates = self.states_actions[old_state.get_hash()][3]
                store = []
                max_num = rates.argmax()
                for index, rate in enumerate(rates):
                    if rate == rates[max_num]:
                        store.append(index)
                index = random.choice(store)
                action = nextactions[index]
                self.actions.append(action)
                return action
        else:
            self.states.pop(-1)
            return None

    def moon_exist(self):
        return self.states[-1].data[self.symbol, 3]

    def feed_reward(self, reward):
        self.states = [state.get_hash() for state in self.states]
        for index0, state in enumerate(self.states):
            action = self.actions[index0]
            for index1, action1 in enumerate(self.states_actions[state][2]):
                if (action == action1).all():
                    self.states_actions[state][3][index1] = max(0, self.states_actions[state][3][index1] + reward)
                    # self.states_actions[state][3][index1] += reward
                    self.states_actions[state][3] = np.array([x / sum(self.states_actions[state][3])
                                                     for x in self.states_actions[state][3]])
                    break
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []

    def save_policy(self):
        with open('policy_2.1', 'wb') as f:
            pickle.dump(self.states_actions, f)

    def load_policy(self):
        with open('policy_2.1', 'rb') as f:
            self.states_actions = pickle.load(f)


class PlayerAi(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.states = []  # 是state不是哈希值
        self.ip = ip

    def feed_state(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def moon_exist(self):
        return self.states[-1].data[self.symbol, 3]

    # 1 是随机会落空
    # 2 是随机不会落空
    # 3 是随机不打空
    # 4 是一心打城市
    # 5 先打卫星，再打城市
    def select_action(self, model):
        old_state = self.states[-1]
        if model == 1:
            actions = old_state.get_actions(self.symbol, fool=True)
            if actions:
                np.random.shuffle(actions)
                action = actions[0]
                return action
            else:
                self.states.pop(-1)
                return None
        else:
            actions = old_state.get_actions(self.symbol, fool=False)
            if actions:
                if model == 2:
                    np.random.shuffle(actions)
                    action = actions[0]
                    return action
                elif model == 3:
                    pass
                elif model == 4:
                    pass
                elif model == 5:
                    pass
            else:
                self.states.pop(-1)
                return None


def calculate_damage(state_action1, state_action2):
    state = State()
    win_times = 0
    times = 10
    for i in range(times):
        damages = [np.random.random(), np.random.random()]
        for action in state_action1:
            if np.random.random() < action[2]:
                damages[1] += state.hit[action[0], action[1]][1]
        for action in state_action2:
            if np.random.random() < action[2]:
                damages[0] += state.hit[action[0], action[1]][1]
        if damages[0] < damages[1]:
            win_times += 1
        # damages[1] += damage1
    return win_times/times


def Train():
    global player1
    global player2
    train_times = 200000
    for i in range(train_times):
        player1.reset()
        player2.reset()
        model = random.randint(1 , 2)
        state = State()
        store1 = []
        store2 = []
        flag = state.is_end()
        # damages = [np.random.random(), np.random.random()]
        while flag == 0:
            player1.feed_state(state)
            player2.feed_state(state)
            action1 = player1.select_action()
            action2 = player2.select_action(model)
            flag1 = player1.moon_exist()
            flag2 = player2.moon_exist()
            temp1 = flag1
            temp2 = flag2
            if action1 is not None:
                missile1, target1 = action1.argmax(1)
                state.data[0, missile1] -= 1
            if action2 is not None:
                target2, missile2 = action2.argmax(1)
                state.data[1, missile2] -= 1
            if action1 is not None:
                hit1 = min(1, state.hit[missile1, target1][0] * (1 + flag1 * state.set_hit_help[missile1]))
                stop2 = min(1, player2.ip * (1 + flag2 * state.set_stop_help))
                if target1 == store_numbers + 1:
                    store1.append((missile1, target1, hit1*(1-stop2)))
                elif np.random.random() < hit1 * (1 - stop2):
                    if target1 < store_numbers:
                        num_left = state.data[1, target1]
                        sub = 0
                        for num in range(num_left):
                            if np.random.random() < state.hit[missile1, target1][1]:
                                sub += 1
                        num_left -= sub
                        state.data[1, target1] = num_left
                    elif target1 == store_numbers:
                        temp2 = 0
            if action2 is not None:
                hit2 = min(1, state.hit[missile2, target2][0] * (1 + flag2 * state.set_hit_help[missile2]))
                stop1 = min(1, player1.ip * (1 + flag1 * state.set_stop_help))
                if target2 == store_numbers + 1:
                    store2.append((missile2, target2, hit2*(1-stop1)))
                elif np.random.random() < hit2 * (1 - stop1):
                    if target2 < store_numbers:
                        num_left = state.data[0, target2]
                        sub = 0
                        for num in range(num_left):
                            if np.random.random() < state.hit[missile2, target2][1]:
                                sub += 1
                        num_left -= sub
                        state.data[0, target2] = num_left
                    elif target2 == store_numbers:
                        temp1 = 0
            state = state.new()
            state.data[:, store_numbers] = temp1, temp2
            flag = state.is_end()
        rate = calculate_damage(store1, store2)
        reward = (rate-0.5)*learn
        player1.feed_reward(reward)
    player1.save_policy()
    return


def Test():
    global player1
    global player2
    player1.train = False
    test_times = 1
    win_times = 0
    lose_times = 0
    equal_times = 0
    if debug:
        print('0号导弹是近程弹\n1号导弹是中程弹\n2号导弹是远程弹\n3号目标是卫星\n4号目标是基地')
    for i in range(test_times):
        player1.reset()
        player2.reset()
        model = random.randint(1, 2)
        state = State()
        flag = state.is_end()
        damages = [0, 0]
        while flag == 0:
            player1.feed_state(state)
            player2.feed_state(state)
            action1 = player1.select_action()
            action2 = player2.select_action(model)
            flag1 = player1.moon_exist()
            flag2 = player2.moon_exist()
            if debug:
                print(flag1, flag2)
            temp1 = flag1
            temp2 = flag2
            if action1 is not None:
                missile1, target1 = action1.argmax(1)
                if debug:
                    print("a的动作是利用%d号导弹，攻击%d目标" % (missile1, target1))
                state.data[0, missile1] -= 1
            if action2 is not None:
                target2, missile2 = action2.argmax(1)
                if debug:
                    print("b的动作是利用%d号导弹，攻击%d目标" % (missile2, target2))
                state.data[1, missile2] -= 1
            if action1 is not None:
                hit1 = min(1, state.hit[missile1, target1][0] * (1 + flag1 * state.set_hit_help[missile1]))
                stop2 = min(1, player2.ip * (1 + flag2 * state.set_stop_help))
                if np.random.random() < hit1 * (1 - stop2):
                    if target1 < store_numbers:
                        num_left = state.data[1, target1]
                        sub = 0
                        for num in range(num_left):
                            if np.random.random() < state.hit[missile1, target1][1]:
                                sub += 1
                        num_left -= sub
                        state.data[1, target1] = num_left
                    elif target1 == store_numbers:
                        temp2 = 0
                    elif target1 == store_numbers + 1:
                        damages[1] += state.hit[missile1, target1][1]
            if action2 is not None:
                hit2 = min(1, state.hit[missile2, target2][0] * (1 + flag2 * state.set_hit_help[missile2]))
                stop1 = min(1, player1.ip * (1 + flag1 * state.set_stop_help))
                if np.random.random() < hit2 * (1 - stop1):
                    if target2 < store_numbers:
                        num_left = state.data[0, target2]
                        sub = 0
                        for num in range(num_left):
                            if np.random.random() < state.hit[missile2, target2][1]:
                                sub += 1
                        num_left -= sub
                        state.data[0, target2] = num_left
                    elif target2 == store_numbers:
                        temp1 = 0
                    elif target2 == store_numbers + 1:
                        damages[0] += state.hit[missile2, target2][1]
            state = state.new()
            state.data[:, store_numbers] = temp1, temp2
            if debug:
                print("本轮结束,现在的状态是\n%s,现在的伤害是\n%s" % (state.data, damages))
            flag = state.is_end()
        if damages[0] < damages[1]:
            win_times += 1
        elif damages[0] > damages[1]:
            lose_times += 1
        else:
            equal_times += 1
    print(win_times/test_times, lose_times/test_times, equal_times/test_times)


if __name__ == '__main__':
    player1 = Player(0)
    player2 = PlayerAi(1)
    time.clock()
    if load:
        player1.load_policy()
    else:
        Train()
    Test()
    time_go = time.clock()
    url = 'http://sc.ftqq.com/SCU7896Td21f33141f474e9dfec81d7da26daacc5900906769c16.send'
    title = '程序员'
    text = '您的程序结束了，用时%.2f s没有错误，恭喜恭喜！' % time_go
    key_word = {'text': title, 'desp': text}
    requests.get(url, params=key_word)