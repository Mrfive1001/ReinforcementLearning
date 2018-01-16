import time
import numpy as np
import pickle
# 策略梯度
time0 = time.time()
person_numbers = 2
store_numbers = 3
near = 6
mid = 4
long = 3
load = 1
debug = 0


class State(object):
    def __init__(self):
        self.data = np.array([[near, mid, long], [near, mid, long]])
        # self.winner = None
        self.hashVal = None
        self.end = None
        self.actions = []
        self.hit = np.array([[[1, 1], [1, 1], [0, 0], [0, 0]],
                             [[1, 1], [1, 1], [1, 1], [1, 60]],
                             [[1, 1], [1, 1], [1, 1], [1, 100]]])  # 90%     200000次数时间
        # self.hit = np.array([[[0.9, 0.7], [0.75, 0.5], [0, 0], [0, 0]],
        #                      [[0.8, 0.8], [0.7, 0.7], [0.7, 0.6], [0.7, 60]],
        #                      [[0.7, 0.9], [0.65, 0.8], [0.6, 0.75], [0.55, 100]]])   # 69%胜率
        # self.hit = np.array([[[0.4, 0.4], [0.3, 0.4], [0, 0], [0, 0]],
        #                      [[0.5, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 60]],
        #                      [[0.7, 0.6], [0.6, 0.6], [0.6, 0.6], [0.6, 100]]])# 73%胜率   200000次数时间 578s

        # hit[i,j]第i个导弹命中j个地方的概率
        self.damage = 0

    def get_hash(self):
        if self.hashVal is None:
            self.hashVal = 0
            for i in self.data.reshape(person_numbers*store_numbers):
                self.hashVal = self.hashVal * 10 + i
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
    def get_actions(self, symbol):
        if self.actions:
            return self.actions
        for num, i in enumerate(self.data[symbol, :]):
            if i == 0:
                continue
            for j in range(store_numbers+1):
                a = np.zeros((person_numbers, store_numbers+1))
                a[symbol, num] = 1
                a[0 if symbol else 1, j] = 1
                self.actions.append(a)
        return self.actions

    def nextstate_hit(self, action, symbol):
        result = action.argmax(1)
        newState = State()
        newState.data = np.copy(self.data)
        target = result[0 if symbol else 1]
        if target < 3:
            rate = self.hit[result[symbol], target]
            if np.random.binomial(1, rate[0]):
                num_left = newState.data[0 if symbol else 1,target]
                sub = 0
                for i in range(num_left):
                    if np.random.binomial(1, rate[1]):
                        sub += 1
                num_left -= sub
                newState.data[0 if symbol else 1, target] = num_left
        else:
            rate = self.hit[result[symbol], target]
            if np.random.binomial(1, rate[0]):
                damage = rate[1]
                newState.damage = -damage-np.random.random()
        return newState
        # return newState

    def nextstate_action(self,action,symbol):
        result = action.argmax(1)
        newState = State()
        newState.data = np.copy(self.data)
        newState.data[symbol, result[symbol]] -= 1
        return newState



def getAllStates():
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


if load == 1:
    fr = open('state', 'rb')
    allStates = pickle.load(fr)
    fr.close()
else:
    allStates = getAllStates()
    fw = open('state', 'wb')
    pickle.dump(allStates, fw)
    fw.close()


class Player(object):
    def __init__(self, symbol, Feedback = True):
        self.symbol = symbol
        self.states_actions = {}
        for state in allStates.values():
            state[0].actions = []
        for key, val in allStates.items():
            self.states_actions[key] = [val[0], val[1], val[0].get_actions(self.symbol)]
            length = len(self.states_actions[key][2])
            if length:
                rates = np.linspace(1/length, 1/length, length)
                values = np.zeros(length)
                self.states_actions[key].append(rates)
                self.states_actions[key].append(values)
        # self.states_actions[key][state,isend,actions,rate,values]
        self.states = []  # 是state不是哈希值
        self.stepSize = 1
        self.exploreRate = 0.1
        self.actions = []
        self.feedback = Feedback

    def reset(self):
        self.states = []
        self.actions = []

    def feedState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        self.states = [state.get_hash() for state in self.states]
        for index0,state in enumerate(self.states):
            action = self.actions[index0]
            for index1, action1 in enumerate(self.states_actions[state][2]):
                if (action == action1).all():
                    self.states_actions[state][3][index1] += reward
                    self.states_actions[state][3] = [x / sum(self.states_actions[state][3])
                                                     for x in self.states_actions[state][3]]
                    break
        # for ind1, latestState in enumerate(reversed(self.states)):
        #     self.actions.reverse()
        #     action = self.actions[ind1]
        #     for ind2, action1 in enumerate(self.states_actions[latestState][2]):
        #         if (action == action1).all():
        #             value = self.states_actions[latestState][4][ind2] + self.stepSize * (target - self.states_actions[latestState][4][ind2])
        #             self.states_actions[latestState][4][ind2] = value
        #             target = value
        #             break
        self.reset()

    def takeAction(self):
        old_state = self.states[-1]
        nextactions = self.states_actions[old_state.get_hash()][2]
        # if nextactions:
        #     # print(nextactions)
        #     values = self.states_actions[old_state.get_hash()][4]
        #     if np.random.binomial(1, self.exploreRate) or not self.feedback:
        #         np.random.shuffle(nextactions)
        #         action = nextactions[0]
        #         if debug:
        #             print(0)
        #     else:
        #         if debug:
        #             print(1)
        #         if sum(values) == values[0] * len(values):
        #             np.random.shuffle(nextactions)
        #             action = nextactions[0]
        #         else:
        #             action = nextactions[np.argmax(values)]
        #     # if self.states_actions[old_state.get_hash()][2]:
        if nextactions:
            if self.feedback:
                rates = self.states_actions[old_state.get_hash()][3]
                ran = np.random.random()
                temp = 0
                for index, rate in enumerate(rates):
                    temp += rate
                    if ran < temp:
                        action = nextactions[index]
                        break
                if debug:
                    print(action)
            else:
                np.random.shuffle(nextactions)
                action = nextactions[0]
            self.actions.append(action)
            return action
        else:
            self.states.pop(-1)
            return None

    def next_state_action(self, state, action):
        return state.nextstate_action(action,self.symbol)

    def nextstatehit(self, state, action):
        return state.nextstate_hit(action, self.symbol)


A = Player(0, Feedback=True)
B = Player(1, Feedback=False)
# A = Player(0, Feedback=False)
# B = Player(1, Feedback=True)
cuunt = 200000
win = 0
lose = 0
equal = 0

for i in range(cuunt):
    A.reset()
    B.reset()
    state = State()
    if debug:
        print(state.data)
    flag = state.is_end()
    damages = [0, 0]
    while flag == 0:
        A.feedState(state)
        B.feedState(state)
        actionA = A.takeAction()
        actionB = B.takeAction()
        if actionA is not None:
            state = A.next_state_action(state, actionA)
        if actionB is not None:
            state = B.next_state_action(state, actionB)
        if actionA is not None:
            state = A.nextstatehit(state, actionA)
            damages[1] += state.damage
        if actionB is not None:
            state = B.nextstatehit(state, actionB)
            damages[0] += state.damage
        if debug:
            print(state.data)
        if debug:
            print(state.data)
        flag = state.is_end()
#    print(damages),
    if damages[0] > damages[1]:
        win += 1
        A.feedReward(0.01)
    elif damages[1] > damages[0]:
        lose += 1
        B.feedReward(0.01)
    else:
        equal += 1
        A.feedReward(0.005)
        B.feedReward(0.005)
    # print(win, lose, equal)
print('平局率为', equal/cuunt)
print('赢率为', win/cuunt)
print('输率为', lose/cuunt)
print(time.time()-time0)