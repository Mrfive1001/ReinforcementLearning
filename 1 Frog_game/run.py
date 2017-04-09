class run:
    def __init__(self):
        # 设置游戏的规则
        # states为总共有的路径状态
        # terminal_states为结束点的集合
        # actions为可以采取的动作
        # rewards为每走一步的奖励，也就是规则的设定
        # gamma衰减因子
        # values为每一个状态的价值 self.values[state] 等于状态state的价值
        # rules[state]某个策略产生的动作
        self.states = range(1,10)
        self.terminal_states = {3: 1, 7: 1, 9: 1}

        self.actions = ['s', 'x', 'z', 'y']

        self.rewards = {'1s': -0.5, '2y': -1,
                        '4s': -1, '6x': -1,
                        '5z': -0.5, '5s': -0.5,
                        '6s': 1, '8y': 1}

        self.transfer = {'1s': 4, '1y': 2, '2z': 1, '2y': 3, '2s': 5, '4x': 1,
                         '4s': 7, '4y': 5, '5s': 8, '5x': 2, '5z': 4, '5y': 6,
                         '6s': 9, '6x': 3, '6z': 5, '8z': 7, '8y': 9, '8x': 5}
        self.gamma = 0.8

        self.values = [0.0 for i in range(len(self.states) + 1)]

        self.rules = {}
        for state in self.states:
            if state in self.terminal_states:
                continue
            self.rules[state] = self.actions[2]

    # return is_terminal,state, reward
    # 状态转移子函数
    def transform(self, state, action):
        if state in self.terminal_states:
            return True, state, 0

        key = '%d%s' % (state, action)
        if key in self.transfer:
            next_state = self.transfer[key]
        else:
            next_state = state

        is_terminal = False
        if next_state in self.terminal_states:
            is_terminal = True
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return is_terminal, next_state, r

    # 利用价值评估来更新策略最后得到最佳策略
    def value_iteration(self):
        for i in range(1000):
            delta = 0.0
            for state in self.states:

                if state in self.terminal_states:
                    continue
                a1 = self.actions[0]
                transfer, s, r = self.transform(state,a1)
                v1 = r + self.gamma*self.values[s]

                for action in self.actions:
                    transfer, s, r = self.transform(state, action)
                    if v1 < (r + self.gamma * self.values[s]):
                        a1 = action
                        v1 = r + self.gamma * self.values[s]
                delta += abs(v1 - self.values[state])

                self.rules[state] = a1
                self.values[state] = v1
            if delta < 1e-6:
                break