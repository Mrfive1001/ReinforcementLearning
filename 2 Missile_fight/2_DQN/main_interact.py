import D3QN
import Missile
import pygame
from pygame.locals import *
import math
import numpy as np

env = Missile.MissileAI()

RL_B = D3QN.DQN(env.action_dim, env.state_dim, load=True,
                id='A', dueling=True, double=True, units=50,
                train=False)


class Game(object):
    def __init__(self):
        self.AI = RL_B  # 人工智能选手
        self.env = env  # 环境
        self.time_stop = 200  # 每隔多少时间刷新
        self.ai_mode = None  # 人工智能与人对战或者与电脑

        self.width = 640
        self.height = 480
        pygame.init()  # 初始化pygame
        self.font = pygame.font.SysFont("simsunnsimsun", 30)
        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        pygame.display.set_caption("Missile_game!")
        # self.background = pygame.image.load(r'source\back0.jpg').convert()  # 背景图片
        self.weixing = pygame.image.load(r'source\weixing.jpg').convert()

        self.corlors = {'white': (255, 255, 255), 'black': (0, 0, 0), 'orange': (255, 69, 0), 'gold': (255, 215, 0),
                        'blue': (0, 191, 255), 'red': (255, 0, 0), 'deepbule': (0, 0, 255)}
        self.interest = 0  # 图形是否进行转动
        self.pos = None  # 鼠标点的位置
        self.text_surface_zuo = self.font.render(u"点击进行人机对战", True, self.corlors['black'])  # 选项
        self.text_surface_you = self.font.render(u"点击进行电脑对战", True, self.corlors['black'])
        self.text_change = u''
        self.blood_a = u'0'
        self.blood_b = u'0'
        self.missi_range = []
        self.tar_range = []
        self.rect_zuo = (0, self.height - 80, self.text_surface_zuo.get_width(), self.text_surface_zuo.get_height())
        self.rect_you = ((self.width - self.text_surface_you.get_width()), self.height - 80, self.text_surface_you.
                         get_width(), self.text_surface_you.get_height())
        self.player_color = [self.corlors['red'], self.corlors['deepbule']]

        self.game_state = 'start'  # 初始状态
        self.winner = None  # 赢者
        self.state = None
        self.action_record1 = [None, None]
        self.action_record2 = [None, None]
        self.action_flag = 0  # 选取动作进行到了第几步

        for index1, val1 in enumerate((self.missi_range, self.tar_range)):  # 增加鼠标点击的时候矩形
            for i in range(3):  # 左边为起始，右边为目标
                val1.append((110 + index1 * 320, 10 + i * 120, 100, 100))
            val1.append((index1 * (self.width - self.weixing.get_width()), 70,
                         self.weixing.get_width(), self.weixing.get_height()))
            val1.append((index1 * 540, 240, 100, 100))

    def state_reset(self):  # 初始化
        self.winner = None
        self.game_state = 'start_1'
        self.action_record1 = [None, None]
        self.action_record2 = [None, None]
        self.action_flag = 0
        self.state = self.env.reset()
        self.text_change = u''

    def paint(self):  # 主要函数处理鼠标点击
        for event in pygame.event.get():
            if event.type == QUIT:  # 接收到退出事件后退出程序
                return 0
            elif event.type == MOUSEBUTTONDOWN:
                self.pos = event.pos  # 保存点击的地方
            else:
                self.pos = None
        if self.game_state == 'start':  # 初始界面
            # self.screen.blit(self.background, (0, 0))
            self.screen.fill(self.corlors['white'])  # 设置背景为白色
            text_surface = self.font.render(u"点击开始游戏吧！", True, self.corlors['black'])
            w = text_surface.get_width()
            h = text_surface.get_height()
            rec = ((self.width - w) / 2, (self.height - h) / 2, w, h)
            self.screen.blit(text_surface, (rec[0], rec[1]))
            if self.pos is not None:
                if pygame.Rect(rec).collidepoint(self.pos):
                    pygame.time.delay(self.time_stop)
                    self.state_reset()
            pygame.display.update()
            return 1
        elif self.game_state == 'start_1':  # 对战界面
            if self.pos is not None:  # 鼠标点击了
                if pygame.Rect(self.rect_you).collidepoint(self.pos):  # 选择电脑对战
                    pygame.time.delay(self.time_stop)
                    self.ai_mode = True
                    self.state_reset()
                    return 1
                elif pygame.Rect(self.rect_zuo).collidepoint(self.pos):  # 选择电脑和人对战
                    pygame.time.delay(self.time_stop)
                    self.ai_mode = False
                    self.state_reset()
                    return 1
                if self.ai_mode == None:
                    return 1
                if self.ai_mode:  # 电脑之间对战
                    state_now = self.state  # 一轮开始的状态
                    action1 = self.env.robot_action(mode='rand_smart', first=True)  # 选择出来a1 整型
                    action2 = self.AI.choose_action(state_now, first=False)  # AI选择动作a2 整型
                    self.action_record1 = [action1 // 5, action1 % 5]
                    self.action_record2 = [action2 // 5, action2 % 5]
                    self.text_change = '玩家1选择了导弹%d,选择目标是%d' % ((self.action_record1[0], self.action_record1[1]))
                    print('玩家2选择了导弹%d,选择目标是%d' % ((self.action_record2[0], self.action_record2[1])))
                    state_next, reward, done, info = env.step(np.array([action1, action2]))
                else:  # 人机对战
                    state_now = self.state  # 一轮开始的状态
                    if self.action_flag == 0:  # 人类选择使用哪一颗导弹
                        for inde, vav in enumerate(self.missi_range):
                            if pygame.Rect(vav).collidepoint(self.pos):
                                pygame.time.delay(self.time_stop)
                                self.action_record1[0] = inde
                                self.text_change = '你选择了导弹%d' % (inde)
                                self.action_flag += 1
                        pygame.time.delay(self.time_stop)
                        self.draw()
                        return 1
                    elif self.action_flag == 1:  # 人类选择打击那一个目标
                        for inde, vav in enumerate(self.tar_range):
                            if pygame.Rect(vav).collidepoint(self.pos):
                                pygame.time.delay(self.time_stop)
                                self.action_record1[1] = inde
                                self.text_change += ('你选择了目标%d' % (inde))
                                self.action_flag += 1
                        if self.action_record1[0] >= 3:
                            self.action_record1 = [None, None]
                            self.action_flag = 0
                            self.text_change = '选择动作不存在，重新选择'
                        pygame.time.delay(self.time_stop)
                        self.draw()
                        return 1
                    elif self.action_flag == 2:  # 确认按钮
                        self.action_flag = 0
                        action1 = self.action_record1[0] * 5 + self.action_record1[1]
                        action2 = self.AI.choose_action(state_now, first=False)  # AI选择动作a2 整型
                        self.action_record2 = [action2 // 5, action2 % 5]
                        print('玩家1选择了导弹%d,选择目标是%d' % ((self.action_record1[0], self.action_record1[1])))
                        print('玩家2选择了导弹%d,选择目标是%d' % ((self.action_record2[0], self.action_record2[1])))
                        self.action_record1 = [None, None]
                        self.text_change = u''
                        state_next, reward, done, info = env.step(np.array([action1, action2]))
                    else:  # 点击地方错误
                        pygame.time.delay(self.time_stop)
                        self.draw()
                        return 1

                # 状态step一步之后
                pygame.time.delay(self.time_stop)  # 小小的延迟
                self.state = state_next
                if done:
                    print('玩家1收到伤害%.2f，玩家2受到伤害%.2f,因此赢者是玩家%d' %
                          (info['damage1'], info['damage2'], info['winner'] + 1))
                    self.winner = info['winner']
                    if not self.ai_mode:
                        self.game_state = 'end'
                    else:
                        self.state_reset()
            self.draw()  # 鼠标没点击就正常显示
            return 1
        elif self.game_state == 'end':
            self.screen.fill(self.corlors['white'])  # 设置背景为白色
            # self.screen.blit(self.background, (0, 0))
            if self.winner == 0:
                text_surface = self.font.render(u"恭喜你打赢了了电脑！！", True, self.corlors['blue'])
            else:
                text_surface = self.font.render(u"很遗憾，你败给了人工智能！", True, self.corlors['orange'])
            w = text_surface.get_width()
            h = text_surface.get_height()
            rec = ((self.width - w) / 2, (self.height - h) / 2, w, h)
            self.screen.blit(text_surface, (rec[0], rec[1]))
            self.screen.blit(text_surface, (rec[0], rec[1]))
            self.screen.blit(self.text_surface_zuo, (self.rect_zuo[0], self.rect_zuo[1]))
            self.screen.blit(self.text_surface_you, (self.rect_you[0], self.rect_you[1]))
            if self.pos is not None:
                if pygame.Rect(self.rect_you).collidepoint(self.pos):
                    pygame.time.delay(self.time_stop)
                    self.state_reset()
                    return 1
                elif pygame.Rect(self.rect_zuo).collidepoint(self.pos):
                    pygame.time.delay(self.time_stop)
                    self.state_reset()
                    return 1
            pygame.time.delay(self.time_stop)
            pygame.display.update()
            return 1

    def draw(self):
        self.screen.fill(self.corlors['white'])  # 设置背景为白色
        pygame.draw.line(self.screen, (0, 0, 0), (320, 0), (320, 400))
        pygame.draw.line(self.screen, (0, 0, 0), (0, 400), (self.width, 400))

        text_surface_temp = self.font.render(self.text_change, True, self.corlors['blue'])
        rect_temp = ((self.width - text_surface_temp.get_width()) / 2, self.height - 40,
                     (self.width + text_surface_temp.get_width()) / 2, self.height)

        self.screen.blit(self.text_surface_zuo, (self.rect_zuo[0], self.rect_zuo[1]))
        self.screen.blit(self.text_surface_you, (self.rect_you[0], self.rect_you[1]))
        self.screen.blit(text_surface_temp, (rect_temp[0], rect_temp[1]))
        situation1 = self.state[:5]
        situation2 = self.state[5:]  # 将当前状态画出来
        banjing = 50
        for index0, val1 in enumerate((situation1, situation2)):  # 分别画出来两个情况
            for index1, val2 in enumerate(val1[:3]):
                pygame.draw.circle(self.screen, self.corlors['black'], (160 + index0 * 320, 60 + index1 * 120), banjing,
                                   1)
                if val2:
                    color1 = self.player_color[index0]
                    tem = self.corlors['black']
                    if self.ai_mode == False:
                        if self.action_record1[0] != None:
                            mi = self.action_record1[0]
                            if index1 == mi and index0 == 0:
                                color1 = tem
                        if self.action_record1[1] != None:
                            tar = self.action_record1[1]
                            if index1 == tar and index0 == 1:
                                color1 = tem
                    for i in range(val2):  # 画出有多少导弹
                        degree = math.pi * 2 * i / val2 + self.interest * 2 * math.pi / 360
                        self.interest += 0
                        if self.interest == 360:
                            self.interest = 1
                        pygame.draw.circle(self.screen, color1, (
                            int(160 + index0 * 320 + banjing / 2 * math.sin(degree)),
                            int(60 + index1 * 120 - banjing / 2 * math.cos(degree))), int(banjing / 2 / val2))
                else:
                    continue
            if val1[3]:  # 画出卫星
                self.screen.blit(self.weixing, (index0 * (self.width - self.weixing.get_width()), 70))
            _length = 90  # 基地的长宽
            _dis = 530  # 两基地中心点的距离
            pygame.draw.rect(self.screen, self.player_color[index0],
                             ((self.width - _length + (2 * index0 - 1) * _dis) / 2, 240, _length, _length), 1)
            blood_temp = self.font.render('%d' % self.state[index0 * 5 + 4], True, self.corlors['black'])
            _blood_temp = ((self.width + (2 * index0 - 1) * _dis - blood_temp.get_width()) / 2
                            , 240 + (_length - blood_temp.get_height()) / 2
                            , (self.width + (2 * index0 - 1) * _dis + blood_temp.get_width()) / 2
                            , 240 + (_length + blood_temp.get_height()) / 2)
            self.screen.blit(blood_temp, (_blood_temp[0], _blood_temp[1]))
        pygame.display.update()

    def ai_fight(self):
        while True:
            if not self.paint():
                return


if __name__ == '__main__':
    game = Game()
    game.ai_fight()
