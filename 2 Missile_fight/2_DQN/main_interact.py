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


def take_action(state):
    return 1


class Game(object):
    def __init__(self):
        self.AI = RL_B
        self.env = env
        self.time_stop = 300
        self.ai_mode = True
        pygame.init()
        self.font = pygame.font.SysFont("simsunnsimsun", 30)
        self.screen = pygame.display.set_mode((640, 480), 0, 32)
        pygame.display.set_caption("Missile_game!")
        self.background = pygame.image.load(r'source\back0.jpg').convert()
        self.weixing = pygame.image.load(r'source\weixing.jpg').convert()

        self.interest = 0
        self.pos = None
        self.text_surface_zuo = self.font.render(u"点击进行人机对战", True, (255, 255, 0))
        self.text_surface_you = self.font.render(u"点击进行电脑对战", True, (255, 255, 0))
        self.missi_range = []
        self.tar_range = []
        self.rect_zuo = (0, 400, self.text_surface_zuo.get_width(), self.text_surface_zuo.get_height())
        self.rect_you = ((640 - self.text_surface_you.get_width()), 400, self.text_surface_you.
                         get_width(), self.text_surface_you.get_height())
        self.player_color = [(255, 0, 0), (0, 0, 255)]

        self.game_state = 'start'
        self.winner = None
        self.state = None

        for index1, val1 in enumerate((self.tar_range, self.missi_range)):
            for i in range(3):
                val1.append((110 + index1 * 320, 10 + i * 120, 100, 100))
            val1.append((index1 * (640 - self.weixing.get_width()), 70,
                         self.weixing.get_width(), self.weixing.get_height()))
            val1.append((index1 * 540, 240, 100, 100))

    def state_reset(self):
        self.winner = None
        self.game_state = 'start_1'
        self.state = self.env.reset()

    def paint(self):
        for event in pygame.event.get():
            if event.type == QUIT:  # 接收到退出事件后退出程序
                return 0
            elif event.type == MOUSEBUTTONDOWN:
                self.pos = event.pos
            else:
                self.pos = None
        if self.game_state == 'start':  # 初始界面
            self.screen.blit(self.background, (0, 0))
            text_surface = self.font.render(u"点击开始游戏吧！", True, (167, 255, 0))
            w = text_surface.get_width()
            h = text_surface.get_height()
            rec = ((640 - w) / 2, (480 - h) / 2, w, h)
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

                state_now = self.state  # 一轮开始的状态
                action1 = self.env.robot_action(mode='rand_smart', first=True)  # 选择出来a1 整型
                action2 = self.AI.choose_action(state_now, first=False)  # AI选择动作a2 整型
                print(action2)
                self.action_record1 = [action1 // 5, action1 % 5]
                self.action_record2 = [action2 // 5, action2 % 5]
                state_next, reward, done, info = env.step(np.array([action1, action2]))

                # if self.action_flag == 0:
                #     for inde, vav in enumerate(self.missi_range):
                #         if pygame.Rect(vav).collidepoint(self.pos):
                #             pygame.time.delay(self.time_stop)
                #             tem = [0 for i in range(5)]
                #             tem[inde] = 1
                #             print('你选择了导弹%d' % (inde + 1))
                #             self.human_2 = (tem, inde)
                #             self.action_flag += 1
                #     self.draw()
                #     return 1
                # elif self.action_flag == 1:
                #     for inde, vav in enumerate(self.tar_range):
                #         if pygame.Rect(vav).collidepoint(self.pos):
                #             pygame.time.delay(self.time_stop)
                #             tem = [0 for i in range(5)]
                #             tem[inde] = 1
                #             print('你选择了目标%d' % (inde + 1))
                #             self.human_1 = (tem, inde)
                #             action2 = np.array([self.human_1[0], self.human_2[0]])
                #             for action in action2s:
                #                 if (action == action2).all():
                #                     self.action_flag += 1
                #                     self.draw()
                #                     return 1
                #             print('选择错误，请重新选择')
                #             self.action_flag = 0
                #             self.human_1 = []
                #             self.human_2 = []
                #             self.draw()
                #             return 1
                # elif self.action_flag == 2:
                #     self.action_flag = 0
                #     action2 = np.array([self.human_1[0], self.human_2[0]])
                #     self.human_1 = []
                #     self.human_2 = []
                pygame.time.delay(self.time_stop)
                self.state = state_next
                if done:
                    self.game_state = 'end'
                    self.winner = info['winner']
            self.draw()
            return 1
        elif self.game_state == 'end':
            self.screen.fill((255, 255, 255))  # 设置背景为白色
            self.screen.blit(self.background, (0, 0))
            if self.winner == 0:
                text_surface = self.font.render(u"哇，你赢了！！", True, (255, 0, 0))
            else:
                text_surface = self.font.render(u"呵呵，电脑都打不过！", True, (167, 255, 255))
            w = text_surface.get_width()
            h = text_surface.get_height()
            rec = ((640 - w) / 2, (480 - h) / 2, w, h)
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
        self.screen.fill((255, 255, 255))  # 设置背景为白色
        pygame.draw.line(self.screen, (0, 0, 0), (320, 0), (320, 400))
        pygame.draw.line(self.screen, (0, 0, 0), (0, 400), (640, 400))
        self.screen.blit(self.text_surface_zuo, (self.rect_zuo[0], self.rect_zuo[1]))
        self.screen.blit(self.text_surface_you, (self.rect_you[0], self.rect_you[1]))

        situation1 = self.state[:5]
        situation2 = self.state[5:]
        banjing = 50
        for index0, val1 in enumerate((situation1, situation2)):
            for index1, val2 in enumerate(val1[:3]):
                pygame.draw.circle(self.screen, (0, 0, 255), (160 + index0 * 320, 60 + index1 * 120), banjing, 1)
                if val2:
                    color1 = self.player_color[index0]
                    # tem = (85, 102, 0)
                    # if len(self.human_2):
                    #     mi = self.human_2[1]
                    #     if index1 == mi and index0 == 1:
                    #         color1 = tem
                    # if len(self.human_1):
                    #     mi = self.human_1[1]
                    #     if index1 == mi and index0 == 0:
                    #         color1 = tem
                    for i in range(val2):
                        degree = math.pi * 2 * i / val2 + self.interest * 2 * math.pi / 360
                        self.interest += 0
                        if self.interest == 360:
                            self.interest = 1
                        pygame.draw.circle(self.screen, color1, (
                            int(160 + index0 * 320 + banjing / 2 * math.sin(degree)),
                            int(60 + index1 * 120 - banjing / 2 * math.cos(degree))), int(banjing / 2 / val2))
                else:
                    continue
            if val1[3]:
                self.screen.blit(self.weixing, (index0 * (640 - self.weixing.get_width()), 70))
            pygame.draw.rect(self.screen, self.player_color[index0], (index0 * 540, 240, 100, 100))
        pygame.display.update()

    def ai_fight(self):
        while True:
            if not self.paint():
                return


if __name__ == '__main__':
    game = Game()
    game.ai_fight()
