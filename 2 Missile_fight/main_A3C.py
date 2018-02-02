import Missile
import numpy as np
import matplotlib
import A3C_dis as A3C
import matplotlib
import matplotlib.pyplot as plt



env = Missile.MissileAI()
para = A3C.Para(env,
                units_a=30,
                units_c=100,
                MAX_GLOBAL_EP=6000,
                UPDATE_GLOBAL_ITER=30,
                gamma=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001,
                oppo = 'rand_smart')
RL = A3C.A3C(para)
RL.run()

# state_now = env.reset()
# i_index = 0
# state_track = []
# action_track = []
# time_track = []
# action_ori_track = []
# reward_track = []
# omega_track = []
# reward_me = 0
# omega_old = 0
# while True:
#
#     omega = RL.choose_action(state_now)
#     state_next, reward, done, info = env.step(omega)
#     state_track.append(state_now.copy())
#     action_track.append(info['action'])
#     time_track.append(info['time'])
#     action_ori_track.append(info['u_ori'])
#     reward_track.append(info['reward'])
#     omega_track.append(float(omega))
#
#
#     state_now = state_next
#     reward_me += reward
#
#     if done:
#         break
#
# print(reward_me)
# plt.figure(1)
# plt.plot(time_track, [x[0] for x in state_track])
# plt.grid()
# plt.title('x')
#
# #
# plt.figure(2)
# plt.plot(time_track, action_track)
# plt.plot(time_track, action_ori_track)
# plt.title('action')
# plt.grid()
#
# plt.figure(3)
# plt.plot(time_track, reward_track)
# plt.grid()
# plt.title('reward')
#
# plt.figure(4)
# plt.plot(time_track, omega_track)
# plt.grid()
# plt.title('omega')
#
# plt.show()


