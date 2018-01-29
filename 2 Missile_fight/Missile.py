import time
import numpy as np
import requests
import pickle
import random

class MissileAI:
    def __init__(self):
        near,mid,long,moon,blood = 6,4,3,1,0
        self.init_state = np.array([near,mid,long,moon,blood]*2) # 双方仓库导弹数目、卫星个数，血量

    def step(self):
        pass

    def reset(self):
        pass


    def render(self):
        pass
