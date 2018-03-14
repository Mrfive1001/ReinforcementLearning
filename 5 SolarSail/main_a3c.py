import numpy as np
import matplotlib.pyplot as plt
from SolarSail import Env
import A3C

if __name__ == '__main__':
    env = Env()
    para = A3C.Para(env,
                    a_constant = True,
                    units_a=10,
                    units_c=20,
                    MAX_GLOBAL_EP=40000,
                    UPDATE_GLOBAL_ITER=2,
                    gamma=0.9,
                    ENTROPY_BETA=0.1,
                    LR_A=0.0007,
                    LR_C=0.001,
                    train = True)
    RL = A3C.A3C(para)
    RL.run()