from TSP_Burma14 import ENV
import numpy as np
import A3C_Dis as A3C
import matplotlib.pyplot as plt

env = ENV()
para = A3C.Para(env,
                units_a=10,
                units_c=20,
                MAX_GLOBAL_EP=50000,
                UPDATE_GLOBAL_ITER=14,
                gamma=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0007,
                LR_C=0.001)
RL = A3C.A3C(para)
RL.run()
# RL.display()