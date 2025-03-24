import numpy as np
import math

idx = np.array([1,2])

state = np.zeros((3,4))
newState = np.ones((3,4))
newState[:, idx] = state[:, idx]

print(newState)

idx = np.ones(10, dtype=bool)
print(np.sum(idx))

import numpy as np
import warnings

# 设置 NumPy 浮动点错误处理策略
#np.seterr(over='ignore')  # 禁止显示溢出警告

print(20 * math.pi/180)
