import numpy as np
import inspect

def resample(weight, N):


    if len(inspect.signature(resample).parameters) == 1:
        N = weight.size

    index = np.zeros(N, dtype=int)

    c = np.cumsum(weight)

    i = 0
    u = np.zeros(N)
    u[0] = np.random.rand() / N

    for j in range(N):
        u[j] = u[0] + j / N

        while u[j] > c[i]:
            i += 1
        index[j] = i
    
    w_par = np.ones(N) / N
    
    return w_par, index