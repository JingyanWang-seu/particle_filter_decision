from ClassType import Theta
from ClassType import Source
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def draw_param(theta):

    

    params = {
    'x': theta.x,
    'y': theta.y,
    'q': theta.q,
    'u': theta.u,
    'phi': theta.phi,
    'd': theta.d,
    'tao': theta.tao
    }

    plt.figure(figsize=(10, 7))

    for i, (name, values) in enumerate(params.items(), 1):
        plt.subplot(3, 3, i)
        plt.hist(values, bins=60, alpha=0.7)
        #sns.kdeplot(values, fill=True)
        #plt.title(f'Distribution of {name}')
        plt.xlabel(name)
        #plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

'''
N=10000
S = Source(40, 60, 5, 4, 90 * math.pi/180, 1, 8)
theta = Theta(80 * np.random.rand(N),
                   80 * np.random.rand(N),
                   np.random.gamma(2, S.q, N),
                   S.u + 2 * np.random.randn(N),
                   S.phi * 0.9 + 10 * math.pi/180 * np.random.randn(N),
                   S.d + 2 * np.random.rand(N),
                   S.tao + 2 * np.random.rand(N) -2)
draw_param(theta)
'''