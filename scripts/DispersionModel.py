import math
import numpy as np
from ClassType import Source
from ClassType import Theta

def dispersion(S, p):
    
    if isinstance(S, Source) or isinstance(S, Theta):
        lamda = np.sqrt(S.d * S.tao / (1 + (S.u**2 * S.tao) / (4 * S.d)))
        dis = np.sqrt((p[0] - S.x)**2 + (p[1] - S.y)**2)

        if isinstance(dis, np.float64):
            dis = np.array([dis])
        
        dis[dis == 0] = 1e-4

        #print((-dis/lamda).shape)
        #print(-(p[0] - S.x) * S.u)
        #print(np.cos(S.phi))
        #print(S.d)

        M  = S.q / (4 * np.pi * S.d * dis) * np.exp(-dis/lamda + \
            + (-(p[0] - S.x) * S.u * np.cos(S.phi) / (2 * S.d)) \
            + (-(p[1] - S.y) * S.u * np.sin(S.phi) / (2 * S.d)))
        
        #print((-dis/lamda))
        #print((-(p[0] - S.x) * S.u * np.cos(S.phi) / (2 * S.d)))
        #print(-(p[1] - S.y) * S.u * np.sin(S.phi) / (2 * S.d))
        #print(M)
        #print(S.phi *180/np.pi)


    


        
        
    else:
        
        x = S[0:1,:].T
        y = S[1:2,:].T
        q = S[2:3,:].T
        u = S[3:4,:].T
        phi = S[4:5,:].T
        d = S[5:6,:].T
        tao = S[6:7,:].T

        lamda = np.sqrt(d * tao / (1 + (u**2 * tao) / (4 * d)))
        #print(lamda)
        dis = np.sqrt((p[0] - x)**2 + (p[1] - y)**2)
        dis[dis==0] = 1e-4
        M  = q / (4 * np.pi * d * dis) * np.exp(-dis/lamda \
            + (-(p[0] - x) * u * np.cos(phi) / (2 * d)) \
            + (-(p[1] - y) * u * np.sin(phi) / (2 * d)))

    return M

