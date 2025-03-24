import numpy as np
from ClassType import Theta

def gCon(theta):

    if isinstance(theta, Theta):

        gVal = np.ones((len(theta.q), 4), dtype=bool)
        gVal[:, 0] = theta.q >= 0
        gVal[:, 1] = theta.u >= 0
        gVal[:, 2] = theta.d > 0
        gVal[:, 3] = theta.tao > 0
        
        consTrue = np.prod(gVal, axis=1)
    else:
        
        gVal = np.ones((theta.shape[1], 4), dtype=bool)
        gVal[:, 0] = theta[2, :] >= 0
        gVal[:, 1] = theta[3, :] >= 0  
        gVal[:, 2] = theta[5, :] > 0   
        gVal[:, 3] = theta[6, :] > 0   
        
        consTrue = np.prod(gVal, axis=1)
    
    return consTrue

