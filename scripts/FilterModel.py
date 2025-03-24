import numpy as np
from scipy.linalg import cholesky
from ResampleModel import resample
from LikelyModel import likelyhood
import ConstraintModel


def ESS(w):
    sum_w2 = np.sum(w**2)
    ess = 1 / sum_w2
    return ess
    

def filter(theta_par_minus, w_par_minus, data_obv, likely, m, constraint):

    N = w_par_minus.size
    n_par =  len(vars(theta_par_minus))

    theta_par = theta_par_minus

    w_par_update = likely(theta_par, data_obv, m)
    w_par_constraint = constraint(theta_par)
    w_par = w_par_minus * w_par_update * w_par_constraint

    w_par = w_par / np.sum(w_par)

    ess = ESS(w_par)

    '''ReSampling with MCMC step'''
    if ess < 0.7*N:
        print('mcmc')
        # state is a 7*10000 size numpy
        state = np.vstack([theta_par.x, theta_par.y, theta_par.q, 
                           theta_par.u, theta_par.phi, theta_par.d, theta_par.tao])
        # print(state.shape)
        avg_state = np.sum(np.ones((n_par, 1)) @ w_par.reshape(1, -1) * state, axis=1, keepdims=True)
        # cov_state = (state - avg_state @ np.ones((1, N))) @ np.diag(w_par.flatten()) @ (state - avg_state @ np.ones((1, N))).T
        # D = cholesky(cov_state)

        # the independence of different terms
        diffPos = state[:2] - avg_state[:2] @ np.ones((1,N))
        covPos = diffPos @ np.diag(w_par.flatten()) @ diffPos.T
        # print(covPos)

        diffq = state[2:3] - avg_state[2:3] @ np.ones((1, N))
        covq = diffq @ np.diag(w_par.flatten()) @ diffq.T
        # print(avg_state)
        # print(diffq.shape)

        diffu = state[3:5] - avg_state[3:5] @ np.ones((1, N))
        covu = diffu @ np.diag(w_par.flatten()) @ diffu.T

        diffHarm = state[5:7] - avg_state[5:7] @ np.ones((1, N))
        covHarm = diffHarm @ np.diag(w_par.flatten()) @ diffHarm.T

        Dpos = cholesky(covPos, lower=True)  
        Dq = cholesky(covq, lower=True)      
        Du = cholesky(covu, lower=True)  
        DHarm = cholesky(covHarm, lower=True)

        w_par, index = resample(w_par, N)
        state = state[:, index]
        w_par = w_par.T

        A = (4 / (n_par + 2)) ** (1 / (N + 4))
        hopt = A * (N ** (-1 / (n_par + 4)))

        idx = np.ones(N, dtype=bool)
        newState = state.copy()

        for jj in range(5):
            
            newState[0:2, idx] = state[0:2, idx] + hopt * Dpos @ np.random.randn(2, np.sum(idx))
            newState[2, idx] = state[2, idx] + hopt * Dq @ np.random.randn(1, np.sum(idx))
            newState[3:5, idx] = state[3:5, idx] + hopt * Du @ np.random.randn(2, np.sum(idx))
            newState[5:7, idx] = state[5:7, idx] + hopt * DHarm @ np.random.randn(2, np.sum(idx))

            idx = constraint(newState) != 1
            print(idx)
            print(idx.shape)
            if np.sum(idx) == 0:
                break
            else:
                newState[:, idx] = state[:, idx]
                print("same!!!")

        newerr = newState - state
        # print(covPos)
        # print(covq.shape)
        SIG = hopt ** 2 * np.block([[covPos, np.zeros((covPos.shape[0],covq.shape[1])),  np.zeros((covPos.shape[0],covu.shape[1])),  np.zeros((covPos.shape[0],covHarm.shape[1]))],
                                    [np.zeros((covq.shape[0],covPos.shape[1])), covq, np.zeros((covq.shape[0],covu.shape[1])), np.zeros((covq.shape[0],covHarm.shape[1]))],
                                    [np.zeros((covu.shape[0],covPos.shape[1])), np.zeros((covu.shape[0],covq.shape[1])), covu, np.zeros((covu.shape[0],covHarm.shape[1]))],
                                    [np.zeros((covHarm.shape[0],covPos.shape[1])), np.zeros((covHarm.shape[0],covq.shape[1])), np.zeros((covHarm.shape[0],covu.shape[1])), covHarm]])
        # print(newerr.T.shape)
        # print(np.linalg.inv(SIG).shape)
        

        logratio = -0.5 * np.sum((newerr.T @ np.linalg.inv(SIG)).T * newerr, axis=0, keepdims=True) + \
                    0.5 * np.sum((np.zeros((n_par, N)).T @ np.linalg.inv(SIG)).T * np.zeros((n_par, N)), axis=0, keepdims=True)

        xupdate = likely(state, data_obv, m)

        # print(xupdate.shape)

        xnewupdate = likely(newState, data_obv, m)
        # print(xnewupdate.shape)

        alpha = xnewupdate / xupdate * np.exp(logratio).T
        # print(logratio)
        # print(alpha.shape)

        mcrand = np.random.rand(N, 1)

        #accept = alpha >= mcrand
        reject = alpha < mcrand

        newState[:, reject.flatten()] = state[:, reject.flatten()]
        
        theta_par.x = newState[0,:].T
        theta_par.y = newState[1,:].T
        theta_par.q = newState[2,:].T
        theta_par.u = newState[3,:].T
        theta_par.phi = newState[4,:].T
        theta_par.d = newState[5,:].T
        theta_par.tao = newState[6,:].T
    
    return theta_par, w_par