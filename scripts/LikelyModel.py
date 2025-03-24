import numpy as np
from scipy.stats import norm
import DispersionModel

def likelyhood(xpart, yObv, m, pos):


    # Call the plumeModel function (assumed to be defined elsewhere in your code)
    conc = DispersionModel.dispersion(xpart, pos)

    # m.sigma - Standard deviation of sensor noise
    sigma0 = m.thresh  # or m.sig
    
    sigmaN = m.sig_pct * conc + m.sig
    
    # Case for yObv <= m.thresh
    if yObv <= m.thresh:
        likelihood = m.pd * 0.5 * (1 + norm.cdf((m.thresh - conc) / (sigma0 * np.sqrt(2)))) + (1 - m.pd)
    else:
        # Case for yObv > m.thresh
        likelihood = 1 / (sigmaN * np.sqrt(2 * np.pi)) * np.exp(-(yObv - conc) ** 2 / (2 * sigmaN ** 2))

    return likelihood