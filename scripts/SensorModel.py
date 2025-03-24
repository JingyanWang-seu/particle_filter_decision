import DispersionModel
import random

def sensor(S, p, m):
    M = DispersionModel.dispersion(S, p)
    err = m.sig_pct * M * random.gauss(0, 1)
    sensor_M = M + err

    if sensor_M < m.thresh:
        sensor_M = 0
    else:
        rand = random.uniform(0, 1)
        if rand < (1 - m.pd):
            sensor_M = 0
    return sensor_M


