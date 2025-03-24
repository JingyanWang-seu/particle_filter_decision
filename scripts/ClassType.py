## source parmeter
class Source:
    def __init__(self, x, y, q, u, phi, d, tao):
        self.x = x       # x position
        self.y = y       # y position
        self.q = q       # release rate
        self.u = u       # wind speed
        self.phi = phi   # wind direction
        self.d = d       # diffusivity of the hazard
        self.tao = tao   # Lifetime of the emitted material

## the particle of the source pamater
class Theta:
    def __init__(self, x, y, q, u, phi, d, tao):
        self.x = x       # x position
        self.y = y       # y position
        self.q = q       # release rate
        self.u = u       # wind speed
        self.phi = phi   # wind direction
        self.d = d       # diffusivity of the hazard
        self.tao = tao   # Lifetime of the emitted material


class Sensor:
    def __init__(self, thresh, pd, sig, sig_pct):
        self.thresh = thresh     # sensor threshold
        self.pd = pd             # probability of detection
        self.sig = sig           # minimum sensor noise
        self.sig_pct = sig_pct   # the standard deviation
