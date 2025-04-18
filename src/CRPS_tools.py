import numpy as np
from scipy.special import erf

@np.vectorize
def CRPS_gaussian(mu, sig, obs):

    if sig == 0:
        z = np.nan
    else:    
        z = (obs - mu) / sig
    
    CRPS = sig / np.sqrt(np.pi) * (
        np.sqrt(np.pi) * z * erf( z / np.sqrt(2)) + np.sqrt(2) * np.exp(- z**2 / 2) - 1
    )

    return CRPS
