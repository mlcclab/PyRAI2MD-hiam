"""
Created on Thu Oct 29 12:52:36 2020

@author: Patrick
"""

import numpy as np

def _predict_uncertainty_mlp_nac(out):
    out_mean = np.mean(np.array(out),axis=0)
    out_std = np.std(np.array(out),axis=0,ddof=1)
    return out_mean,out_std 