"""
Scaling of in and output.

@author: Patrick
"""

import numpy as np
import json
import os


class NACStandardScaler():
    def __init__(self):
        self.x_mean = np.zeros((1,1,1))
        self.x_std = np.ones((1,1,1))
        self.nac_mean = np.zeros((1,1,1,1))
        self.nac_std = np.ones((1,1,1,1))

    def scale_x(self,x=None):
        x_res = (x-self.x_mean)/self.x_std 
        return x_res
    
    def rescale_y(self,y=None):
        out_nac = y * self.nac_std + self.nac_mean
        return out_nac
    
    def fit(self,x,y,auto_scale = {'x_mean':False,'x_std':False,'nac_std':True,'nac_mean':False} ):
        npeps = np.finfo(float).eps
        if(auto_scale['x_mean'] == True):
            self.x_mean = np.mean(x)
        if(auto_scale['x_std'] == True):
            self.x_std = np.std(x) + npeps
        if(auto_scale['nac_std'] == True):
            self.nac_std = np.std(y,axis=(0,3),keepdims=True)+ npeps
            self.nac_mean = np.zeros_like(y_nac_std)
    
    def save(self,filepath):
        outdict = {'x_mean' : self.x_mean.tolist(),
                    'x_std' : self.x_std.tolist(),
                    'nac_mean' : self.nac_mean.tolist(),
                    'nac_std' : self.nac_std.tolist()
                    }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)
            
    def load(self,filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)
        
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.nac_mean = np.array(indict['nac_mean'])
        self.nac_std = np.array(indict['nac_std'])

    def get(self):
        outdict = {'x_mean' : self.x_mean.tolist(),
                    'x_std' : self.x_std.tolist(),
                    'nac_mean' : self.nac_mean.tolist(),
                    'nac_std' : self.nac_std.tolist(),
                    }
        return outdict
    
    def set_dict(self,indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.nac_mean = np.array(indict['nac_mean'])
        self.nac_std = np.array(indict['nac_std'])
