"""
Scaling of in and output

@author: Patrick
"""

import numpy as np
import json

# DEFAULT_STD_SCALER_ENERGY_GRADS = {'x_mean' : np.zeros((1,1,1)),
#                                    'x_std' : np.ones((1,1,1)),
#                                    'energy_mean' : np.zeros((1,1)),
#                                    'energy_std' : np.ones((1,1)),
#                                    'gradient_mean' : np.zeros((1,1,1,1)),
#                                    'gradient_std' : np.ones((1,1,1,1))
#                                    }


class EnergyGradientStandardScaler():
    def __init__(self):
        self.x_mean = np.zeros((1,1,1))
        self.x_std = np.ones((1,1,1))
        self.energy_mean = np.zeros((1,1))
        self.energy_std = np.ones((1,1))
        self.gradient_mean = np.zeros((1,1,1,1))
        self.gradient_std = np.ones((1,1,1,1))

    def scale_x(self,x=None):
        x_res = (x-self.x_mean)/self.x_std 
        return x_res
    
    def rescale_y(self,y=None):
        energy = y[0]
        gradient = y[1]
        out_e = energy * self.energy_std + self.energy_mean 
        out_g = gradient  * self.gradient_std
        return out_e,out_g
    
    def fit(self,x=None,y=None,auto_scale = {'x_mean':True,'x_std':True,'energy_std':True,'energy_mean':True}):
        npeps = np.finfo(float).eps
        if(auto_scale['x_mean'] == True):
            self.x_mean = np.mean(x)
        if(auto_scale['x_std'] == True):
            self.x_std = np.std(x) + npeps
        if(auto_scale['energy_mean'] == True):
            y1 = y[0]
            self.energy_mean = np.mean(y1,axis=0,keepdims=True)
        if(auto_scale['energy_std'] == True):
            y1 = y[0]
            self.energy_std = np.std(y1,axis=0,keepdims=True) + npeps
        self.gradient_std = np.expand_dims(np.expand_dims(self.energy_std,axis=-1),axis=-1) /x_std + npeps
        self.gradient_mean = np.zeros_like(self.gradient_std, dtype=np.float32) #no mean shift expected
    
    def save(self,filepath):
        outdict = {'x_mean' : self.x_mean.tolist(),
                    'x_std' : self.x_std.tolist(),
                    'energy_mean' : self.energy_mean.tolist(),
                    'energy_std' : self.energy_std.tolist(),
                    'gradient_mean' : self.gradient_mean.tolist(),
                    'gradient_std' : self.gradient_std.tolist()
                    }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)
            
    def load(self,filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)
        
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def get(self):
        outdict = {'x_mean' : self.x_mean.tolist(),
                    'x_std' : self.x_std.tolist(),
                    'energy_mean' : self.energy_mean.tolist(),
                    'energy_std' : self.energy_std.tolist(),
                    'gradient_mean' : self.gradient_mean.tolist(),
                    'gradient_std' : self.gradient_std.tolist()
                    }
        return outdict
    
    def set_dict(self,indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

                                 
# def rescale_eg(output, scaler = DEFAULT_STD_SCALER_ENERGY_GRADS ):
#     """
#     Rescale Energy and gradients.

#     Args:
#         output (np.array): [Energy,Gradients]
#         scaler (dict, optional): Scale to revert. The default is DEFAULT_STD_SCALER_ENERGY_GRADS.

#     Returns:
#         out_e (np.array): Rescaled energy.
#         out_g (np.array): gradient.

#     """
#     eng = output[0]
#     grad = output[1]
#     y_energy_std = scaler['energy_std']
#     y_energy_mean = scaler['energy_mean']
#     y_gradient_std = scaler['gradient_std']
#     #y_gradient_mean = scaler['gradient_mean']
    
#     #Scaling
#     out_e = eng * y_energy_std + y_energy_mean 
#     out_g = grad  * y_gradient_std

#     return out_e,out_g

