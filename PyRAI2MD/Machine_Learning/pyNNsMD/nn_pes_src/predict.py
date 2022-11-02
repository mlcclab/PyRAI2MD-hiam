"""
Created on Sat Oct 10 19:48:15 2020

@author: Patrick
"""

import numpy as np
import tensorflow as tf


from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.predicting.predict_mlp_nac import _predict_uncertainty_mlp_nac
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.predicting.predict_mlp_eg import _predict_uncertainty_mlp_eg


def _predict_uncertainty(model_type,out):
    if(model_type == 'mlp_nac'):
        return _predict_uncertainty_mlp_nac(out)
    elif(model_type == 'mlp_nac2'):
        return _predict_uncertainty_mlp_nac(out)
    elif(model_type == 'mlp_eg'):
        return _predict_uncertainty_mlp_eg(out)
    elif(model_type == 'mlp_e'):
        return _predict_uncertainty_mlp_eg(out)
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")


       

def _call_convert_x_to_tensor(model_type,x):
    if(model_type == 'mlp_eg'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    elif(model_type == 'mlp_nac'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    elif(model_type == 'mlp_nac2'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    elif(model_type == 'mlp_e'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
    
    
    
def _call_convert_y_to_numpy(model_type,temp):
    if(model_type == 'mlp_nac'):
        return temp.numpy()
    if(model_type == 'mlp_nac2'):
        return temp.numpy()
    elif(model_type == 'mlp_eg'):
        return [temp[0].numpy(),temp[1].numpy()]
    elif(model_type == 'mlp_e'):
        return [temp[0].numpy(),temp[1].numpy()]
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
        