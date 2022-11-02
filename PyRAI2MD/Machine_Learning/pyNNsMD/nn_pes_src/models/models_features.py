# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:26:10 2020

@author: Patrick
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC as hyper_create_model_nac
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.layers import MLP,ConstLayerNormalization,PropagateNACGradient2,FeatureGeometric
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.loss import get_lr_metric,r2_metric,NACphaselessLoss,ScaledMeanAbsoluteError



class FeatureModel(ks.Model):
    def __init__(self, **kwargs):
        super(FeatureModel, self).__init__(**kwargs)
    #No training here possible
        
    def predict_step(self, data):
        #Precompute features with gradient:
        x,_,_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            feat_pred = self(x, training=False)  # Forward pass 
        grad = tape2.batch_jacobian(feat_pred, x)
        return [feat_pred,grad]

    def predict_in_chunks(self,x,batch_size):
        np_x = []
        np_grad = []
        for j in range(int(np.ceil(len(x)/batch_size))):
            a = int(batch_size*j)
            b = int(batch_size*j + batch_size)
            invd,grad = self.predict(x[a:b])
            np_x.append(np.array(invd)) 
            np_grad.append(np.array(grad))
            
        np_x = np.concatenate(np_x,axis=0) 
        np_grad = np.concatenate(np_grad,axis=0)
        return np_x,np_grad


def create_feature_models(hyper,model_name="feat",run_eagerly=False,use_derivative = True):
    """
    Model to precompute features feat = model(x).

    Args:
        hyper (dict): Hyper dictionary.
        model_name (str, optional): Name of the Model. Defaults to "feat".
        run_eagerly (bool, optional): Whether to run eagerly. Defaults to False.

    Returns:
        model (keras.model): tf.keras model with coordinate input.

    """
    indim = int( hyper['atoms'])
    invd_index = hyper['invd_index']
    angle_index = hyper['angle_index']
    dihyd_index = hyper['dihyd_index']
    
    use_invd_index = len(invd_index)>0 if isinstance(invd_index,list) or isinstance(invd_index,np.ndarray) else False
    use_angle_index = len(angle_index)>0 if isinstance(angle_index,list) or isinstance(angle_index,np.ndarray) else False
    use_dihyd_index = len(dihyd_index)>0 if isinstance(dihyd_index,list) or isinstance(dihyd_index,np.ndarray) else False
    
    invd_index = np.array(invd_index,dtype = np.int64) if use_invd_index else None
    angle_index = np.array(angle_index ,dtype = np.int64) if use_angle_index else None
    dihyd_index = np.array(dihyd_index,dtype = np.int64) if use_dihyd_index else None
    
    invd_shape = invd_index.shape if use_invd_index else None
    angle_shape = angle_index.shape if use_angle_index else None
    dihyd_shape = dihyd_index.shape if use_dihyd_index else None
    
    geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
    #Features precompute layer  

    feat_layer = FeatureGeometric(invd_shape = invd_shape,
                                  angle_shape = angle_shape,
                                  dihyd_shape = dihyd_shape,
                                  )
    feat_layer.set_mol_index(invd_index, angle_index , dihyd_index)      
    feat = feat_layer(geo_input)
    
    feat = ks.layers.Flatten(name='feat_flat')(feat)
    if(use_derivative == True):
        model = FeatureModel(inputs=geo_input, outputs=feat,name=model_name)
    else:
        model = ks.Model(inputs=geo_input, outputs=feat,name=model_name)
    
    model.compile(run_eagerly=run_eagerly) 
    #Strange bug with tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [0] [Op:Assert] name: EagerVariableNameReuse
  
    return model



# def create_feature_models(hyper,model_name="feat",run_eagerly=False):
#     """
#     Model to precompute features feat = model(x).

#     Args:
#         hyper (dict): Hyper dictionary.
#         model_name (str, optional): Name of the Model. Defaults to "feat".
#         run_eagerly (bool, optional): Whether to run eagerly. Defaults to False.

#     Returns:
#         model (keras.model): tf.keras model with coordinate input.

#     """
#     indim = int( hyper['atoms'])
#     use_invdist = hyper['invd_index'] != []
#     use_bond_angles = hyper['angle_index'] != []
#     angle_index = hyper['angle_index'] 
#     use_dihyd_angles = hyper['dihyd_index'] != []
#     dihyd_index = hyper['dihyd_index']
    
#     geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
#     #Features precompute layer        
#     if(use_invdist==True):
#         invdlayer = InverseDistance()
#         feat = invdlayer(geo_input)
#     if(use_bond_angles==True):
#         if(use_invdist==False):
#             feat = Angles(angle_list=angle_index)(geo_input)
#         else:
#             angs = Angles(angle_list=angle_index)(geo_input)
#             feat = ks.layers.concatenate([feat,angs], axis=-1)
#     if(use_dihyd_angles==True):
#         if(use_invdist==False and use_bond_angles==False):
#             feat = Dihydral(angle_list=dihyd_index)(geo_input)
#         else:
#             dih = Dihydral(angle_list=dihyd_index)(geo_input)
#             feat = ks.layers.concatenate([feat,dih], axis=-1)
    
#     feat = ks.layers.Flatten(name='feat_flat')(feat)
#     model = FeatureModel(inputs=geo_input, outputs=feat,name=model_name)
    
#     model.compile(run_eagerly=run_eagerly) 
#     #Strange bug with tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [0] [Op:Assert] name: EagerVariableNameReuse
  
#     return model