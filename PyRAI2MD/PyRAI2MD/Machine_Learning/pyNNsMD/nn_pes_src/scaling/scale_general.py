"""
Created on Tue Nov 10 11:47:49 2020

@author: Patrick
"""
import numpy as np



def scale_feature(feats,hyper):
    """
    Scale features.
    
    This rquires knowledge on how the featue vector is composed. 
    Must be changed with the feautre description.

    Args:
        feats (np.array): DESCRIPTION.
        hyp (dict): DESCRIPTION.

    Returns:
        out_mean (np.array): DESCRIPTION.
        out_scale (np.array): DESCRIPTION.

    """
    print("Feature shape", feats.shape)

    out_dim = int( hyper['states'])
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
    
    in_model_dim = 0
    out_scale = []
    out_mean = []
    
    if(use_invd_index == True):
        invd_dim = invd_shape[0]
    else:
        invd_dim = int(indim*(indim-1)/2)
    if(invd_dim>0):
        invd_mean = np.mean(feats[:,0:invd_dim])
        invd_std  = np.std(feats[:,0:invd_dim])
        out_scale.append(np.tile(np.expand_dims(invd_std,axis=-1),(1,invd_dim)))
        out_mean.append(np.tile(np.expand_dims(invd_mean,axis=-1),(1,invd_dim)))
    
    if(use_angle_index == True):
        angle_dim = angle_shape[0]
    else:
        angle_dim = 0
    if(angle_dim>0):
        angle_mean = np.mean(feats[:,invd_dim:(invd_dim+angle_dim)])
        angle_std = np.std(feats[:,invd_dim:(invd_dim+angle_dim)])
        out_scale.append(np.tile(np.expand_dims(angle_std,axis=-1),(1,angle_dim)))
        out_mean.append(np.tile(np.expand_dims(angle_mean,axis=-1),(1,angle_dim)))
        
    if(use_dihyd_index == True):
        dihyd_dim = dihyd_shape[0]
    else:
        dihyd_dim = 0
    if(dihyd_dim > 0 ):
        dihyd_mean = np.mean(feats[:,(invd_dim+angle_dim):(invd_dim+angle_dim+dihyd_dim)])
        dihyd_std = np.std(feats[:,(invd_dim+angle_dim):(invd_dim+angle_dim+dihyd_dim)])
        out_scale.append(np.tile(np.expand_dims(dihyd_std,axis=-1),(1,dihyd_dim)))
        out_mean.append(np.tile(np.expand_dims(dihyd_mean,axis=-1),(1,dihyd_dim)))
    
    
    out_scale = np.concatenate(out_scale,axis=-1)
    out_mean = np.concatenate(out_mean,axis=-1)
    return out_mean,out_scale