"""
Model specific handling and storing data. For example to save to folder etc.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
import os

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.datasets.data_general import make_random_shuffle,save_data_to_folder,merge_np_arrays_in_chunks


def index_make_random_shuffle(x):
    """
    Shuffle indexarray.

    Args:
        x (np.array): Index to shuffle.

    Returns:
        np.array: Shuffled index.

    """
    return shuffle(x)    


def model_save_data_to_folder(model_type,x,y,
               target_model,
               mod_dir,
               random_shuffle = False):
    """
    Save Data to model folder. Always dumps data_x and data_y as pickle.

    Args:
        model_type (str): Model type identifier.
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        target_model (str): Name of the Model to save data for.
        mod_dir (str): Path of model directory.
        random_shuffle (bool, optional): Whether to shuffle data before save. The default is False.
        
    Returns:
        None.

    """
    if(model_type == 'mlp_nac'):
        return save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)
    elif(model_type == 'mlp_nac2'):
        return save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)
    elif(model_type == 'mlp_eg'):
        return save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)
    elif(model_type == 'mlp_e'):
        return save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)
    else:
        print("Error: Unknown model type for data",model_type)
        raise TypeError(f"Error: Unknown model type for predict {target_model}")

  
def model_make_random_shuffle(model_type,x,y,shuffle_ind):
    """
    Shuffle data according to model.    

    Args:
        model_type (str): Type of the model.
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        shuffle_ind (np.array): Index order of datapoints in dataset to shuffle after.

    Returns:
        None.

    """
    if(model_type == 'mlp_nac'):
        return make_random_shuffle([x,y],shuffle_ind)[1]
    elif(model_type == 'mlp_nac2'):
        return make_random_shuffle([x,y],shuffle_ind)[1]
    elif(model_type == 'mlp_eg'):
        _,temp = make_random_shuffle([x]+y,shuffle_ind)
        return temp[0],temp[1:]
    elif(model_type == 'mlp_e'):
        return make_random_shuffle([x,y],shuffle_ind)[1]
    else:
        print("Error: Unknown model type for data",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
   
       
def model_merge_data_in_chunks(model_type,mx1,my1,mx2,my2,val_split=0.1):
    """
    Merge Data in chunks.

    Args:
        model_type (str): Type of the model.
        mx1 (list,np.array): Coordinates as x-data.
        my1 (list,np.array): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        mx2 (list,np.array): Coordinates as x-data.
        my2 (list,np.array): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        val_split (float, optional): Validation split. Defaults to 0.1.

    Raises:
        TypeError: Unkown model type.

    Returns:
        x: Merged x data. Depending on model.
        y: Merged y data. Depending on model.
            
    """
    if(model_type == 'mlp_nac'):
        x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
        y_merge = merge_np_arrays_in_chunks(my1,my2,val_split)
        return x_merge,y_merge
    if(model_type == 'mlp_nac2'):
        x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
        y_merge = merge_np_arrays_in_chunks(my1,my2,val_split)
        return x_merge,y_merge
    elif(model_type == 'mlp_eg'):
        x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
        y1_merge = merge_np_arrays_in_chunks(my1[0],my2[0],val_split)
        y2_merge = merge_np_arrays_in_chunks(my1[1],my2[1],val_split)       
        return x_merge,[y1_merge,y2_merge]
    elif(model_type == 'mlp_e'):
        x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
        y_merge = merge_np_arrays_in_chunks(my1,my2,val_split)
        return x_merge,y_merge
    else:
        print("Error: Unknown model type for data",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
        




          