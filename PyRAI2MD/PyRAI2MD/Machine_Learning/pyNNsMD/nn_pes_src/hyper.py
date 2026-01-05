"""
Default hyperparameters
"""

import json

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hypers.hyper_mlp_e import DEFAULT_HYPER_PARAM_ENERGY
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC



def _get_default_hyperparameters_by_modeltype(model_type):
    """
    Select the default parameters for each model

    Args:
        model_type (str): Model identifier.

    Returns:
        dict: Default hyper parameters for model.

    """
    if(model_type == 'mlp_eg'):
        return DEFAULT_HYPER_PARAM_ENERGY_GRADS
    elif(model_type == 'mlp_e'):
        return DEFAULT_HYPER_PARAM_ENERGY
    elif(model_type == 'mlp_nac'):
        return DEFAULT_HYPER_PARAM_NAC
    elif(model_type == 'mlp_nac2'):
        return DEFAULT_HYPER_PARAM_NAC
    else:
        print("Error: Unknown model type",model_type)
        raise TypeError(f"Error: Unknown model type for default hyper parameter {model_type}")



def _save_hyp(HYPERPARAMETER,filepath): 
    with open(filepath, 'w') as f:
        json.dump(HYPERPARAMETER, f)

def _load_hyp(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)