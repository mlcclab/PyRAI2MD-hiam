"""
Selecting different tf.keras models.

@author: Patrick
"""

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_nac import NACModel
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_nac2 import NACModel2
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_eg import EnergyGradientModel
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_e import EnergyModel



def _get_model_by_type(model_type, hyper):
    """
    Find the implemented model by its string identifier.

    Args:
        model_type (str): Model type.
        hyper (dict): Dict with hyper parameters.

    Returns:
        tf.keras.model: Defult initialized tf.keras.model.

    """
    if(model_type == 'mlp_eg'):
        return EnergyGradientModel(hyper)
    elif(model_type == 'mlp_e'):
        return EnergyModel(hyper)
    elif(model_type == 'mlp_nac'):
       return NACModel(hyper)
    elif(model_type == 'mlp_nac2'):
       return NACModel2(hyper)
    else:
        print(f"Error: Unknwon Model type in hyper dict for {model_type}")
        raise TypeError(f"Error: Unknown model type {model_type}")