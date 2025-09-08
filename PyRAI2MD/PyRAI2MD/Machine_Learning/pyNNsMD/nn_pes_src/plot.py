"""
Functions to plot fitresults.
They include training and resampling.
"""
import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os


from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.plotting.plot_mlp_nac import plot_resampling_nac
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.plotting.plot_mlp_eg import plot_resampling_gradient

def _plot_resampling(model_type,dirpath,out_index,out_error,out_fiterr,out_testerr,plotdict):
    if(model_type == 'mlp_eg'):
        plot_resampling_gradient(dirpath,
                                     out_index,
                                     np.array(out_error),
                                     np.array(out_fiterr) ,
                                     np.array(out_testerr),
                                     unit_energy=plotdict['unit_energy'],
                                     unit_force=plotdict['unit_gradient']
                                     )
    if(model_type == 'mlp_nac'):
        plot_resampling_nac(dirpath,
                                 out_index,
                                 np.array(out_error),
                                 np.array(out_fiterr),
                                 np.array(out_testerr),
                                 unit_nac = plotdict['unit_nac']
                                 )

    else:
        print("Error: Can not find model type",model_type)