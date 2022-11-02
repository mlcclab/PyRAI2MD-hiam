"""
Main interface to start training_??.py scripts in parallel. This can be solved in many different ways.

Possible are server solutions with slurm and MPI. Here only python subprocess are started to local machine.
The training scripts are supposed to read all necessary information from folder. 
NOTE: Path information of folder and training scripts as well as os info are made fetchable but could fail in certain
circumstances.
"""
import numpy as np
import time
import os
import sys
import subprocess
import sys


def get_path_for_fit_script(model_type):
    """
    Interface to find the path of training scripts.
    
    For now they are expected to be in the same folder-system as calling .py script.
    
    Args:
        model_type (str): Name of the model.

    Returns:
        filepath (str): Filepath pointing to training scripts.

    """
    #Ways of finding path either os.getcwd() or __file__ or just set static path with install...
    locdiR = os.getcwd()
    filepath = os.path.abspath(os.path.dirname(__file__) )
    STATIC_PATH_FIT_SCRIPT = ""
    fit_script = {"mlp_eg" : "training_mlp_eg.py",
                  "mlp_nac" : "training_mlp_nac.py",
                  "mlp_nac2" : "training_mlp_nac2.py",
                  "mlp_e" : "training_mlp_e.py"}
    outpath = os.path.join(filepath,"training",fit_script[model_type])
    return outpath


def fit_model_get_python_cmd_os():
    """
    Return proper commandline command for pyhton depending on os.

    Returns:
        str: Python command either python or pyhton3.

    """
    # python or python3 to run
    if(sys.platform[0:3] == 'win'):
        return 'python' # or 'python.exe'
    else:
        return 'python3'


def _fit_model_by_modeltype(model_type,dist_method,i,filepath,g,m):
    """
    Run the training script in subprocess.

    Args:
        model_type (str): Name of the model.
        dist_method (tba): Method to call training scripts on cluster.
        i (int): Index of model.
        filepath (str): Filepath to model.
        g (int): GPU index to use.
        m (str): Fitmode.

    Returns:
        None.

    """
    print("Run:",filepath, "Instance:",i, "on GPU:",g,m)
    py_script = get_path_for_fit_script(model_type)
    py_cmd = fit_model_get_python_cmd_os()
    if(os.path.exists(py_script) == False):
        print("Error: Can not find trainingsript, please check path",py_script)
    if(dist_method==True):
        proc = subprocess.Popen([py_cmd,py_script,"-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)]) 
        return proc
    if(dist_method==False):
        proc = subprocess.run([py_cmd,py_script, "-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)],capture_output=False,shell = False)
        return proc
    