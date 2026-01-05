"""
The main training script for energy gradient model. Called with ArgumentParse.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
#from sklearn.utils import shuffle
#import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import pickle
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus",default=-1 ,required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode",default="training" ,required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
#args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/energy_gradient_0",'index' : 0,"gpus":0}


fstdout =  open(os.path.join(args['filepath'],"fitlog_"+str(args['index'])+".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:",args)

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:",tf.config.experimental.list_logical_devices('GPU'))

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.callbacks import EarlyStopping,lr_lin_reduction,lr_exp_reduction,lr_step_reduction
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.plotting.plot_mlp_eg import plot_energy_gradient_fit_result
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_features import create_feature_models
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_eg import create_model_energy_gradient_precomputed,EnergyGradientModel
#from pyNNsMD.nn_pes_src.legacy import compute_feature_derivative
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hyper import _load_hyp
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.datasets.data_general import split_validation_training_index
#from pyNNsMD.nn_pes_src.scaler import save_std_scaler_dict
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.scaling.scale_mlp_eg import EnergyGradientStandardScaler
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.scaling.scale_general import scale_feature


def train_model_energy_gradient(i = 0, outdir=None,  mode='training'): 
    """
    Train an energy plus gradient model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        outdir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for (energy,gradient).

    """
    i = int(i)
    #Load everything from folder
    try:
        with open(os.path.join(outdir,'data_y'),'rb') as f: y = pickle.load(f)
        with open(os.path.join(outdir,'data_x'),'rb') as f: x = pickle.load(f)
    except:
        print("Error: Can not load data for fit",outdir)
        return
    hyperall  = None
    try:    
        hyperall = _load_hyp(os.path.join(outdir,'hyper'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load hyper for fit",outdir)
    
    scaler = EnergyGradientStandardScaler()
    try:
        scaler.load(os.path.join(outdir,'scaler'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load scaling info for fit",outdir)
        
    #Model
    hypermodel  = hyperall['model']
    #plots
    unit_label_energy = hyperall['plots']['unit_energy']
    unit_label_grad = hyperall['plots']['unit_gradient']
    #Fit
    hyper = hyperall[mode]
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split'] 
    initialize_weights = hyper['initialize_weights']
    learning_rate = hyper['learning_rate']
    loss_weights = hyper['loss_weights']
    auto_scale = hyper['auto_scaling']
    normalize_feat = int(hyper['normalization_mode'])
    #step
    use_step_callback = hyper['step_callback']['use']
    epoch_step_reduction = hyper['step_callback']['epoch_step_reduction']
    learning_rate_step = hyper['step_callback']['learning_rate_step']
    #lin
    use_linear_callback = hyper['linear_callback']['use']
    learning_rate_start = hyper['linear_callback']['learning_rate_start']
    learning_rate_stop = hyper['linear_callback']['learning_rate_stop']
    epomin_lin = hyper['linear_callback']['epomin']
    #exp
    use_exp_callback = hyper['exp_callback']['use']    
    epomin_exp = hyper['exp_callback']['epomin']
    factor_lr_exp = hyper['exp_callback']['factor_lr']
    #early
    use_early_callback = hyper['early_callback']['use']
    epomin_early = hyper['early_callback']['epomin']
    factor_lr_early = hyper['early_callback']['factor_lr']
    patience =  hyper['early_callback']['patience']
    max_time = hyper['early_callback']['max_time']
    delta_loss = hyper['early_callback']['delta_loss']
    loss_monitor = hyper['early_callback']['loss_monitor']
    learning_rate_start_early = hyper['linear_callback']['learning_rate_start']
    learning_rate_stop_early = hyper['linear_callback']['learning_rate_stop']

    #scaler
    y_energy_std = scaler.energy_std
    y_energy_mean = scaler.energy_mean
    y_gradient_std = scaler.gradient_std
    y_gradient_mean = scaler.gradient_mean
    x_mean = scaler.x_mean
    x_std = scaler.x_std

    #Data Check here:
    if(len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of",x.shape)
    if(isinstance(y,list)==False):
        raise ValueError("Input y must be list of [energy,gradient]")
    if(len(y[0].shape) != 2):
        raise ValueError("Input energy-shape must be (batch,states)")
    else:
        print("Found energy-shape of",y[0].shape)
    if(len(y[1].shape) != 4):
        raise ValueError("Input gradient-shape must be (batch,states,atoms,3)")
    else:
        print("Found gradient-shape of",y[1].shape)

    #Fit stats dir
    dir_save = os.path.join(outdir,"fit_stats")
    os.makedirs(dir_save,exist_ok=True)    

    #cbks,Learning rate schedule  
    cbks = []
    if(use_early_callback == True):
        es_cbk = EarlyStopping(patience = patience,
                       minutes=max_time,
                       epochs=epo,
                       learning_rate=learning_rate_start_early,
                       min_delta=delta_loss,
                       epostep=epostep,
                       min_lr=learning_rate_stop_early,
                       monitor=loss_monitor,
                       factor=factor_lr_early,
                       minEpoch=epomin_early) 
        cbks.append(es_cbk)
    if(use_linear_callback == True):
        lr_sched = lr_lin_reduction(learning_rate_start,learning_rate_stop,epo,epomin_lin)
        lr_cbk = tf.keras.callbacks.LearningRateScheduler(lr_sched)
        cbks.append(lr_cbk)
    if(use_exp_callback == True):
        lr_exp = lr_exp_reduction(learning_rate_start,epomin_exp,epostep,factor_lr_exp)
        exp_cbk = tf.keras.callbacks.LearningRateScheduler(lr_exp)
        cbks.append(exp_cbk)
    if(use_step_callback == True):
        lr_step = lr_step_reduction(learning_rate_step,epoch_step_reduction)
        step_cbk = tf.keras.callbacks.LearningRateScheduler(lr_step)
        cbks.append(step_cbk)
    
    # Index train test split
    lval = int(len(x)*val_split)
    allind = np.arange(0,len(x))
    i_train,i_val = split_validation_training_index(allind,lval,val_disjoint,i)
    print("Info: Train-Test split at Train:",len(i_train),"Test",len(i_val),"Total",len(x))
    
    #Make all Model
    out_model = EnergyGradientModel(hypermodel)
    temp_model = create_model_energy_gradient_precomputed(hypermodel,learning_rate,loss_weights)
    temp_model_feat = create_feature_models(hypermodel)
    
    #Look for loading weights
    npeps = np.finfo(float).eps
    if(initialize_weights==False):
        try:
            out_model.load_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Load old weights at:",os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Transferring weights...")
            temp_model.get_layer('mlp').set_weights(out_model.get_layer('mlp').get_weights())
            temp_model.get_layer('energy').set_weights(out_model.get_layer('energy').get_weights())
            print("Info: Reading standardization...")
            temp_model.get_layer('feat_std').set_weights(out_model.get_layer('feat_std').get_weights())
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Making new initialized weights.")
    
    #Recalculate standardization    
    if(auto_scale['x_mean'] == True):
        x_mean = np.mean(x)
        print("Info: Calculating x-mean.")
    if(auto_scale['x_std'] == True):
        x_std = np.std(x) + npeps
    if(auto_scale['energy_mean'] == True):
        print("Info: Calculating energy-mean.")
        y1 = y[0][i_train]
        y_energy_mean = np.mean(y1,axis=0,keepdims=True)
    if(auto_scale['energy_std'] == True):
        print("Info: Calculating energy-std.")
        y1 = y[0][i_train]
        y_energy_std = np.std(y1,axis=0,keepdims=True) + npeps
    print("Info: Adjusting gradient std.")    
    #Gradient is not independent!!
    y_gradient_std = np.expand_dims(np.expand_dims(y_energy_std,axis=-1),axis=-1) /x_std + npeps
    y_gradient_mean = np.zeros_like(y_gradient_std, dtype=np.float32) #no mean shift expected
    
    #Apply Scaling
    x_rescale = (x-x_mean) / (x_std)
    y1 = (y[0] - y_energy_mean)/(y_energy_std)
    y2 = y[1]/(y_gradient_std)
    
    #Model + Model precompute layer +feat
    feat_x, feat_grad = temp_model_feat.predict_in_chunks(x_rescale,batch_size=batch_size)
    
    #Finding Normalization
    feat_x_mean,feat_x_std = temp_model.get_layer('feat_std').get_weights()
    if(normalize_feat==1):
        print("Info: Making new feature normalization for last dimension.")
        feat_x_mean = np.mean(feat_x[i_train],axis=0,keepdims=True)
        feat_x_std = np.std(feat_x[i_train],axis=0,keepdims=True)
    elif(normalize_feat==2):
        feat_x_mean,feat_x_std = scale_feature(feat_x[i_train],hypermodel)
    else:
        print("Info: Keeping old normalization (default/unity or loaded from file).")
        
    #Train Test split
    xtrain = [feat_x[i_train],feat_grad[i_train]]
    ytrain = [y1[i_train],y2[i_train]]
    xval = [feat_x[i_val],feat_grad[i_val]]
    yval = [y1[i_val],y2[i_val]]
    
    #Setting constant feature normalization
    temp_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])  
    # This is only for metric to without std.
    temp_model.metrics_y_gradient_std = tf.constant(y_gradient_std,dtype=tf.float32)
    temp_model.metrics_y_energy_std = tf.constant(y_energy_std,dtype=tf.float32)

    
    print("Info: Total-Data gradient std",y[1].shape,":",np.std(y[1],axis=(0,2,3)))
    print("Info: Total-Data energy std",y[0].shape,":",np.std(y[0],axis=0))
    print("Info: Using energy-std",y_energy_std.shape,":", y_energy_std[0])
    print("Info: Using energy-mean" ,y_energy_mean.shape,":", y_energy_mean[0])
    print("Info: Using gradient-std" ,y_gradient_std.shape,":", y_gradient_std[0,:,0,0])
    print("Info: Using gradient-mean" ,y_gradient_mean.shape,":", y_gradient_mean[0,:,0,0])
    print("Info: Using x-scale",x_std.shape,":", x_std)
    print("Info: Using x-offset", x_mean.shape,":",x_mean)
    print("Info: Using feature-scale", feat_x_std.shape,":",feat_x_std)
    print("Info: Using feature-offset", feat_x_mean.shape,":",feat_x_mean)
    temp_model.summary()
    print("")
    print("Start fit.")
    hist = temp_model.fit(x=xtrain, y=ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_freq=epostep,validation_data=(xval,yval),verbose=2)
    print("End fit.")
    print("")
    
    try:
        outname = os.path.join(dir_save,"history_"+".json")           
        outhist = {a: np.array(b,dtype=np.float64).tolist() for a,b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
          print("Warning: Cant save history")
    
    try:
        #Save weights
        out_model.get_layer('mlp').set_weights(temp_model.get_layer('mlp').get_weights())
        out_model.get_layer('energy').set_weights(temp_model.get_layer('energy').get_weights())
        out_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])
        out_model.save_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
        #print(out_model.get_weights())
    except:
          print("Warning: Cant save weights")
          
    try:
        print("Info: Saving auto-scaling to file...")
        outscaler = {'x_mean' : x_mean,'x_std' : x_std,
                     'energy_mean' : y_energy_mean, 'energy_std' : y_energy_std,
                     'gradient_mean' : y_gradient_mean, 'gradient_std' : y_gradient_std
                     }
        scaler.set_dict(outscaler)
        scaler.save(os.path.join(outdir,"scaler"+'_v%i'%i+'.json'))
    except:
        print("Error: Can not export scaling info. Model prediciton will be wrongly scaled.")
    
    try:
        #Plot and Save
        yval_plot = [y[0][i_val] , y[1][i_val] ]
        ytrain_plot = [y[0][i_train] , y[1][i_train] ]
        # Convert back scaling
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = [pval[0] * y_energy_std + y_energy_mean,
                pval[1] * y_gradient_std ]
        ptrain = [ptrain[0] * y_energy_std + y_energy_mean,
                  ptrain[1] * y_gradient_std]
    

        print("Info: Predicted Energy shape:",ptrain[0].shape)
        print("Info: Predicted Gradient shape:",ptrain[1].shape)
        print("Info: Plot fit stats...")        
        
        #Plot
        plot_energy_gradient_fit_result(i,xval,xtrain,
                yval_plot,ytrain_plot,
                pval,ptrain,
                hist,
                epostep = epostep,
                dir_save= dir_save,
                unit_energy=unit_label_energy,
                unit_force=unit_label_grad)     
    except:
        print("Error: Could not plot fitting stats")
        
    error_val = None
    try:
        #Safe fitting Error MAE
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = [pval[0]  * y_energy_std + y_energy_mean,
                pval[1] * y_gradient_std ]
        ptrain = [ptrain[0] * y_energy_std+ y_energy_mean,
                  ptrain[1]  * y_gradient_std ]
        ptrain2 = out_model.predict(x_rescale[i_train])
        ptrain2 = [ptrain2[0] * y_energy_std+ y_energy_mean,
                  ptrain2[1]  * y_gradient_std ]
        print("Info: Max error between precomputed and full keras model:")
        print("Energy",np.max(np.abs(ptrain[0]-ptrain2[0])))        
        print("Gradient",np.max(np.abs(ptrain[1]-ptrain2[1]))) 
        error_val = [np.mean(np.abs(pval[0]-y[0][i_val])),np.mean(np.abs(pval[1]-y[1][i_val])) ]
        error_train = [np.mean(np.abs(ptrain[0]-y[0][i_train])),np.mean(np.abs(ptrain[1]-y[1][i_train])) ]
        np.save(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"),error_val)
        np.save(os.path.join(outdir,"fiterr_train" +'_v%i'%i+".npy"),error_train)
        print("error_val:" ,error_val)
        print("error_train:",error_train )
    except:
        print("Error: Can not save fiterror")
    
    return error_val



if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    out = train_model_energy_gradient(args['index'],args['filepath'],args['mode'])
    
fstdout.close()
