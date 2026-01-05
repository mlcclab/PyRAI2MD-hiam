"""
The main training script for NAC model. Called with ArgumentParse.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
#from sklearn.utils import shuffle
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import pickle
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a nac model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus",default=-1 ,required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode",default="training" ,required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
#args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/nac_0",'index' : 0,"gpus":0}


fstdout =  open(os.path.join(args['filepath'],"fitlog_"+str(args['index'])+".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout
    
print("Input argpars:",args)

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:",tf.config.experimental.list_logical_devices('GPU'))


from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.callbacks import EarlyStopping,lr_lin_reduction,lr_exp_reduction,lr_step_reduction
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.plotting.plot_mlp_nac import plot_nac_fit_result
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_features import create_feature_models
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.models.models_mlp_nac2 import create_model_nac_precomputed,NACModel2
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hyper import _load_hyp
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.datasets.data_general import split_validation_training_index
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.scaling.scale_mlp_nac import NACStandardScaler
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.scaling.scale_general import scale_feature

def train_model_nac(i=0, outdir=None, mode = 'training'):
    """
    Train NAC model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        outdir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for NAC.

    """
    i = int(i)
    #Load everything from folder
    try:
        with open(os.path.join(outdir,'data_y'),'rb') as f: y_in = pickle.load(f)
        with open(os.path.join(outdir,'data_x'),'rb') as f: x = pickle.load(f)
    except:
        print("Error: Can not load data for fit",outdir)
        return
    hyperall  = None
    try:    
        hyperall = _load_hyp(os.path.join(outdir,'hyper'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load hyper for fit",outdir)
    
    scaler = NACStandardScaler()
    try:
        scaler.load(os.path.join(outdir,'scaler'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load scaling info for fit",outdir)
        
    #Model
    hypermodel = hyperall['model']
    #plots
    unit_label_nac = hyperall['plots']['unit_nac']
    #Fit
    hyper = hyperall[mode]
    phase_less_loss= ['phase_less_loss']
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    pre_epo = hyper['pre_epo']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split'] 
    initialize_weights = hyper['initialize_weights']
    learning_rate = hyper['learning_rate']
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

    #Transfer scaler
    y_nac_std = scaler.nac_std
    y_nac_mean = scaler.nac_mean
    x_mean = scaler.x_mean
    x_std = scaler.x_std
    
    #Data Check here:
    if(len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of",x.shape)
    if(len(y_in.shape) != 4):
        raise ValueError("Input nac-shape must be (batch,states,atoms,3)")
    else:
        print("Found nac-shape of",y_in.shape)
    
    #Set stat dir    
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
    
    
    #Data selection
    lval = int(len(x)*val_split)
    allind = np.arange(0,len(x))
    i_train,i_val = split_validation_training_index(allind,lval,val_disjoint,i)
    print("Info: Train-Test split at Train:",len(i_train),"Test",len(i_val),"Total",len(x))
    
    #Make all Models
    out_model = NACModel2(hypermodel)
    temp_model_feat = create_feature_models(hypermodel)
    temp_model,scaled_mae = create_model_nac_precomputed(hypermodel,learning_rate,phase_less_loss)
    
    npeps = np.finfo(float).eps
    if(initialize_weights==False):
        try:
            out_model.load_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Load old weights at:",os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            #Transfer weights
            print("Info: Transferring weights...")
            temp_model.get_layer('mlp').set_weights(out_model.get_layer('mlp').get_weights())
            temp_model.get_layer('virt').set_weights(out_model.get_layer('virt').get_weights())
            print("Info: Reading Normalization...")
            temp_model.get_layer('feat_std').set_weights(out_model.get_layer('feat_std').get_weights())
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Making new initialized weights..")
        
    if(auto_scale['x_mean'] == True):
        x_mean = np.mean(x)
        print("Info: Calculating x-mean.")
    if(auto_scale['x_std'] == True):
        x_std = np.std(x) + npeps
    if(auto_scale['nac_std'] == True):
        print("Info: Calculating nac-std.")
        yit = y_in[i_train]
        y_nac_std = np.std(yit,axis=(0,3),keepdims=True)+ npeps
        y_nac_mean = np.zeros_like(y_nac_std)
        
    # No x-scale for the moment
    y = (y_in - y_nac_mean) / (y_nac_std )
    #x_rescale = (x - x_mean) / (x_std )
    x_rescale = x
    
    #Calculate features
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
        
    xtrain = [feat_x[i_train],feat_grad[i_train]]
    ytrain = y[i_train]
    xval = [feat_x[i_val],feat_grad[i_val]]
    yval = y[i_val]
    
    #Actutal Fitting
    temp_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])   
    ks.backend.set_value(scaled_mae.scale,y_nac_std )
    temp_model.summary()
    
    print("Info: All-data NAC std",y.shape,":",np.std(y_in,axis=(0,3),keepdims=True)[0,:,:,0])
    print("Info: Using nac-std",y_nac_std.shape,":", y_nac_std[0,:,:,0])
    print("Info: Using nac-mean",y_nac_mean.shape,":", y_nac_mean[0,:,:,0])
    print("Info: Using x-scale",x_std.shape,":" , x_std)
    print("Info: Using x-offset",x_mean.shape,":" , x_mean)
    print("Info: Using feature-scale",feat_x_std.shape,":", feat_x_std)
    print("Info: Using feature-offset",feat_x_mean.shape,":", feat_x_mean)
    
    print("")
    print("Start fit.")   
    
    #Prefit if necessary
    if(pre_epo>0):
        temp_model_prefit,_ = create_model_nac_precomputed(hypermodel,learning_rate,False)
        temp_model_prefit.set_weights(temp_model.get_weights())
        temp_model_prefit.fit(x=xtrain, y=ytrain, epochs=pre_epo,batch_size=batch_size,verbose=2)
        temp_model.set_weights(temp_model_prefit.get_weights())
    
    hist = temp_model.fit(x=xtrain, y=ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_freq=epostep,validation_data=(xval,yval),verbose=2)
    print("End fit.")
    print("")
    
    try:
        print("Info: Saving history...")
        outname = os.path.join(dir_save,"history_"+".json")
        outhist = {a: np.array(b,dtype=np.float64).tolist() for a,b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
          print("Warning: Cant save history")
        
    try:
        #Save Weights
        print("Info: Saving weights...")
        out_model.get_layer('mlp').set_weights(temp_model.get_layer('mlp').get_weights())
        out_model.get_layer('virt').set_weights(temp_model.get_layer('virt').get_weights())
        out_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])
        out_model.save_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
    except:
        print("Error: Cant save weights")
      
    try:
        print("Info: Saving auto-scaling to file...")
        outscaler = {'x_mean' : x_mean,'x_std' : x_std,
                     'nac_mean' : y_nac_mean, 'nac_std' : y_nac_std}
        scaler.set_dict(outscaler)
        scaler.save(os.path.join(outdir,"scaler"+'_v%i'%i+'.json'))
    except:
        print("Error: Can not export scaling info. Model prediciton will be wrongly scaled.")
    
    try:
        #Plot stats
        yval_plot = y_in[i_val] 
        ytrain_plot  = y_in[i_train] 
        #Revert standard but keep unit conversion
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = pval * (y_nac_std )+ y_nac_mean 
        ptrain = ptrain * (y_nac_std)+ y_nac_mean 

        print("Info: Predicted NAC shape:",ptrain.shape)
        print("Info: Plot fit stats...")        
        
        plot_nac_fit_result(i,xval,xtrain,
                            yval_plot,ytrain_plot,
                            pval,ptrain,
                            hist,
                            epostep = epostep,
                            dir_save= dir_save,
                            unit_nac=unit_label_nac)   
    except:
        print("Warning: Could not plot fitting stats")
    
    #error out
    error_val = None

    try:
        print("Info: saving fitting error...")
        #Safe fitting Error MAE
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = pval * y_nac_std + y_nac_mean
        ptrain = ptrain  * y_nac_std + y_nac_mean
        ptrain2 = out_model.predict(x_rescale[i_train])
        ptrain2 = ptrain2  * y_nac_std + y_nac_mean
        print("Info: MAE between precomputed and full keras model:")      
        print("NAC", np.mean(np.abs(ptrain-ptrain2))) 
        error_val = np.mean(np.abs(pval-y_in[i_val]))
        error_train = np.mean(np.abs(ptrain-y_in[i_train]))
        print("error_val:" ,error_val)
        print("error_train:",error_train )
        np.save(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"),error_val)
        np.save(os.path.join(outdir,"fiterr_train" +'_v%i'%i+".npy"),error_train)
    except:
        print("Error: Can not save fiterror")
        
    return error_val
        



if __name__ == "__main__":

    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    train_model_nac(args['index'],args['filepath'],args['mode'])
    
fstdout.close()