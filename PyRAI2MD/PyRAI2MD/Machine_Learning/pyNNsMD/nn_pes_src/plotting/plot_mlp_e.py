"""
Plot of mlp_e model.

@author: Patrick
"""

import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os


def find_max_relative_error(preds,yval):
    """
    Find maximum error and its relative value if possible.

    Args:
        preds (np.array): Prediction array.
        yval (np.array): Validation array.

    Returns:
        pred_err (np.array): Flatten maximum error along axis=0
        prelm (np.array): Flatten Relative maximum error along axis=0

    """
    pred = np.reshape(preds,(preds.shape[0],-1))
    flat_yval = np.reshape(yval,(yval.shape[0],-1))
    maxerr_ind = np.expand_dims(np.argmax(np.abs(pred-flat_yval),axis=0),axis=0)
    pred_err = np.abs(np.take_along_axis(pred,maxerr_ind,axis=0)-
                      np.take_along_axis(flat_yval,maxerr_ind,axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        prelm = pred_err / np.abs(np.take_along_axis(flat_yval,maxerr_ind,axis=0))
    pred_err = pred_err.flatten()
    prelm = prelm.flatten()
    return pred_err,prelm



def plot_energy_fit_result(i,xval,xtrain,yval,ytrain,
                                    predval,predtrain,
                                    hist,
                                    epostep=1,
                                    dir_save=None,
                                    unit_energy="eV"):
    """
    Plot and store fit.

    Args:
        i (int): index of the nerual network.
        xval (np.array): Validation Data.
        xtrain (np.array): Training Data.
        yval (list): True Validation data. [energy, gradient]
        ytrain (list): True Training data. [energy, gradient]
        predval (list): Model prediction for validation.
        predtrain (list): Model prediction for training.
        hist (dict): histogram from keras.
        epostep (int, optional): Step size of evals. The default is 1.
        dir_save (str, optional): Path of dictionary to store plots. The default is None.
        unit_energy (str, optional): Unit of energy. The default is 'eV'.
        unit_force (str, optional): DESCRIPTION. Unit of force. The default is 'eV/A'.

    Returns:
        int: 0.

    """
    filetypeout = '.pdf'
    #timestamp = f"{round(time.time())}"
    
    if(os.path.exists(dir_save) == False):
        print("Error: Output directory does not exist")
        return
        
    try:
        #Training curve
        trainlossall_energy = hist.history['mean_absolute_error']
        testlossall_energy = hist.history['val_mean_absolute_error']
        outname = os.path.join(dir_save,"fit"+str(i)+"_loss"+filetypeout)
        plt.figure()
        plt.plot(np.arange(1,len(trainlossall_energy)+1),trainlossall_energy,label='Training energy',color='c')
        plt.plot(np.array(range(1,len(testlossall_energy)+1))*epostep,testlossall_energy,label='Test energy',color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Mean absolute Error ' +"["+ unit_energy+ "]")
        plt.title("Mean absolute Error vs. epochs")
        plt.legend(loc='upper right',fontsize='x-large')
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot loss curve")
    
    try:
        #Predicted vs Actual    
        plt.figure()
        outname = os.path.join(dir_save,"fit"+str(i)+"_energy_predict"+filetypeout)
        preds = predval.flatten()
        engval = yval.flatten()
        engval_min = np.amin(engval)
        engval_max = np.amax(engval)
        plt.plot(np.arange(engval_min, engval_max,np.abs(engval_min-engval_max)/100), np.arange(engval_min, engval_max,np.abs(engval_min-engval_max)/100), color='red')
        plt.scatter(preds, engval, alpha=0.3)        
        plt.xlabel('Predicted '+"["+unit_energy+"]")
        plt.ylabel('Actual ' +"["+unit_energy+"]")
        plt.title("Prediction engery")
        plt.text(engval_min,engval_max,"MAE: {0:0.6f} ".format(np.mean(np.abs(preds-engval))) +"["+unit_energy+"]")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot energy prediction") 
    
    try:
        #Learing rate
        learningall = hist.history['lr']
        plt.figure()
        outname = os.path.join(dir_save,"fit"+str(i)+"_lr"+filetypeout)
        plt.plot(np.arange(1,len(learningall)+1),learningall,label='Learning rate',color='r')
        plt.xlabel("Epochs")
        plt.ylabel('Learning rate')
        plt.title("Learning rate decrease")                
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot learning rate") 
    
    return 0

