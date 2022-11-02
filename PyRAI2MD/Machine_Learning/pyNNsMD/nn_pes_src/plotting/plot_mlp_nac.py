# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:00:22 2020

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



def plot_nac_fit_result(i,xval,xtrain,yval,ytrain,
                        predval,predtrain,
                        hist,epostep=1,
                        dir_save=None,
                        prename='fit', unit_nac = '1/A'):
    """
    Plot and store fit.

    Args:
        i (int): index of the nerual network.
        xval (np.array): Validation Data.
        xtrain (np.array): Training Data.
        yval (np.array): True Validation data. nac
        ytrain (np.array): True Training data. nac
        predval (np.array): Model prediction for validation.
        predtrain (np.array): Model prediction for training.
        hist (dict): histogram from keras.
        epostep (int, optional): Step size of evals. The default is 1.
        dir_save (str, optional): Path of dictionary to store plots. The default is None.
        prename (str, optional): Start of naming plots. The default is 'fit'.
        unit_nac (str, optional): Unit of NACs. The default is '1/A'.

    Returns:
        int: 0.

    """
    filetypeout = ".pdf"
    #timestamp = f"{round(time.time())}"
    
    if(os.path.exists(dir_save) == False):
        print("Error: Output directory does not exist")
        return
            
    try:
        trainlossall_nac = hist.history['mean_absolute_error']
        testlossall_nac = hist.history['val_mean_absolute_error']
        outname = os.path.join(dir_save,prename+str(i)+"_nac_loss"+".pdf")
        plt.figure()
        plt.plot(np.arange(1,len(trainlossall_nac)+1),trainlossall_nac,label='Training NAC',color='c')
        plt.plot(np.array(range(0,len(testlossall_nac)))*epostep+epostep,testlossall_nac,label='Test NAC',color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Scaled mean absolute error ' +"["+unit_nac+"]")
        plt.ylim(0,testlossall_nac[0]*1.1)
        plt.title("Standardized")
        plt.legend(loc='upper right',fontsize='x-large')
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot loss curve")       
        
    try:
        #nacs
        outname = os.path.join(dir_save,prename+str(i)+"_nac_mean"+".pdf")
        plt.figure()
        preds = np.mean(np.abs(predval-yval),axis=0).flatten()
        preds2 = np.mean(np.abs(predtrain-ytrain),axis=0).flatten()
        plt.plot(np.arange(len(preds)),preds,label='Validation NAC',color='r')
        plt.plot(np.arange(len(preds2)),preds2,label='Training NAC',color='m')
        plt.ylabel('Mean absolute error ' +"["+unit_nac+"]")
        plt.legend(loc='upper right',fontsize='x-large')
        plt.xlabel('NACs xyz * #atoms * #states ')
        plt.title("NAC mean error")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot mean error") 
    
    try:
        #nacs max
        outname = os.path.join(dir_save,prename+str(i)+"_nac_max"+".pdf")
        pred_err, prelm = find_max_relative_error(predval,yval)
        pred_err2, prelm2 = find_max_relative_error(predtrain,ytrain)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(np.arange(len(pred_err)),pred_err,label='Validation',color='c')
        ax1.plot(np.arange(len(pred_err2)),pred_err2,label='Training',color='m')
        plt.ylabel('Max absolute error ' +"["+unit_nac+"]")
        plt.legend(loc='upper left')
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(np.arange(len(prelm)),prelm,label='Rel. Validation',color='b')
        ax2.plot(np.arange(len(prelm2)),prelm2,label='Rel. Training',color='r')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("Relative Max Error")    
        plt.legend(loc='upper right')
        plt.xlabel('NACs xyz * #atoms * #states ')
        plt.title("NAC max error")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot max error") 
    
    try:
        #Predicted vs Actual
        outname = os.path.join(dir_save,prename+str(i)+"_nac_predict"+filetypeout)
        plt.figure()
        preds = predval.flatten()
        nacval = yval.flatten()
        nacval_max = np.amax(nacval)
        nacval_min = np.amin(nacval)
        plt.plot(np.arange(nacval_min,nacval_max,np.abs(nacval_max-nacval_min)/100), np.arange(nacval_min,nacval_max,np.abs(nacval_max-nacval_min)/100), color='red')
        plt.scatter(preds, nacval, alpha=0.3)        
        plt.xlabel('Predicted '+"["+unit_nac+"]")
        plt.ylabel('Actual ' +"["+unit_nac+"]")
        plt.title("Prediction NAC components")
        plt.text(nacval_min,nacval_max,"MAE: {0:0.6f} ".format(np.mean(np.abs(preds-nacval))) +"["+unit_nac+"]")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot prediction")    
    
    try:
        #Learing rate
        learningall = hist.history['lr']        
        outname = os.path.join(dir_save,prename+str(i)+"_lr"+".pdf")
        plt.figure()
        plt.plot(np.arange(1,len(learningall)+1),learningall,label='Learning rate',color='r')
        plt.xlabel("Epochs")
        plt.ylabel('Learning rate')
        plt.title("Learning rate decrease")                
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot learning rate") 
                
    return 0


def plot_resampling_nac(dir_save,
                        out_index,
                        pool_error,
                        fit_error,
                        test_error,
                        prename='resample', 
                        unit_nac_conv = 1,
                        unit_nac = '1/A'):
    """
    Plot the resampling statistics for energy gradient.

    Args:
        dir_save (str): Directory to save data. The default is None.
        out_index (list): List of indexarrays from data.
        pool_error (np.array): List of error on the remaining data.
        fit_error (np.array): List of error of validation set.
        test_error (np.array): List of error for test set.
        prename (str, optional): Prefix for fit. The default is 'fit'.
        unit_nac (str, optional): Unit of NAC. The default is '1/A'.

    Returns:
        None.

    """
    filetypeout = ".png"
    #timestamp = f"{round(time.time())}"
    
    if(os.path.exists(dir_save) == False):
        print("Error: Output directory does not exist")
        return

    datalen = np.array([len(x) for x in out_index])
    pool_error = np.array(pool_error)
    fit_error = np.array(fit_error)
    test_error = np.array(test_error)
    
    #Active learning plot                
    outname = os.path.join(dir_save,prename+"_nac"+filetypeout)
    plt.figure()
    plt.semilogx(datalen, pool_error,label='Unknown',color='r')
    plt.semilogx(datalen, fit_error[:,0],label='Valid',color='b')
    plt.semilogx(datalen, test_error,label='Test',color='g')
    plt.xlabel("Dataset Size")
    plt.ylabel('Mean absolute error ' +"["+ unit_nac+"]")
    plt.title("NAC data vs. error")      
    plt.legend(loc='upper right')         
    plt.savefig(outname)
    plt.close()
