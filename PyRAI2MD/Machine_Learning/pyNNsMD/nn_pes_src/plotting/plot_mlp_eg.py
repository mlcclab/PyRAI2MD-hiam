"""
Created on Sun Oct 11 11:00:01 2020

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



def plot_energy_gradient_fit_result(i,xval,xtrain,yval,ytrain,
                                    predval,predtrain,
                                    hist,
                                    epostep=1,
                                    dir_save=None,
                                    unit_energy="eV",unit_force="eV/A"):
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
        trainlossall_energy = hist.history['energy_mean_absolute_error']
        trainlossall_force = hist.history['force_mean_absolute_error']
        testlossall_energy = hist.history['val_energy_mean_absolute_error']
        testlossall_force = hist.history['val_force_mean_absolute_error']
        outname = os.path.join(dir_save,"fit"+str(i)+"_loss"+filetypeout)
        plt.figure()
        plt.plot(np.arange(1,len(trainlossall_energy)+1),trainlossall_energy,label='Training energy',color='c')
        plt.plot(np.arange(1,len(trainlossall_force)+1),trainlossall_force,label='Training gradients',color='m')
        plt.plot(np.array(range(1,len(testlossall_energy)+1))*epostep,testlossall_energy,label='Test energy',color='b')
        plt.plot(np.array(range(1,len(testlossall_force)+1))*epostep,testlossall_force,label='Test gradients',color='r')
        plt.xlabel('Epochs')
        plt.ylim(0,max(testlossall_energy[0],testlossall_force[0])*1.1)
        plt.ylabel('Scaled mean absolute error ' +"["+ unit_energy+ ","+ unit_force+"]")
        plt.title("Standardized MAE loss vs. epochs")
        plt.legend(loc='upper right',fontsize='x-large')
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot loss curve")
    
    try:
        #Forces mean
        plt.figure()
        outname = os.path.join(dir_save,"fit"+str(i)+"_grad_mean"+filetypeout)
        preds = np.mean(np.abs(predval[1]-yval[1]),axis=0).flatten()
        preds2 = np.mean(np.abs(predtrain[1]-ytrain[1]),axis=0).flatten()
        plt.plot(np.arange(len(preds)),preds,label='Validation gradients',color='r')
        plt.plot(np.arange(len(preds2)),preds2,label='Training gradients',color='m')
        plt.ylabel('Mean absolute error ' +"["+unit_force+"]")
        plt.legend(loc='upper right',fontsize='x-large')
        plt.xlabel('Gradients xyz * #atoms * #states ')
        plt.title("Gradient mean error")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot Gradients mean")    
    
    try:          
        #Gradients max
        outname = os.path.join(dir_save,"fit"+str(i)+"_grad_max"+filetypeout)
        pred_err, prelm = find_max_relative_error(predval[1],yval[1])
        pred_err2, prelm2 = find_max_relative_error(predtrain[1],ytrain[1])
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(np.arange(len(pred_err)),pred_err,label='Validation',color='c')
        ax1.plot(np.arange(len(pred_err2)),pred_err2,label='Training',color='m')
        plt.ylabel('Max absolute error ' +"["+unit_force+"]")
        plt.legend(loc='upper left')
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        ax2.plot(np.arange(len(prelm)),prelm,label='Rel. validation',color='b')
        ax2.plot(np.arange(len(prelm2)),prelm2,label='Rel. training',color='r')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.ylabel("Relative max error")    
        plt.legend(loc='upper right')
        plt.xlabel('Gradients xyz * #atoms * #states ')
        plt.title("Gradient max error")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot gradients max") 
    
    try:
        #Predicted vs Actual
        plt.figure()
        outname = os.path.join(dir_save,"fit"+str(i)+"_grad_predict"+filetypeout)
        preds = predval[1].flatten()
        gradval = yval[1].flatten()
        gradval_max = np.amax(gradval)
        gradval_min = np.amin(gradval)
        plt.plot(np.arange(gradval_min,gradval_max,np.abs(gradval_min-gradval_max)/100),np.arange(gradval_min,gradval_max,np.abs(gradval_min-gradval_max)/100), color='red')
        plt.scatter(preds, gradval, alpha=0.3)        
        plt.xlabel('Predicted '+"["+unit_force+"]")
        plt.ylabel('Actual ' +"["+unit_force+"]")
        plt.title("Prediction gradient components")
        plt.text(gradval_min,gradval_max,"MAE: {0:0.6f} ".format(np.mean(np.abs(preds-gradval))) +"["+unit_force+"]")
        plt.savefig(outname)
        plt.close()
    except:
        print("Error: Could not plot gradient prediction") 
    
    try:
        #Predicted vs Actual    
        plt.figure()
        outname = os.path.join(dir_save,"fit"+str(i)+"_energy_predict"+filetypeout)
        preds = predval[0].flatten()
        engval = yval[0].flatten()
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
        learningall = hist.history['energy_lr']
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


def plot_resampling_gradient(dir_save,
                        out_index,
                        pool_error,
                        fit_error,
                        test_error,
                        prename='resample', 
                        unit_energy_conv = 1,
                        unit_force_conv = 1,
                        unit_energy="eV",
                        unit_force="eV/A"):
    """
    Plot the resampling statistics for energy gradient.

    Args:
        dir_save (str): Directory to save data. The default is None.
        out_index (list): List of indexarrays from data.
        pool_error (np.array): List of error on the remaining data.
        fit_error (np.array): List of error of validation set.
        test_error (np.array): List of error for test set.
        prename (str, optional): Prefix for fit. The default is 'fit'.
        unit_energy (str, optional):  Unit of energy. The default is 'eV'.
        unit_force (str, optional): Unit of force. The default is 'eV/A'.

    Returns:
        None.

    """
    filetypeout = ".png"
    #timestamp = f"{round(time.time())}"
    
    if(os.path.exists(dir_save) == False):
        print("Error: Output directory does not exist")
        return

    datalen = np.array([len(x) for x in out_index])

    
    #Active learning plot        
    outname = os.path.join(dir_save,prename+"_energy"+filetypeout)
    plt.figure()
    plt.semilogx(datalen, pool_error[:,0],label='Unknown',color='r')
    plt.semilogx(datalen, fit_error[:,0,0],label='Valid',color='b')
    plt.semilogx(datalen, test_error[:,0],label='Test',color='g')
    plt.xlabel("Log Dataset Size")
    plt.ylabel('Mean absolute error ' +"["+ unit_energy+"]")
    plt.title("Gradient data vs. error")      
    plt.legend(loc='upper right')         
    plt.savefig(outname)
    plt.close()
    
    #Active learning plot        
    outname = os.path.join(dir_save,prename+"_gradient"+filetypeout)
    plt.figure()
    plt.semilogx(datalen, pool_error[:,1],label='Unknown',color='r')
    plt.semilogx(datalen, fit_error[:,0,1],label='Valid',color='b')
    plt.semilogx(datalen, test_error[:,1],label='Test',color='g')
    plt.xlabel("Log Dataset Size")
    plt.ylabel('Mean absolute error ' +"["+unit_force+"]")
    plt.title("Energy data vs. error")      
    plt.legend(loc='upper right')         
    plt.savefig(outname)
    plt.close()