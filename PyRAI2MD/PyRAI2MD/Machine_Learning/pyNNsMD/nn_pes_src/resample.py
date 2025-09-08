"""
Methods for active sampling. This was just a test and will not work smoothly at the moment.

This can be improved on how to pick new points.
"""
import numpy as np


def find_samples_with_max_error(y,y_pred):
    """
    Find the sample indices with the combined maximum deviation for all model in y.
    
    y_pred may have more models. Each np.array in y and y_pred must have same size.

    Args:
        y (dict): Dictionary of y values by model key.
        y_pred (dict): Dictionary of predicted values by model key.

    Returns:
        out (np.array): Sorted index array.
        err_total_dict (dict): Dictionary of mean absolute error that matches structure of y.

    """
    #Disassable y_dict
    ylist = []
    yplist = []
    keylist = []
    for key in y.keys():
        if(isinstance(y[key],list)):
            ytemp = y[key]
            yptemp = y_pred[key]
            for x in ytemp:
                ylist.append(x)
                keylist.append(key)
            for x in yptemp:
                yplist.append(x)
        else:
            keylist.append(key)
            ylist.append(y[key])
            yplist.append(y_pred[key])
    
    #Find Max Error
    errout = []
    err_total = []
    for i in range(len(ylist)):
        yplist[i] = np.abs(yplist[i] - ylist[i])
        err_total.append(np.mean(yplist[i]))
        tempmean = [np.mean(yplist[i][j]) for j in range(len(yplist[i]))]
        tempmean = np.array(tempmean)
        tempmean /= max(np.max(tempmean),1e-8)
        errout.append(tempmean)
    
    errout = np.array(errout)   
    errout = np.sum(errout,axis=0)
    out = np.flip(np.argsort(errout))
    
    #Reassamble y_dict for error_total
    err_total_dict = {}
    for key,err in zip(keylist,err_total):
        if(key not in err_total_dict):
            err_total_dict.update({key:err})
        else:
            if(isinstance(err_total_dict[key],list) == False):
                err_total_dict[key] = [err_total_dict[key]]
            err_total_dict[key].append(err)
            
    
    return out,err_total_dict


def index_data_in_y_dict(y,ind):
    """
    Index np.arrays as array[index] in the nested y_dict used in pes.

    Args:
        y (dict): Dcitionary of y-values as y={'energy_gradients' : [np.array,np.array], 'NAC' : np.array}.
        ind (np.array): Index array.

    Returns:
        y_out (dict): Same y_dict with its data as data[index].

    """
    y_out = {}
    for key, value in y.items():
        if(isinstance(value,list)):
            y_out[key] = [x[ind] for x in value]
        else:
            y_out[key] = value[ind]
    return y_out
    