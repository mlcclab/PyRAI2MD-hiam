#####################################################
#
# PyRAI2MD 2 module for hyperparameter template
#
# Author Jingbai Li
# Apr 20 2023
#
######################################################

def set_hyper_eg(hyp):
    """ Generating hyperparameter dict for energy+gradient NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    """
    Note:
    info['natom'] gives the number of atoms
    info['nstate'] gives the number of state
    """

    ## setup hypers dict
    hyp_dict = {
        # assign input hyperparameters in the format of nn package
    }

    return hyp_dict

def set_hyper_nac(model_path, hyp, shuffle, gpu):
    """ Generating hyperparameter dict for nonadiabatic coupling NN

        Parameters:          Type:
            model_path       str      a path to save the model
            hyp              dict     hyperparameter input
            shuffle          bool     shuffle data loader during training
            gpu              int      use gpu for training when the device is available

        Return:              Type:
            hyp_dict         dict     hyperparameter dict for NN

    """

    ## setup hypers dict
    if hyp['model_type'] == 'pp':
        model_type = 'Dimenet++'
    else:
        model_type = 'Dimenet'

    hyp_dict = {
        'model_type': model_type,
        'optimizer': 'Adam',
        'batch_size': hyp['batch_size'],
        'val_size': hyp['val_size'],
        'criterion': 'MAE',
        'model_param': {
            'hidden_channels': hyp['hidden_channels'],
            'out_channels': hyp['out_channels'],
            'num_blocks': hyp['blocks'],
            'num_bilinear': hyp['bilinear'],
            'num_spherical': hyp['spherical'],
            'num_radial': hyp['radial']
        },
        'lr_start': hyp['lr'],
        'nepochs': hyp['epo'],
        'model_path': '%s/nac' % model_path,
        'gpu': gpu,
        'shuffle': shuffle
    }

    return hyp_dict

def set_hyper_soc(hyp):
    """ Generating hyperparameter dict for spin-orbit coupling  NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    """
    Note:
    info['natom'] gives the number of atoms
    info['nstate'] gives the number of state
    """

    ## setup hypers dict
    hyp_dict = {
        # assign input hyperparameters in the format of nn package
    }

    return hyp_dict
