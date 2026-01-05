#####################################################
#
# PyRAI2MD 2 module for hyperparameter template
#
# Author Jingbai Li
# Sep 22 2021
#
######################################################

def set_hyper_eg(hyp, unit, info, splits):
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

def set_hyper_nac(hyp, unit, info, splits):
    """ Generating hyperparameter dict for nonadiabatic coupling NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits

        Return:              Type:
            hyp_dict         dict     hyperparameter dict for NN

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

def set_hyper_soc(hyp, unit, info, splits):
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
