######################################################
#
# PyRAI2MD 2 module for utility tools - permutation
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import numpy as np

def permute_map(x, y_dict, pmap, val_split):
    ## This function permute data following the map P.
    ## x is M x N x 3, M entries, N atoms, x,y,z
    ## y_dict has possible three keys 'energy_gradient', 'nac' and 'soc'
    ## energy is M x n, M batches, n states
    ## gradient is M x n x N x 3, M batches, n states, N atoms, x, y, z
    ## nac is M x m x N x 3, M batches, m state pairs, N atoms, x, y, z
    ## soc is M x l, M batches, l state pairs
    ## permute_map is a file including all permutation

    # early stop the function
    if pmap == 'No':
        return x, y_dict

    if permute_map != 'No' and os.path.exists(pmap) is False:
        return x, y_dict

    # load permutation map
    p = np.loadtxt(pmap) - 1
    p = p.astype(int)
    if len(p.shape) == 1:
        p = p.reshape((1, -1))

    x_new = np.zeros([0, x.shape[1], x.shape[2]])  # initialize coordinates list
    y_dict_new = {}

    # pick energy and gradient, note permutation does not change energy
    if 'energy_gradient' in y_dict.keys():  # check energy gradient
        energy = y_dict['energy_gradient'][0]
        grad = y_dict['energy_gradient'][1]

        # initialize energy and gradient list
        y_dict_new['energy_gradient'] = [np.zeros([0, energy.shape[1]]),
                                         np.zeros([0, grad.shape[1], grad.shape[2], grad.shape[3]])]
        per_eg = 1
    else:
        energy = []
        grad = []
        per_eg = 0

    # pick nac
    if 'nac' in y_dict.keys():
        nac = y_dict['nac']

        # initialize nac list
        y_dict_new['nac'] = np.zeros([0, nac.shape[1], nac.shape[2], nac.shape[3]])
        per_nac = 1
    else:
        nac = []
        per_nac = 0

    # pick soc, permutation does not change soc
    if 'soc' in y_dict.keys():
        soc = y_dict['soc']

        # initialize soc list
        y_dict_new['soc'] = np.zeros([0, soc.shape[1]])
        per_soc = 1
    else:
        soc = []
        per_soc = 0

    kfold = np.ceil(1 / val_split).astype(int)
    portion = int(len(x) * val_split)

    ## determine the range of k-fold
    kfoldrange = []
    for k in range(kfold):
        if k < kfold - 1:
            kfoldrange.append([k * portion, (k + 1) * portion])
        else:
            kfoldrange.append([k * portion, len(x)])

    ## permute data per k-fold
    for k in kfoldrange:
        # separate data in kfold
        a, b = k
        kx = x[a: b]
        new_x = kx
        if per_eg == 1:
            kenergy = energy[a: b]
            kgrad = grad[a: b]
            new_e = kenergy
            new_g = kgrad
        else:
            kenergy = []
            kgrad = []
            new_e = []
            new_g = []

        if per_nac == 1:
            knac = nac[a: b]
            new_n = knac
        else:
            knac = []
            new_n = []

        if per_soc == 1:
            ksoc = soc[a: b]
            new_s = ksoc
        else:
            ksoc = []
            new_s = []

        for index in p:
            # permute coord along N atoms
            per_x = kx[:, index, :]
            new_x = np.concatenate((new_x, per_x), axis=0)
            if per_eg == 1:
                # permute grad along N atoms
                per_e = kenergy
                per_g = kgrad[:, :, index, :]
                new_e = np.concatenate((new_e, per_e), axis=0)
                new_g = np.concatenate((new_g, per_g), axis=0)
            if per_nac == 1:
                # permute nac along N atoms
                per_n = knac[:, :, index, :]
                new_n = np.concatenate((new_n, per_n), axis=0)
            if per_soc == 1:
                per_s = ksoc
                new_s = np.concatenate((new_s, per_s), axis=0)

        # merge the new data
        x_new = np.concatenate((x_new, new_x), axis=0)
        if per_eg == 1:
            y_dict_new['energy_gradient'][0] = np.concatenate((y_dict_new['energy_gradient'][0], new_e), axis=0)
            y_dict_new['energy_gradient'][1] = np.concatenate((y_dict_new['energy_gradient'][1], new_g), axis=0)
        if per_nac == 1:
            y_dict_new['nac'] = np.concatenate((y_dict_new['nac'], new_n), axis=0)
        if per_soc == 1:
            y_dict_new['soc'] = np.concatenate((y_dict_new['soc'], new_s), axis=0)

    return x_new, y_dict_new

def permute_map2(geos, energy, grad, nac, soc, pmap, splits):
    ## This function permute data following the map P.
    ## x is M x N x 3, M entries, N atoms, x,y,z
    ## y_dict has possible three keys 'energy_gradient', 'nac' and 'soc'
    ## energy is M x n, M batches, n states
    ## gradient is M x n x N x 3, M batches, n states, N atoms, x, y, z
    ## nac is M x m x N x 3, M batches, m state pairs, N atoms, x, y, z
    ## soc is M x l, M batches, l state pairs
    ## permute_map is a file including all permutation

    # early stop the function
    if pmap == 'No':
        return geos, energy, grad, nac, soc

    if permute_map != 'No' and os.path.exists(pmap) is False:
        return geos, energy, grad, nac, soc

    # load permutation map
    p = np.loadtxt(pmap) - 1
    p = p.astype(int)
    if len(p.shape) == 1:
        p = p.reshape((1, -1))

    geos_new = np.zeros([0, geos.shape[1], geos.shape[2]])  # initialize coordinates list

    # pick energy and gradient, note permutation does not change energy
    if len(energy) > 0:
        energy_new = np.zeros([0, energy.shape[1]])
        grad_new = np.zeros([0, grad.shape[1], grad.shape[2]. grad.shape[3]])
        per_eg = 1
    else:
        energy_new = np.zeros(0)
        grad_new = np.zeros(0)
        per_eg = 0

    # pick nac
    if len(nac) > 0:
        nac_new = np.zeros([0, nac.shape[1], nac.shape[2], nac.shape[3]])
        per_nac = 1
    else:
        nac_new = np.zeros(0)
        per_nac = 0

    # pick soc, permutation does not change soc
    if len(soc) > 0:
        soc_new = np.zeros([0, soc.shape[1]])
        per_soc = 1
    else:
        soc_new = np.zeros(0)
        per_soc = 0

    kfold = splits
    portion = int(len(geos) / splits)

    ## determine the range of k-fold
    kfoldrange = []
    for k in range(kfold):
        if k < kfold - 1:
            kfoldrange.append([k * portion, (k + 1) * portion])
        else:
            kfoldrange.append([k * portion, len(geos)])

    ## permute data per k-fold
    for k in kfoldrange:
        # separate data in kfold
        a, b = k
        kgeos = geos[a: b]
        new_geos = kgeos
        if per_eg == 1:
            kenergy = energy[a: b]
            kgrad = grad[a: b]
            new_e = kenergy
            new_g = kgrad
        else:
            kenergy = []
            kgrad = []
            new_e = []
            new_g = []

        if per_nac == 1:
            knac = nac[a: b]
            new_n = knac
        else:
            knac = []
            new_n = []

        if per_soc == 1:
            ksoc = soc[a: b]
            new_s = ksoc
        else:
            ksoc = []
            new_s = []

        for index in p:
            # permute coord along N atoms
            per_geos = kgeos[:, index, :]
            new_geos = np.concatenate((new_geos, per_geos), axis=0)
            if per_eg == 1:
                # permute grad along N atoms
                per_e = kenergy
                per_g = kgrad[:, :, index, :]
                new_e = np.concatenate((new_e, per_e), axis=0)
                new_g = np.concatenate((new_g, per_g), axis=0)
            if per_nac == 1:
                # permute nac along N atoms
                per_n = knac[:, :, index, :]
                new_n = np.concatenate((new_n, per_n), axis=0)
            if per_soc == 1:
                per_s = ksoc
                new_s = np.concatenate((new_s, per_s), axis=0)

        # merge the new data
        geos_new = np.concatenate((geos_new, new_geos), axis=0)
        if per_eg == 1:
            energy_new = np.concatenate((energy_new, new_e), axis=0)
            grad_new = np.concatenate((grad_new, new_g), axis=0)
        if per_nac == 1:
            nac_new = np.concatenate((nac_new, new_n), axis=0)
        if per_soc == 1:
            soc_new = np.concatenate((soc_new, new_s), axis=0)

    return geos_new, energy_new, grad_new, nac_new, soc_new
