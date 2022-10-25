######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 29 2021
#
######################################################

import os
import sys
from PyRAI2MD.Utils.read_tools import ReadVal
from PyRAI2MD.Utils.read_tools import ReadIndex

def read_control(keywords, values):
    ## This function read variables from &control
    keyfunc = {
        'title': ReadVal('s'),
        'ml_ncpu': ReadVal('i'),
        'qc_ncpu': ReadVal('i'),
        'ms_ncpu': ReadVal('i'),
        'gl_seed': ReadVal('i'),
        'jobtype': ReadVal('s'),
        'qm': ReadVal('sl'),
        'abinit': ReadVal('sl'),
        'refine': ReadVal('i'),
        'refine_num': ReadVal('i'),
        'refine_start': ReadVal('i'),
        'refine_end': ReadVal('i'),
        'maxiter': ReadVal('i'),
        'maxsample': ReadVal('i'),
        'dynsample': ReadVal('i'),
        'maxdiscard': ReadVal('i'),
        'maxenergy': ReadVal('f'),
        'minenergy': ReadVal('f'),
        'dynenergy': ReadVal('f'),
        'inienergy': ReadVal('f'),
        'fwdenergy': ReadVal('i'),
        'bckenergy': ReadVal('i'),
        'maxgrad': ReadVal('f'),
        'mingrad': ReadVal('f'),
        'dyngrad': ReadVal('f'),
        'inigrad': ReadVal('f'),
        'fwdgrad': ReadVal('i'),
        'bckgrad': ReadVal('i'),
        'maxnac': ReadVal('f'),
        'minnac': ReadVal('f'),
        'dynnac': ReadVal('f'),
        'ininac': ReadVal('f'),
        'fwdnac': ReadVal('i'),
        'bcknac': ReadVal('i'),
        'maxsoc': ReadVal('f'),
        'minsoc': ReadVal('f'),
        'dynsoc': ReadVal('f'),
        'inisoc': ReadVal('f'),
        'fwdsoc': ReadVal('i'),
        'bcksoc': ReadVal('i'),
        'load': ReadVal('i'),
        'transfer': ReadVal('i'),
        'pop_step': ReadVal('i'),
        'verbose': ReadVal('i'),
        'silent': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in $control' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_molecule(keywords, values):
    ## This function read variables from &molecule
    keyfunc = {
        'ci': ReadVal('il'),
        'spin': ReadVal('il'),
        'coupling': ReadIndex('g'),
        'qmmm_key': ReadVal('s'),
        'qmmm_xyz': ReadVal('s'),
        'highlevel': ReadIndex('s', start=1),
        'embedding': ReadVal('i'),
        'boundary': ReadIndex('g'),
        'freeze': ReadIndex('s'),
        'constrain': ReadIndex('s'),
        'shape': ReadVal('s'),
        'factor': ReadVal('i'),
        'cavity': ReadVal('fl'),
        'center': ReadIndex('s'),
        'primitive': ReadIndex('g'),
        'lattice': ReadIndex('s'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molecule' % key)
        keywords[key] = keyfunc[key](val)

    return keywords


def read_molcas(keywords, values):
    ## This function read variables from &molcas
    keyfunc = {
        'molcas': ReadVal('s'),
        'molcas_nproc': ReadVal('s'),
        'molcas_mem': ReadVal('s'),
        'molcas_print': ReadVal('s'),
        'molcas_project': ReadVal('s'),
        'molcas_calcdir': ReadVal('s'),
        'molcas_workdir': ReadVal('s'),
        'track_phase': ReadVal('i'),
        'basis': ReadVal('i'),
        'omp_num_threads': ReadVal('s'),
        'use_hpc': ReadVal('i'),
        'keep_tmp': ReadVal('i'),
        'verbose': ReadVal('i'),
        'tinker': ReadVal('s'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molcas' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_bagel(keywords, values):
    ## This function read variables from &bagel
    keyfunc = {
        'bagel': ReadVal('s'),
        'bagel_nproc': ReadVal('s'),
        'bagel_project': ReadVal('s'),
        'bagel_workdir': ReadVal('s'),
        'bagel_archive': ReadVal('s'),
        'mpi': ReadVal('s'),
        'blas': ReadVal('s'),
        'lapack': ReadVal('s'),
        'boost': ReadVal('s'),
        'mkl': ReadVal('s'),
        'arch': ReadVal('s'),
        'omp_num_threads': ReadVal('s'),
        'use_mpi': ReadVal('i'),
        'use_hpc': ReadVal('i'),
        'keep_tmp': ReadVal('i'),
        'verbose': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &bagel' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_orca(keywords, values):
    ## This function read variables from &orca
    keyfunc = {
        'orca': ReadVal('s'),
        'orca_project': ReadVal('s'),
        'orca_workdir': ReadVal('s'),
        'dft_type': ReadVal('s'),
        'mpi': ReadVal('s'),
        'use_hpc': ReadVal('i'),
        'keep_tmp': ReadVal('i'),
        'verbose': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &orca' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_xtb(keywords, values):
    ## This function read variables from &xtb
    keyfunc = {
        'xtb': ReadVal('s'),
        'xtb_nproc': ReadVal('s'),
        'xtb_project': ReadVal('s'),
        'xtb_workdir': ReadVal('s'),
        'use_hpc': ReadVal('i'),
        'keep_tmp': ReadVal('i'),
        'verbose': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &xtb' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_md(keywords, values):
    ## This function read variables from &md
    keyfunc = {
        'initcond': ReadVal('i'),
        'excess': ReadVal('f'),
        'scale': ReadVal('f'),
        'target': ReadVal('f'),
        'graddesc': ReadVal('i'),
        'reset': ReadVal('i'),
        'resetstep': ReadVal('i'),
        'ninitcond': ReadVal('i'),
        'method': ReadVal('s'),
        'format': ReadVal('s'),
        'temp': ReadVal('f'),
        'step': ReadVal('i'),
        'size': ReadVal('f'),
        'root': ReadVal('i'),
        'activestate': ReadVal('i'),
        'sfhp': ReadVal('s'),
        'nactype': ReadVal('s'),
        'phasecheck': ReadVal('i'),
        'gap': ReadVal('f'),
        'gapsoc': ReadVal('f'),
        'substep': ReadVal('i'),
        'integrate': ReadVal('i'),
        'deco': ReadVal('s'),
        'adjust': ReadVal('i'),
        'reflect': ReadVal('i'),
        'maxh': ReadVal('i'),
        'dosoc': ReadVal('i'),
        'thermo': ReadVal('s'),
        'thermodelay': ReadVal('i'),
        'silent': ReadVal('i'),
        'verbose': ReadVal('i'),
        'direct': ReadVal('i'),
        'buffer': ReadVal('i'),
        'record': ReadVal('s'),
        'record_step': ReadVal('i'),
        'checkpoint': ReadVal('i'),
        'restart': ReadVal('i'),
        'addstep': ReadVal('i'),
        'ref_energy': ReadVal('i'),
        'ref_grad': ReadVal('i'),
        'ref_nac': ReadVal('i'),
        'ref_soc': ReadVal('i'),
        'datapath': ReadVal('s'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &md' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_nn(keywords, values):
    ## This function read variables from &nn
    keyfunc = {
        'train_mode': ReadVal('s'),
        'train_data': ReadVal('s'),
        'pred_data': ReadVal('s'),
        'modeldir': ReadVal('s'),
        'nsplits': ReadVal('i'),
        'nn_eg_type': ReadVal('i'),
        'nn_nac_type': ReadVal('i'),
        'nn_soc_type': ReadVal('i'),
        'multiscale': ReadIndex('g'),
        'shuffle': ReadVal('b'),
        'eg_unit': ReadVal('s'),
        'nac_unit': ReadVal('s'),
        'soc_unit': ReadVal('s'),
        'permute_map': ReadVal('s'),
        'gpu': ReadVal('i'),
        'silent': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &nn' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_grid_search(keywords, values):
    ## This function read variables form &search
    keyfunc = {
        'depth': ReadVal('il'),
        'nn_size': ReadVal('il'),
        'batch_size': ReadVal('il'),
        'reg_l1': ReadVal('fl'),
        'reg_l2': ReadVal('fl'),
        'dropout': ReadVal('fl'),
        'node_features': ReadVal('il'),
        'n_features': ReadVal('il'),
        'n_edges': ReadVal('il'),
        'n_filters': ReadVal('il'),
        'n_blocks': ReadVal('il'),
        'n_rbf': ReadVal('il'),
        'use_hpc': ReadVal('i'),
        'retrieve': ReadVal('i'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &search' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_mlp(keywords, values):
    ## This function read variables from &eg1,&eg2,&nac1,&nac2,&soc,&soc2
    keyfunc = {
        'invd_index': ReadIndex('g'),
        'angle_index': ReadIndex('g'),
        'dihed_index': ReadIndex('g'),
        'depth': ReadVal('i'),
        'nn_size': ReadVal('i'),
        'activ': ReadVal('s'),
        'activ_alpha': ReadVal('f'),
        'loss_weights': ReadVal('fl'),
        'use_dropout': ReadVal('b'),
        'dropout': ReadVal('f'),
        'use_reg_activ': ReadVal('s'),
        'use_reg_weight': ReadVal('s'),
        'use_reg_bias': ReadVal('s'),
        'reg_l1': ReadVal('f'),
        'reg_l2': ReadVal('f'),
        'use_step_callback': ReadVal('b'),
        'use_linear_callback': ReadVal('b'),
        'use_early_callback': ReadVal('b'),
        'use_exp_callback': ReadVal('b'),
        'scale_x_mean': ReadVal('b'),
        'scale_x_std': ReadVal('b'),
        'scale_y_mean': ReadVal('b'),
        'scale_y_std': ReadVal('b'),
        'normalization_mode': ReadVal('i'),
        'learning_rate': ReadVal('f'),
        'phase_less_loss': ReadVal('b'),
        'initialize_weights': ReadVal('b'),
        'val_disjoint': ReadVal('b'),
        'epo': ReadVal('i'),
        'epomin': ReadVal('i'),
        'pre_epo': ReadVal('i'),
        'patience': ReadVal('i'),
        'max_time': ReadVal('i'),
        'batch_size': ReadVal('i'),
        'delta_loss': ReadVal('f'),
        'loss_monitor': ReadVal('s'),
        'factor_lr': ReadVal('f'),
        'epostep': ReadVal('i'),
        'learning_rate_start': ReadVal('f'),
        'learning_rate_stop': ReadVal('f'),
        'learning_rate_step': ReadVal('fl'),
        'epoch_step_reduction': ReadVal('il'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &eg/&nac/&soc' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_sch(keywords, values):
    ## This function read variables from &sch_eg,&sch_nac,&sch_soc
    keyfunc = {
        'node_features': ReadVal('i'),
        'n_features': ReadVal('i'),
        'n_edges': ReadVal('i'),
        'n_filters': ReadVal('i'),
        'use_filter_bias': ReadVal('b'),
        'cfc_activ': ReadVal('s'),
        'n_blocks': ReadVal('i'),
        'n_rbf': ReadVal('i'),
        'maxradius': ReadVal('i'),
        'offset': ReadVal('f'),
        'sigma': ReadVal('f'),
        'mlp': ReadVal('il'),
        'use_mlp_bias': ReadVal('b'),
        'mlp_activ': ReadVal('s'),
        'use_output_bias': ReadVal('b'),
        'use_step_callback': ReadVal('b'),
        'use_linear_callback': ReadVal('b'),
        'use_early_callback': ReadVal('b'),
        'use_exp_callback': ReadVal('b'),
        'loss_weights': ReadVal('fl'),
        'phase_less_loss': ReadVal('b'),
        'initialize_weights': ReadVal('b'),
        'epo': ReadVal('i'),
        'epomin': ReadVal('i'),
        'epostep': ReadVal('i'),
        'pre_epo': ReadVal('i'),
        'patience': ReadVal('i'),
        'max_time': ReadVal('i'),
        'batch_size': ReadVal('i'),
        'delta_loss': ReadVal('f'),
        'loss_monitor': ReadVal('s'),
        'factor_lr': ReadVal('f'),
        'learning_rate': ReadVal('f'),
        'learning_rate_start': ReadVal('f'),
        'learning_rate_stop': ReadVal('f'),
        'learning_rate_step': ReadVal('fl'),
        'epoch_step_reduction': ReadVal('il'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &sch_eg/&sch_nac/&sch_soc' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_e2n2(keywords, values):
    ## This function read variables from &e2n2_eg,&e2n2_nac,&e2n2_soc
    keyfunc = {
        'n_edges': ReadVal('i'),
        'maxradius': ReadVal('i'),
        'n_features': ReadVal('i'),
        'n_blocks': ReadVal('i'),
        'l_max': ReadVal('i'),
        'parity': ReadVal('b'),
        'n_rbf': ReadVal('i'),
        'trainable_rbf': ReadVal('b'),
        'rbf_cutoff': ReadVal('i'),
        'rbf_layers': ReadVal('i'),
        'rbf_neurons': ReadVal('i'),
        'rbf_act': ReadVal('s'),
        'rbf_act_a': ReadVal('f'),
        'normalization_y': ReadVal('s'),
        'normalize_y': ReadVal('b'),
        'resnet': ReadVal('b'),
        'gate': ReadVal('b'),
        'act_scalars_e': ReadVal('s'),
        'act_scalars_o': ReadVal('s'),
        'act_gates_e': ReadVal('s'),
        'act_gates_o': ReadVal('s'),
        'use_step_callback': ReadVal('b'),
        'initialize_weights': ReadVal('b'),
        'loss_weights': ReadVal('fl'),
        'use_reg_loss': ReadVal('s'),
        'reg_l1': ReadVal('f'),
        'reg_l2': ReadVal('f'),
        'epo': ReadVal('i'),
        'epostep': ReadVal('i'),
        'subset': ReadVal('f'),
        'batch_size': ReadVal('i'),
        'learning_rate': ReadVal('f'),
        'learning_rate_step': ReadVal('fl'),
        'epoch_step_reduction': ReadVal('il'),
        'scaler': ReadVal('s'),
        'grad_type': ReadVal('s'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &e2n2_eg/e2n2_&nac/&e2n2_soc' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_file(keywords, values):
    ## This function read variables form &file
    keyfunc = {
        'natom': ReadVal('i'),
        'file': ReadVal('s'),
    }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &file' % key)
        keywords[key] = keyfunc[key](val)

    return keywords

def read_input(ld_input):
    ## This function store all default values for variables
    ## This function read variable from input
    ## This function is expected to be expanded in future as more methods added

    ## default values
    variables_control = {
        'title': None,
        'ml_ncpu': 1,
        'qc_ncpu': 1,
        'ms_ncpu': 1,
        'gl_seed': 1,
        'jobtype': 'sp',
        'qm': 'nn',
        'abinit': ['molcas'],
        'refine': 0,
        'refine_num': 4,
        'refine_start': 0,
        'refine_end': 200,
        'maxiter': 1,
        'maxsample': 1,
        'dynsample': 0,
        'maxdiscard': 0,
        'maxenergy': 0.05,
        'minenergy': 0.02,
        'dynenergy': 0.1,
        'inienergy': 0.3,
        'fwdenergy': 1,
        'bckenergy': 1,
        'maxgrad': 0.15,
        'mingrad': 0.06,
        'dyngrad': 0.1,
        'inigrad': 0.3,
        'fwdgrad': 1,
        'bckgrad': 1,
        'maxnac': 0.15,
        'minnac': 0.06,
        'dynnac': 0.1,
        'ininac': 0.3,
        'fwdnac': 1,
        'bcknac': 1,
        'maxsoc': 50,
        'minsoc': 20,
        'dynsoc': 0.1,
        'inisoc': 0.3,
        'fwdsoc': 1,
        'bcksoc': 1,
        'load': 1,
        'transfer': 0,
        'pop_step': 200,
        'verbose': 2,
        'silent': 1,
    }

    variables_molecule = {
        'qmmm_key': None,
        'qmmm_xyz': 'Input',
        'ci': [1],
        'spin': [0],
        'coupling': [],
        'highlevel': [],
        'embedding': 1,
        'boundary': [],
        'freeze': [],
        'constrain': [],
        'shape': 'ellipsoid',
        'factor': 10,
        'cavity': [],
        'center': [],
        'primitive': [],
        'lattice': [],
    }

    variables_molcas = {
        'molcas': '',
        'molcas_nproc': '1',
        'molcas_mem': '2000',
        'molcas_print': '2',
        'molcas_project': None,
        'molcas_calcdir': os.getcwd(),
        'molcas_workdir': None,
        'track_phase': 0,
        'basis': 2,
        'omp_num_threads': '1',
        'use_hpc': 0,
        'group': None,  # Caution! Not allow user to set.
        'keep_tmp': 1,
        'verbose': 0,
        'tinker': '',
    }

    variables_bagel = {
        'bagel': '',
        'bagel_nproc': 1,
        'bagel_project': None,
        'bagel_workdir': os.getcwd(),
        'bagel_archive': 'default',
        'mpi': '',
        'blas': '',
        'lapack': '',
        'boost': '',
        'mkl': '',
        'arch': '',
        'omp_num_threads': '1',
        'use_mpi': 0,
        'use_hpc': 0,
        'group': None,  # Caution! Not allow user to set.
        'keep_tmp': 1,
        'verbose': 0,
    }

    variables_orca = {
        'orca': '',
        'orca_project': None,
        'orca_workdir': os.getcwd(),
        'dft_type': 'tddft',
        'mpi': '',
        'use_hpc': 0,
        'keep_tmp': 1,
        'verbose': 0,
    }

    variables_xtb = {
        'xtb': '',
        'xtb_nproc': 1,
        'xtb_project': None,
        'xtb_workdir': os.getcwd(),
        'use_hpc': 0,
        'keep_tmp': 1,
        'verbose': 0,
    }

    variables_md = {
        'gl_seed': 1,  # Caution! Not allow user to set.
        'initcond': 0,
        'excess': 0,
        'scale': 1,
        'target': 0,
        'graddesc': 0,
        'reset': 0,
        'resetstep': 0,
        'ninitcond': 20,
        'method': 'wigner',
        'format': 'molden',
        'temp': 300,
        'step': 10,
        'size': 20.67,
        'root': 1,
        'activestate': 0,
        'sfhp': 'nosh',
        'nactype': 'ktdc',
        'phasecheck': 0,
        'gap': 0.5,
        'gapsoc': 0.5,
        'substep': 20,
        'integrate': 0,
        'deco': '0.1',
        'adjust': 1,
        'reflect': 1,
        'maxh': 10,
        'dosoc': 0,
        'thermo': 'off',
        'thermodelay': 200,
        'silent': 1,
        'verbose': 0,
        'direct': 2000,
        'buffer': 500,
        'record': 'whole',
        'record_step': 0,
        'checkpoint': 0,
        'restart': 0,
        'addstep': 0,
        'group': None,  # Caution! Not allow user to set.
        'ref_energy': 0,
        'ref_grad': 0,
        'ref_nac': 0,
        'ref_soc': 0,
        'datapath': None,
    }

    variables_nn = {
        'train_mode': 'training',
        'train_data': None,
        'pred_data': None,
        'modeldir': None,
        'silent': 1,
        'nsplits': 10,
        'nn_eg_type': 1,
        'nn_nac_type': 0,
        'nn_soc_type': 0,
        'multiscale': [],
        'shuffle': False,
        'eg_unit': 'si',
        'nac_unit': 'si',
        'soc_unit': 'si',
        'ml_seed': 1,  # Caution! Not allow user to set.
        'data': None,  # Caution! Not allow user to set.
        'search': None,  # Caution! Not allow user to set.
        'eg': None,  # Caution! This value will be updated later. Not allow user to set.
        'nac': None,  # Caution! This value will be updated later. Not allow user to set.
        'eg2': None,  # Caution! This value will be updated later. Not allow user to set.
        'nac2': None,  # Caution! This value will be updated later. Not allow user to set.
        'soc': None,  # Caution! This value will be updated later. Not allow user to set.
        'soc2': None,  # Caution! This value will be updated later. Not allow user to set.
        'permute_map': 'No',
        'gpu': 0,
    }

    variables_search = {
        'depth': [],
        'nn_size': [],
        'batch_size': [],
        'reg_l1': [],
        'reg_l2': [],
        'dropout': [],
        'node_features': [],
        'n_features': [],
        'n_edges': [],
        'n_filters': [],
        'n_blocks': [],
        'n_rbf': [],
        'use_hpc': 0,
        'retrieve': 0,
    }

    variables_eg = {
        'invd_index': [],
        'angle_index': [],
        'dihed_index': [],
        'depth': 4,
        'nn_size': 100,
        'activ': 'leaky_softplus',
        'activ_alpha': 0.03,
        'loss_weights': [1, 1],
        'use_dropout': False,
        'dropout': 0.005,
        'use_reg_activ': None,
        'use_reg_weight': None,
        'use_reg_bias': None,
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'scale_x_mean': False,
        'scale_x_std': False,
        'scale_y_mean': True,
        'scale_y_std': True,
        'normalization_mode': 1,
        'learning_rate': 1e-3,
        'initialize_weights': True,
        'val_disjoint': True,
        'epo': 2000,
        'epomin': 1000,
        'patience': 300,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'epostep': 10,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [500, 500, 500, 500],
    }

    variables_nac = {
        'invd_index': [],
        'angle_index': [],
        'dihed_index': [],
        'depth': 4,
        'nn_size': 100,
        'activ': 'leaky_softplus',
        'activ_alpha': 0.03,
        'use_dropout': False,
        'dropout': 0.005,
        'use_reg_activ': None,
        'use_reg_weight': None,
        'use_reg_bias': None,
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'scale_x_mean': False,
        'scale_x_std': False,
        'scale_y_mean': True,
        'scale_y_std': True,
        'normalization_mode': 1,
        'learning_rate': 1e-3,
        'phase_less_loss': False,
        'initialize_weights': True,
        'val_disjoint': True,
        'epo': 2000,
        'epomin': 1000,
        'pre_epo': 100,
        'patience': 300,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'epostep': 10,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [500, 500, 500, 500],
    }

    variables_soc = {
        'invd_index': [],
        'angle_index': [],
        'dihed_index': [],
        'depth': 4,
        'nn_size': 100,
        'activ': 'leaky_softplus',
        'activ_alpha': 0.03,
        'use_dropout': False,
        'dropout': 0.005,
        'use_reg_activ': None,
        'use_reg_weight': None,
        'use_reg_bias': None,
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'scale_x_mean': False,
        'scale_x_std': False,
        'scale_y_mean': True,
        'scale_y_std': True,
        'normalization_mode': 1,
        'learning_rate': 1e-3,
        'initialize_weights': True,
        'val_disjoint': True,
        'epo': 2000,
        'epomin': 1000,
        'patience': 300,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'epostep': 10,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [500, 500, 500, 500],
    }

    variables_sch_eg = {
        'node_features': 128,
        'n_features': 64,
        'n_edges': 10,
        'n_filters': 64,
        'use_filter_bias': True,
        'cfc_activ': 'shifted_softplus',
        'n_blocks': 3,
        'n_rbf': 20,
        'maxradius': 4,
        'offset': 0.0,
        'sigma': 0.4,
        'mlp': [64],
        'use_mlp_bias': True,
        'mlp_activ': 'shifted_softplus',
        'use_output_bias': True,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'loss_weights': [1, 1],
        'initialize_weights': True,
        'epo': 400,
        'epomin': 200,
        'epostep': 10,
        'patience': 200,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'learning_rate': 1e-3,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
    }

    variables_sch_nac = {
        'node_features': 128,
        'n_features': 64,
        'n_edges': 10,
        'n_filters': 64,
        'use_filter_bias': True,
        'cfc_activ': 'shifted_softplus',
        'n_blocks': 3,
        'n_rbf': 20,
        'maxradius': 4,
        'offset': 0.0,
        'sigma': 0.4,
        'mlp': [64],
        'use_mlp_bias': True,
        'mlp_activ': 'shifted_softplus',
        'use_output_bias': True,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'phase_less_loss': False,
        'initialize_weights': True,
        'epo': 400,
        'epomin': 200,
        'epostep': 10,
        'patience': 200,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'learning_rate': 1e-3,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
    }

    variables_sch_soc = {
        'node_features': 128,
        'n_features': 64,
        'n_edges': 10,
        'n_filters': 64,
        'use_filter_bias': True,
        'cfc_activ': 'shifted_softplus',
        'n_blocks': 3,
        'n_rbf': 20,
        'maxradius': 4,
        'offset': 0.0,
        'sigma': 0.4,
        'mlp': [64],
        'use_mlp_bias': True,
        'mlp_activ': 'shifted_softplus',
        'use_output_bias': True,
        'use_step_callback': True,
        'use_linear_callback': False,
        'use_early_callback': False,
        'use_exp_callback': False,
        'callbacks': [],
        'initialize_weights': True,
        'epo': 400,
        'epomin': 200,
        'epostep': 10,
        'patience': 200,
        'max_time': 300,
        'batch_size': 64,
        'delta_loss': 1e-5,
        'loss_monitor': 'val_loss',
        'factor_lr': 0.1,
        'learning_rate': 1e-3,
        'learning_rate_start': 1e-3,
        'learning_rate_stop': 1e-6,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
    }

    variables_e2n2_eg = {
        'n_edges': 10,
        'maxradius': 4,
        'n_features': 64,
        'n_blocks': 3,
        'l_max': 1,
        'parity': True,
        'n_rbf': 20,
        'trainable_rbf': True,
        'rbf_cutoff': 6,
        'rbf_layers': 2,
        'rbf_neurons': 64,
        'rbf_act': 'silu',
        'rbf_act_a': 0.03,
        'normalization_y': 'component',
        'normalize_y': True,
        'self_connection': True,
        'resnet': False,
        'gate': True,
        'act_scalars_e': 'silu',
        'act_scalars_o': 'tanh',
        'act_gates_e': 'silu',
        'act_gates_o': 'tanh',
        'use_step_callback': True,
        'callbacks': [],
        'initialize_weights': True,
        'loss_weights': [10, 1],
        'use_reg_loss': 'l2',
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'epo': 400,
        'epostep': 10,
        'subset': 0,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
        'scaler': 'total_energy_mean_std',
        'grad_type': 'grad',
    }

    variables_e2n2_nac = {
        'n_edges': 10,
        'maxradius': 4,
        'n_features': 64,
        'n_blocks': 3,
        'l_max': 1,
        'parity': True,
        'n_rbf': 20,
        'trainable_rbf': True,
        'rbf_cutoff': 6,
        'rbf_layers': 2,
        'rbf_neurons': 64,
        'rbf_act': 'shifted_softplus',
        'rbf_act_a': 0.03,
        'normalization_y': 'component',
        'normalize_y': True,
        'self_connection': True,
        'resnet': False,
        'gate': True,
        'act_scalars_e': 'silu',
        'act_scalars_o': 'tanh',
        'act_gates_e': 'silu',
        'act_gates_o': 'tanh',
        'use_step_callback': True,
        'callbacks': [],
        'initialize_weights': True,
        'use_reg_loss': 'l2',
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'epo': 400,
        'epostep': 10,
        'subset': 0,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
        'scaler': 'no',
        'grad_type': 'grad',
    }

    variables_e2n2_soc = {
        'n_edges': 10,
        'maxradius': 4,
        'n_features': 64,
        'n_blocks': 3,
        'l_max': 1,
        'parity': True,
        'n_rbf': 20,
        'trainable_rbf': True,
        'rbf_cutoff': 6,
        'rbf_layers': 2,
        'rbf_neurons': 64,
        'rbf_act': 'shifted_softplus',
        'rbf_act_a': 0.03,
        'normalization_y': 'component',
        'normalize_y': True,
        'self_connection': True,
        'resnet': False,
        'gate': True,
        'act_scalars_e': 'silu',
        'act_scalars_o': 'tanh',
        'act_gates_e': 'silu',
        'act_gates_o': 'tanh',
        'use_step_callback': True,
        'callbacks': [],
        'initialize_weights': True,
        'use_reg_loss': 'l2',
        'reg_l1': 1e-5,
        'reg_l2': 1e-5,
        'epo': 400,
        'epostep': 10,
        'subset': 0,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction': [100, 100, 100, 100],
        'scaler': 'total_energy_mean_std',
        'grad_type': 'grad',
    }

    variables_file = {
        'natom': 0,
        'file': None,
    }

    ## More default will be added below

    ## ready to read input

    variables_input = {
        'control': variables_control,
        'molecule': variables_molecule,
        'molcas': variables_molcas,
        'bagel': variables_bagel,
        'orca': variables_orca,
        'xtb': variables_xtb,
        'md': variables_md,
        'nn': variables_nn,
        'mlp': variables_nn,
        'schnet': variables_nn,
        'e2n2': variables_nn,
        'search': variables_search,
        'eg': variables_eg.copy(),
        'nac': variables_nac.copy(),
        'soc': variables_soc.copy(),
        'eg2': variables_eg.copy(),
        'nac2': variables_nac.copy(),
        'soc2': variables_soc.copy(),
        'sch_eg': variables_sch_eg.copy(),
        'sch_nac': variables_sch_nac.copy(),
        'sch_soc': variables_sch_soc.copy(),
        'e2n2_eg': variables_e2n2_eg.copy(),
        'e2n2_nac': variables_e2n2_nac.copy(),
        'e2n2_soc': variables_e2n2_soc.copy(),
        'file': variables_file.copy(),
    }

    variables_readfunc = {
        'control': read_control,
        'molecule': read_molecule,
        'molcas': read_molcas,
        'bagel': read_bagel,
        'orca': read_orca,
        'xtb': read_xtb,
        'md': read_md,
        'nn': read_nn,
        'mlp': read_nn,
        'schnet': read_nn,
        'e2n2': read_nn,
        'search': read_grid_search,
        'eg': read_mlp,
        'nac': read_mlp,
        'soc': read_mlp,
        'eg2': read_mlp,
        'nac2': read_mlp,
        'soc2': read_mlp,
        'sch_eg': read_sch,
        'sch_nac': read_sch,
        'sch_soc': read_sch,
        'e2n2_eg': read_e2n2,
        'e2n2_nac': read_e2n2,
        'e2n2_soc': read_e2n2,
        'file': read_file,
    }

    ## read variable if input is a list of string
    ## skip reading if input is a json dict
    if isinstance(ld_input, list):
        for line in ld_input:
            line = line.splitlines()
            if len(line) == 0:
                continue
            variable_name = line[0].lower()
            variables_input[variable_name] = variables_readfunc[variable_name](variables_input[variable_name], line)

    ## assemble variables
    variables_all = {
        'control': variables_input['control'],
        'molecule': variables_input['molecule'],
        'molcas': variables_input['molcas'],
        'bagel': variables_input['bagel'],
        'orca': variables_input['orca'],
        'xtb': variables_input['xtb'],
        'md': variables_input['md'],
        'nn': variables_input['nn'],
        'mlp': variables_input['mlp'],
        'schnet': variables_input['schnet'],
        'e2n2': variables_input['e2n2'],
        'file': variables_input['file'],
    }

    ## update variables_nn
    variables_all['nn']['search'] = variables_input['search']
    variables_all['nn']['eg'] = variables_input['eg']
    variables_all['nn']['nac'] = variables_input['nac']
    variables_all['nn']['soc'] = variables_input['soc']
    variables_all['nn']['eg2'] = variables_input['eg2']
    variables_all['nn']['nac2'] = variables_input['nac2']
    variables_all['nn']['soc2'] = variables_input['soc2']
    variables_all['mlp']['eg'] = variables_input['eg']
    variables_all['mlp']['nac'] = variables_input['nac']
    variables_all['mlp']['soc'] = variables_input['soc']
    variables_all['mlp']['eg2'] = variables_input['eg2']
    variables_all['mlp']['nac2'] = variables_input['nac2']
    variables_all['mlp']['soc2'] = variables_input['soc2']
    variables_all['schnet']['sch_eg'] = variables_input['sch_eg']
    variables_all['schnet']['sch_nac'] = variables_input['sch_nac']
    variables_all['schnet']['sch_soc'] = variables_input['sch_soc']
    variables_all['e2n2']['e2n2_eg'] = variables_input['e2n2_eg']
    variables_all['e2n2']['e2n2_nac'] = variables_input['e2n2_nac']
    variables_all['e2n2']['e2n2_soc'] = variables_input['e2n2_soc']
    variables_all['nn']['ml_seed'] = variables_all['control']['gl_seed']
    variables_all['mlp']['ml_seed'] = variables_all['control']['gl_seed']
    variables_all['schnet']['ml_seed'] = variables_all['control']['gl_seed']
    variables_all['e2n2']['ml_seed'] = variables_all['control']['gl_seed']
    variables_all['md']['gl_seed'] = variables_all['control']['gl_seed']
    variables_all['molcas']['molcas_project'] = variables_all['control']['title']
    variables_all['molcas']['verbose'] = variables_all['md']['verbose']
    variables_all['bagel']['bagel_project'] = variables_all['control']['title']
    variables_all['bagel']['verbose'] = variables_all['md']['verbose']
    variables_all['orca']['orca_project'] = variables_all['control']['title']
    variables_all['orca']['verbose'] = variables_all['md']['verbose']
    variables_all['xtb']['xtb_project'] = variables_all['control']['title']
    variables_all['xtb']['verbose'] = variables_all['md']['verbose']

    ## update variables if input is a dict
    ## be caution that the input dict must have the same data structure
    ## this is only use to load pre-stored input in json
    if isinstance(ld_input, dict):
        variables_all = deep_update(variables_all, ld_input)

    return variables_all

def deep_update(a, b):
    ## recursively update a with b
    for key, val in b.items():
        if key in a.keys():
            if isinstance(val, dict) and isinstance(a[key], dict):
                a[key] = deep_update(a[key], val)
            else:
                a[key] = val
        else:
            a[key] = val

    return a

def start_info(variables_all):
    ##  This function print start information

    variables_control = variables_all['control']
    variables_molecule = variables_all['molecule']
    variables_molcas = variables_all['molcas']
    variables_bagel = variables_all['bagel']
    variables_orca = variables_all['orca']
    variables_xtb = variables_all['xtb']
    variables_md = variables_all['md']
    variables_nn = variables_all['nn']
    variables_mlp = variables_all['mlp']
    variables_schnet = variables_all['schnet']
    variables_e2n2 = variables_all['e2n2']
    variables_eg = variables_nn['eg']
    variables_nac = variables_nn['nac']
    variables_soc = variables_nn['soc']
    variables_eg2 = variables_nn['eg2']
    variables_nac2 = variables_nn['nac2']
    variables_soc2 = variables_nn['soc2']
    variables_mlp_eg = variables_mlp['eg']
    variables_mlp_nac = variables_mlp['nac']
    variables_mlp_soc = variables_mlp['soc']
    variables_mlp_eg2 = variables_mlp['eg2']
    variables_mlp_nac2 = variables_mlp['nac2']
    variables_mlp_soc2 = variables_mlp['soc2']
    variables_sch_eg = variables_schnet['sch_eg']
    variables_sch_nac = variables_schnet['sch_nac']
    variables_sch_soc = variables_schnet['sch_soc']
    variables_e2n2_eg = variables_e2n2['e2n2_eg']
    variables_e2n2_nac = variables_e2n2['e2n2_nac']
    variables_e2n2_soc = variables_e2n2['e2n2_soc']

    variables_search = variables_nn['search']

    control_info = """
  &control
-------------------------------------------------------
  Title:                      %-10s
  NPROCS for ML:              %-10s
  NPROCS for QC:              %-10s
  NPROCS for Multiscale:      %-10s 
  Seed:                       %-10s
  Job: 	                      %-10s
  QM:          	       	      %-10s
  Ab initio:                  %-10s
-------------------------------------------------------

""" % (
        variables_control['title'],
        variables_control['ml_ncpu'],
        variables_control['qc_ncpu'],
        variables_control['ms_ncpu'],
        variables_control['gl_seed'],
        variables_control['jobtype'],
        ' '.join(variables_control['qm']),
        ' '.join(variables_control['abinit'])
    )

    molecule_info = """
  &molecule
-------------------------------------------------------
  States:                     %-10s
  Spin:                       %-10s
  Interstates:                %-10s
  QMMM keyfile:               %-10s
  QMMM xyzfile:               %-10s
  High level region:          %-10s ...
  Boundary:                   %-10s ...
  Embedding charges:          %-10s
  Frozen atoms:               %-10s
  Constrained atoms:          %-10s
  External potential shape:   %-10s
  External potential factor:  %-10s
  External potential radius:  %-10s
  External potential center:  %-10s
  Primitive vectors:          %-10s
  Lattice constant:           %-10s
-------------------------------------------------------

""" % (
        variables_molecule['ci'],
        variables_molecule['spin'],
        variables_molecule['coupling'],
        variables_molecule['qmmm_key'],
        variables_molecule['qmmm_xyz'],
        variables_molecule['highlevel'][0:10],
        variables_molecule['boundary'][0:5],
        variables_molecule['embedding'],
        variables_molecule['freeze'],
        variables_molecule['constrain'],
        variables_molecule['shape'],
        variables_molecule['factor'],
        variables_molecule['cavity'],
        variables_molecule['center'],
        variables_molecule['primitive'],
        variables_molecule['lattice']
    )

    adaptive_info = """
  &adaptive sampling method
-------------------------------------------------------
  Ab initio:                  %-10s
  Load trained model:         %-10s
  Transfer learning:          %-10s
  Maxiter:                    %-10s
  Sampling number per traj:   %-10s
  Use dynamical Std:          %-10s
  Max discard range           %-10s
  Refine crossing:            %-10s
  Refine points/range: 	      %-10s %-10s %-10s
  MaxStd  energy:             %-10s
  MinStd  energy:             %-10s
  InitStd energy:             %-10s
  Dynfctr energy:             %-10s
  Forward delay energy:       %-10s
  Backward delay energy:      %-10s
  MaxStd  gradient:           %-10s
  MinStd  gradient:           %-10s
  InitStd gradient:           %-10s
  Dynfctr gradient:           %-10s
  Forward delay	gradient:     %-10s
  Backward delay gradient:    %-10s
  MaxStd  nac:                %-10s
  MinStd  nac:                %-10s
  InitStd nac:                %-10s
  Dynfctr nac:                %-10s
  Forward delay	nac:          %-10s
  Backward delay nac:         %-10s
  MaxStd  soc:                %-10s
  MinStd  soc:                %-10s
  InitStd soc:                %-10s
  Dynfctr soc:                %-10s
  Forward delay	soc:   	      %-10s
  Backward delay soc:  	      %-10s
-------------------------------------------------------

""" % (
        ' '.join(variables_control['abinit']),
        variables_control['load'],
        variables_control['transfer'],
        variables_control['maxiter'],
        variables_control['maxsample'],
        variables_control['dynsample'],
        variables_control['maxdiscard'],
        variables_control['refine'],
        variables_control['refine_num'],
        variables_control['refine_start'],
        variables_control['refine_end'],
        variables_control['maxenergy'],
        variables_control['minenergy'],
        variables_control['inienergy'],
        variables_control['dynenergy'],
        variables_control['fwdenergy'],
        variables_control['bckenergy'],
        variables_control['maxgrad'],
        variables_control['mingrad'],
        variables_control['inigrad'],
        variables_control['dyngrad'],
        variables_control['fwdgrad'],
        variables_control['bckgrad'],
        variables_control['maxnac'],
        variables_control['minnac'],
        variables_control['ininac'],
        variables_control['dynnac'],
        variables_control['fwdnac'],
        variables_control['bcknac'],
        variables_control['maxsoc'],
        variables_control['minsoc'],
        variables_control['inisoc'],
        variables_control['dynsoc'],
        variables_control['fwdsoc'],
        variables_control['bcksoc']
    )

    md_info = """
  &initial condition
-------------------------------------------------------
  Generate initial condition: %-10s
  Number:                     %-10s
  Method:                     %-10s 
  Format:                     %-10s
-------------------------------------------------------

""" % (
        variables_md['initcond'],
        variables_md['ninitcond'],
        variables_md['method'],
        variables_md['format']
    )

    md_info += """
  &md
-------------------------------------------------------
  Initial state:              %-10s
  Temperature (K):            %-10s
  Step:                       %-10s
  Dt (au):                    %-10s
  Only active state grad      %-10s
  Surface hopping:            %-10s
  NAC type:                   %-10s
  Phase correction            %-10s
  Substep:                    %-10s
  Integrate probability       %-10s
  Decoherence:                %-10s
  Adjust velocity:            %-10s
  Reflect velocity:           %-10s
  Maxhop:                     %-10s
  Thermodynamic:              %-10s
  Thermodynamic delay:        %-10s
  Print level:                %-10s
  Direct output:              %-10s
  Buffer output:              %-10s
  Record MD data:             %-10s
  Record MD steps:            %-10s
  Checkpoint steps:           %-10s 
  Restart function:           %-10s
  Additional steps:           %-10s
-------------------------------------------------------

""" % (
        variables_md['root'],
        variables_md['temp'],
        variables_md['step'],
        variables_md['size'],
        variables_md['activestate'],
        variables_md['sfhp'],
        variables_md['nactype'],
        variables_md['phasecheck'],
        variables_md['substep'],
        variables_md['integrate'],
        variables_md['deco'],
        variables_md['adjust'],
        variables_md['reflect'],
        variables_md['maxh'],
        variables_md['thermo'],
        variables_md['thermodelay'],
        variables_md['verbose'],
        variables_md['direct'],
        variables_md['buffer'],
        variables_md['record'],
        variables_md['record_step'],
        variables_md['checkpoint'],
        variables_md['restart'],
        variables_md['addstep']
    )

    md_info += """
  &md velocity control
-------------------------------------------------------
  Excess kinetic energy       %-10s
  Scale kinetic energy        %-10s
  Target kinetic energy       %-10s
  Gradient descent path       %-10s
  Reset velocity:             %-10s
  Reset step:                 %-10s
-------------------------------------------------------

""" % (
        variables_md['excess'],
        variables_md['scale'],
        variables_md['target'],
        variables_md['graddesc'],
        variables_md['reset'],
        variables_md['resetstep']
    )

    hybrid_info = """
  &hybrid namd
-------------------------------------------------------
  Mix Energy                  %-10s
  Mix Gradient                %-10s
  Mix NAC                     %-10s
  Mix SOC                     %-10s
-------------------------------------------------------

""" % (
        variables_md['ref_energy'],
        variables_md['ref_grad'],
        variables_md['ref_nac'],
        variables_md['ref_soc']
    )

    nn_info = """
  &nn
-------------------------------------------------------
  Train data:                 %-10s
  Prediction data:            %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  Data splits:                %-10s
  NN EG type:                 %-10s
  NN NAC type:                %-10s
  NN SOC type:                %-10s
  Multiscale:                 %-10s
  Shuffle data:               %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
  Data permutation            %-10s
-------------------------------------------------------

""" % (
        variables_nn['train_data'],
        variables_nn['pred_data'],
        variables_nn['train_mode'],
        variables_nn['silent'],
        variables_nn['nsplits'],
        variables_nn['nn_eg_type'],
        variables_nn['nn_nac_type'],
        variables_nn['nn_soc_type'],
        [[x[: 5], '...'] for x in variables_nn['multiscale']],
        variables_nn['shuffle'],
        variables_nn['eg_unit'],
        variables_nn['nac_unit'],
        variables_nn['permute_map']
    )

    nn_info += """

  multilayer perceptron (native)
  
  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s 
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:      	              %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:         	              %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Validation disjoint:        %-20s %-20s %-20s
  Validation split:           %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Shift_X:     	       	      %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % (
        len(variables_eg['invd_index']),
        len(variables_nac['invd_index']),
        len(variables_soc['invd_index']),
        len(variables_eg['angle_index']),
        len(variables_nac['angle_index']),
        len(variables_soc['angle_index']),
        len(variables_eg['dihed_index']),
        len(variables_nac['dihed_index']),
        len(variables_soc['dihed_index']),
        variables_eg['activ'],
        variables_nac['activ'],
        variables_soc['activ'],
        variables_eg['activ_alpha'],
        variables_nac['activ_alpha'],
        variables_soc['activ_alpha'],
        variables_eg['depth'],
        variables_nac['depth'],
        variables_soc['depth'],
        variables_eg['nn_size'],
        variables_nac['nn_size'],
        variables_soc['nn_size'],
        variables_eg['use_dropout'],
        variables_nac['use_dropout'],
        variables_soc['use_dropout'],
        variables_eg['dropout'],
        variables_nac['dropout'],
        variables_soc['dropout'],
        variables_eg['use_reg_activ'],
        variables_nac['use_reg_activ'],
        variables_soc['use_reg_activ'],
        variables_eg['use_reg_weight'],
        variables_nac['use_reg_weight'],
        variables_soc['use_reg_weight'],
        variables_eg['use_reg_bias'],
        variables_nac['use_reg_bias'],
        variables_soc['use_reg_bias'],
        variables_eg['reg_l1'],
        variables_nac['reg_l1'],
        variables_soc['reg_l1'],
        variables_eg['reg_l2'],
        variables_nac['reg_l2'],
        variables_soc['reg_l2'],
        variables_eg['loss_weights'],
        '',
        '',
        '',
        variables_nac['phase_less_loss'],
        '',
        variables_eg['initialize_weights'],
        variables_nac['initialize_weights'],
        variables_soc['initialize_weights'],
        variables_eg['val_disjoint'],
        variables_nac['val_disjoint'],
        variables_soc['val_disjoint'],
        1 / variables_nn['nsplits'],
        1 / variables_nn['nsplits'],
        1 / variables_nn['nsplits'],
        variables_eg['epo'],
        variables_nac['epo'],
        variables_soc['epo'],
        '',
        variables_nac['pre_epo'],
        '',
        variables_eg['epomin'],
        variables_nac['epomin'],
        variables_soc['epomin'],
        variables_eg['patience'],
        variables_nac['patience'],
        variables_soc['patience'],
        variables_eg['max_time'],
        variables_nac['max_time'],
        variables_soc['max_time'],
        variables_eg['epostep'],
        variables_nac['epostep'],
        variables_soc['epostep'],
        variables_eg['batch_size'],
        variables_nac['batch_size'],
        variables_soc['batch_size'],
        variables_eg['delta_loss'],
        variables_nac['delta_loss'],
        variables_soc['delta_loss'],
        variables_eg['scale_x_mean'],
        variables_nac['scale_x_mean'],
        variables_soc['scale_x_mean'],
        variables_eg['scale_x_std'],
        variables_nac['scale_x_std'],
        variables_soc['scale_x_std'],
        variables_eg['scale_y_mean'],
        variables_nac['scale_y_mean'],
        variables_soc['scale_y_mean'],
        variables_eg['scale_y_std'],
        variables_nac['scale_y_std'],
        variables_soc['scale_y_std'],
        variables_eg['normalization_mode'],
        variables_nac['normalization_mode'],
        variables_soc['normalization_mode']
    )

    nn_info += """
  &hyperparameters            Energy+Gradient(2)   Nonadiabatic(2)      Spin-orbit(2)
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:                     %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:                         %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Validation disjoint:        %-20s %-20s %-20s
  Validation split:           %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Shift_X:                    %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % (
        len(variables_eg2['invd_index']),
        len(variables_nac2['invd_index']),
        len(variables_soc2['invd_index']),
        len(variables_eg2['angle_index']),
        len(variables_nac2['angle_index']),
        len(variables_soc2['angle_index']),
        len(variables_eg2['dihed_index']),
        len(variables_nac2['dihed_index']),
        len(variables_soc2['dihed_index']),
        variables_eg2['activ'],
        variables_nac2['activ'],
        variables_soc2['activ'],
        variables_eg2['activ_alpha'],
        variables_nac2['activ_alpha'],
        variables_soc2['activ_alpha'],
        variables_eg2['depth'],
        variables_nac2['depth'],
        variables_soc2['depth'],
        variables_eg2['nn_size'],
        variables_nac2['nn_size'],
        variables_soc2['nn_size'],
        variables_eg2['use_dropout'],
        variables_nac2['use_dropout'],
        variables_soc2['use_dropout'],
        variables_eg2['dropout'],
        variables_nac2['dropout'],
        variables_soc2['dropout'],
        variables_eg2['use_reg_activ'],
        variables_nac2['use_reg_activ'],
        variables_soc2['use_reg_activ'],
        variables_eg2['use_reg_weight'],
        variables_nac2['use_reg_weight'],
        variables_soc2['use_reg_weight'],
        variables_eg2['use_reg_bias'],
        variables_nac2['use_reg_bias'],
        variables_soc2['use_reg_bias'],
        variables_eg2['reg_l1'],
        variables_nac2['reg_l1'],
        variables_soc2['reg_l1'],
        variables_eg2['reg_l2'],
        variables_nac2['reg_l2'],
        variables_soc2['reg_l2'],
        variables_eg['loss_weights'],
        '',
        '',
        '',
        variables_nac2['phase_less_loss'],
        '',
        variables_eg2['initialize_weights'],
        variables_nac2['initialize_weights'],
        variables_soc2['initialize_weights'],
        variables_eg2['val_disjoint'],
        variables_nac2['val_disjoint'],
        variables_soc2['val_disjoint'],
        1 / variables_nn['nsplits'],
        1 / variables_nn['nsplits'],
        1 / variables_nn['nsplits'],
        variables_eg2['epo'],
        variables_nac2['epo'],
        variables_soc2['epo'],
        '',
        variables_nac2['pre_epo'],
        '',
        variables_eg2['epomin'],
        variables_nac2['epomin'],
        variables_soc2['epomin'],
        variables_eg2['patience'],
        variables_nac2['patience'],
        variables_soc2['patience'],
        variables_eg2['max_time'],
        variables_nac2['max_time'],
        variables_soc2['max_time'],
        variables_eg2['epostep'],
        variables_nac2['epostep'],
        variables_soc2['epostep'],
        variables_eg2['batch_size'],
        variables_nac2['batch_size'],
        variables_soc2['batch_size'],
        variables_eg2['delta_loss'],
        variables_nac2['delta_loss'],
        variables_soc2['delta_loss'],
        variables_eg2['scale_x_mean'],
        variables_nac2['scale_x_mean'],
        variables_soc2['scale_x_mean'],
        variables_eg2['scale_x_std'],
        variables_nac2['scale_x_std'],
        variables_soc2['scale_x_std'],
        variables_eg2['scale_y_mean'],
        variables_nac2['scale_y_mean'],
        variables_soc2['scale_y_mean'],
        variables_eg2['scale_y_std'],
        variables_nac2['scale_y_std'],
        variables_soc2['scale_y_std'],
        variables_eg2['normalization_mode'],
        variables_nac2['normalization_mode'],
        variables_soc2['normalization_mode']
    )

    mlp_info = """
  &mlp
-------------------------------------------------------
  Train data:                 %-10s
  Prediction data:            %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  Data splits:                %-10s
  NN EG type:                 %-10s
  NN NAC type:                %-10s
  NN SOC type:                %-10s
  Multiscale:                 %-10s
  Shuffle data:               %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
  Data permutation            %-10s
-------------------------------------------------------

    """ % (
        variables_mlp['train_data'],
        variables_mlp['pred_data'],
        variables_mlp['train_mode'],
        variables_mlp['silent'],
        variables_mlp['nsplits'],
        variables_mlp['nn_eg_type'],
        variables_mlp['nn_nac_type'],
        variables_mlp['nn_soc_type'],
        [[x[: 5], '...'] for x in variables_mlp['multiscale']],
        variables_mlp['shuffle'],
        variables_mlp['eg_unit'],
        variables_mlp['nac_unit'],
        variables_mlp['permute_map']
    )

    mlp_info += """

  multilayer perceptron (pyNNsMD)
  
  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s 
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:      	              %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:         	              %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % (
        len(variables_mlp_eg['invd_index']),
        len(variables_mlp_nac['invd_index']),
        len(variables_mlp_soc['invd_index']),
        len(variables_mlp_eg['angle_index']),
        len(variables_mlp_nac['angle_index']),
        len(variables_mlp_soc['angle_index']),
        len(variables_mlp_eg['dihed_index']),
        len(variables_mlp_nac['dihed_index']),
        len(variables_mlp_soc['dihed_index']),
        variables_mlp_eg['activ'],
        variables_mlp_nac['activ'],
        variables_mlp_soc['activ'],
        variables_mlp_eg['activ_alpha'],
        variables_mlp_nac['activ_alpha'],
        variables_mlp_soc['activ_alpha'],
        variables_mlp_eg['depth'],
        variables_mlp_nac['depth'],
        variables_mlp_soc['depth'],
        variables_mlp_eg['nn_size'],
        variables_mlp_nac['nn_size'],
        variables_mlp_soc['nn_size'],
        variables_mlp_eg['use_dropout'],
        variables_mlp_nac['use_dropout'],
        variables_mlp_soc['use_dropout'],
        variables_mlp_eg['dropout'],
        variables_mlp_nac['dropout'],
        variables_mlp_soc['dropout'],
        variables_mlp_eg['use_reg_activ'],
        variables_mlp_nac['use_reg_activ'],
        variables_mlp_soc['use_reg_activ'],
        variables_mlp_eg['use_reg_weight'],
        variables_mlp_nac['use_reg_weight'],
        variables_mlp_soc['use_reg_weight'],
        variables_mlp_eg['use_reg_bias'],
        variables_mlp_nac['use_reg_bias'],
        variables_mlp_soc['use_reg_bias'],
        variables_mlp_eg['reg_l1'],
        variables_mlp_nac['reg_l1'],
        variables_mlp_soc['reg_l1'],
        variables_mlp_eg['reg_l2'],
        variables_mlp_nac['reg_l2'],
        variables_mlp_soc['reg_l2'],
        variables_mlp_eg['loss_weights'],
        '',
        '',
        '',
        variables_mlp_nac['phase_less_loss'],
        '',
        variables_mlp_eg['initialize_weights'],
        variables_mlp_nac['initialize_weights'],
        variables_mlp_soc['initialize_weights'],
        variables_mlp_eg['epo'],
        variables_mlp_nac['epo'],
        variables_mlp_soc['epo'],
        '',
        variables_mlp_nac['pre_epo'],
        '',
        variables_mlp_eg['epomin'],
        variables_mlp_nac['epomin'],
        variables_mlp_soc['epomin'],
        variables_mlp_eg['patience'],
        variables_mlp_nac['patience'],
        variables_mlp_soc['patience'],
        variables_mlp_eg['max_time'],
        variables_mlp_nac['max_time'],
        variables_mlp_soc['max_time'],
        variables_mlp_eg['epostep'],
        variables_mlp_nac['epostep'],
        variables_mlp_soc['epostep'],
        variables_mlp_eg['batch_size'],
        variables_mlp_nac['batch_size'],
        variables_mlp_soc['batch_size'],
        variables_mlp_eg['delta_loss'],
        variables_mlp_nac['delta_loss'],
        variables_mlp_soc['delta_loss'],
        variables_mlp_eg['normalization_mode'],
        variables_mlp_nac['normalization_mode'],
        variables_mlp_soc['normalization_mode']
    )

    mlp_info += """
  &hyperparameters            Energy+Gradient(2)   Nonadiabatic(2)      Spin-orbit(2)
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:                     %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:                         %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % (
        len(variables_mlp_eg2['invd_index']),
        len(variables_mlp_nac2['invd_index']),
        len(variables_mlp_soc2['invd_index']),
        len(variables_mlp_eg2['angle_index']),
        len(variables_mlp_nac2['angle_index']),
        len(variables_mlp_soc2['angle_index']),
        len(variables_mlp_eg2['dihed_index']),
        len(variables_mlp_nac2['dihed_index']),
        len(variables_mlp_soc2['dihed_index']),
        variables_mlp_eg2['activ'],
        variables_mlp_nac2['activ'],
        variables_mlp_soc2['activ'],
        variables_mlp_eg2['activ_alpha'],
        variables_mlp_nac2['activ_alpha'],
        variables_mlp_soc2['activ_alpha'],
        variables_mlp_eg2['depth'],
        variables_mlp_nac2['depth'],
        variables_mlp_soc2['depth'],
        variables_mlp_eg2['nn_size'],
        variables_mlp_nac2['nn_size'],
        variables_mlp_soc2['nn_size'],
        variables_mlp_eg2['use_dropout'],
        variables_mlp_nac2['use_dropout'],
        variables_mlp_soc2['use_dropout'],
        variables_mlp_eg2['dropout'],
        variables_mlp_nac2['dropout'],
        variables_mlp_soc2['dropout'],
        variables_mlp_eg2['use_reg_activ'],
        variables_mlp_nac2['use_reg_activ'],
        variables_mlp_soc2['use_reg_activ'],
        variables_mlp_eg2['use_reg_weight'],
        variables_mlp_nac2['use_reg_weight'],
        variables_mlp_soc2['use_reg_weight'],
        variables_mlp_eg2['use_reg_bias'],
        variables_mlp_nac2['use_reg_bias'],
        variables_mlp_soc2['use_reg_bias'],
        variables_mlp_eg2['reg_l1'],
        variables_mlp_nac2['reg_l1'],
        variables_mlp_soc2['reg_l1'],
        variables_mlp_eg2['reg_l2'],
        variables_mlp_nac2['reg_l2'],
        variables_mlp_soc2['reg_l2'],
        variables_mlp_eg['loss_weights'],
        '',
        '',
        '',
        variables_mlp_nac2['phase_less_loss'],
        '',
        variables_mlp_eg2['initialize_weights'],
        variables_mlp_nac2['initialize_weights'],
        variables_mlp_soc2['initialize_weights'],
        variables_mlp_eg2['epo'],
        variables_mlp_nac2['epo'],
        variables_mlp_soc2['epo'],
        '',
        variables_mlp_nac2['pre_epo'],
        '',
        variables_mlp_eg2['epomin'],
        variables_mlp_nac2['epomin'],
        variables_mlp_soc2['epomin'],
        variables_mlp_eg2['patience'],
        variables_mlp_nac2['patience'],
        variables_mlp_soc2['patience'],
        variables_mlp_eg2['max_time'],
        variables_mlp_nac2['max_time'],
        variables_mlp_soc2['max_time'],
        variables_mlp_eg2['epostep'],
        variables_mlp_nac2['epostep'],
        variables_mlp_soc2['epostep'],
        variables_mlp_eg2['batch_size'],
        variables_mlp_nac2['batch_size'],
        variables_mlp_soc2['batch_size'],
        variables_mlp_eg2['delta_loss'],
        variables_mlp_nac2['delta_loss'],
        variables_mlp_soc2['delta_loss'],
        variables_mlp_eg2['normalization_mode'],
        variables_mlp_nac2['normalization_mode'],
        variables_mlp_soc2['normalization_mode']
    )

    sch_info = """
  &schnet
-------------------------------------------------------
  Train data:                 %-10s
  Prediction data:            %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  Data splits:                %-10s
  Shuffle data:               %-10s
  Multiscale:                 %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
-------------------------------------------------------

    """ % (
        variables_schnet['train_data'],
        variables_schnet['pred_data'],
        variables_schnet['train_mode'],
        variables_schnet['silent'],
        variables_schnet['nsplits'],
        variables_schnet['shuffle'],
        [[x[: 5], '...'] for x in variables_schnet['multiscale']],
        variables_schnet['eg_unit'],
        variables_schnet['nac_unit'],
    )

    sch_info += """

  Schnet (pyNNsMD)

  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  Node features:              %-20s %-20s %-20s
  Generated features:         %-20s %-20s %-20s  
  Edges:                      %-20s %-20s %-20s
  Filter features:            %-20s %-20s %-20s
  Use filter bias:            %-20s %-20s %-20s
  ConvLayer activation:       %-20s %-20s %-20s
  Interaction blocks:      	  %-20s %-20s %-20s
  Radial basis:               %-20s %-20s %-20s
  Maxradius:                  %-20s %-20s %-20s
  Radial offset:              %-20s %-20s %-20s
  Radial sigma:               %-20s %-20s %-20s
  MLP layers:                 %-20s %-20s %-20s
  Use MLP bias:               %-20s %-20s %-20s
  MLP activation:             %-20s %-20s %-20s
  Use output bias:         	  %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

    """ % (
        variables_sch_eg['node_features'],
        variables_sch_nac['node_features'],
        variables_sch_soc['node_features'],
        variables_sch_eg['n_features'],
        variables_sch_nac['n_features'],
        variables_sch_soc['n_features'],
        variables_sch_eg['n_edges'],
        variables_sch_nac['n_edges'],
        variables_sch_soc['n_edges'],
        variables_sch_eg['n_filters'],
        variables_sch_nac['n_filters'],
        variables_sch_soc['n_filters'],
        variables_sch_eg['use_filter_bias'],
        variables_sch_nac['use_filter_bias'],
        variables_sch_soc['use_filter_bias'],
        variables_sch_eg['cfc_activ'],
        variables_sch_nac['cfc_activ'],
        variables_sch_soc['cfc_activ'],
        variables_sch_eg['n_blocks'],
        variables_sch_nac['n_blocks'],
        variables_sch_soc['n_blocks'],
        variables_sch_eg['n_rbf'],
        variables_sch_nac['n_rbf'],
        variables_sch_soc['n_rbf'],
        variables_sch_eg['maxradius'],
        variables_sch_nac['maxradius'],
        variables_sch_soc['maxradius'],
        variables_sch_eg['offset'],
        variables_sch_nac['offset'],
        variables_sch_soc['offset'],
        variables_sch_eg['sigma'],
        variables_sch_nac['sigma'],
        variables_sch_soc['sigma'],
        variables_sch_eg['mlp'],
        variables_sch_nac['mlp'],
        variables_sch_soc['mlp'],
        variables_sch_eg['use_mlp_bias'],
        variables_sch_nac['use_mlp_bias'],
        variables_sch_soc['use_mlp_bias'],
        variables_sch_eg['mlp_activ'],
        variables_sch_nac['mlp_activ'],
        variables_sch_soc['mlp_activ'],
        variables_sch_eg['use_output_bias'],
        variables_sch_nac['use_output_bias'],
        variables_sch_soc['use_output_bias'],
        variables_sch_eg['loss_weights'],
        '',
        '',
        '',
        variables_sch_nac['phase_less_loss'],
        '',
        variables_sch_eg['initialize_weights'],
        variables_sch_nac['initialize_weights'],
        variables_sch_soc['initialize_weights'],
        variables_sch_eg['epo'],
        variables_sch_nac['epo'],
        variables_sch_soc['epo'],
        variables_sch_eg['epomin'],
        variables_sch_nac['epomin'],
        variables_sch_soc['epomin'],
        variables_sch_eg['patience'],
        variables_sch_nac['patience'],
        variables_sch_soc['patience'],
        variables_sch_eg['max_time'],
        variables_sch_nac['max_time'],
        variables_sch_soc['max_time'],
        variables_sch_eg['epostep'],
        variables_sch_nac['epostep'],
        variables_sch_soc['epostep'],
        variables_sch_eg['batch_size'],
        variables_sch_nac['batch_size'],
        variables_sch_soc['batch_size'],
        variables_sch_eg['delta_loss'],
        variables_sch_nac['delta_loss'],
        variables_sch_soc['delta_loss'],
    )

    e2n2_info = """
  &e2n2
-------------------------------------------------------
  Train data:                 %-10s
  Prediction data:            %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  Data splits:                %-10s
  Shuffle data:               %-10s
  Multiscale:                 %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
-------------------------------------------------------

    """ % (
        variables_e2n2['train_data'],
        variables_e2n2['pred_data'],
        variables_e2n2['train_mode'],
        variables_e2n2['silent'],
        variables_e2n2['nsplits'],
        variables_e2n2['shuffle'],
        [[x[: 5], '...'] for x in variables_e2n2['multiscale']],
        variables_e2n2['eg_unit'],
        variables_e2n2['nac_unit'],
    )

    e2n2_info += """

      E2N2 (GCNNP)

      &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
    ----------------------------------------------------------------------------------------------
      Edges:                      %-20s %-20s %-20s
      Maxradius:                  %-20s %-20s %-20s
      Node features:              %-20s %-20s %-20s 
      Interaction blocks:      	  %-20s %-20s %-20s
      Rotation order              %-20s %-20s %-20s
      Irreps parity:              %-20s %-20s %-20s
      Radial basis:               %-20s %-20s %-20s
      Radial basis trainable:     %-20s %-20s %-20s
      Envelop func cutoff:        %-20s %-20s %-20s
      Radial net layers:          %-20s %-20s %-20s
      Radial net neurons:         %-20s %-20s %-20s
      Radial net activation:      %-20s %-20s %-20s
      Radial net activation a:    %-20s %-20s %-20s
      Y normalization scheme:     %-20s %-20s %-20s
      Normalize Y:                %-20s %-20s %-20s
      Self connection:            %-20s %-20s %-20s
      Resnet update:              %-20s %-20s %-20s
      Use gate activation:        %-20s %-20s %-20s
      Even scalars activation:    %-20s %-20s %-20s
      Odd scalars activation:     %-20s %-20s %-20s
      Even gates activation:      %-20s %-20s %-20s
      Odd gates activation:       %-20s %-20s %-20s
      Initialize weight:          %-20s %-20s %-20s
      Loss weights:               %-20s %-20s %-20s
      Epoch:                      %-20s %-20s %-20s
      Epoch step:                 %-20s %-20s %-20s
      Subset:                     %-20s %-20s %-20s
      Scaler:                     %-20s %-20s %-20s
      Batch:                      %-20s %-20s %-20s
    ----------------------------------------------------------------------------------------------

        """ % (
        variables_e2n2_eg['n_edges'],
        variables_e2n2_nac['n_edges'],
        variables_e2n2_soc['n_edges'],
        variables_e2n2_eg['maxradius'],
        variables_e2n2_nac['maxradius'],
        variables_e2n2_soc['maxradius'],
        variables_e2n2_eg['n_features'],
        variables_e2n2_nac['n_features'],
        variables_e2n2_soc['n_features'],
        variables_e2n2_eg['n_blocks'],
        variables_e2n2_nac['n_blocks'],
        variables_e2n2_soc['n_blocks'],
        variables_e2n2_eg['l_max'],
        variables_e2n2_nac['l_max'],
        variables_e2n2_soc['l_max'],
        variables_e2n2_eg['parity'],
        variables_e2n2_nac['parity'],
        variables_e2n2_soc['parity'],
        variables_e2n2_eg['n_rbf'],
        variables_e2n2_nac['n_rbf'],
        variables_e2n2_soc['n_rbf'],
        variables_e2n2_eg['trainable_rbf'],
        variables_e2n2_nac['trainable_rbf'],
        variables_e2n2_soc['trainable_rbf'],
        variables_e2n2_eg['rbf_cutoff'],
        variables_e2n2_nac['rbf_cutoff'],
        variables_e2n2_soc['rbf_cutoff'],
        variables_e2n2_eg['rbf_layers'],
        variables_e2n2_nac['rbf_layers'],
        variables_e2n2_soc['rbf_layers'],
        variables_e2n2_eg['rbf_neurons'],
        variables_e2n2_nac['rbf_neurons'],
        variables_e2n2_soc['rbf_neurons'],
        variables_e2n2_eg['rbf_act'],
        variables_e2n2_nac['rbf_act'],
        variables_e2n2_soc['rbf_act'],
        variables_e2n2_eg['rbf_act_a'],
        variables_e2n2_nac['rbf_act_a'],
        variables_e2n2_soc['rbf_act_a'],
        variables_e2n2_eg['normalization_y'],
        variables_e2n2_nac['normalization_y'],
        variables_e2n2_soc['normalization_y'],
        variables_e2n2_eg['normalize_y'],
        variables_e2n2_nac['normalize_y'],
        variables_e2n2_soc['normalize_y'],
        variables_e2n2_eg['self_connection'],
        variables_e2n2_nac['self_connection'],
        variables_e2n2_soc['self_connection'],
        variables_e2n2_eg['resnet'],
        variables_e2n2_nac['resnet'],
        variables_e2n2_soc['resnet'],
        variables_e2n2_eg['gate'],
        variables_e2n2_nac['gate'],
        variables_e2n2_soc['gate'],
        variables_e2n2_eg['act_scalars_e'],
        variables_e2n2_nac['act_scalars_e'],
        variables_e2n2_soc['act_scalars_e'],
        variables_e2n2_eg['act_scalars_o'],
        variables_e2n2_nac['act_scalars_o'],
        variables_e2n2_soc['act_scalars_o'],
        variables_e2n2_eg['act_gates_e'],
        variables_e2n2_nac['act_gates_e'],
        variables_e2n2_soc['act_gates_e'],
        variables_e2n2_eg['act_gates_o'],
        variables_e2n2_nac['act_gates_o'],
        variables_e2n2_soc['act_gates_o'],
        variables_e2n2_eg['initialize_weights'],
        variables_e2n2_nac['initialize_weights'],
        variables_e2n2_soc['initialize_weights'],
        variables_e2n2_eg['loss_weights'],
        '',
        '',
        variables_e2n2_eg['epo'],
        variables_e2n2_nac['epo'],
        variables_e2n2_soc['epo'],
        variables_e2n2_eg['epostep'],
        variables_e2n2_nac['epostep'],
        variables_e2n2_soc['epostep'],
        variables_e2n2_eg['subset'],
        variables_e2n2_nac['subset'],
        variables_e2n2_soc['subset'],
        variables_e2n2_eg['scaler'],
        variables_e2n2_nac['scaler'],
        variables_e2n2_soc['scaler'],
        variables_e2n2_eg['batch_size'],
        variables_e2n2_nac['batch_size'],
        variables_e2n2_soc['batch_size'],
    )

    search_info = """
  &grid search
-------------------------------------------------------
  Layers:                     %-10s
  Neurons/layer::             %-10s
  Batch:                      %-10s
  L1:                         %-10s
  L2:                         %-10s
  Dropout:                    %-10s
  Job distribution            %-10s
  Retrieve data               %-10s
-------------------------------------------------------

""" % (
        variables_search['depth'],
        variables_search['nn_size'],
        variables_search['batch_size'],
        variables_search['reg_l1'],
        variables_search['reg_l2'],
        variables_search['dropout'],
        variables_search['use_hpc'],
        variables_search['retrieve']
    )

    molcas_info = """
  &molcas
-------------------------------------------------------
  Molcas:                   %-10s
  Molcas_nproc:             %-10s
  Molcas_mem:               %-10s
  Molcas_print:      	    %-10s
  Molcas_project:      	    %-10s
  Molcas_workdir:      	    %-10s
  Molcas_calcdir:           %-10s
  Tinker interface:         %-10s
  Omp_num_threads:          %-10s
  Keep tmp_molcas:          %-10s
  Track phase:              %-10s
  Job distribution:         %-10s
-------------------------------------------------------
""" % (
        variables_molcas['molcas'],
        variables_molcas['molcas_nproc'],
        variables_molcas['molcas_mem'],
        variables_molcas['molcas_print'],
        variables_molcas['molcas_project'],
        variables_molcas['molcas_workdir'],
        variables_molcas['molcas_calcdir'],
        variables_molcas['tinker'],
        variables_molcas['omp_num_threads'],
        variables_molcas['keep_tmp'],
        variables_molcas['track_phase'],
        variables_molcas['use_hpc']
    )

    bagel_info = """
  &bagel
-------------------------------------------------------
  BAGEL:                    %-10s
  BAGEL_nproc:              %-10s
  BAGEL_project:            %-10s
  BAGEL_workdir:            %-10s
  BAGEL_archive:            %-10s
  MPI:                      %-10s
  BLAS:                     %-10s
  LAPACK:                   %-10s
  BOOST:                    %-10s
  MKL:                      %-10s
  Architecture:             %-10s
  Omp_num_threads:          %-10s
  Keep tmp_bagel:           %-10s
  Job distribution:         %-10s
-------------------------------------------------------
""" % (
        variables_bagel['bagel'],
        variables_bagel['bagel_nproc'],
        variables_bagel['bagel_project'],
        variables_bagel['bagel_workdir'],
        variables_bagel['bagel_archive'],
        variables_bagel['mpi'],
        variables_bagel['blas'],
        variables_bagel['lapack'],
        variables_bagel['boost'],
        variables_bagel['mkl'],
        variables_bagel['arch'],
        variables_bagel['omp_num_threads'],
        variables_bagel['keep_tmp'],
        variables_bagel['use_hpc']
    )

    orca_info = """
  &orca
-------------------------------------------------------
  ORCA:                     %-10s
  ORCA_project:             %-10s
  ORCA_workdir:             %-10s
  DFT type:                 %-10s
  MPI:                      %-10s
  Keep tmp_orca:            %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
        variables_orca['orca'],
        variables_orca['orca_project'],
        variables_orca['orca_workdir'],
        variables_orca['dft_type'],
        variables_orca['mpi'],
        variables_orca['keep_tmp'],
        variables_orca['use_hpc']
    )

    xtb_info = """
  &xtb
-------------------------------------------------------
  XTB:                      %-10s
  XTB_project:              %-10s
  XTB_workdir:              %-10s
  Omp_num_threads:          %-10s
  Keep tmp_xtb:             %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
        variables_xtb['xtb'],
        variables_xtb['xtb_project'],
        variables_xtb['xtb_workdir'],
        variables_xtb['xtb_nproc'],
        variables_orca['keep_tmp'],
        variables_orca['use_hpc']
    )

    info_method = {
        'nn': nn_info,
        'mlp': mlp_info,
        'schnet': sch_info,
        'e2n2': e2n2_info,
        'molcas': molcas_info,
        'mlctkr': molcas_info,
        'bagel': bagel_info,
        'orca': orca_info,
        'xtb': xtb_info,
    }

    ## unpack control variables
    jobtype = variables_all['control']['jobtype']
    qm = variables_control['qm']
    abinit = variables_control['abinit']
    qm_info = ''.join([info_method[m] for m in qm])
    ab_info = ''.join([info_method[m] for m in abinit])
    info_jobtype = {
        'sp': control_info + molecule_info + qm_info,
        'md': control_info + molecule_info + md_info + qm_info,
        'hop': control_info + molecule_info + md_info,
        'hybrid': control_info + molecule_info + md_info + qm_info + ab_info + hybrid_info,
        'adaptive': control_info + molecule_info + adaptive_info + md_info + qm_info + ab_info,
        'train': control_info + molecule_info + qm_info,
        'prediction': control_info + molecule_info + qm_info,
        'predict': control_info + molecule_info + qm_info,
        'search': control_info + molecule_info + qm_info + search_info,
    }

    log_info = info_jobtype[jobtype]

    return log_info
