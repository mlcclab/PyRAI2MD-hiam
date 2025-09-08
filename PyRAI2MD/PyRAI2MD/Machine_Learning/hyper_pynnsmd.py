#####################################################
#
# PyRAI2MD 2 module for pyNNsMD hyperparameter
#
# Author Jingbai Li
# Aug 31 2022
#
######################################################

import numpy as np

def set_mlp_hyper_eg(hyp, unit, info):
    """ Generating hyperparameter dict for energy+gradient NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ', 'use_reg_weight', 'use_reg_bias']:
        penalty_key = '%s_dict' % penalty
        if hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name': 'l1', 'config': {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name': 'l2', 'config': {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name': 'l1_l2', 'config': {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = ['eV', 'eV/A']
    else:
        hyp['unit'] = ['Eh', 'Eh/Bohr']

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'EnergyGradientModel',
            'config': {
                'atoms': info['natom'],
                'states': info['nstate'],
                'nn_size': hyp['nn_size'],
                'depth': hyp['depth'],
                'activ': {
                    'class_name': 'pyNNsMD>%s' % hyp['activ'],
                    'config': {
                        'alpha': hyp['activ_alpha']
                    }
                },
                'use_dropout': hyp['use_dropout'],
                'dropout': hyp['dropout'],
                'use_reg_activ': hyp['use_reg_activ_dict'],
                'use_reg_weight': hyp['use_reg_weight_dict'],
                'use_reg_bias': hyp['use_reg_bias_dict'],
                'invd_index': (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp['angle_index']) - 1).tolist(),
                'dihed_index': (np.array(hyp['dihed_index']) - 1).tolist(),
                'normalization_mode': hyp['normalization_mode'],
                'model_module': "mlp_eg"
            }
        },
        'scaler': {
            'class_name': 'EnergyGradientStandardScaler',
            'config': {
                'scaler_module': 'energy'
            }
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'energy_only': False,
            'loss_weights': hyp['loss_weights'],
            'learning_rate': hyp['learning_rate'],
            'epo': hyp['epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_energy': hyp['unit'][0],
            'unit_gradient': hyp['unit'][1],
        }
    }

    return hyp_dict

def set_mlp_hyper_nac(hyp, unit, info):
    """ Generating hyperparameter dict for nac NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ', 'use_reg_weight', 'use_reg_bias']:
        penalty_key = '%s_dict' % penalty
        if hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name': 'l1', 'config': {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name': 'l2', 'config': {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name': 'l1_l2', 'config': {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = 'eV/A'
    else:
        hyp['unit'] = 'Eh/Bohr'

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'NACModel2',
            'config': {
                'atoms': info['natom'],
                'states': info['nnac'],
                'nn_size': hyp['nn_size'],
                'depth': hyp['depth'],
                'activ': {
                    'class_name': 'pyNNsMD>%s' % hyp['activ'],
                    'config': {
                        'alpha': hyp['activ_alpha']
                    }
                },
                'use_dropout': hyp['use_dropout'],
                'dropout': hyp['dropout'],
                'use_reg_activ': hyp['use_reg_activ_dict'],
                'use_reg_weight': hyp['use_reg_weight_dict'],
                'use_reg_bias': hyp['use_reg_bias_dict'],
                'invd_index': (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp['angle_index']) - 1).tolist(),
                'dihed_index': (np.array(hyp['dihed_index']) - 1).tolist(),
                'normalization_mode': hyp['normalization_mode'],
                'model_module': "mlp_nac2"
            }
        },
        'scaler': {
            'class_name': 'NACStandardScaler',
            'config': {
                'scaler_module': 'nac'
            }
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'learning_rate': hyp['learning_rate'],
            'phase_less_loss': hyp['phase_less_loss'],
            'epo': hyp['epo'],
            'pre_epo': hyp['pre_epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_nac': hyp['unit'],
        }
    }

    return hyp_dict

def set_mlp_hyper_soc(hyp, unit, info):
    """ Generating hyperparameter dict for soc NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ', 'use_reg_weight', 'use_reg_bias']:
        penalty_key = '%s_dict' % penalty
        if hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name': 'l1', 'config': {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name': 'l2', 'config': {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name': 'l1_l2', 'config': {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = 'cm-1'
    else:
        hyp['unit'] = 'cm-1'

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'EnergyModel',
            'config': {
                'atoms': info['natom'],
                'states': info['nsoc'],
                'nn_size': hyp['nn_size'],
                'depth': hyp['depth'],
                'activ': {
                    'class_name': 'pyNNsMD>%s' % hyp['activ'],
                    'config': {
                        'alpha': hyp['activ_alpha']
                    }
                },
                'use_dropout': hyp['use_dropout'],
                'dropout': hyp['dropout'],
                'use_reg_activ': hyp['use_reg_activ_dict'],
                'use_reg_weight': hyp['use_reg_weight_dict'],
                'use_reg_bias': hyp['use_reg_bias_dict'],
                'invd_index': (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp['angle_index']) - 1).tolist(),
                'dihed_index': (np.array(hyp['dihed_index']) - 1).tolist(),
                'normalization_mode': hyp['normalization_mode'],
                'model_module': "mlp_e"
            }
        },
        'scaler': {
            'class_name': 'EnergyStandardScaler',
            'config': {
                'scaler_module': 'energy'
            }
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'energy_only': True,
            'learning_rate': hyp['learning_rate'],
            'epo': hyp['epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_energy': hyp['unit'],
            'unit_gradient': '',
        }
    }

    return hyp_dict

def set_sch_hyper_eg(hyp, unit, info):
    """ Generating hyperparameter dict for energy+gradient NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = ['eV', 'eV/A']
    else:
        hyp['unit'] = ['Eh', 'Eh/Bohr']

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'SchNetEnergy',
            'config': {
                'model_module': "schnet_eg",
                'output_as_dict': False,
                'energy_only': False,
                'name': 'Schnet',
                'inputs': [{'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32'},
                           {'shape': (None,), 'name': "node_number", 'dtype': 'int64'},
                           {'shape': (None, None), 'name': "edge_indices", 'dtype': 'int64'},
                           {'shape': (None, 1), 'name': "mask_node", 'dtype': 'bool'},
                           {'shape': (None, None, 1), 'name': "mask_edge", 'dtype': 'bool'}],
                'input_embedding': {
                    'node': {
                        'input_dim': hyp['node_features'],
                        'output_dim': hyp['n_features'],
                    }
                },
                'interaction_args': {
                    'units': hyp['n_filters'],
                    'use_bias': hyp['use_filter_bias'],
                    'activation': 'kgcnn>%s' % hyp['cfc_activ']
                },
                'node_pooling_args': {"pooling_method": "sum"},
                'depth': hyp['n_blocks'],
                'gauss_args': {
                    'bins': hyp['n_rbf'],
                    'distance': hyp['maxradius'],
                    'offset': hyp['offset'],
                    'sigma': hyp['sigma'],
                },
                'verbose': 10,
                'max_neighbours': hyp['n_edges'],
                'last_mlp': {
                    "use_bias": [hyp['use_mlp_bias'] for _ in range(len(hyp['mlp']))],
                    "units": hyp['mlp'],
                    "activation": ['kgcnn>%s' % hyp['mlp_activ'] for _ in range(len(hyp['mlp']))]
                },
                "use_output_mlp": True,
                'output_mlp': {
                    "use_bias": [hyp['use_output_bias']],
                    "units": [info['nstate']],
                    "activation": ['linear']
                }
            }
        },
        'scaler': {
            'class_name': 'EnergyGradientStandardScaler',
            'config': {
                'scaler_module': 'energy',
            },
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'loss_weights': hyp['loss_weights'],
            'learning_rate': hyp['learning_rate'],
            'epo': hyp['epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_energy': hyp['unit'][0],
            'unit_gradient': hyp['unit'][1],
        },
    }

    return hyp_dict

def set_sch_hyper_nac(hyp, unit, info):
    """ Generating hyperparameter dict for nac NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = 'eV/A'
    else:
        hyp['unit'] = 'Eh/Bohr'

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'N/A',
            'config': {
                'model_module': "n/a",
                'output_as_dict': False,
                'energy_only': False,
                'name': 'Schnet',
                'inputs': [{'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32'},
                           {'shape': (None,), 'name': "node_number", 'dtype': 'int64'},
                           {'shape': (None, None), 'name': "edge_indices", 'dtype': 'int64'},
                           {'shape': (None, 1), 'name': "mask_node", 'dtype': 'bool'},
                           {'shape': (None, None, 1), 'name': "mask_edge", 'dtype': 'bool'}],
                'input_embedding': {
                    'node': {
                        'input_dim': hyp['node_features'],
                        'output_dim': hyp['n_features'],
                    }
                },
                'interaction_args': {
                    'units': hyp['n_filters'],
                    'use_bias': hyp['use_filter_bias'],
                    'activation': 'kgcnn>%s' % hyp['cfc_activ']
                },
                'node_pooling_args': {"pooling_method": "sum"},
                'depth': hyp['n_blocks'],
                'gauss_args': {
                    'bins': hyp['n_rbf'],
                    'distance': hyp['maxradius'],
                    'offset': hyp['offset'],
                    'sigma': hyp['sigma'],
                },
                'verbose': 10,
                'max_neighbours': hyp['n_edges'],
                'last_mlp': {
                    "use_bias": [hyp['use_mlp_bias'] for _ in range(len(hyp['mlp']))],
                    "units": hyp['mlp'],
                    "activation": ['kgcnn>%s' % hyp['mlp_activ'] for _ in range(len(hyp['mlp']))]
                },
                "use_output_mlp": True,
                'output_mlp': {
                    "use_bias": [hyp['use_output_bias']],
                    "units": [info['nstate']],
                    "activation": ['linear']
                }
            }
        },

        'scaler': {
            'class_name': 'NACStandardScaler',
            'config': {
                'scaler_module': 'nac',
            },
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'learning_rate': hyp['learning_rate'],
            'epo': hyp['epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_nac': hyp['unit'],
        },
    }

    return hyp_dict

def set_sch_hyper_soc(hyp, unit, info):
    """ Generating hyperparameter dict for soc NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>StepWiseLearningScheduler',
            'config': {
                'epoch_step_reduction': hyp['epoch_step_reduction'],
                'learning_rate_step': hyp['learning_rate_step'],
            }
        }

    elif hyp['use_linear_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearLearningRateScheduler',
            'config': {
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epomin': hyp['epomin'],
                'epo': hyp['epo']
            }
        }

    elif hyp['use_early_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>EarlyStopping',
            'config': {
                'use': hyp['use_early_callback'],
                'epomin': hyp['epomin'],
                'patience': hyp['patience'],
                'max_time': hyp['max_time'],
                'delta_loss': hyp['delta_loss'],
                'loss_monitor': hyp['loss_monitor'],
                'factor_lr': hyp['factor_lr'],
                'learning_rate_start': hyp['learning_rate_start'],
                'learning_rate_stop': hyp['learning_rate_stop'],
                'epostep': 1
            }
        }

    elif hyp['use_exp_callback']:
        hyp['callbacks'] = {
            'class_name': 'pyNNsMD>LinearWarmupExponentialLearningRateScheduler',
            'config': {
                'epo_warmup': hyp['epomin'],
                'decay_gamma': hyp['factor_lr'],
                'lr_start': hyp['learning_rate_start'],
                'ls_min': hyp['learning_rate_stop'],
            }
        }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = 'cm-1'
    else:
        hyp['unit'] = 'cm-1'

    ## setup hypers
    hyp_dict = {
        'model': {
            'class_name': 'SchNetEnergy',
            'config': {
                'model_module': "schnet_eg",
                'output_as_dict': False,
                'energy_only': True,
                'name': 'Schnet',
                'inputs': [{'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32'},
                           {'shape': (None,), 'name': "node_number", 'dtype': 'int64'},
                           {'shape': (None, None), 'name': "edge_indices", 'dtype': 'int64'},
                           {'shape': (None, 1), 'name': "mask_node", 'dtype': 'bool'},
                           {'shape': (None, None, 1), 'name': "mask_edge", 'dtype': 'bool'}],
                'input_embedding': {
                    'node': {
                        'input_dim': hyp['node_features'],
                        'output_dim': hyp['n_features'],
                    }
                },
                'interaction_args': {
                    'units': hyp['n_filters'],
                    'use_bias': hyp['use_filter_bias'],
                    'activation': 'kgcnn>%s' % hyp['cfc_activ']
                },
                'node_pooling_args': {"pooling_method": "sum"},
                'depth': hyp['n_blocks'],
                'gauss_args': {
                    'bins': hyp['n_rbf'],
                    'distance': hyp['maxradius'],
                    'offset': hyp['offset'],
                    'sigma': hyp['sigma'],
                },
                'verbose': 10,
                'max_neighbours': hyp['n_edges'],
                'last_mlp': {
                    "use_bias": [hyp['use_mlp_bias'] for _ in range(len(hyp['mlp']))],
                    "units": hyp['mlp'],
                    "activation": ['kgcnn>%s' % hyp['mlp_activ'] for _ in range(len(hyp['mlp']))]
                },
                "use_output_mlp": True,
                'output_mlp': {
                    "use_bias": [hyp['use_output_bias']],
                    "units": [info['nsoc']],
                    "activation": ['linear']
                }
            }
        },
        'scaler': {
            'class_name': 'EnergyStandardScaler',
            'config': {
                'scaler_module': 'energy',
            }
        },
        'training': {
            'initialize_weights': hyp['initialize_weights'],
            'learning_rate': hyp['learning_rate'],
            'epo': hyp['epo'],
            'epostep': hyp['epostep'],
            'batch_size': hyp['batch_size'],
            'callbacks': hyp['callbacks'],
            'unit_energy': hyp['unit'],
            'unit_gradient': '',
        }
    }

    return hyp_dict
