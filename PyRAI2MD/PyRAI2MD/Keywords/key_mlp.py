######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 20 2023
#
######################################################

import sys
from PyRAI2MD.Utils.read_tools import ReadVal
from PyRAI2MD.Utils.read_tools import ReadIndex


class KeyMLP:

    def __init__(self, key_type='eg'):
        eg = {
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

        nac = {
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

        soc = {
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

        keywords = {
            'eg': eg,
            'nac': nac,
            'soc': soc,
        }

        self.keywords = keywords[key_type]
        self.key_type = key_type

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &eg1,&eg2,&nac1,&nac2,&soc,&soc2
        keywords = self.keywords.copy()
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
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &%s' % (key, self.key_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(eg, nac, soc, idx, nn_type='native'):
        if idx == 1:
            title = '\n\n  multilayer perceptron (%s)' % nn_type
        else:
            title = ''
        summary = """%s
  
  &hyperparameters(%s)         Energy+Gradient      Nonadiabatic         Spin-orbit
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
  Shift_X:     	       	      %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % (
            title,
            idx,
            len(eg['invd_index']),
            len(nac['invd_index']),
            len(soc['invd_index']),
            len(eg['angle_index']),
            len(nac['angle_index']),
            len(soc['angle_index']),
            len(eg['dihed_index']),
            len(nac['dihed_index']),
            len(soc['dihed_index']),
            eg['activ'],
            nac['activ'],
            soc['activ'],
            eg['activ_alpha'],
            nac['activ_alpha'],
            soc['activ_alpha'],
            eg['depth'],
            nac['depth'],
            soc['depth'],
            eg['nn_size'],
            nac['nn_size'],
            soc['nn_size'],
            eg['use_dropout'],
            nac['use_dropout'],
            soc['use_dropout'],
            eg['dropout'],
            nac['dropout'],
            soc['dropout'],
            eg['use_reg_activ'],
            nac['use_reg_activ'],
            soc['use_reg_activ'],
            eg['use_reg_weight'],
            nac['use_reg_weight'],
            soc['use_reg_weight'],
            eg['use_reg_bias'],
            nac['use_reg_bias'],
            soc['use_reg_bias'],
            eg['reg_l1'],
            nac['reg_l1'],
            soc['reg_l1'],
            eg['reg_l2'],
            nac['reg_l2'],
            soc['reg_l2'],
            eg['loss_weights'],
            '',
            '',
            '',
            nac['phase_less_loss'],
            '',
            eg['initialize_weights'],
            nac['initialize_weights'],
            soc['initialize_weights'],
            eg['epo'],
            nac['epo'],
            soc['epo'],
            '',
            nac['pre_epo'],
            '',
            eg['epomin'],
            nac['epomin'],
            soc['epomin'],
            eg['patience'],
            nac['patience'],
            soc['patience'],
            eg['max_time'],
            nac['max_time'],
            soc['max_time'],
            eg['epostep'],
            nac['epostep'],
            soc['epostep'],
            eg['batch_size'],
            nac['batch_size'],
            soc['batch_size'],
            eg['delta_loss'],
            nac['delta_loss'],
            soc['delta_loss'],
            eg['scale_x_mean'],
            nac['scale_x_mean'],
            soc['scale_x_mean'],
            eg['scale_x_std'],
            nac['scale_x_std'],
            soc['scale_x_std'],
            eg['scale_y_mean'],
            nac['scale_y_mean'],
            soc['scale_y_mean'],
            eg['scale_y_std'],
            nac['scale_y_std'],
            soc['scale_y_std'],
            eg['normalization_mode'],
            nac['normalization_mode'],
            soc['normalization_mode']
        )

        return summary
