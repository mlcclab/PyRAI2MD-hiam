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


class KeyE2N2Demo:

    def __init__(self, key_type='eg'):
        eg = {
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
            'val_batch_size': 0,
            'nbatch': 0,
            'learning_rate': 1e-3,
            'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
            'epoch_step_reduction': [100, 100, 100, 100],
            'scaler': 'total_energy_mean_std',
            'grad_type': 'grad',
        }

        nac = {
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
            'val_batch_size': 0,
            'nbatch': 0,
            'learning_rate': 1e-3,
            'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
            'epoch_step_reduction': [100, 100, 100, 100],
            'scaler': 'no',
            'grad_type': 'grad',
        }

        soc = {
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
            'val_batch_size': 0,
            'nbatch': 0,
            'learning_rate': 1e-3,
            'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6],
            'epoch_step_reduction': [100, 100, 100, 100],
            'scaler': 'total_energy_mean_std',
            'grad_type': 'grad',
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
        ## This function read variables from &e2n2_eg,&e2n2_nac,&e2n2_soc
        keywords = self.keywords.copy()
        keyfunc = {
            'n_edges': ReadVal('i'),
            'maxradius': ReadVal('f'),
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
            'val_batch_size': ReadVal('i'),
            'nbatch': ReadVal('i'),
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
                sys.exit(
                    '\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &e2n2_%s' % (key, self.key_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(eg, nac, soc):
        summary = """

  E2N2 (esnnp)

  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  Edges:                      %-20s %-20s %-20s
  Maxradius:                  %-20s %-20s %-20s
  Node features:              %-20s %-20s %-20s 
  Interaction blocks:         %-20s %-20s %-20s
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
  Validation batch            %-20s %-20s %-20s
  Nbatch:                     %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

    """ % (
            eg['n_edges'],
            nac['n_edges'],
            soc['n_edges'],
            eg['maxradius'],
            nac['maxradius'],
            soc['maxradius'],
            eg['n_features'],
            nac['n_features'],
            soc['n_features'],
            eg['n_blocks'],
            nac['n_blocks'],
            soc['n_blocks'],
            eg['l_max'],
            nac['l_max'],
            soc['l_max'],
            eg['parity'],
            nac['parity'],
            soc['parity'],
            eg['n_rbf'],
            nac['n_rbf'],
            soc['n_rbf'],
            eg['trainable_rbf'],
            nac['trainable_rbf'],
            soc['trainable_rbf'],
            eg['rbf_cutoff'],
            nac['rbf_cutoff'],
            soc['rbf_cutoff'],
            eg['rbf_layers'],
            nac['rbf_layers'],
            soc['rbf_layers'],
            eg['rbf_neurons'],
            nac['rbf_neurons'],
            soc['rbf_neurons'],
            eg['rbf_act'],
            nac['rbf_act'],
            soc['rbf_act'],
            eg['rbf_act_a'],
            nac['rbf_act_a'],
            soc['rbf_act_a'],
            eg['normalization_y'],
            nac['normalization_y'],
            soc['normalization_y'],
            eg['normalize_y'],
            nac['normalize_y'],
            soc['normalize_y'],
            eg['self_connection'],
            nac['self_connection'],
            soc['self_connection'],
            eg['resnet'],
            nac['resnet'],
            soc['resnet'],
            eg['gate'],
            nac['gate'],
            soc['gate'],
            eg['act_scalars_e'],
            nac['act_scalars_e'],
            soc['act_scalars_e'],
            eg['act_scalars_o'],
            nac['act_scalars_o'],
            soc['act_scalars_o'],
            eg['act_gates_e'],
            nac['act_gates_e'],
            soc['act_gates_e'],
            eg['act_gates_o'],
            nac['act_gates_o'],
            soc['act_gates_o'],
            eg['initialize_weights'],
            nac['initialize_weights'],
            soc['initialize_weights'],
            eg['loss_weights'],
            '',
            '',
            eg['epo'],
            nac['epo'],
            soc['epo'],
            eg['epostep'],
            nac['epostep'],
            soc['epostep'],
            eg['subset'],
            nac['subset'],
            soc['subset'],
            eg['scaler'],
            nac['scaler'],
            soc['scaler'],
            eg['batch_size'],
            nac['batch_size'],
            soc['batch_size'],
            eg['val_batch_size'],
            nac['val_batch_size'],
            soc['val_batch_size'],
            eg['nbatch'],
            nac['nbatch'],
            soc['nbatch'],
        )

        return summary
