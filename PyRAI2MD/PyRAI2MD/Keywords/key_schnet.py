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


class KeySchNet:

    def __init__(self, key_type='eg'):
        eg = {
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

        nac = {
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

        soc = {
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
        ## This function read variables from &sch_eg,&sch_nac,&sch_soc
        keywords = self.keywords.copy()
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
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &sch_%s' % (key, self.key_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(eg, nac, soc):
        summary = """

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
            eg['node_features'],
            nac['node_features'],
            soc['node_features'],
            eg['n_features'],
            nac['n_features'],
            soc['n_features'],
            eg['n_edges'],
            nac['n_edges'],
            soc['n_edges'],
            eg['n_filters'],
            nac['n_filters'],
            soc['n_filters'],
            eg['use_filter_bias'],
            nac['use_filter_bias'],
            soc['use_filter_bias'],
            eg['cfc_activ'],
            nac['cfc_activ'],
            soc['cfc_activ'],
            eg['n_blocks'],
            nac['n_blocks'],
            soc['n_blocks'],
            eg['n_rbf'],
            nac['n_rbf'],
            soc['n_rbf'],
            eg['maxradius'],
            nac['maxradius'],
            soc['maxradius'],
            eg['offset'],
            nac['offset'],
            soc['offset'],
            eg['sigma'],
            nac['sigma'],
            soc['sigma'],
            eg['mlp'],
            nac['mlp'],
            soc['mlp'],
            eg['use_mlp_bias'],
            nac['use_mlp_bias'],
            soc['use_mlp_bias'],
            eg['mlp_activ'],
            nac['mlp_activ'],
            soc['mlp_activ'],
            eg['use_output_bias'],
            nac['use_output_bias'],
            soc['use_output_bias'],
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
        )

        return summary
