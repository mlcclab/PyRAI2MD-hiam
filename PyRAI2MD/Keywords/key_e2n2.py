######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Mar 20 2024
#
######################################################

import sys
from PyRAI2MD.Utils.read_tools import ReadVal


class KeyE2N2:

    def __init__(self, key_type='eg'):
        eg = {
            'model': 'distance',
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
            'trainable_atom': False,
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
            'mlp_act': 'silu',
            'mlp_init': 'uniform',
            'mlp_norm': False,
            'edge_neurons': [64, 128, 64],
            'latent_neurons': [64, 64],
            'embedding_neurons': [],
            'output_neurons': [32],
            'resnet_ratio': 0.0,
            'resnet_trainable': False,
            'loss_weights': [10, 1],
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
            'model': 'distance',
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
            'trainable_atom': False,
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
            'mlp_act': 'silu',
            'mlp_init': 'uniform',
            'mlp_norm': False,
            'edge_neurons': [64, 128, 64],
            'latent_neurons': [64, 64],
            'embedding_neurons': [],
            'output_neurons': [32],
            'resnet_ratio': 0.0,
            'resnet_trainable': False,
            'loss_weights': 'N/A',
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
            'model': 'distance',
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
            'trainable_atom': False,
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
            'mlp_act': 'silu',
            'mlp_init': 'uniform',
            'mlp_norm': False,
            'edge_neurons': [64, 128, 64],
            'latent_neurons': [64, 64],
            'embedding_neurons': [],
            'output_neurons': [32],
            'resnet_ratio': 0.0,
            'resnet_trainable': False,
            'loss_weights': 'N/A',
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
            'model': ReadVal('s'),
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
            'trainable_atom': ReadVal('b'),
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
            'mlp_act': ReadVal('s'),
            'mlp_init': ReadVal('s'),
            'mlp_norm': ReadVal('b'),
            'edge_neurons': ReadVal('il'),
            'latent_neurons': ReadVal('il'),
            'embedding_neurons': ReadVal('il'),
            'output_neurons': ReadVal('il'),
            'resnet_ratio': ReadVal('f'),
            'resnet_trainable': ReadVal('b'),
            'loss_weights': ReadVal('fl'),
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
        labels = ['Energy+Gradient', 'Nonadiabatic', 'Spin-orbit']
        summary = ''
        for n, key in enumerate((eg, nac, soc)):
            model = key['model']
            label = labels[n]
            if model == 'atomic':
                summary += """

  ESNNP (atomic version)

  &hyperparameters            %s
----------------------------------------------------------------------------------------------
  Edges:                      %-20s
  Maxradius:                  %-20s
  Node features:              %-20s 
  Interaction blocks:         %-20s
  Rotation order              %-20s
  Irreps parity:              %-20s
  Radial basis:               %-20s
  Radial basis trainable:     %-20s
  Envelop func cutoff:        %-20s
  Radial net layers:          %-20s
  Radial net neurons:         %-20s
  Radial net activation:      %-20s
  Radial net activation a:    %-20s
  Atomic scaling trainable:   %-20s
  Y normalization scheme:     %-20s
  Normalize Y:                %-20s
  Self connection:            %-20s
  Resnet update:              %-20s
  Use gate activation:        %-20s
  Even scalars activation:    %-20s
  Odd scalars activation:     %-20s
  Even gates activation:      %-20s
  Odd gates activation:       %-20s
  Initialize weight:          %-20s
  Loss weights:               %-20s
  Epoch:                      %-20s
  Epoch step:                 %-20s
  Subset:                     %-20s
  Scaler:                     %-20s
  Batch:                      %-20s
  Validation batch            %-20s
  Nbatch:                     %-20s
----------------------------------------------------------------------------------------------

    """ % (
                    label,
                    key['n_edges'],
                    key['maxradius'],
                    key['n_features'],
                    key['n_blocks'],
                    key['l_max'],
                    key['parity'],
                    key['n_rbf'],
                    key['trainable_rbf'],
                    key['rbf_cutoff'],
                    key['rbf_layers'],
                    key['rbf_neurons'],
                    key['rbf_act'],
                    key['rbf_act_a'],
                    key['trainable_atom'],
                    key['normalization_y'],
                    key['normalize_y'],
                    key['self_connection'],
                    key['resnet'],
                    key['gate'],
                    key['act_scalars_e'],
                    key['act_scalars_o'],
                    key['act_gates_e'],
                    key['act_gates_o'],
                    key['initialize_weights'],
                    key['loss_weights'],
                    key['epo'],
                    key['epostep'],
                    key['subset'],
                    key['scaler'],
                    key['batch_size'],
                    key['val_batch_size'],
                    key['nbatch'],
                )
            elif model == 'distance':
                summary += """

  ESNNP (distance version)

  &hyperparameters            %s
----------------------------------------------------------------------------------------------
  Edges:                      %-20s
  Maxradius:                  %-20s
  Edge features:              %-20s 
  Interaction blocks:         %-20s
  Rotation order              %-20s
  Irreps parity:              %-20s
  Radial basis:               %-20s
  Radial basis trainable:     %-20s
  Envelop func cutoff:        %-20s
  Y normalization scheme:     %-20s
  Normalize Y:                %-20s
  MLP_activation:             %-20s
  MLP initialization:         %-20s
  MLP batch normalization:    %-20s
  Edge neurons:               %-20s
  Latent neurons:             %-20s
  Embedding neurons:          %-20s
  Output neurons:             %-20s
  Resnet update:              %-20s
  Resnet ratio                %-20s
  Resnet trainable            %-20s
  Loss weights:               %-20s
  Epoch:                      %-20s
  Epoch step:                 %-20s
  Subset:                     %-20s
  Scaler:                     %-20s
  Batch:                      %-20s
  Validation batch            %-20s
  Nbatch:                     %-20s
----------------------------------------------------------------------------------------------

    """ % (
                    label,
                    key['n_edges'],
                    key['maxradius'],
                    key['n_features'],
                    key['n_blocks'],
                    key['l_max'],
                    key['parity'],
                    key['n_rbf'],
                    key['trainable_rbf'],
                    key['rbf_cutoff'],
                    key['normalization_y'],
                    key['normalize_y'],
                    key['mlp_act'],
                    key['mlp_init'],
                    key['mlp_norm'],
                    key['edge_neurons'],
                    key['latent_neurons'],
                    key['embedding_neurons'],
                    key['output_neurons'],
                    key['resnet'],
                    key['resnet_ratio'],
                    key['resnet_trainable'],
                    key['loss_weights'],
                    key['epo'],
                    key['epostep'],
                    key['subset'],
                    key['scaler'],
                    key['batch_size'],
                    key['val_batch_size'],
                    key['nbatch'],
                )

        return summary
