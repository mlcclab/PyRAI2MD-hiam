#####################################################
#
# PyRAI2MD 2 module for esnnp hyperparameter
#
# Author Jingbai Li
# Mar 24 2025
#
######################################################

def set_e2n2_hyper_eg(hyp, unit, info, splits, shuffle):
    """ Generating hyperparameter dict for energy+gradient NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits
            shuffle          bool     shuffle data loader during training

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'stepwise': True,
            'epoch_step_reduction': hyp['epoch_step_reduction'],
            'learning_rate_step': hyp['learning_rate_step'],
            }
    else:
        hyp['callbacks'] = []

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = ['kcal/mol', 'kcal/mol/A']
    else:
        hyp['unit'] = ['kcal/mol', 'kal/mol/A']

    ## setup hypers
    hyp_dict = {
        'model': {
            'model_option': hyp['model'],  # model option
            'class_name': 'energy_grad',  # name of the class
            'class_module': 'scalar_grad',  # name of the model
            'model_id': 0,  # index of the model
            # Properties
            'states': info['nstate'],  # number of electronic states
            'elements': None,  # the largest number of elements
            'nedges': hyp['n_edges'],  # number of edges
            'maxradius': hyp['maxradius'],  # maximum atom-centered radius in Angstrom
            'atomic': {
                # NN architecture
                'n_features': hyp['n_features'],  # number of node features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'rbf_layers': hyp['rbf_layers'],  # number of rbf layers
                'rbf_neurons': hyp['rbf_neurons'],  # number of rbf neuron per layer
                'rbf_act': hyp['rbf_act'],  # activation in rbf net
                'rbf_act_a': hyp['rbf_act_a'],  # parameter for leakysoftplus function
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],  # normalization scheme in spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'resnet': hyp['resnet'],  # use resnet feature update
                'self_connection': hyp['self_connection'],  # compute self connection in feature convolution
                # Convolutional layer activation
                'gate': hyp['gate'],  # use gated activation or norm activation
                'act_scalars': {
                    'e': hyp['act_scalars_e'],
                    'o': hyp['act_scalars_o']
                },  # activation for scalars
                'act_gates': {
                    'e': hyp['act_gates_e'],
                    'o': hyp['act_gates_o']
                },  # activation for gated tensors
            },
            'distance': {
                # distance e2n2 architecture
                'n_features': hyp['n_features'],  # number of edge features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],
                # normalization scheme for projecting edge vectors to spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'two_body_mlp_nonlinearity': hyp['mlp_act'],
                'two_body_mlp_initialization': hyp['mlp_init'],
                'two_body_mlp_dropout': 0.0,
                'two_body_mlp_batchnorm': hyp['mlp_norm'],
                'two_body_mlp_latent_dimension': hyp['edge_neurons'],
                'latent_mlp_nonlinearity': hyp['mlp_act'],
                'latent_mlp_initialization': hyp['mlp_init'],
                'latent_mlp_dropout': 0.0,
                'latent_mlp_batchnorm': hyp['mlp_norm'],
                'latent_mlp_latent_dimension': hyp['latent_neurons'],
                'embedding_mlp_nonlinearity': hyp['mlp_act'],
                'embedding_mlp_initialization': hyp['mlp_init'],
                'embedding_mlp_dropout': 0.0,
                'embedding_mlp_batchnorm': hyp['mlp_norm'],
                'embedding_mlp_latent_dimension': hyp['embedding_neurons'],
                'output_mlp_nonlinearity': hyp['mlp_act'],
                'output_mlp_initialization': hyp['mlp_init'],
                'output_mlp_dropout': 0.0,
                'output_mlp_batchnorm': hyp['mlp_norm'],
                'output_mlp_latent_dimension': hyp['output_neurons'],
                'latent_resnet_update_ratios': hyp['resnet_ratio'],
                'latent_resnet_update_ratios_learnable': hyp['resnet_trainable'],
                'resnet': hyp['resnet'],  # use resnet feature update
            },
        },
        'training': {
            'device': 'cpu',  # training device
            'val_split': 1 / splits,  # validation training set ratio
            'initialize_weights': hyp['initialize_weights'],  # initialize weight to retrain
            'loss_weights': hyp['loss_weights'],  # weight between scalar and grad loss
            'learning_rate': hyp['learning_rate'],  # learning rate
            'epo': hyp['epo'],  # number of epoch
            'scaler': hyp['scaler'],  # scale method
            'shuffle': shuffle,  # shuffle full training set or subset
            'subset': hyp['subset'],  # ratio of train data used for training
            'batch_size': hyp['batch_size'],  # batch size
            'val_batch_size': hyp['val_batch_size'],  # batch size in validation
            'nbatch': hyp['nbatch'],  # number of batch, larger than 0 will overwrite batch size
            'epo_step': hyp['epostep'],  # steps of epochs for validation
            'callbacks': hyp['callbacks'],
            'unit_scalar': hyp['unit'][0],  # unit of scalar
            'unit_grad': hyp['unit'][1],  # unit of grad
            'grad_type': hyp['grad_type'],  # type of grad or force
        }
    }

    return hyp_dict

def set_e2n2_hyper_nac(hyp, unit, info, splits, shuffle):
    """ Generating hyperparameter dict for soc NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits
            shuffle          bool     shuffle data loader during training

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'stepwise': True,
            'epoch_step_reduction': hyp['epoch_step_reduction'],
            'learning_rate_step': hyp['learning_rate_step'],
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
            'model_option': hyp['model'],  # model option
            'class_name': 'nac',  # name of the class
            'class_module': 'grad',  # name of the model
            'model_id': 0,  # index of the model
            # Properties
            'states': info['nstate'],  # number of electronic states
            'elements': None,  # the largest number of elements
            'nedges': hyp['n_edges'],  # number of edges
            'maxradius': hyp['maxradius'],  # maximum atom-centered radius in Angstrom
            'atomic': {
                # NN architecture
                'n_features': hyp['n_features'],  # number of node features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'rbf_layers': hyp['rbf_layers'],  # number of rbf layers
                'rbf_neurons': hyp['rbf_neurons'],  # number of rbf neuron per layer
                'rbf_act': hyp['rbf_act'],  # activation in rbf net
                'rbf_act_a': hyp['rbf_act_a'],  # parameter for leakysoftplus function
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],  # normalization scheme in spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'resnet': hyp['resnet'],  # use resnet feature update
                'self_connection': hyp['self_connection'],  # compute self connection in feature convolution
                # Convolutional layer activation
                'gate': hyp['gate'],  # use gated activation or norm activation
                'act_scalars': {
                    'e': hyp['act_scalars_e'],
                    'o': hyp['act_scalars_o']
                },  # activation for scalars
                'act_gates': {
                    'e': hyp['act_gates_e'],
                    'o': hyp['act_gates_o']
                },  # activation for gated tensors
            },
            'distance': {
                # distance e2n2 architecture
                'n_features': hyp['n_features'],  # number of edge features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],
                # normalization scheme for projecting edge vectors to spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'two_body_mlp_nonlinearity': hyp['mlp_act'],
                'two_body_mlp_initialization': hyp['mlp_init'],
                'two_body_mlp_dropout': 0.0,
                'two_body_mlp_batchnorm': hyp['mlp_norm'],
                'two_body_mlp_latent_dimension': hyp['edge_neurons'],
                'latent_mlp_nonlinearity': hyp['mlp_act'],
                'latent_mlp_initialization': hyp['mlp_init'],
                'latent_mlp_dropout': 0.0,
                'latent_mlp_batchnorm': hyp['mlp_norm'],
                'latent_mlp_latent_dimension': hyp['latent_neurons'],
                'embedding_mlp_nonlinearity': hyp['mlp_act'],
                'embedding_mlp_initialization': hyp['mlp_init'],
                'embedding_mlp_dropout': 0.0,
                'embedding_mlp_batchnorm': hyp['mlp_norm'],
                'embedding_mlp_latent_dimension': hyp['embedding_neurons'],
                'output_mlp_nonlinearity': hyp['mlp_act'],
                'output_mlp_initialization': hyp['mlp_init'],
                'output_mlp_dropout': 0.0,
                'output_mlp_batchnorm': hyp['mlp_norm'],
                'output_mlp_latent_dimension': hyp['output_neurons'],
                'latent_resnet_update_ratios': hyp['resnet_ratio'],
                'latent_resnet_update_ratios_learnable': hyp['resnet_trainable'],
                'resnet': hyp['resnet'],  # use resnet feature update
            },
        },
        'training': {
            'device': 'cpu',  # training device
            'val_split': 1 / splits,  # validation training set ratio
            'initialize_weights': hyp['initialize_weights'],  # initialize weight to retrain
            'learning_rate': hyp['learning_rate'],  # learning rate
            'epo': hyp['epo'],  # number of epoch
            'scaler': hyp['scaler'],  # scale method
            'shuffle': shuffle,  # shuffle full training set or subset
            'subset': hyp['subset'],  # ratio of train data used for training
            'batch_size': hyp['batch_size'],  # batch size
            'val_batch_size': hyp['val_batch_size'],  # batch size in validation
            'nbatch': hyp['nbatch'],  # number of batch, larger than 0 will overwrite batch size
            'epo_step': hyp['epostep'],  # steps of epochs for validation
            'callbacks': hyp['callbacks'],
            'unit_scalar': '',  # unit of scalar
            'unit_grad': hyp['unit'],  # unit of grad
            'grad_type': hyp['grad_type'],  # type of grad or force
        }
    }

    return hyp_dict

def set_e2n2_hyper_soc(hyp, unit, info, splits, shuffle):
    """ Generating hyperparameter dict for soc NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information
            splits           int      train valid splits
            shuffle          bool     shuffle data loader during training

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup callbacks
    if hyp['use_step_callback']:
        hyp['callbacks'] = {
            'stepwise': True,
            'epoch_step_reduction': hyp['epoch_step_reduction'],
            'learning_rate_step': hyp['learning_rate_step'],
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
            'model_option': hyp['model'],  # model option
            'class_name': 'soc',  # name of the class
            'class_module': 'scalar',  # name of the model
            'model_id': 0,  # index of the model
            # Properties
            'states': info['nstate'],  # number of electronic states
            'elements': None,  # the largest number of elements
            'nedges': hyp['n_edges'],  # number of edges
            'edge_list': None,  # a list of pairwise atom indices in edges
            'maxradius': hyp['maxradius'],  # maximum atom-centered radius in Angstrom
            'atomic': {
                # NN architecture
                'n_features': hyp['n_features'],  # number of node features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'rbf_layers': hyp['rbf_layers'],  # number of rbf layers
                'rbf_neurons': hyp['rbf_neurons'],  # number of rbf neuron per layer
                'rbf_act': hyp['rbf_act'],  # activation in rbf net
                'rbf_act_a': hyp['rbf_act_a'],  # parameter for leakysoftplus function
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],  # normalization scheme in spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'resnet': hyp['resnet'],  # use resnet feature update
                'self_connection': hyp['self_connection'],  # compute self connection in feature convolution
                # Convolutional layer activation
                'gate': hyp['gate'],  # use gated activation or norm activation
                'act_scalars': {
                    'e': hyp['act_scalars_e'],
                    'o': hyp['act_scalars_o']
                },  # activation for scalars
                'act_gates': {
                    'e': hyp['act_gates_e'],
                    'o': hyp['act_gates_o']
                },  # activation for gated tensors
            },
            'distance': {
                # distance e2n2 architecture
                'n_features': hyp['n_features'],  # number of edge features
                'n_blocks': hyp['n_blocks'],  # number of interaction blocks
                'l_max': hyp['l_max'],  # the largest rotation order
                'parity': hyp['parity'],  # use parity or not in o3 irreps
                # Radial net
                'n_rbf': hyp['n_rbf'],  # number of radial basis function
                'trainable_rbf': hyp['trainable_rbf'],  # trainable radial basis function
                'rbf_cutoff': hyp['rbf_cutoff'],  # rbf envelop function cutoff
                'trainable_atom': hyp['trainable_atom'],  # trainable atomic scaling factor
                'normalization_y': hyp['normalization_y'],
                # normalization scheme for projecting edge vectors to spherical harmonics
                'normalize_y': hyp['normalize_y'],  # normalize edge vectors when projecting to spherical harmonics
                # Convolution
                'two_body_mlp_nonlinearity': hyp['mlp_act'],
                'two_body_mlp_initialization': hyp['mlp_init'],
                'two_body_mlp_dropout': 0.0,
                'two_body_mlp_batchnorm': hyp['mlp_norm'],
                'two_body_mlp_latent_dimension': hyp['edge_neurons'],
                'latent_mlp_nonlinearity': hyp['mlp_act'],
                'latent_mlp_initialization': hyp['mlp_init'],
                'latent_mlp_dropout': 0.0,
                'latent_mlp_batchnorm': hyp['mlp_norm'],
                'latent_mlp_latent_dimension': hyp['latent_neurons'],
                'embedding_mlp_nonlinearity': hyp['mlp_act'],
                'embedding_mlp_initialization': hyp['mlp_init'],
                'embedding_mlp_dropout': 0.0,
                'embedding_mlp_batchnorm': hyp['mlp_norm'],
                'embedding_mlp_latent_dimension': hyp['embedding_neurons'],
                'output_mlp_nonlinearity': hyp['mlp_act'],
                'output_mlp_initialization': hyp['mlp_init'],
                'output_mlp_dropout': 0.0,
                'output_mlp_batchnorm': hyp['mlp_norm'],
                'output_mlp_latent_dimension': hyp['output_neurons'],
                'latent_resnet_update_ratios': hyp['resnet_ratio'],
                'latent_resnet_update_ratios_learnable': hyp['resnet_trainable'],
                'resnet': hyp['resnet'],  # use resnet feature update
            },
        },
        'training': {
            'device': 'cpu',  # training device
            'val_split': 1 / splits,  # validation training set ratio
            'initialize_weights': hyp['initialize_weights'],  # initialize weight to retrain
            'learning_rate': hyp['learning_rate'],  # learning rate
            'epo': hyp['epo'],  # number of epoch
            'scaler': hyp['scaler'],  # scale method
            'shuffle': shuffle,  # shuffle full training set or subset
            'subset': hyp['subset'],  # ratio of train data used for training
            'batch_size': hyp['batch_size'],  # batch size
            'val_batch_size': hyp['val_batch_size'],  # batch size in validation
            'nbatch': hyp['nbatch'],  # number of batch, larger than 0 will overwrite batch size
            'epo_step': hyp['epostep'],  # steps of epochs for validation
            'callbacks': hyp['callbacks'],
            'unit_scalar': hyp['unit'],  # unit of scalar
            'unit_grad': '',  # unit of grad
            'grad_type': hyp['grad_type'],  # type of grad or force
        }
    }

    return hyp_dict
