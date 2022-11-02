#####################################################
#
# PyRAI2MD 2 module for interfacing to pyNNsMD (demo)
#
# Author Jingbai Li
# Jul 30 2021
#
######################################################

import time
import numpy as np

from PyRAI2MD.Machine_Learning.permutation import permute_map
from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes import NeuralNetPes
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.device import set_gpu


class Demo:
    """
        data      : dict
                    All data from training data
        pred_data : str
                    Filename for test set
        hyp_eg/nac: dict
                  : Hyperparameters for NNs
        x         : np.ndarray
                    Inverse distance in shape of (batch,(atom*atom-1)/2)
        y_dict    : dict
                    Dictionary of y values for each model. Energy in Bohr, Gradients in Hartree/Bohr. Nac are unchanged.
    """
    def __init__(self, keywords=None, job_id=None, runtype='qm'):
        ## unpack variables
        self.runtype = runtype
        self.jobtype = keywords['control']['jobtype']
        set_gpu([])  # No GPU for prediction
        title = keywords['control']['title']
        variables = keywords['nn']
        modeldir = variables['modeldir']
        data = variables['data']
        nn_eg_type = variables['nn_eg_type']
        nn_nac_type = variables['nn_nac_type']
        hyp_eg = variables['eg'].copy()
        hyp_nac = variables['nac'].copy()
        hyp_eg2 = variables['eg2'].copy()
        hyp_nac2 = variables['nac2'].copy()
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        splits = variables['nsplits']
        permute = variables['permute_map']
        gpu = variables['gpu']

        ## setup l1 l2 dict
        for model_dict in [hyp_eg, hyp_nac, hyp_eg2, hyp_nac2]:
            for penalty in ['use_reg_activ', 'use_reg_weight', 'use_reg_bias']:
                penalty_key = '%s_dict' % penalty
                if model_dict[penalty] == 'l1':
                    model_dict[penalty_key] = {
                        'class_name': 'l1',
                        'config': {
                            'l1': model_dict['reg_l1'],
                        },
                    }
                elif model_dict[penalty] == 'l2':
                    model_dict[penalty_key] = {
                        'class_name': 'l2',
                        'config': {
                            'l2': model_dict['reg_l2'],
                        },
                    }
                elif model_dict[penalty] == 'l1_l2':
                    model_dict[penalty_key] = {
                        'class_name': 'l1_l2',
                        'config': {
                            'l1': model_dict['reg_l1'],
                            'l2': model_dict['reg_l2'],
                        },
                    }
                else:
                    model_dict[penalty_key] = None

        ## setup unit scheme
        if variables['eg_unit'] == 'si':
            hyp_eg['unit'] = ['eV', 'eV/A']
            hyp_eg2['unit'] = ['eV', 'eV/A']
        else:
            hyp_eg['unit'] = ['Eh', 'Eh/Bohr']
            hyp_eg2['unit'] = ['Eh', 'Eh/Bohr']
        if variables['nac_unit'] == 'si':
            hyp_nac['unit'] = 'eV/A'
            hyp_nac2['unit'] = 'eV/A'
        elif variables['nac_unit'] == 'au':
            hyp_nac['unit'] = 'Eh/A'
            hyp_nac2['unit'] = 'Eh/A'
        elif variables['nac_unit'] == 'eha':
            hyp_nac['unit'] = 'Eh/Bohr'
            hyp_nac2['unit'] = 'Eh/Bohr'

        ## setup hypers
        hyp_dict_eg = {
            'general': {
                'model_type': 'mlp_eg',
            },
            'model': {
                'atoms': data.natom,
                'states': data.nstate,
                'nn_size': hyp_eg['nn_size'],
                'depth': hyp_eg['depth'],
                'activ': {
                    'class_name': hyp_eg['activ'],
                    'config': {
                        'alpha': hyp_eg['activ_alpha'],
                    },
                },
                'use_dropout': hyp_eg['use_dropout'],
                'dropout': hyp_eg['dropout'],
                'use_reg_activ': hyp_eg['use_reg_activ_dict'],
                'use_reg_weight': hyp_eg['use_reg_weight_dict'],
                'use_reg_bias': hyp_eg['use_reg_bias_dict'],
                'invd_index': (np.array(hyp_eg['invd_index']) - 1).tolist() if len(hyp_eg['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp_eg['angle_index']) - 1).tolist(),
                'dihyd_index': (np.array(hyp_eg['dihed_index']) - 1).tolist(),
            },
            'training': {
                'auto_scaling': {
                    'x_mean': hyp_eg['scale_x_mean'],
                    'x_std': hyp_eg['scale_x_std'],
                    'energy_mean': hyp_eg['scale_y_mean'],
                    'energy_std': hyp_eg['scale_y_std'],
                },

                'normalization_mode': hyp_eg['normalization_mode'],
                'loss_weights': hyp_eg['loss_weights'],
                'learning_rate': hyp_eg['learning_rate'],
                'initialize_weights': hyp_eg['initialize_weights'],
                'val_disjoint': True,
                'val_split': 1/splits,
                'epo': hyp_eg['epo'],
                'batch_size': hyp_eg['batch_size'],
                'epostep': hyp_eg['epostep'],
                'step_callback': {
                    'use': hyp_eg['use_step_callback'],
                    'epoch_step_reduction': hyp_eg['epoch_step_reduction'],
                    'learning_rate_step': hyp_eg['learning_rate_step'],
                },
                'linear_callback': {
                    'use': hyp_eg['use_linear_callback'],
                    'learning_rate_start': hyp_eg['learning_rate_start'],
                    'learning_rate_stop': hyp_eg['learning_rate_stop'],
                    'epomin': hyp_eg['epomin'],
                },
                'early_callback': {
                    'use': hyp_eg['use_early_callback'],
                    'epomin': hyp_eg['epomin'],
                    'patience': hyp_eg['patience'],
                    'max_time': hyp_eg['max_time'],
                    'delta_loss': hyp_eg['delta_loss'],
                    'loss_monitor': hyp_eg['loss_monitor'],
                    'factor_lr': hyp_eg['factor_lr'],
                    'learning_rate_start': hyp_eg['learning_rate_start'],
                    'learning_rate_stop': hyp_eg['learning_rate_stop'],
                },
                'exp_callback': {
                    'use': hyp_eg['use_exp_callback'],
                    'epomin': hyp_eg['epomin'],
                    'factor_lr': hyp_eg['factor_lr'],
                },
            },
            'plots': {
                'unit_energy': hyp_eg['unit'][0],
                'unit_gradient': hyp_eg['unit'][1],
            },
        }

        hyp_dict_nac = {
            'general': {
                'model_type': 'mlp_nac',
            },
            'model': {
                'atoms': data.natom,
                'states': data.nstate,
                'nn_size': hyp_nac['nn_size'],
                'depth': hyp_nac['depth'],
                'activ': {
                    'class_name': hyp_nac['activ'],
                    'config': {
                        'alpha': hyp_nac['activ_alpha'],
                    },
                },
                'use_dropout': hyp_nac['use_dropout'],
                'dropout': hyp_nac['dropout'],
                'use_reg_activ': hyp_nac['use_reg_activ_dict'],
                'use_reg_weight': hyp_nac['use_reg_weight_dict'],
                'use_reg_bias': hyp_nac['use_reg_bias_dict'],
                'invd_index': (np.array(hyp_nac['invd_index']) - 1).tolist() if len(hyp_nac['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp_nac['angle_index']) - 1).tolist(),
                'dihyd_index': (np.array(hyp_nac['dihed_index']) - 1).tolist(),
            },
            'training': {
                'auto_scaling': {
                    'x_mean': hyp_nac['scale_x_mean'],
                    'x_std': hyp_nac['scale_x_std'],
                    'nac_mean': hyp_nac['scale_y_mean'],
                    'nac_std': hyp_nac['scale_y_std'],
                },
                'normalization_mode': hyp_nac['normalization_mode'],
                'learning_rate': hyp_nac['learning_rate'],
                'phase_less_loss': hyp_nac['phase_less_loss'],
                'initialize_weights': hyp_nac['initialize_weights'],
                'val_disjoint': True,
                'val_split': 1/splits,
                'epo': hyp_nac['epo'],
                'pre_epo': hyp_nac['pre_epo'],
                'batch_size': hyp_nac['batch_size'],
                'epostep': hyp_nac['epostep'],
                'step_callback': {
                    'use': hyp_nac['use_step_callback'],
                    'epoch_step_reduction': hyp_nac['epoch_step_reduction'],
                    'learning_rate_step': hyp_nac['learning_rate_step'],
                },
                'linear_callback': {
                    'use': hyp_nac['use_linear_callback'],
                    'learning_rate_start': hyp_nac['learning_rate_start'],
                    'learning_rate_stop': hyp_nac['learning_rate_stop'],
                    'epomin': hyp_nac['epomin'],
                },
                'early_callback': {
                    'use': hyp_nac['use_early_callback'],
                    'epomin': hyp_nac['epomin'],
                    'patience': hyp_nac['patience'],
                    'max_time': hyp_nac['max_time'],
                    'delta_loss': hyp_nac['delta_loss'],
                    'loss_monitor': hyp_nac['loss_monitor'],
                    'factor_lr': hyp_nac['factor_lr'],
                    'learning_rate_start': hyp_nac['learning_rate_start'],
                    'learning_rate_stop': hyp_nac['learning_rate_stop'],
                },
                'exp_callback': {
                    'use': hyp_nac['use_exp_callback'],
                    'epomin': hyp_nac['epomin'],
                    'factor_lr': hyp_nac['factor_lr'],
                },
            },
            'plots': {
                'unit_nac': hyp_nac['unit'],
            },
        }

        hyp_dict_eg2 = {
            'general': {
                'model_type': 'mlp_eg',
            },
            'model': {
                'atoms': data.natom,
                'states': data.nstate,
                'nn_size': hyp_eg2['nn_size'],
                'depth': hyp_eg2['depth'],
                'activ': {
                    'class_name': hyp_eg2['activ'],
                    'config': {
                        'alpha': hyp_eg2['activ_alpha'],
                    },
                },
                'use_dropout': hyp_eg2['use_dropout'],
                'dropout': hyp_eg2['dropout'],
                'use_reg_activ': hyp_eg2['use_reg_activ_dict'],
                'use_reg_weight': hyp_eg2['use_reg_weight_dict'],
                'use_reg_bias': hyp_eg2['use_reg_bias_dict'],
                'invd_index': (np.array(hyp_eg2['invd_index']) - 1).tolist() if len(hyp_eg2['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp_eg2['angle_index']) - 1).tolist(),
                'dihyd_index': (np.array(hyp_eg2['dihed_index']) - 1).tolist(),
            },
            'training': {
                'auto_scaling': {
                    'x_mean': hyp_eg2['scale_x_mean'],
                    'x_std': hyp_eg2['scale_x_std'],
                    'energy_mean': hyp_eg2['scale_y_mean'],
                    'energy_std': hyp_eg2['scale_y_std'],
                },
                'normalization_mode': hyp_eg2['normalization_mode'],
                'loss_weights': hyp_eg2['loss_weights'],
                'learning_rate': hyp_eg2['learning_rate'],
                'initialize_weights': hyp_eg2['initialize_weights'],
                'val_disjoint': True,
                'val_split': 1/splits,
                'epo': hyp_eg2['epo'],
                'batch_size': hyp_eg2['batch_size'],
                'epostep': hyp_eg2['epostep'],
                'step_callback': {
                    'use': hyp_eg2['use_step_callback'],
                    'epoch_step_reduction': hyp_eg2['epoch_step_reduction'],
                    'learning_rate_step': hyp_eg2['learning_rate_step'],
                },
                'linear_callback': {
                    'use': hyp_eg2['use_linear_callback'],
                    'learning_rate_start': hyp_eg2['learning_rate_start'],
                    'learning_rate_stop': hyp_eg2['learning_rate_stop'],
                    'epomin': hyp_eg2['epomin'],
                },
                'early_callback': {
                    'use': hyp_eg2['use_early_callback'],
                    'epomin': hyp_eg2['epomin'],
                    'patience': hyp_eg2['patience'],
                    'max_time': hyp_eg2['max_time'],
                    'delta_loss': hyp_eg2['delta_loss'],
                    'loss_monitor': hyp_eg2['loss_monitor'],
                    'factor_lr': hyp_eg2['factor_lr'],
                    'learning_rate_start': hyp_eg2['learning_rate_start'],
                    'learning_rate_stop': hyp_eg2['learning_rate_stop'],
                },
                'exp_callback': {
                    'use': hyp_eg2['use_exp_callback'],
                    'epomin': hyp_eg2['epomin'],
                    'factor_lr': hyp_eg2['factor_lr'],
                },
            },
            'plots': {
                'unit_energy': hyp_eg2['unit'][0],
                'unit_gradient': hyp_eg2['unit'][1],
            },
        }

        hyp_dict_nac2 = {
            'general': {
                'model_type': 'mlp_nac',
            },
            'model': {
                'atoms': data.natom,
                'states': data.nstate,
                'nn_size': hyp_nac2['nn_size'],
                'depth': hyp_nac2['depth'],
                'activ': {
                    'class_name': hyp_nac2['activ'],
                    'config': {
                        'alpha': hyp_nac2['activ_alpha'],
                    },
                },
                'use_dropout': hyp_nac2['use_dropout'],
                'dropout': hyp_nac2['dropout'],
                'use_reg_activ': hyp_nac2['use_reg_activ_dict'],
                'use_reg_weight': hyp_nac2['use_reg_weight_dict'],
                'use_reg_bias': hyp_nac2['use_reg_bias_dict'],
                'invd_index': (np.array(hyp_nac2['invd_index']) - 1).tolist() if len(hyp_nac2['invd_index']) > 0 else True,
                'angle_index': (np.array(hyp_nac2['angle_index']) - 1).tolist(),
                'dihyd_index': (np.array(hyp_nac2['dihed_index']) - 1).tolist(),
            },
            'training': {
                'auto_scaling': {
                    'x_mean': hyp_nac2['scale_x_mean'],
                    'x_std': hyp_nac2['scale_x_std'],
                    'nac_mean': hyp_nac2['scale_y_mean'],
                    'nac_std': hyp_nac2['scale_y_std'],
                },
                'normalization_mode': hyp_nac2['normalization_mode'],
                'learning_rate': hyp_nac2['learning_rate'],
                'phase_less_loss': hyp_nac2['phase_less_loss'],
                'initialize_weights': hyp_nac2['initialize_weights'],
                'val_disjoint': True,
                'val_split': 1/splits,
                'epo': hyp_nac2['epo'],
                'pre_epo': hyp_nac2['pre_epo'],
                'batch_size': hyp_nac2['batch_size'],
                'epostep': hyp_nac2['epostep'],
                'step_callback': {
                    'use': hyp_nac2['use_step_callback'],
                    'epoch_step_reduction': hyp_nac2['epoch_step_reduction'],
                    'learning_rate_step': hyp_nac2['learning_rate_step'],
                },
                'linear_callback': {
                    'use': hyp_nac2['use_linear_callback'],
                    'learning_rate_sta': hyp_nac2['learning_rate_start'],
                    'learning_rate_stop': hyp_nac2['learning_rate_stop'],
                    'epomin': hyp_nac2['epomin'],
                },
                'early_callback': {
                    'use': hyp_nac2['use_early_callback'],
                    'epomin': hyp_nac2['epomin'],
                    'patience': hyp_nac2['patience'],
                    'max_time': hyp_nac2['max_time'],
                    'delta_loss': hyp_nac2['delta_loss'],
                    'loss_monitor': hyp_nac2['loss_monitor'],
                    'factor_lr': hyp_nac2['factor_lr'],
                    'learning_rate_sta': hyp_nac2['learning_rate_start'],
                    'learning_rate_stop': hyp_nac2['learning_rate_stop'],
                },
                'exp_callback': {
                    'use': hyp_nac2['use_exp_callback'],
                    'epomin': hyp_nac2['epomin'],
                    'factor_lr': hyp_nac2['factor_lr'],
                },
            },
            'plots': {
                'unit_nac': hyp_nac2['unit'],
            },
        }

        hyp_dict_eg['retraining'] = hyp_dict_eg['training']
        hyp_dict_eg2['retraining'] = hyp_dict_eg2['training']
        hyp_dict_nac['retraining'] = hyp_dict_nac['training']
        hyp_dict_nac2['retraining'] = hyp_dict_nac2['training']

        # hyp_dict_eg['retraining']['initialize_weights'] = False
        # hyp_dict_eg2['retraining']['initialize_weights'] = False
        # hyp_dict_nac['retraining']['initialize_weights'] = False
        # hyp_dict_nac2['retraining']['initialize_weights'] = False

        ## prepare training data
        self.natom = data.natom
        self.nstate = data.nstate
        self.version = keywords['version']
        self.ncpu = keywords['control']['ml_ncpu']
        self.pred_data = variables['pred_data']
        self.train_mode = variables['train_mode']
        self.shuffle = variables['shuffle']
        self.eg_unit = variables['eg_unit']
        self.nac_unit = variables['nac_unit']
        self.pred_geos = data.pred_geos
        self.pred_energy = data.pred_energy
        self.pred_grad = data.pred_grad
        self.pred_nac = data.pred_nac
        self.pred_soc = data.pred_soc

        ## retraining has some bug at the moment, do not use
        if self.train_mode not in ['train ing', 'retrain ing', 'resample']:
            self.train_mode = 'training'

        if job_id is None or job_id == 1:
            self.name = f"NN-{title}"
        else:
            self.name = f"NN-{title}-{job_id}"
        self.silent = variables['silent']
        self.x = data.geos

        ## convert unit of energy and force. au or si. data are in au.
        h_to_ev = 27.211396132
        h_bohr_to_ev_a = 27.211396132 / 0.529177249

        if eg_unit == 'si':
            self.f_e = h_to_ev
            self.f_g = h_bohr_to_ev_a
            self.k_e = 1
            self.k_g = 1
        else:
            self.f_e = 1
            self.f_g = 1
            self.k_e = h_to_ev
            self.k_g = h_bohr_to_ev_a

        if nac_unit == 'si':
            self.f_n = h_bohr_to_ev_a  # convert to eV/A
            self.k_n = 1
        else:
            self.f_n = 1  # convert to Eh/B
            self.k_n = h_bohr_to_ev_a

        ## combine y_dict
        self.y_dict = {}
        if nn_eg_type > 0:
            y_energy = data.energy * self.f_e
            y_grad = data.grad * self.f_g
            self.y_dict['energy_gradient'] = [y_energy, y_grad]
        if nn_nac_type > 0:
            y_nac = data.nac * self.f_n
            self.y_dict['nac'] = y_nac

        ##  check permutation map
        self.x, self.y_dict = permute_map(self.x, self.y_dict, permute, 1/splits)

        ## combine hypers
        self.hyper = {}
        if nn_eg_type == 1:  # same architecture with different weight
            self.hyper['energy_gradient'] = hyp_dict_eg
        else:
            self.hyper['energy_gradient'] = [hyp_dict_eg, hyp_dict_eg2]
        if nn_nac_type == 1:  # same architecture with differ e nt weight
            self.hyper['nac'] = hyp_dict_nac
        elif nn_nac_type > 1:
            self.hyper['nac'] = [hyp_dict_nac, hyp_dict_nac2]

        ## setup GP ist
        self.gpu_list = {}
        if gpu == 1:
            self.gpu_list['energy_gradient'] = [0, 0]
            self.gpu_list['nac'] = [0, 0]
        elif gpu == 2:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [0, 1]
        elif gpu == 3:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [2, 2]
        elif gpu == 4:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [2, 3]

        ## initialize model
        if modeldir is None or job_id not in [None, 1]:
            self.model = NeuralNetPes(self.name)
        else:
            self.model = NeuralNetPes(modeldir)

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |                  Neural Networks                  |
 |                                                   |
 *---------------------------------------------------*

""" % self.version

        return headline

    def train(self):
        ## ferr      : dict
        ##            Fitting errors, shar e  the same keys as y_dict

        start = time.time()

        self.model.create(self.hyper)

        topline = 'Neural Networks  Start: %20s\n%s' % (what_is_time(), self._heading())
        runinfo = """\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        log = open('%s.log' % self.name, 'w')
        log.write(topline)
        log.write(runinfo)
        log.close()

        if self.train_mode == 'resample':
            # out_index, out_errr, out_fiterr, out_testerr = self.model.resample(
            #     self.x,
            #     self.y_dict,
            #     gpu_dist=self.gpu_list,
            #     proc_async=self.ncpu>=4
            # )
            train_info = 'resample'
        else:
            ferr = self.model.fit(
                self.x,
                self.y_dict,
                gpu_dist=self.gpu_list,
                proc_async=self.ncpu >= 4,
                fitmode=self.train_mode,
                random_shuffle=self.shuffle
            )
            print(ferr)  # self.model.save()
            err_eg1 = ferr['energy_gradient'][0]
            err_eg2 = ferr['energy_gradient'][1]

            if 'nac' in ferr.keys():
                err_n = ferr['nac']
            else:
                err_n = np.zeros(2)

            train_info = """
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac(interstate)
        eV           eV/A         eV/A
  %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f

""" % (
                err_eg1[0] * self.k_e,
                err_eg1[1] * self.k_g,
                err_n[0] * self.k_n,
                err_eg2[0] * self.k_e,
                err_eg2[1] * self.k_g,
                err_n[1] * self.k_n
            )

        end = time.time()
        walltime = how_long(start, end)
        endline = 'Neural Networks End: %20s Total: %20s\n' % (what_is_time(), walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        log = open('%s.log' % self.name, 'a')
        log.write(train_info)
        log.write(endline)
        log.close()

        return self

    def load(self):
        self.model.load()

        return self

    def appendix(self, _):
        ## fake	function does nothing

        return self

    def _qmmm(self, traj):
        ## run psnnsmd for QMQM2 calculation
        traj = traj.apply_qmmm()

        xyz = traj.qm_coord.reshape((1, self.natom, 3))
        y_pred, y_std = self.model.call(xyz)

        ## initialize return values
        energy = []
        gradient = []
        nac = []
        soc = []
        err_e = 0
        err_g = 0
        err_n = 0
        err_s = 0

        ## update return values
        if 'energy_gradient' in y_pred.keys():
            e_pred = y_pred['energy_gradient'][0] / self.f_e
            g_pred = y_pred['energy_gradient'][1] / self.f_g
            e_std = y_std['energy_gradient'][0] / self.f_e
            g_std = y_std['energy_gradient'][1] / self.f_g
            energy = e_pred[0]
            gradient = g_pred[0]
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)

        if 'nac' in y_pred.keys():
            n_pred = y_pred['nac'] / self.f_n
            n_std = y_std['nac'] / self.f_n
            nac = n_pred[0]
            err_n = np.amax(n_std)

        # if 'soc' in y_pred.keys():
        #     s_pred = y_pred['soc']
        #     s_std = y_std['soc']
        #     soc = s_pred[0]
        #     err_s = np.amax(s_std)

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _qm(self, traj):
        ## run psnnsmd for QM calculation

        xyz = traj.coord.reshape((1, self.natom, 3))
        y_pred, y_std = self.model.call(xyz)

        ## initialize return values
        energy = []
        gradient = []
        nac = []
        soc = []
        err_e = 0
        err_g = 0
        err_n = 0
        err_s = 0

        ## update return values
        if 'energy_gradient' in y_pred.keys():
            e_pred = y_pred['energy_gradient'][0] / self.f_e
            g_pred = y_pred['energy_gradient'][1] / self.f_g
            e_std = y_std['energy_gradient'][0] / self.f_e
            g_std = y_std['energy_gradient'][1] / self.f_g
            energy = e_pred[0]
            gradient = g_pred[0]
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)

        if 'nac' in y_pred.keys():
            n_pred = y_pred['nac'] / self.f_n
            n_std = y_std['nac'] / self.f_n
            nac = n_pred[0]
            err_n = np.amax(n_std)

        # if 'soc' in y_pred.keys():
        #     s_pred = y_pred['soc']
        #     s_std = y_std['soc']
        #     soc = s_pred[0]
        #     err_s = np.amax(s_std)

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x):
        ## run psnnsmd for model testing

        batch = len(x)

        y_pred, y_std = self.model.predict(x)

        ## load values from prediction set
        pred_e = self.pred_energy
        pred_g = self.pred_grad
        pred_n = self.pred_nac
        # pred_s = self.pred_soc

        ## initialize errors
        de_max = np.zeros(batch)
        dg_max = np.zeros(batch)
        dn_max = np.zeros(batch)
        ds_max = np.zeros(batch)

        ## update errors
        if 'energy_gradient' in y_pred.keys():
            e_pred = y_pred['energy_gradient'][0] / self.f_e
            g_pred = y_pred['energy_gradient'][1] / self.f_g
            e_std = y_std['energy_gradient'][0] / self.f_e
            g_std = y_std['energy_gradient'][1] / self.f_g
            de = np.abs(pred_e - e_pred)
            dg = np.abs(pred_g - g_pred)
            de_max = np.amax(de.reshape((batch, -1)), axis=1)
            dg_max = np.amax(dg.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((pred_e.reshape((batch, -1)), e_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((de.reshape((batch, -1)), e_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-e.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

            val_out = np.concatenate((pred_g.reshape((batch, -1)), g_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dg.reshape((batch, -1)), g_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-g.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

        if 'nac' in y_pred.keys():
            n_pred = y_pred['nac'] / self.f_n
            n_std = y_std['nac'] / self.f_n
            dn = np.abs(pred_n - n_pred)
            dn_max = np.amax(dn.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((pred_n.reshape((batch, -1)), n_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dn.reshape((batch, -1)), n_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-n.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

        # if 'soc' in y_pred.keys():
        #     s_pred = y_pred['soc']
        #     s_std = y_std['soc']
        #     ds = np.abs(pred_s - s_pred)
        #     ds_max = np.amax(ds.reshape((batch, -1)), axis=1)
        #
        #     val_out = np.concatenate((pred_s.reshape((batch, -1)), s_pred.reshape((batch, -1))), axis=1)
        #     std_out = np.concatenate((ds.reshape((batch, -1)), s_std.reshape((batch, -1))), axis=1)
        #     np.savetxt('%s-s.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

        output = ''
        for i in range(batch):
            output += '%5s %8.4f %8.4f %8.4f %8.4f\n' % (i + 1, de_max[i], dg_max[i], dn_max[i], ds_max[i])

        with open('max_abs_dev.txt', 'w') as out:
            out.write(output)

        return self

    def evaluate(self, traj):
        ## main function to run pyNNsMD and communicate with other PyRAI2MD modules

        if self.jobtype == 'prediction' or self.jobtype == 'predict':
            self._predict(self.pred_geos)
        else:
            if self.runtype == 'qmmm':
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._qmmm(traj)
            else:
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._qm(traj)

            traj.energy = np.copy(energy)
            traj.grad = np.copy(gradient)
            traj.nac = np.copy(nac)
            traj.soc = np.copy(soc)
            traj.err_energy = err_energy
            traj.err_grad = err_grad
            traj.err_nac = err_nac
            traj.err_soc = err_soc
            traj.status = 1

            return traj
