#####################################################
#
# PyRAI2MD 2 module for interfacing to pyNNsMD (KGCNN)
#
# Author Jingbai Li
# Sep 1 2022
#
######################################################

import os
import sys
import time
import numpy as np

from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_mlp_hyper_eg
from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_mlp_hyper_nac
from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_mlp_hyper_soc
from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_sch_hyper_eg
from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_sch_hyper_nac
from PyRAI2MD.Machine_Learning.hyper_pynnsmd import set_sch_hyper_soc
from PyRAI2MD.Machine_Learning.permutation import permute_map2
from PyRAI2MD.Machine_Learning.model_helper import Multiregions
from PyRAI2MD.Machine_Learning.NNsMD.nn_pes_src.device import set_gpu
from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long

from pyNNsMD.NNsMD import NeuralNetEnsemble

class MLP:
    """ pyNNsMD interface

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            hyp_eg           dict        Hyperparameters of energy gradient NN
       	    hyp_nac          dict        Hyperparameters of nonadiabatic coupling NN
       	    hyp_soc          dict     	 Hyperparameters of spin-orbit coupling NN
            energy           ndarray     energy array
            grad             ndarray     gradient array
            nac              ndarray     nonadiabatic coupling array
            soc              ndarray     spin-orbit coupling array
            atoms            list        atom list
            geos             ndarray     training set coordinates
            pred_atoms       list        atom list
            pred_geos        ndarray     prediction set coordinates
            pred_energy      ndarray     prediction set target energy
            pred_grad        ndarray     prediction set target grad
            pred_nac         ndarray     prediction set target nac
            pred_soc         ndarray     prediction set target soc

        Functions:           Returns:
            train            self        train NN for a given training set
            load             self        load trained NN for prediction
            appendix         self        fake function
            evaluate         self        run prediction

    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):

        set_gpu([])  # No GPU for prediction
        self.runtype = runtype
        title = keywords['control']['title']
        variables = keywords['mlp'].copy()
        modeldir = variables['modeldir']
        data = variables['data']
        nn_eg_type = variables['nn_eg_type']
        nn_nac_type = variables['nn_nac_type']
        nn_soc_type = variables['nn_soc_type']
        hyp_eg = variables['eg'].copy()
        hyp_nac = variables['nac'].copy()
        hyp_eg2 = variables['eg2'].copy()
        hyp_nac2 = variables['nac2'].copy()
        hyp_soc = variables['soc'].copy()
        hyp_soc2 = variables['soc2'].copy()
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        soc_unit = variables['soc_unit']
        permute = variables['permute_map']
        gpu = variables['gpu']
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.ncpu = keywords['control']['ml_ncpu']
        self.train_mode = variables['train_mode']
        self.shuffle = variables['shuffle']
        self.splits = variables['nsplits']
        self.natom = data.natom
        self.nstate = data.nstate
        self.nnac = data.nnac
        self.nsoc = data.nsoc

        # set output value range
        if 0 < len(variables['select_eg_out']) < self.nstate:
            self.select_eg_out = variables['select_eg_out']
        else:
            self.select_eg_out = np.arange(self.nstate)

        if 0 < len(variables['select_nac_out']) < self.nnac:
            self.select_eg_out = variables['select_nac_out']
        else:
            self.select_nac_out = np.arange(self.nnac)

        if 0 < len(variables['select_soc_out']) < self.nsoc:
            self.select_soc_out = variables['select_soc_out']
        else:
            self.select_soc_out = np.arange(self.nsoc)

        ## set hyperparameters
        hyp_dict_eg = set_mlp_hyper_eg(hyp_eg, eg_unit, data.info)
        hyp_dict_eg2 = set_mlp_hyper_eg(hyp_eg2, eg_unit, data.info)
        hyp_dict_nac = set_mlp_hyper_nac(hyp_nac, nac_unit, data.info)
        hyp_dict_nac2 = set_mlp_hyper_nac(hyp_nac2, nac_unit, data.info)
        hyp_dict_soc = set_mlp_hyper_soc(hyp_soc, soc_unit, data.info)
        hyp_dict_soc2 = set_mlp_hyper_soc(hyp_soc2, soc_unit, data.info)

        ## retraining has some bug at the moment, do not use
        if self.train_mode not in ['training', 'retraining', 'resample']:
            self.train_mode = 'training'

        if job_id is None or job_id == 1:
            self.name = f"NN-{title}"
        else:
            self.name = f"NN-{title}-{job_id}"

        self.silent = variables['silent']

        ## prepare unit conversion for energy and force. au or si. The data units are in au.
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

        ## unpack data
        self.atoms = data.atoms
        self.geos = data.geos
        self.energy = data.energy * self.f_e
        self.grad = data.grad * self.f_g
        self.nac = data.nac * self.f_n
        self.soc = data.soc
        self.pred_atoms = data.pred_atoms
        self.pred_geos = data.pred_geos
        self.pred_energy = data.pred_energy
        self.pred_grad = data.pred_grad
        self.pred_nac = data.pred_nac
        self.pred_soc = data.pred_soc

        ## check permutation map
        self.geos, self.energy, self.grad, self.nac, self.soc = permute_map2(
            self.geos,
            self.energy,
            self.grad,
            self.nac,
            self.soc,
            permute,
            self.splits
        )

        ## setup GPU list
        if gpu == 1:
            self.gpu_eg = [0, 0]
            self.gpu_n = [0, 0]
            self.gpu_s = [0, 0]
        elif gpu == 2:
            self.gpu_eg = [0, 1]
            self.gpu_n = [0, 1]
            self.gpu_s = [0, 1]
        else:
            self.gpu_eg = [0, 0]
            self.gpu_n = [0, 0]
            self.gpu_s = [0, 0]

        ## initialize model path
        if modeldir is None or job_id not in [None, 1]:
            ensemble_path = self.name
        else:
            ensemble_path = modeldir

        self.model_register = {
            'energy_grad': False,
            'nac': False,
            'soc': False
        }

        ## initialize models
        if nn_eg_type == 1:
            self.model_register['energy_grad'] = True
            self.hyper_eg = [hyp_dict_eg, hyp_dict_eg]
            self.model_eg = NeuralNetEnsemble('%s/energy_grad' % ensemble_path, 2)
        elif nn_eg_type > 1:
            self.model_register['energy_grad'] = True
            self.hyper_eg = [hyp_dict_eg, hyp_dict_eg2]
            self.model_eg = NeuralNetEnsemble('%s/energy_grad' % ensemble_path, 2)
        else:
            self.model_register['energy_grad'] = False
            self.hyper_eg = None
            self.model_eg = None

        if nn_nac_type == 1:
            self.model_register['nac'] = True
            self.hyper_nac = [hyp_dict_nac, hyp_dict_nac]
            self.model_nac = NeuralNetEnsemble('%s/nac' % ensemble_path, 2)
        elif nn_nac_type > 1:
            self.model_register['nac'] = True
            self.hyper_nac = [hyp_dict_nac, hyp_dict_nac2]
            self.model_nac = NeuralNetEnsemble('%s/nac' % ensemble_path, 2)
        else:
            self.model_register['nac'] = False
            self.hyper_nac = None
            self.model_nac = None

        if nn_soc_type == 1:  # same architecture with different weight
            self.model_register['soc'] = True
            self.hyper_soc = [hyp_dict_soc, hyp_dict_soc]
            self.model_soc = NeuralNetEnsemble('%s/soc' % ensemble_path, 2)
        elif nn_soc_type > 1:
            self.model_register['soc'] = True
            self.hyper_soc = [hyp_dict_soc, hyp_dict_soc2]
            self.model_soc = NeuralNetEnsemble('%s/soc' % ensemble_path, 2)
        else:
            self.model_register['soc'] = False
            self.hyper_soc = None
            self.model_soc = None

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |             Multilayer Perceptron                 |
 |                                                   |
 |              powered by pyNNsMD                   |
 |                                                   |
 *---------------------------------------------------*

 Number of atoms:  %s
 Number of state:  %s
 Number of NAC:    %s
 Number of SOC:    %s

""" % (
            self.version,
            self.natom,
            self.nstate,
            self.nnac,
            self.nsoc
        )

        return headline

    def train(self):
        start = time.time()
        topline = 'Neural Networks Start: %20s\n%s' % (what_is_time(), self._heading())
        runinfo = """\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        with open('%s.log' % self.name, 'w') as log:
            log.write(topline)
            log.write(runinfo)

        if self.model_register['energy_grad']:

            self.model_eg.create(
                models=[m['model'] for m in self.hyper_eg],
                scalers=[m['scaler'] for m in self.hyper_eg],
            )

            self.model_eg.save()

            self.model_eg.data(
                atoms=self.atoms,
                geometries=self.geos,
                energies=self.energy,
                forces=self.grad
            )

            self.model_eg.train_test_split(
                dataset_size=len(self.energy),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_eg.training(
                [m['training'] for m in self.hyper_eg],
                fit_mode=self.train_mode
            )

            eg_error = self.model_eg.fit(
                ['training_mlp_eg'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_eg,
                proc_async=self.ncpu >= 2
            )
            err_e1 = eg_error[0]['valid'][0]
            err_e2 = eg_error[1]['valid'][0]
            err_g1 = eg_error[0]['valid'][1]
            err_g2 = eg_error[1]['valid'][1]
        else:
            err_e1 = 0
            err_e2 = 0
            err_g1 = 0
            err_g2 = 0

        if self.model_register['nac']:

            self.model_nac.create(
                models=[m['model'] for m in self.hyper_nac],
                scalers=[m['scaler'] for m in self.hyper_nac],
            )

            self.model_nac.save()

            self.model_nac.data(
                atoms=self.atoms,
                geometries=self.geos,
                couplings=self.nac,
            )

            self.model_nac.train_test_split(
                dataset_size=len(self.nac),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_nac.training(
                [m['training'] for m in self.hyper_nac],
                fit_mode=self.train_mode
            )

            nac_error = self.model_nac.fit(
                ['training_mlp_nac2'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_n,
                proc_async=self.ncpu >= 2
            )
            err_n1 = nac_error[0]['valid']
            err_n2 = nac_error[1]['valid']
        else:
            err_n1 = 0
            err_n2 = 0

        if self.model_register['soc']:

            self.model_soc.create(
                models=[m['model'] for m in self.hyper_soc],
                scalers=[m['scaler'] for m in self.hyper_soc],
            )

            self.model_soc.save()

            self.model_soc.data(
                atoms=self.atoms,
                geometries=self.geos,
                energies=self.soc,
            )

            self.model_soc.train_test_split(
                dataset_size=len(self.soc),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_soc.training(
                [m['training'] for m in self.hyper_soc],
                fit_mode=self.train_mode
            )

            soc_error = self.model_soc.fit(
                ['training_mlp_e'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_s,
                proc_async=self.ncpu >= 2
            )
            err_s1 = soc_error[0]['valid']
            err_s2 = soc_error[1]['valid']
        else:
            err_s1 = 0
            err_s2 = 0

        metrics = {
            'e1': err_e1 * self.k_e,
            'g1': err_g1 * self.k_g,
            'n1': err_n1 / self.k_n,
            's1': err_s1,
            'e2': err_e2 * self.k_e,
            'g2': err_g2 * self.k_g,
            'n2': err_n2 / self.k_n,
            's2': err_s2
        }

        train_info = """
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac          soc
        eV           eV/A         eV/A         cm-1
  %12.8f %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f %12.8f

""" % (
            metrics['e1'], metrics['g1'], metrics['n1'], metrics['s1'],
            metrics['e2'], metrics['g2'], metrics['n2'], metrics['s2']
        )

        end = time.time()
        walltime = how_long(start, end)
        endline = 'Neural Networks End: %20s Total: %20s\n' % (what_is_time(), walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        with open('%s.log' % self.name, 'a') as log:
            log.write(train_info)
            log.write(endline)

        metrics['time'] = end - start
        metrics['walltime'] = walltime
        metrics['path'] = os.getcwd()
        metrics['status'] = 1

        return metrics

    def load(self):

        if self.model_register['energy_grad']:
            self.model_eg.create(
                models=[m['model'] for m in self.hyper_eg],
                scalers=[m['scaler'] for m in self.hyper_eg],
            )
            self.model_eg.load()

        if self.model_register['nac']:
            self.model_nac.create(
                models=[m['model'] for m in self.hyper_nac],
                scalers=[m['scaler'] for m in self.hyper_nac],
            )
            self.model_nac.load()

        if self.model_register['soc']:
            self.model_soc.create(
                models=[m['model'] for m in self.hyper_soc],
                scalers=[m['scaler'] for m in self.hyper_soc],
            )
            self.model_soc.load()

        return self

    def appendix(self, _):
        ## fake	function does nothing

        return self

    def _high_mid_low(self, traj):
        ## run psnnsmd for high level region, middle level region, and low level region in QM calculation

        xyz = traj.coord.reshape((1, self.natom, 3))

        if self.model_register['energy_grad']:
            pred = self.model_eg.call(xyz)
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = self.model_nac.predict(xyz)
            nac = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = self.model_soc.predict(xyz)
            soc = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _high(self, traj):
        ## run psnnsmd for high level region QM calculation
        traj = traj.apply_qmmm()

        xyz = traj.qm_coord.reshape((1, self.natom, 3))

        if self.model_register['energy_grad']:
            pred = self.model_eg.call(xyz)
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = self.model_nac.predict(xyz)
            nac = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = self.model_soc.predict(xyz)
            soc = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x):
        ## run psnnsmd for model testing

        batch = len(x)

        if self.model_register['energy_grad']:
            pred = self.model_eg.predict(x)
            e_pred = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            g_pred = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g

            de = np.abs(self.pred_energy - e_pred)
            dg = np.abs(self.pred_grad - g_pred)
            de_max = np.amax(de.reshape((batch, -1)), axis=1)
            dg_max = np.amax(dg.reshape((batch, -1)), axis=1)
            val_out = np.concatenate((self.pred_energy.reshape((batch, -1)), e_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((de.reshape((batch, -1)), e_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-e.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

            val_out = np.concatenate((self.pred_grad.reshape((batch, -1)), g_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dg.reshape((batch, -1)), g_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-g.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            de_max = np.zeros(batch)
            dg_max = np.zeros(batch)

        if self.model_register['nac']:
            pred = self.model_nac.predict(x)
            n_pred = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n

            dn = np.abs(self.pred_nac - n_pred)
            dn_max = np.amax(dn.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((self.pred_nac.reshape((batch, -1)), n_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dn.reshape((batch, -1)), n_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-n.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            dn_max = np.zeros(batch)

        if self.model_register['soc']:
            pred = self.model_soc.predict(x)
            s_pred = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)

            ds = np.abs(self.pred_soc - s_pred)
            ds_max = np.amax(ds.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((self.pred_soc.reshape((batch, -1)), s_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((ds.reshape((batch, -1)), s_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-s.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            ds_max = np.zeros(batch)

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
            if self.runtype == 'qm_high':
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high(traj)
            else:
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high_mid_low(traj)

            traj.energy = np.copy(energy)[self.select_eg_out]
            traj.grad = np.copy(gradient)[self.select_eg_out]
            traj.nac = np.copy(nac)[self.select_nac_out]
            traj.soc = np.copy(soc)[self.select_soc_out]
            traj.err_energy = err_energy
            traj.err_grad = err_grad
            traj.err_nac = err_nac
            traj.err_soc = err_soc
            traj.status = 1

            return traj

class Schnet:
    """ pyNNsMD interface

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            hyp_eg           dict        Hyperparameters of energy gradient NN
       	    hyp_nac          dict        Hyperparameters of nonadiabatic coupling NN
       	    hyp_soc          dict     	 Hyperparameters of spin-orbit coupling NN
            energy           ndarray     energy array
            grad             ndarray     gradient array
            nac              ndarray     nonadiabatic coupling array
            soc              ndarray     spin-orbit coupling array
            atoms            list        atom list
            geos             ndarray     training set coordinates
            pred_atoms       list        atom list
            pred_geos        ndarray     prediction set coordinates
            pred_energy      ndarray     prediction set target energy
            pred_grad        ndarray     prediction set target grad
            pred_nac         ndarray     prediction set target nac
            pred_soc         ndarray     prediction set target soc
            atomic_numbers   list        atomic number list
            multiscale       list        atom indices in multiscale regions

        Functions:           Returns:
            train            self        train NN for a given training set
            load             self        load trained NN for prediction
            appendix         self        fake function
            evaluate         self        run prediction

    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):

        set_gpu([])  # No GPU for prediction
        self.runtype = runtype
        title = keywords['control']['title']
        variables = keywords['schnet'].copy()
        modeldir = variables['modeldir']
        data = variables['data']
        nn_eg_type = variables['nn_eg_type']
        nn_nac_type = variables['nn_nac_type']
        nn_soc_type = variables['nn_soc_type']
        multiscale = variables['multiscale']
        hyp_eg = variables['sch_eg'].copy()
        hyp_nac = variables['sch_nac'].copy()
        hyp_soc = variables['sch_soc'].copy()
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        soc_unit = variables['soc_unit']
        gpu = variables['gpu']
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.ncpu = keywords['control']['ml_ncpu']
        self.train_mode = variables['train_mode']
        self.shuffle = variables['shuffle']
        self.splits = variables['nsplits']
        self.natom = data.natom
        self.nstate = data.nstate
        self.nnac = data.nnac
        self.nsoc = data.nsoc

        # set output value range
        if 0 < len(variables['select_eg_out']) < self.nstate:
            self.select_eg_out = variables['select_eg_out']
        else:
            self.select_eg_out = np.arange(self.nstate)

        if 0 < len(variables['select_nac_out']) < self.nnac:
            self.select_eg_out = variables['select__out']
        else:
            self.select_nac_out = np.arange(self.nnac)

        if 0 < len(variables['select_eg_out']) < self.nsoc:
            self.select_soc_out = variables['select_eg_out']
        else:
            self.select_soc_out = np.arange(self.nsoc)

        ## set hyperparameters
        hyp_dict_eg = set_sch_hyper_eg(hyp_eg, eg_unit, data.info)
        hyp_dict_nac = set_sch_hyper_nac(hyp_nac, nac_unit, data.info)
        hyp_dict_soc = set_sch_hyper_soc(hyp_soc, soc_unit, data.info)

        ## retraining has some bug at the moment, do not use
        if self.train_mode not in ['training', 'retraining', 'resample']:
            self.train_mode = 'training'

        if job_id is None or job_id == 1:
            self.name = f"NN-{title}"
        else:
            self.name = f"NN-{title}-{job_id}"

        self.silent = variables['silent']

        ## prepare unit conversion for energy and force. au or si. The data units are in au.
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

        ## unpack data
        self.atomic_numbers = np.array(data.atomic_numbers[0])
        self.atoms = data.atoms
        self.geos = data.geos
        self.energy = data.energy * self.f_e
        self.grad = data.grad * self.f_g
        self.nac = data.nac * self.f_n
        self.soc = data.soc
        self.pred_atoms = data.pred_atoms
        self.pred_geos = data.pred_geos
        self.pred_energy = data.pred_energy
        self.pred_grad = data.pred_grad
        self.pred_nac = data.pred_nac
        self.pred_soc = data.pred_soc

        if len(multiscale) > 0:
            self.mr = Multiregions(self.atoms[0], multiscale)
            atoms = self.mr.partition_atoms(self.atoms[0])
            self.atoms = [atoms] * len(self.atoms)
            self.pred_atoms = [atoms] * len(self.pred_atoms)
            self.atomic_numbers = self.mr.partition_atomic_numbers(self.atomic_numbers)
        else:
            self.mr = None

        ## setup GPU list
        if gpu == 1:
            self.gpu_eg = [0, 0]
            self.gpu_n = [0, 0]
            self.gpu_s = [0, 0]
        elif gpu == 2:
            self.gpu_eg = [0, 1]
            self.gpu_n = [0, 1]
            self.gpu_s = [0, 1]
        else:
            self.gpu_eg = [0, 0]
            self.gpu_n = [0, 0]
            self.gpu_s = [0, 0]

        ## initialize model path
        if modeldir is None or job_id not in [None, 1]:
            ensemble_path = self.name
        else:
            ensemble_path = modeldir

        self.model_register = {
            'energy_grad': False,
            'nac': False,
            'soc': False
        }

        ## initialize models
        if nn_eg_type > 0:
            self.model_register['energy_grad'] = True
            self.hyper_eg = [hyp_dict_eg, hyp_dict_eg]
            self.model_eg = NeuralNetEnsemble('%s/energy_grad' % ensemble_path, 2)
        else:
            self.model_register['energy_grad'] = False
            self.hyper_eg = None
            self.model_eg = None

        if nn_nac_type > 0:
            self.model_register['nac'] = False
            self.hyper_nac = [hyp_dict_nac, hyp_dict_nac]
            sys.exit('\n  RuntimeError\n  PyRAI2MD: Schnet nac model is unavailable\n')
        else:
            self.model_register['nac'] = False
            self.hyper_nac = None
            self.model_nac = None

        if nn_soc_type > 0:  # same architecture with different weight
            self.model_register['soc'] = True
            self.hyper_soc = [hyp_dict_soc, hyp_dict_soc]
            self.model_soc = NeuralNetEnsemble('%s/soc' % ensemble_path, 2)
        else:
            self.model_register['soc'] = False
            self.hyper_soc = None
            self.model_soc = None

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |                     Schnet                        |
 | A continuous-filter convolutional neural network  |
 |                                                   |
 |              powered by pyNNsMD                   |
 |                                                   |
 *---------------------------------------------------*

 Number of atoms:  %s
 Number of state:  %s
 Number of NAC:    %s
 Number of SOC:    %s

""" % (
            self.version,
            self.natom,
            self.nstate,
            self.nnac,
            self.nsoc
        )

        return headline

    def train(self):
        start = time.time()
        topline = 'Neural Networks Start: %20s\n%s' % (what_is_time(), self._heading())
        runinfo = """\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        with open('%s.log' % self.name, 'w') as log:
            log.write(topline)
            log.write(runinfo)

        if self.model_register['energy_grad']:

            self.model_eg.create(
                models=[m['model'] for m in self.hyper_eg],
                scalers=[m['scaler'] for m in self.hyper_eg],
            )

            self.model_eg.save()

            self.model_eg.data(
                atoms=self.atoms,
                geometries=self.geos,
                energies=self.energy,
                forces=self.grad
            )

            self.model_eg.train_test_split(
                dataset_size=len(self.energy),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_eg.training(
                [m['training'] for m in self.hyper_eg],
                fit_mode=self.train_mode
            )

            eg_error = self.model_eg.fit(
                ['training_schnet_eg'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_eg,
                proc_async=self.ncpu >= 2
            )
            err_e1 = eg_error[0]['valid'][0]
            err_e2 = eg_error[1]['valid'][0]
            err_g1 = eg_error[0]['valid'][1]
            err_g2 = eg_error[1]['valid'][1]
        else:
            err_e1 = 0
            err_e2 = 0
            err_g1 = 0
            err_g2 = 0

        if self.model_register['nac']:

            self.model_nac.create(
                models=[m['model'] for m in self.hyper_nac],
                scalers=[m['scaler'] for m in self.hyper_nac],
            )

            self.model_nac.save()

            self.model_nac.data(
                atoms=self.atoms,
                geometries=self.geos,
                couplings=self.nac,
            )

            self.model_nac.train_test_split(
                dataset_size=len(self.nac),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_nac.training(
                [m['training'] for m in self.hyper_nac],
                fit_mode=self.train_mode
            )

            nac_error = self.model_nac.fit(
                ['training_schnet_nac'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_n,
                proc_async=self.ncpu >= 2
            )
            err_n1 = nac_error[0]['valid']
            err_n2 = nac_error[1]['valid']
        else:
            err_n1 = 0
            err_n2 = 0

        if self.model_register['soc']:

            self.model_soc.create(
                models=[m['model'] for m in self.hyper_soc],
                scalers=[m['scaler'] for m in self.hyper_soc],
            )

            self.model_soc.save()

            self.model_soc.data(
                atoms=self.atoms,
                geometries=self.geos,
                energies=self.soc,
            )

            self.model_soc.train_test_split(
                dataset_size=len(self.soc),
                n_splits=self.splits,
                shuffle=self.shuffle
            )

            self.model_soc.training(
                [m['training'] for m in self.hyper_soc],
                fit_mode=self.train_mode
            )

            soc_error = self.model_soc.fit(
                ['training_schnet_e'] * 2,
                fit_mode=self.train_mode,
                gpu_dist=self.gpu_s,
                proc_async=self.ncpu >= 2
            )
            err_s1 = soc_error[0]['valid']
            err_s2 = soc_error[1]['valid']
        else:
            err_s1 = 0
            err_s2 = 0

        metrics = {
            'e1': err_e1 * self.k_e,
            'g1': err_g1 * self.k_g,
            'n1': err_n1 / self.k_n,
            's1': err_s1,
            'e2': err_e2 * self.k_e,
            'g2': err_g2 * self.k_g,
            'n2': err_n2 / self.k_n,
            's2': err_s2
        }

        train_info = """
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac          soc
        eV           eV/A         eV/A         cm-1
  %12.8f %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f %12.8f

""" % (
            metrics['e1'], metrics['g1'], metrics['n1'], metrics['s1'],
            metrics['e2'], metrics['g2'], metrics['n2'], metrics['s2']
        )

        end = time.time()
        walltime = how_long(start, end)
        endline = 'Neural Networks End: %20s Total: %20s\n' % (what_is_time(), walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        with open('%s.log' % self.name, 'a') as log:
            log.write(train_info)
            log.write(endline)

        metrics['time'] = end - start
        metrics['walltime'] = walltime
        metrics['path'] = os.getcwd()
        metrics['status'] = 1

        return metrics

    def load(self):

        if self.model_register['energy_grad']:
            self.model_eg.create(
                models=[m['model'] for m in self.hyper_eg],
                scalers=[m['scaler'] for m in self.hyper_eg],
            )
            self.model_eg.load()
            for i in range(len(self.model_eg)):
                self.model_eg[i].energy_only = False
                self.model_eg[i].output_as_dict = False

        if self.model_register['nac']:
            self.model_nac.create(
                models=[m['model'] for m in self.hyper_nac],
                scalers=[m['scaler'] for m in self.hyper_nac],
            )
            self.model_nac.load()

        if self.model_register['soc']:
            self.model_soc.create(
                models=[m['model'] for m in self.hyper_soc],
                scalers=[m['scaler'] for m in self.hyper_soc],
            )
            self.model_soc.load()
            for i in range(len(self.model_soc)):
                self.model_soc[i].energy_only = True
                self.model_soc[i].output_as_dict = False

        return self

    def appendix(self, _):
        ## fake	function does nothing

        return self

    def _high_mid_low(self, traj):
        ## run psnnsmd for high level region, middle level region, and low level region in QM calculation

        xyz = traj.coord.reshape((1, self.natom, 3))
        atomic_numbers = [self.atomic_numbers]

        if self.model_register['energy_grad']:
            pred = self.model_eg.call([atomic_numbers, xyz])
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = self.model_nac.predict([atomic_numbers, xyz])
            nac = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = self.model_soc.predict([atomic_numbers, xyz])
            soc = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _high(self, traj):
        ## run psnnsmd for high level region in QM calculation
        traj = traj.apply_qmmm()

        xyz = traj.qm_coord.reshape((1, self.natom, 3))
        atomic_numbers = [self.atomic_numbers]

        if self.model_register['energy_grad']:
            pred = self.model_eg.call([atomic_numbers, xyz])
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = self.model_nac.predict([atomic_numbers, xyz])
            nac = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = self.model_soc.predict([atomic_numbers, xyz])
            soc = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x):
        ## run psnnsmd for model testing

        batch = len(x)
        atomic_numbers = [self.atomic_numbers] * batch

        if self.model_register['energy_grad']:
            pred = self.model_eg.predict([atomic_numbers, x])
            e_pred = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            g_pred = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g
            g_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_g

            de = np.abs(self.pred_energy - e_pred)
            dg = np.abs(self.pred_grad - g_pred)
            de_max = np.amax(de.reshape((batch, -1)), axis=1)
            dg_max = np.amax(dg.reshape((batch, -1)), axis=1)
            val_out = np.concatenate((self.pred_energy.reshape((batch, -1)), e_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((de.reshape((batch, -1)), e_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-e.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

            val_out = np.concatenate((self.pred_grad.reshape((batch, -1)), g_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dg.reshape((batch, -1)), g_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-g.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            de_max = np.zeros(batch)
            dg_max = np.zeros(batch)

        if self.model_register['nac']:
            pred = self.model_nac.predict([atomic_numbers, x])
            n_pred = np.mean([pred[0], pred[1]], axis=0) / self.f_n
            n_std = np.std([pred[0], pred[1]], axis=0, ddof=1) / self.f_n

            dn = np.abs(self.pred_nac - n_pred)
            dn_max = np.amax(dn.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((self.pred_nac.reshape((batch, -1)), n_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dn.reshape((batch, -1)), n_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-n.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            dn_max = np.zeros(batch)

        if self.model_register['soc']:
            pred = self.model_soc.predict([atomic_numbers, x])
            s_pred = np.mean([pred[0], pred[1]], axis=0)
            s_std = np.std([pred[0], pred[1]], axis=0, ddof=1)

            ds = np.abs(self.pred_soc - s_pred)
            ds_max = np.amax(ds.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((self.pred_soc.reshape((batch, -1)), s_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((ds.reshape((batch, -1)), s_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-s.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            ds_max = np.zeros(batch)

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
            if self.runtype == 'qm_high':
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high(traj)
            else:
                energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._high_mid_low(traj)

            traj.energy = np.copy(energy)[self.select_eg_out]
            traj.grad = np.copy(gradient)[self.select_eg_out]
            traj.nac = np.copy(nac)[self.select_nac_out]
            traj.soc = np.copy(soc)[self.select_soc_out]
            traj.err_energy = err_energy
            traj.err_grad = err_grad
            traj.err_nac = err_nac
            traj.err_soc = err_soc
            traj.status = 1

            return traj
