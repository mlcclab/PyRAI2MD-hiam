#####################################################
#
# PyRAI2MD 2 module for interfacing to esnnp
#
# Author Jingbai Li
# Aug 6 2025
#
######################################################

import os
import time
import copy
import numpy as np
import torch.cuda

from PyRAI2MD.Molecule.atom import Atom
from PyRAI2MD.Machine_Learning.hyper_esnnp import set_e2n2_hyper_eg
from PyRAI2MD.Machine_Learning.hyper_esnnp import set_e2n2_hyper_nac
from PyRAI2MD.Machine_Learning.hyper_esnnp import set_e2n2_hyper_soc
from PyRAI2MD.Machine_Learning.model_helper import Multiregions
from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long

from esnnp.esnnp import ESNNP

class E2N2:
    """ esnnp interface

        Parameters:          Type:
            keywords         dict        keywords dict
            job_id           int         calculation index

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

        self.runtype = runtype
        title = keywords['control']['title']
        variables = keywords['e2n2'].copy()
        modeldir = variables['modeldir']
        data = variables['data']
        train_mode = variables['train_mode']
        nn_eg_type = variables['nn_eg_type']
        nn_nac_type = variables['nn_nac_type']
        nn_soc_type = variables['nn_soc_type']
        multiscale = variables['multiscale']
        hyp_eg = variables['e2n2_eg'].copy()
        hyp_nac = variables['e2n2_nac'].copy()
        hyp_soc = variables['e2n2_soc'].copy()
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        soc_unit = variables['soc_unit']
        shuffle = variables['shuffle']
        splits = variables['nsplits']
        gpu = variables['gpu']
        elements = variables['elements']
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.ncpu = keywords['control']['ml_ncpu']
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
        hyp_dict_eg = set_e2n2_hyper_eg(hyp_eg, eg_unit, data.info, splits, shuffle)
        hyp_dict_nac = set_e2n2_hyper_nac(hyp_nac, nac_unit, data.info, splits, shuffle)
        hyp_dict_soc = set_e2n2_hyper_soc(hyp_soc, soc_unit, data.info, splits, shuffle)

        ## retraining has some bug at the moment, do not use
        self.retrain = True if train_mode == 'retrain' else False

        if job_id is None or job_id == 1:
            self.name = f"NN-{title}"
        else:
            self.name = f"NN-{title}-{job_id}"

        self.silent = variables['silent']

        ## prepare unit conversion for energy and force. au or si. The data units are in au.
        kcal_mol_to_ev = 27.211 / 627.5
        h_to_kcal_mol = 627.5
        h_bohr_to_kcal_mol_a = 627.5 / 0.529177249
        h_bohr_to_ev_a = 27.211 / 0.529177249

        # energy grad unit are force to be kcal/mol and kcal/mol/A during training
        self.f_e = h_to_kcal_mol
        self.f_g = h_bohr_to_kcal_mol_a
        self.k_e = kcal_mol_to_ev
        self.k_g = kcal_mol_to_ev

        if nac_unit == 'si':
            self.f_n = h_bohr_to_ev_a  # convert to eV/A
            self.k_n = 1
        else:
            self.f_n = 1  # convert to Eh/B
            self.k_n = h_bohr_to_ev_a

        ## unpack data
        self.atoms = data.atoms
        self.geos = data.geos
        self.xyz = data.xyz
        self.charges = data.charges
        self.cell = data.cell
        self.pbc = data.pbc
        self.energy = data.energy * self.f_e
        self.grad = [(np.array(x) * self.f_g).tolist() for x in data.grad]
        self.nac = [(np.array(x) * self.f_n).tolist() for x in data.nac]
        self.soc = data.soc
        self.pred_atoms = data.pred_atoms
        self.pred_geos = data.pred_geos
        self.pred_xyz = data.pred_xyz
        self.pred_charges = data.pred_charges
        self.pred_cell = data.pred_cell
        self.pred_pbc = data.pred_pbc
        self.pred_energy = data.pred_energy
        self.pred_grad = data.pred_grad
        self.pred_nac = data.pred_nac
        self.pred_soc = data.pred_soc

        if len(elements) == 0:
            unique_atom = list(set(np.concatenate(self.atoms).tolist()))
        else:
            unique_atom = list(set(elements))

        self.elements = sorted([Atom(x).name for x in unique_atom])

        ## find node type
        if len(multiscale) > 0:
            # to differentiate the same element in different regions
            self.mr = Multiregions(self.atoms[0], multiscale)
            atoms = self.mr.partition_atoms(self.atoms[0])
            self.atoms = np.array([atoms] * len(self.atoms))
            self.pred_atoms = np.array([atoms] * len(self.pred_atoms))
        else:
            self.mr = None
            self.atoms = np.array(self.atoms)
            self.pred_atoms = np.array(self.pred_atoms)

        ## initialize model path
        if modeldir is None or job_id not in [None, 1]:
            self.model_path = self.name
        else:
            self.model_path = modeldir

        self.model_register = {
            'energy_grad': False,
            'nac': False,
            'soc': False
        }

        ## initialize hypers and train data
        self.y_dict = {
            'energy_grad': [],
            'nac': [],
            'soc': []
        }

        if nn_eg_type > 0:
            self.y_dict['energy_grad'] = [self.energy.tolist(), self.grad]
            self.model_register['energy_grad'] = True
            hyper_eg = [copy.deepcopy(hyp_dict_eg), copy.deepcopy(hyp_dict_eg)]
        else:
            del self.y_dict['energy_grad']
            self.model_register['energy_grad'] = False
            hyper_eg = []

        if nn_nac_type > 0:
            self.y_dict['nac'] = [None, self.nac]
            self.model_register['nac'] = True
            hyper_nac = [copy.deepcopy(hyp_dict_nac), copy.deepcopy(hyp_dict_nac)]
        else:
            del self.y_dict['nac']
            self.model_register['nac'] = False
            hyper_nac = []

        if nn_soc_type > 0:
            self.y_dict['soc'] = [self.soc.tolist(), None]
            self.model_register['soc'] = True
            hyper_soc = [copy.deepcopy(hyp_dict_soc), copy.deepcopy(hyp_dict_soc)]
        else:
            del self.y_dict['soc']
            self.model_register['soc'] = False
            hyper_soc = []

        self.hypers = hyper_eg + hyper_nac + hyper_soc

        # initialize a model to load a trained method
        ngpu = torch.cuda.device_count()
        if ngpu > 0 and gpu > 0:
            self.device = 'gpu'
        else:
            self.device = 'cpu'

        if ngpu > 0:
            self.device_name = '%s gpu' % ngpu
        else:
            self.device_name = '%s cpu' % os.environ['OMP_NUM_THREADS']

        if ngpu == 0:
            device = None
        elif 0 < ngpu < 2:
            device = [0, 0, 0, 0, 0, 0][:len(self.hypers)]
        elif 2 <= ngpu < 4:
            device = [0, 1, 0, 1, 0, 1][:len(self.hypers)]
        elif 4 <= ngpu < 6:
            device = [0, 1, 2, 3, 2, 3][:len(self.hypers)]
        else:
            device = [0, 1, 2, 3, 4, 5][:len(self.hypers)]

        self.model = ESNNP(self.model_path, self.hypers, self.elements, device=device, silent=self.silent)

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |                                                   |
 |      Excited-State Neural Network Potential       |
 |                                                   |
 |                powered by esnnp                   |
 |                                                   |
 *---------------------------------------------------*

 Number of atoms:  %s
 Number of state:  %s
 Number of NAC:    %s
 Number of SOC:    %s

 Device found: %s
 Running device: %s
 
""" % (
            self.version,
            self.natom,
            self.nstate,
            self.nnac,
            self.nsoc,
            self.device,
            self.device_name,
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

        # xyz = np.concatenate((self.atoms.reshape((-1, self.natom, 1)), self.geos), axis=-1).tolist()
        self.model.build()
        errors = self.model.train(
            self.xyz, self.charges, self.cell, self.pbc, self.y_dict, remote=True, retrain=self.retrain
        )

        if self.model_register['energy_grad']:
            eg_error = errors['energy_grad']
            err_e1 = eg_error[0][0][0]
            err_e2 = eg_error[1][0][0]
            err_g1 = eg_error[0][0][1]
            err_g2 = eg_error[1][0][1]
        else:
            err_e1 = 0
            err_e2 = 0
            err_g1 = 0
            err_g2 = 0

        if self.model_register['nac']:
            nac_error = errors['nac']
            err_n1 = nac_error[0][0][0]
            err_n2 = nac_error[1][0][0]
        else:
            err_n1 = 0
            err_n2 = 0

        if self.model_register['soc']:
            soc_error = errors['soc']
            err_s1 = soc_error[0][0][0]
            err_s2 = soc_error[1][0][0]
        else:
            err_s1 = 0
            err_s2 = 0

        metrics = {
            'e1': err_e1 * self.k_e,
            'g1': err_g1 * self.k_g,
            'n1': err_n1 * self.k_n,
            's1': err_s1,
            'e2': err_e2 * self.k_e,
            'g2': err_g2 * self.k_g,
            'n2': err_n2 * self.k_n,
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
        # must build before load
        self.model.build()
        self.model.load()

        return self

    def appendix(self, _):
        ## fake	function does nothing

        return self

    def _high(self, traj):
        ## run esnnp for high level region in QM calculation
        traj = traj.apply_qmmm()

        atoms = traj.qm_atoms
        coord = traj.qm_coord
        x = [np.concatenate((atoms.reshape((-1, 1)), coord), axis=-1).tolist()]
        charges = [traj.qm2_charge]
        cell = [traj.cell]
        pbc = [traj.pbc]

        results = self.model.predict(x, charges, cell, pbc)
        if self.model_register['energy_grad']:
            pred = results['energy_grad']
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g  # [nstates, n * natoms, 3]
            gradient = gradient.reshape(gradient.shape[0], -1, len(atoms), gradient.shape[2])  # [nstates, n, natoms, 3]
            gradient = np.transpose(gradient, (1, 0, 2, 3))  # [n, nstates, natoms, 3]
            g_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_g
            energy = energy[0]
            gradient = gradient[0]
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = results['nac']
            nac = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_n  # [nstates, n * natoms, 3]
            nac = nac.reshape(nac.shape[0], -1, len(atoms),  nac.shape[2])  # [n, natoms, nstates, 3]
            nac = np.transpose(nac, (1, 0, 2, 3))  # [n, nstates, natoms, 3]
            n_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_n
            nac = nac[0]
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = results['soc']
            soc = np.mean([pred[0][0], pred[1][0]], axis=0)
            s_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1)
            soc = soc[0]
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _high_mid_low(self, traj):
        ## run esnnp for high level region, middle level region, and low level region in QM calculation
        atoms = traj.atoms
        coord = traj.coord
        x = [np.concatenate((atoms.reshape((-1, 1)), coord), axis=-1).tolist()]
        charges = [traj.qm2_charge]
        cell = [traj.cell]
        pbc = [traj.pbc]

        results = self.model.predict(x, charges, cell, pbc)
        if self.model_register['energy_grad']:
            pred = results['energy_grad']
            energy = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            gradient = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g  # [nstates, n * natoms, 3]
            gradient = gradient.reshape(gradient.shape[0], -1, len(atoms), gradient.shape[2])  # [nstates, n, natoms, 3]
            gradient = np.transpose(gradient, (1, 0, 2, 3))  # [n, nstates, natoms, 3]
            g_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_g
            energy = energy[0]
            gradient = gradient[0]
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)
        else:
            energy = []
            gradient = []
            err_e = 0
            err_g = 0

        if self.model_register['nac']:
            pred = results['nac']
            nac = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_n  # [nstates, n * natoms, 3]
            nac = nac.reshape(nac.shape[0], -1, len(atoms),  nac.shape[2])  # [n, natoms, nstates, 3]
            nac = np.transpose(nac, (1, 0, 2, 3))  # [n, nstates, natoms, 3]
            n_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_n
            nac = nac[0]
            err_n = np.amax(n_std)
        else:
            nac = []
            err_n = 0

        if self.model_register['soc']:
            pred = results['soc']
            soc = np.mean([pred[0][0], pred[1][0]], axis=0)
            s_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1)
            soc = soc[0]
            err_s = np.amax(s_std)
        else:
            soc = []
            err_s = 0

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x, charges, cell, pbc):
        ## run esnnp for model testing
        batch = len(x)
        if isinstance(self.pred_grad, np.ndarray):
            batched = True
        else:
            batched = False

        results = self.model.predict(x, charges, cell, pbc, batched=batched)

        if self.model_register['energy_grad']:
            pred = results['energy_grad']
            e_pred = np.mean([pred[0][0], pred[1][0]], axis=0) / self.f_e
            e_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1) / self.f_e
            g_pred = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_g  # [nstates, n * natoms, 3]
            g_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_g

            ref_grad = np.concatenate(self.pred_grad, axis=1)  # [nbatch, nstate, natom, 3]->[nstate, nbatch * natom, 3]
            de = np.abs(self.pred_energy - e_pred)
            dg = np.abs(ref_grad - g_pred)
            de_max = np.amax(de.reshape((batch, -1)), axis=1)

            idx = 0
            dg_max = []
            for g in self.pred_grad:
                start = idx
                end = start + len(g)
                dg_max.append(np.amax(dg[:, start:end]))
                idx += len(g)

            val_out = np.concatenate((self.pred_energy.reshape((batch, -1)), e_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((de.reshape((batch, -1)), e_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-e.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
            nb = ref_grad.shape[0]
            val_out = np.concatenate((ref_grad.reshape(nb, -1).T, g_pred.reshape(nb, -1).T), axis=1)
            std_out = np.concatenate((dg.reshape(nb, -1).T, g_std.reshape(nb, -1).T), axis=1)
            np.savetxt('%s-g.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            de_max = np.zeros(batch)
            dg_max = np.zeros(batch)

        if self.model_register['nac']:
            pred = results['nac']
            n_pred = np.mean([pred[0][1], pred[1][1]], axis=0) / self.f_n  # [nstates, n * natoms, 3]
            n_std = np.std([pred[0][1], pred[1][1]], axis=0, ddof=1) / self.f_n

            ref_nac = np.concatenate(self.pred_nac, axis=1)  # [nbatch, nstate, natom, 3] -> [nstate, nbatch * natom, 3]
            dn = np.abs(ref_nac - n_pred)

            idx = 0
            dn_max = []
            for g in self.pred_nac:
                start = idx
                end = start + len(g)
                dn_max.append(np.amax(dn[:, start:end]))
                idx += len(g)

            nb = ref_nac[0]
            val_out = np.concatenate((ref_nac.reshape(nb, -1).T, n_pred.reshape(nb, -1).T), axis=1)
            std_out = np.concatenate((dn.reshape(nb, -1).T, n_std.reshape(nb, -1).T), axis=1)
            np.savetxt('%s-n.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))
        else:
            dn_max = np.zeros(batch)

        if self.model_register['soc']:
            pred = results['soc']
            s_pred = np.mean([pred[0][0], pred[1][0]], axis=0)
            s_std = np.std([pred[0][0], pred[1][0]], axis=0, ddof=1)

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
        ## main function to run esnnp and communicate with other PyRAI2MD modules

        if self.jobtype == 'prediction' or self.jobtype == 'predict':
            # xyz = np.concatenate((self.pred_atoms.reshape((-1, self.natom, 1)), self.pred_geos), axis=-1).tolist()
            self._predict(self.pred_xyz, self.pred_charges, self.pred_cell, self.pred_pbc)
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
