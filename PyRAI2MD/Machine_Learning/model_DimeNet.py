#####################################################
#
# PyRAI2MD 2 module for interface template
#
# Author Jingbai Li
# Sep 22 2021
#
######################################################

import os
import time
import torch.cuda
import numpy as np
# from PyRAI2MD.Machine_Learning.hyper_dimenet import set_hyper_eg
from PyRAI2MD.Machine_Learning.hyper_dimenet import set_hyper_nac
# from PyRAI2MD.Machine_Learning.hyper_dimenet import set_hyper_soc

from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long

# Sijin
from PyRAI2MD.Machine_Learning.Dimenet import DimenetNAC


class DimenetModel:
    """ pyNNsMD interface

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            hyp_eg           dict        Hyperparameters of energy gradient NN
       	    hyp_nac          dict        Hyperparameters of nonadiabatic coupling NN
       	    hyp_soc          dict     	 Hyperparameters of spin-orbit coupling NN
       	    atoms            list        atom list
            geos             ndarray     training set coordinates
            energy           ndarray     energy array
            grad             ndarray     gradient array
            nac              ndarray     nonadiabatic coupling array
            soc              ndarray     spin-orbit coupling array
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

        # basic setting, don't change
        self.runtype = runtype
        title = keywords['control']['title']
        variables = keywords['dimenet'].copy()
        modeldir = variables['modeldir']
        data = variables['data']
        # nn_eg_type = variables['nn_eg_type']
        # nn_nac_type = variables['nn_nac_type']
        # nn_soc_type = variables['nn_soc_type']
        eg_unit = variables['eg_unit']
        nac_unit = variables['nac_unit']
        gpu = variables['gpu']
        shuffle = variables['shuffle']
        self.jobtype = keywords['control']['jobtype']
        self.version = keywords['version']
        self.ncpu = keywords['control']['ml_ncpu']
        self.train_mode = variables['train_mode']
        self.shuffle = variables['shuffle']
        self.natom = data.natom
        self.nstate = data.nstate
        self.nnac = data.nnac
        self.nsoc = data.nsoc

        ## assign folder name and parse data, don't change
        self.train_mode = 'training'
        if job_id is None or job_id == 1:
            self.name = f"NN-{title}"
        else:
            self.name = f"NN-{title}-{job_id}"

        ## assume the initialization read a path to create the model
        if modeldir is None or job_id not in [None, 1]:
            modeldir = self.name

        ## update default hyperparameters
        hyp_nac = variables['dime_nac'].copy()
        hyp_dict_nac = set_hyper_nac(modeldir, hyp_nac, data.info, shuffle, gpu)
        """
        ## below are template for updating parameters of the energy_grad, nac, and soc models
        hyp_eg = variables['dime_eg'].copy()
        hyp_nac = variables['dime_nac'].copy()
        hyp_soc = variables['dime_soc'].copy()
        hyp_dict_eg = set_hyper_eg(hyp_eg, eg_unit, data.info, splits)
        hyp_dict_nac = set_hyper_nac(hyp_nac, nac_unit, data.info, splits)
        hyp_dict_soc = set_hyper_soc(hyp_soc, soc_unit, data.info, splits)
        """
        self.silent = variables['silent']

        ## convert unit of energy and force.
        ## data are assumed in au unit (Hartree, Hartree/Bohr), si convert them to eV, eV/A for training
        ## Note the logfile units are always in eV, eV/A, but the prediction values are in their original units.
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

        self.nac = data.nac * self.f_n
        self.atoms = data.atoms
        self.pred_geos = data.pred_geos
        self.pred_energy = data.pred_energy
        self.pred_grad = data.pred_grad
        self.pred_nac = data.pred_nac
        self.pred_soc = data.pred_soc

        ## pack training data
        self.data = {
            'xyz': np.concatenate((np.array(data.atoms).reshape((-1, data.natom, 1)), data.geos), axis=-1).tolist(),
            'nac': self.nac.tolist()
        }

        """
        ## pack data for different models
        self.y = {}
        if nn_eg_type > 0:  # request energy force training
            # self.y =
            pass
        if nn_nac_type > 0:  # request energy force training
            # self.y =
            pass
        if nn_soc_type > 0:  # request energy force training
            # self.y =
            pass


        ## pack hyperparameters for models
        self.hyper = {}
        if nn_eg_type == 1:  # same architecture with different weight
            self.hyper['energy_gradient'] = hyp_dict_eg
        elif nn_eg_type > 1:
            self.hyper['energy_gradient'] = [hyp_dict_eg, hyp_dict_eg2]

        if nn_nac_type == 1:  # same architecture with different weight
            self.hyper['nac'] = hyp_dict_nac
        elif nn_nac_type > 1:
            self.hyper['nac'] = [hyp_dict_nac, hyp_dict_nac2]

        if nn_soc_type == 1:  # same architecture with different weight
            self.hyper['soc'] = hyp_dict_soc
        elif nn_soc_type > 1:
            self.hyper['soc'] = [hyp_dict_soc, hyp_dict_soc2]
        """

        ## setup GPU list
        ngpu = torch.cuda.device_count()
        if ngpu > 0 and gpu > 0:
            self.device = 'gpu'
        else:
            self.device = 'cpu'

        if ngpu > 0:
            self.device_name = torch.cuda.current_device()
        else:
            self.device_name = 'cpu'

        ## initialize model
        self.model = DimenetNAC(hyp_dict_nac)

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |             DimeNet NAC model only                |
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

        # fitting model
        # Sijin: already set my fit functon to return a ferr dictionary, but I only have one model
        # ferr['nac'][model_1_nac_err]
        ferr = self.model.fit(self.data)

        # ferr should be a dict as following
        """
        ferr = {
            'energy_gradient':
                [
                    [model_1_energy_err, model_1_force_err],
                    [model_2_energy_err, model_2_force_err]
                ]
            'nac':
                [model_1_nac_err, model_2_nac_err]
            'soc':
                [model_2_soc_err, model_2_soc_err]
        }
        
        """

        # self.model.save()

        err_e1 = 0
        err_e2 = 0
        err_g1 = 0
        err_g2 = 0
        err_n1 = 0
        err_n2 = 0
        err_s1 = 0
        err_s2 = 0
        if 'energy_gradient' in ferr.keys():
            err_e1 = ferr['energy_gradient'][0][0]
            err_e2 = ferr['energy_gradient'][1][0]
            err_g1 = ferr['energy_gradient'][0][1]
            err_g2 = ferr['energy_gradient'][1][1]

        if 'nac' in ferr.keys():
            err_n1 = ferr['nac'][0]
            err_n2 = 0

        if 'soc' in ferr.keys():
            err_s1 = ferr['soc'][0]
            err_s2 = ferr['soc'][1]

        metrics = {
            'e1': err_e1 * self.k_e,
            'g1': err_g1 * self.k_g,
            'n1': err_n1 * self.k_n,
            's1': err_s1,
            'e2': err_e2 * self.k_e,
            'g2': err_g2 * self.k_g,
            'n2': err_n2 * self.k_n,
            's2': err_s2}

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
        self.model.load_model()

        return self

    def appendix(self, _):
        ## fake	function does nothing

        return self

    def _high(self, traj):
        ## run dimenet for high level region in QM calculation
        traj = traj.apply_qmmm()

        xyz = traj.qm_coord.reshape((1, self.natom, 3))
        y_pred, y_std = self.model.predict(xyz)

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

        if 'soc' in y_pred.keys():
            s_pred = y_pred['soc']
            s_std = y_std['soc']
            soc = s_pred[0]
            err_s = np.amax(s_std)

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _high_mid_low(self, traj):
        ## run dimenet for high level region, middle level region, and low level region in QM calculation

        xyz = traj.coord.reshape((1, self.natom, 3))
        y_pred, y_std = self.model.predict(xyz)

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

        if 'soc' in y_pred.keys():
            s_pred = y_pred['soc']
            s_std = y_std['soc']
            soc = s_pred[0]
            err_s = np.amax(s_std)

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x):
        ## run psnnsmd for model testing

        batch = len(x)
        x = np.concatenate((np.repeat(self.atoms[0], batch).reshape((-1, self.natom, 1)), x), axis=-1).tolist()

        y_pred, y_std = self.model.predict(x)

        ## load values from prediction set
        pred_e = self.pred_energy
        pred_g = self.pred_grad
        pred_n = self.pred_nac
        pred_s = self.pred_soc

        ## initialize errors
        de_max = np.zeros(batch)
        dg_max = np.zeros(batch)
        dn_max = np.zeros(batch)
        ds_max = np.zeros(batch)

        ## update errors
        if 'energy_gradient' in y_pred.keys():
            e_pred = np.array(y_pred['energy_gradient'][0]) / self.f_e
            g_pred = np.array(y_pred['energy_gradient'][1]) / self.f_g
            e_std = np.array(y_std['energy_gradient'][0]) / self.f_e
            g_std = np.array(y_std['energy_gradient'][1]) / self.f_g
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
            n_pred = np.array(y_pred['nac']) / self.f_n
            n_std = np.array(y_std['nac']) / self.f_n
            dn = np.abs(pred_n - n_pred)
            dn_max = np.amax(dn.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((pred_n.reshape((batch, -1)), n_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((dn.reshape((batch, -1)), n_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-n.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

        if 'soc' in y_pred.keys():
            s_pred = np.array(y_pred['soc'])
            s_std = np.array(y_std['soc'])
            ds = np.abs(pred_s - s_pred)
            ds_max = np.amax(ds.reshape((batch, -1)), axis=1)

            val_out = np.concatenate((pred_s.reshape((batch, -1)), s_pred.reshape((batch, -1))), axis=1)
            std_out = np.concatenate((ds.reshape((batch, -1)), s_std.reshape((batch, -1))), axis=1)
            np.savetxt('%s-s.pred.txt' % self.name, np.concatenate((val_out, std_out), axis=1))

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
