#####################################################
#
# PyRAI2MD 2 module for processing training data
#
# Author Jingbai Li
# Sep 22 2021
#
######################################################

import os
import sys
import json
import numpy as np

from PyRAI2MD.Utils.coordinates import atomic_number

class Data:
    """ Training data class

        Parameters:          Type:
            None

        Attribute:           Type:       
            natom            int         number of atoms
            nstate           int         number of states
            nnac             int         number of nonadiabatic couplings
       	    nsoc             int    	 number of spin-orbit couplings
       	    info             dict        data size info dict
            xyz              ndarray     coordinates array
            charges          ndarray     embedding charges
            cell             ndarray     lattice vectors
            pbc              ndarray     periodic boundary condition
            energy           ndarray     energy array
            grad             ndarray     gradient array
            nac              ndarray     nonadiabatic coupling array
            soc              ndarray     spin-orbit coupling array
            atoms            list        atom list
            geos             ndarray     training set coordinates
            pred_xyz         ndarray     prediction set coordinates array
            pred_atoms       list        prediction set atom list
            pred_geos        ndarray     prediction set coordinates
            pred_charges     ndarray     prediction set embedding charges
            pred_cell        ndarray     prediction set lattice vectors
            pred_pbc         ndarray     prediction set periodic boundary condition
            pred_energy      ndarray     prediction set target energy
            pred_grad        ndarray     prediction set target grad
            pred_nac         ndarray     prediction set target nac
            pred_soc         ndarray     prediction set target soc
            atomic_numbers   list        atomic number list
            max_xx           float       maximum value
            min_xx           float       minimum value
            mid_xx           float       middle value
            dev_xx           float       deviation value
            avg_xx           float       mean value
            std_xx           float       standard deviation

        Functions:           Returns:
            load             self        load data
            append           self        add new data
            save             self        save data
            stat             self        update data statistics (max, min, mid, dev, mean, std)
    """

    def __init__(self):

        self.natom = 0
        self.nstate = 0
        self.nnac = 0
        self.nsoc = 0
        self.info = {}
        self.xyz = np.zeros(0)
        self.charges = np.zeros(0)
        self.cell = np.zeros(0)
        self.pbc = np.zeros(0)
        self.energy = np.zeros(0)
        self.grad = np.zeros(0)
        self.nac = np.zeros(0)
        self.soc = np.zeros(0)
        self.atoms = np.zeros(0)
        self.geos = np.zeros(0)
        self.pred_xyz = np.zeros(0)
        self.pred_atoms = np.zeros(0)
        self.pred_geos = np.zeros(0)
        self.pred_charges = np.zeros(0)
        self.pred_cell = np.zeros(0)
        self.pred_pbc = np.zeros(0)
        self.pred_energy = np.zeros(0)
        self.pred_grad = np.zeros(0)
        self.pred_nac = np.zeros(0)
        self.pred_soc = np.zeros(0)
        self.atomic_numbers = []
        self.max_energy = 0
        self.max_grad = 0
        self.max_nac = 0
        self.max_soc = 0
        self.min_energy = 0
        self.min_grad = 0
        self.min_nac = 0
        self.min_soc = 0
        self.mid_energy = 0
        self.mid_grad = 0
        self.mid_nac = 0
        self.mid_soc = 0
        self.dev_energy = 0
        self.dev_grad = 0
        self.dev_nac = 0
        self.dev_soc = 0
        self.avg_energy = 0
        self.avg_grad = 0
        self.avg_nac = 0
        self.avg_soc = 0
        self.std_energy = 0
        self.std_grad = 0
        self.std_nac = 0
        self.std_soc = 0

    def _load_training_data(self, file):
        with open('%s' % file, 'r') as indata:
            data = json.load(indata)

        if isinstance(data, list):  # old format
            natom, nstate, xyz, invr, energy, grad, nac, ci, mo = data
            self.natom = int(natom)
            self.nstate = int(nstate)
            self.nnac = int(nstate * (nstate - 1) / 2)
            self.xyz = np.array(xyz)
            self.energy = np.array(energy)
            self.grad = np.array(grad)
            self.nac = np.array(nac)

        elif isinstance(data, dict):  # new format
            self.natom = int(data['natom'])
            self.nstate = int(data['nstate'])
            self.nnac = int(data['nnac'])
            self.nsoc = int(data['nsoc'])
            self.xyz = np.array(data['xyz'])
            self.energy = np.array(data['energy'])
            self.grad = np.array(data['grad'])
            self.nac = np.array(data['nac'])
            self.soc = np.array(data['soc'])

            try:
                self.charges = np.array(data['charges'])
            except KeyError:
                pass

            try:
                self.cell = np.array(data['cell'])
            except KeyError:
                pass

            try:
                self.pbc = np.array(data['pbc'])
            except KeyError:
                pass

        else:
            sys.exit('\n  FileTypeError\n  PyRAI2MD: cannot recognize training data format %s' % file)

        self.atoms = self.xyz[:, :, 0].astype(str).tolist()
        self.geos = self.xyz[:, :, 1: 4].astype(float)
        self.atomic_numbers = [atomic_number(atom) for atom in self.atoms]
        self.info = {
            'natom': self.natom,
            'nstate': self.nstate,
            'nnac': self.nnac,
            'nsoc': self.nsoc,
        }

        return self

    def _load_prediction_data(self, file):
        with open('%s' % file, 'r') as indata:
            data = json.load(indata)

        if isinstance(data, list):  # old format
            natom, nstate, xyz, invr, energy, grad, nac, ci, mo = data
            self.pred_xyz = np.array(xyz)
            self.pred_energy = np.array(energy)
            self.pred_grad = np.array(grad)
            self.pred_nac = np.array(nac)
            self.pred_soc = 0

        elif isinstance(data, dict):  # new format
            self.pred_xyz = np.array(data['xyz'])
            self.pred_energy = np.array(data['energy'])
            self.pred_grad = np.array(data['grad'])
            self.pred_nac = np.array(data['nac'])
            self.pred_soc = np.array(data['soc'])

            try:
                self.charges = np.array(data['charges'])
            except KeyError:
                pass

            try:
                self.cell = np.array(data['cell'])
            except KeyError:
                pass

            try:
                self.pbc = np.array(data['pbc'])
            except KeyError:
                pass

        else:
            sys.exit('\n  FileTypeError\n  PyRAI2MD: cannot recognize prediction data format %s' % file)

        self.pred_atoms = self.pred_xyz[:, :, 0].astype(str).tolist()
        self.pred_geos = self.pred_xyz[:, :, 1: 4].astype(float)
        return self

    def load(self, file, filetype='train'):
        if not os.path.exists(file):
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for training data  %s for %s' % (file, filetype))

        if filetype == 'train':
            self._load_training_data(file)
        elif filetype == 'prediction':
            self._load_prediction_data(file)
        else:
            sys.exit('\n  TypeError\n  PyRAI2MD: load data for train or prediction, %s was unrecognized' % filetype)

        return self

    def save(self, file):
        batch = len(self.xyz)
        data = {
            'natom': self.natom,
            'nstate': self.nstate,
            'nnac': self.nnac,
            'nsoc': self.nsoc,
            'xyz': self.xyz.tolist(),
            'charges': self.charges.tolist(),
            'cell': sefl.cell.tolist(),
            'pbc': self.pbc.tolist(),
            'energy': self.energy.tolist(),
            'grad': self.grad.tolist(),
            'nac': self.nac.tolist(),
            'soc': self.soc.tolist(),
            }

        with open('New-data%s-%s.json' % (batch, file), 'w') as outdata:
            json.dump(data, outdata)

        return self

    def append(self, newdata):
        new_xyz, new_charges, new_cell, new_pbc, new_energy, new_grad, new_nac, new_soc = newdata
        self.xyz = np.concatenate((self.xyz, new_xyz))
        self.energy = np.concatenate((self.energy, new_energy))
        self.grad = np.concatenate((self.grad, new_grad))
        self.nac = np.concatenate((self.nac, new_nac))
        self.soc = np.concatenate((self.soc, new_soc))
        self.atoms = np.array(self.xyz[:, :, 0]).astype(str).tolist()
        self.geos = np.array(self.xyz[:, :, 1: 4]).astype(float)
        self.charges = np.concatenate((self.charges, new_charges))
        self.cell = np.concatenate((self.cell, new_cell))
        self.pbc = np.concatenate((self.pbc, new_pbc))

        return self

    def stat(self):
        if len(self.energy[0]) > 0:
            self.max_energy = np.amax(self.energy)
            self.min_energy = np.amin(self.energy)
            self.mid_energy = (self.max_energy + self.min_energy) / 2
            self.dev_energy = (self.max_energy - self.min_energy) / 2
            self.avg_energy = np.mean(self.energy)
            self.std_energy = np.std(self.energy)

        if len(self.grad[0]) > 0:
            self.max_grad = np.amax(self.grad)
            self.min_grad = np.amin(self.grad)
            self.mid_grad = (self.max_grad + self.min_grad) / 2
            self.dev_grad = (self.max_grad - self.min_grad) / 2
            self.avg_grad = np.mean(self.grad)
            self.std_grad = np.std(self.grad)

        if len(self.nac[0]) > 0:
            self.max_nac = np.amax(self.nac)
            self.min_nac = np.amin(self.nac)
            self.mid_nac = (self.max_nac + self.min_nac) / 2
            self.dev_nac = (self.max_nac - self.min_nac) / 2
            self.avg_nac = np.mean(self.nac)
            self.std_nac = np.std(self.nac)

        if len(self.soc[0]) > 0:
            self.max_soc = np.amax(self.soc)
            self.min_soc = np.amin(self.soc)
            self.mid_soc = (self.max_soc + self.min_soc) / 2
            self.dev_soc = (self.max_soc - self.min_soc) / 2
            self.avg_soc = np.mean(self.soc)
            self.std_soc = np.std(self.soc)

        return self
