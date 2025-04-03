######################################################
#
# PyRAI2MD 2 module for handle module not found error
#
# Author Jingbai Li
# Aug 31 2022
#
######################################################

import sys
import numpy as np
from PyRAI2MD.Molecule.atom import Atom

class DummyModel:
    """ dummy model to return error after instantiating the model

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            warning          str         exit message

        Functions:           Returns:
            train            None        exit
            load             None        exit
            appendix         None        exit
            evaluate         None        exit

    """
    def __init__(self, keywords=None, job_id=None):
        model_name = keywords['control']['qm'][0]
        lib = {
            'mlp': 'pyNNsMD',
            'schnet': 'pyNNsMD',
            'e2n2_demo': 'GCNNP',
            'e2n2': 'ESNNP',
            'dimenet': 'DimenetNAC',
            'job_id': job_id,

        }

        self.warning = '\n PyRAI2MD: You have not installed %s, %s model is not available\n' % (
            lib[model_name],
            model_name
        )

    def train(self):
        sys.exit(self.warning)

    def load(self):
        sys.exit(self.warning)

    def appendix(self, _):
        sys.exit(self.warning)

    def evaluate(self, _):
        sys.exit(self.warning)

class EmptyModel:
    """ empty model to skip training or prediction

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            warning          str         exit message

        Functions:           Returns:
            train            None        exit
            load             None        exit
            appendix         None        exit
            evaluate         None        exit

    """
    def __init__(self, _, __):
        self.info = ''

    def info(self):
        return self.info

    @staticmethod
    def train():
        return 0

    def load(self):
        return self

    def appendix(self, _):
        return self

    @staticmethod
    def evaluate(_):
        return 0


class Multiregions:
    """ partition atoms into multiregions

        Parameters:          Type:
            atoms            list        atom list
            multiscale_list  list        atom indices in multiscale regions

        Attribute:           Type:
            warning          str         exit message

        Functions:           Returns:
            partition_atoms        list        transformed atom list
            partition_atomic_numbers list      transformed atomic number list
            retrieve_atoms        list   transformed back atom list
            retrieve_atomic_numbers list transformed back atomic number list
            update_xyz          list        transformed xyz array

    """
    def __init__(self, atoms, multiscale_list, flag=50):
        self.flag = flag
        multiscale_index = np.zeros(len(atoms)).astype(int)
        for n, region in enumerate(multiscale_list):
            for indx in region:
                multiscale_index[indx - 1] = n

        self.multiscale_index = multiscale_index

    def update_xyz(self, xyz):
        xyz = np.array(xyz)
        xyz[:, :, 0] += self.multiscale_index * self.flag
        node_type = np.unique(xyz[:, :, 0]).astype(int).tolist()
        xyz = xyz.tolist()

        return node_type, xyz

    def partition_atoms(self, atoms):
        z = np.array([Atom(atom).name for atom in atoms])
        z += self.multiscale_index * self.flag
        atoms = [Atom(x).get_symbol() for x in z]

        return atoms

    def retrieve_atoms(self, atoms):
        z = np.array([Atom(atom).name for atom in atoms])
        z -= self.multiscale_index * self.flag
        atoms = [Atom(x).get_symbol() for x in z]

        return atoms

    def partition_atomic_numbers(self, atomic_numbers):
        atomic_numbers = np.array(atomic_numbers)
        atomic_numbers += self.multiscale_index * self.flag

        return atomic_numbers.tolist()

    def retrieve_atomic_numbers(self, atomic_numbers):
        atomic_numbers = np.array(atomic_numbers)
        atomic_numbers -= self.multiscale_index * self.flag

        return atomic_numbers.tolist()
