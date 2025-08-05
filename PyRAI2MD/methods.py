######################################################
#
# PyRAI2MD 2 module for loading QM and ML method
#
# Author Jingbai Li
# Sep 18 2021
#
######################################################

from PyRAI2MD.Quantum_Chemistry.qc_bagel import Bagel
from PyRAI2MD.Quantum_Chemistry.qc_molcas import Molcas
from PyRAI2MD.Quantum_Chemistry.qc_molcas_tinker import MolcasTinker
from PyRAI2MD.Quantum_Chemistry.qc_orca import Orca
from PyRAI2MD.Quantum_Chemistry.qc_openqp import OpenQP
from PyRAI2MD.Quantum_Chemistry.qc_xtb import Xtb
from PyRAI2MD.Quantum_Chemistry.qmqm2 import QMQM2
from PyRAI2MD.Machine_Learning.model_NN import DNN
from PyRAI2MD.Machine_Learning.model_demo import Demo
from PyRAI2MD.Machine_Learning.model_helper import DummyModel


try:
    from PyRAI2MD.Machine_Learning.model_pyNNsMD import MLP
except ModuleNotFoundError:
    MLP = DummyModel

try:
    from PyRAI2MD.Machine_Learning.model_pyNNsMD import Schnet
except ModuleNotFoundError:
    Schnet = DummyModel

try:
    from PyRAI2MD.Machine_Learning.model_gcnnp import E2N2Demo

except ModuleNotFoundError:
    E2N2Demo = DummyModel

try:
    from PyRAI2MD.Machine_Learning.model_esnnp import E2N2

except ModuleNotFoundError:
    E2N2 = DummyModel

try:
    from PyRAI2MD.Machine_Learning.model_DimeNet import DimenetModel

except ModuleNotFoundError:
    DimenetModel = DummyModel

class QM:
    """ Electronic structure method class

        Parameters:          Type:
            qm               str         electronic structure method
            keywords         dict        input keywords
            job_id           int         calculation ID
            runtype          str         type of calculation, 'qm', 'qmmm', or 'qmmm_low'

        Attribute:           Type:

        Functions:           Returns:
            train            self        train a model if qm == 'nn'
            load             self        load a model if qm == 'nn'
            appendix         self        add more information to the selected method
            evaluate         self        run the selected method

    """

    def __init__(self, qm, keywords=None, job_id=None):
        # methods available for single region calculation

        if qm[0] == 'openqp':
            try:
                from mpi4py import MPI
            except ModuleNotFoundError:
                pass
                
        qm_list = {
            'molcas': Molcas,
            'mlctkr': MolcasTinker,
            'bagel': Bagel,
            'orca': Orca,
            'openqp': OpenQP,
            'xtb': Xtb,
            'nn': DNN,
            'demo': Demo,
            'mlp': MLP,
            'schnet': Schnet,
            'e2n2_demo': E2N2Demo,
            'e2n2': E2N2,
            'dimenet': DimenetModel,
        }

        # methods available for QM 1 region calculation
        qm1_list = {
            'molcas': Molcas,
            'bagel': Bagel,
            'orca': Orca,
            'openqp': OpenQP,
            'xtb': Xtb,
            'nn': DNN,
            'demo': Demo,
            'mlp': MLP,
            'schnet': Schnet,
            'e2n2_demo': E2N2Demo,
            'e2n2': E2N2,
            'dimenet': DimenetModel,
        }

        # methods available for QM 2 region calculation
        qm2_list = {
            'xtb': Xtb,
        }

        # methods available for MM calculation
        mm_list = {
            'xtb': Xtb,
        }

        if len(qm) == 1:
            if isinstance(job_id, list):
                job_id = job_id[0]
            self.method = qm_list[qm[0]](keywords=keywords, job_id=job_id)  # This should pass hypers
        else:
            job_id_1 = None
            job_id_2 = None
            if isinstance(job_id, list):
                job_id_1 = job_id[0]
                job_id_2 = job_id[1]
            qm1 = qm1_list[qm[0]]
            qm2 = qm2_list[qm[1]]

            if len(qm) >= 3:
                mm = mm_list[qm[2]]
            else:
                mm = False

            self.method = QMQM2(methods=[qm1, qm2, mm], keywords=keywords, job_id_1=job_id_1, job_id_2=job_id_2)

    def train(self):
        metrics = self.method.train()
        return metrics

    def load(self):  # This should load model
        self.method.load()
        return self

    def appendix(self, addons):  # appendix function to pass more info for different methods
        self.method.appendix(addons)
        return self

    def evaluate(self, traj):
        traj = self.method.evaluate(traj)
        return traj

    def get_method(self):

        return self.method

    def read_data(self, natom):
        data = self.method.read_data(natom)
        return data
