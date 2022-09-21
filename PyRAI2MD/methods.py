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
from PyRAI2MD.Quantum_Chemistry.qc_xtb import Xtb
from PyRAI2MD.Machine_Learning.model_NN import DNN
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
    from PyRAI2MD.Machine_Learning.model_GCNNP import E2N2

except ModuleNotFoundError:
    E2N2 = DummyModel

class QM:
    """ Electronic structure method class

        Parameters:          Type:
            qm               str         electronic structure method
            keywords         dict        input keywords
            id               int         calculation ID

        Attribute:           Type:

        Functions:           Returns:
            train            self        train a model if qm == 'nn'
            load             self        load a model if qm == 'nn'
            appendix         self        add more information to the selected method
            evaluate         self        run the selected method

    """

    def __init__(self, qm, keywords=None, job_id=None):
        qm_list = {
            'molcas': Molcas,
            'mlctkr': MolcasTinker,
            'bagel': Bagel,
            'orca': Orca,
            'xtb': Xtb,
            'nn': DNN,
            'mlp': MLP,
            'schnet': Schnet,
            'e2n2': E2N2,
        }

        self.method = qm_list[qm](keywords=keywords, job_id=job_id)  # This should pass hypers

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
