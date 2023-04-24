######################################################
#
# PyRAI2MD 2 module for ML-QC mixed molecular dynamics
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

import copy
import numpy as np

from PyRAI2MD.Dynamics.aimd import AIMD
from PyRAI2MD.Utils.coordinates import print_coord

class MIXAIMD(AIMD):
    """ Ab initial molecular dynamics class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            ref              class       reference QM method
            job_id           int         trajectory id index
            job_dir          boolean     create a subdirectory

        Attributes:          Type:
            ref_energy       int         use reference energy for hybrid ML/QM molecular dynamics
            ref_grad         int         use reference gradient for hybrid ML/QM molecular dynamics
            ref_nac          int         use reference nac for hybrid ML/QM molecular dynamics
            ref_soc          int         use reference soc for hybrid ML/QM molecular dynamics

        Functions:           Returns:
            run              class       run molecular dynamics simulation

    """

    def __init__(self, trajectory=None, keywords=None, qm=None, ref=None, job_id=None, job_dir=None):
        super().__init__(trajectory=trajectory, keywords=keywords, qm=qm, job_id=job_id, job_dir=job_dir)
        ## initialize variables for mixed dynamics
        self.ref_energy = keywords['md']['ref_energy']
        self.ref_grad = keywords['md']['ref_grad']
        self.ref_nac = keywords['md']['ref_nac']
        self.ref_soc = keywords['md']['ref_soc']

        ## create a reference electronic method object
        self.REF = ref

    def _potential_energies(self, traj):
        ## modify the potential energy calculation to mixed mode
        traj_qm = self.QM.evaluate(traj)
        traj_ref = self.REF.evaluate(copy.deepcopy(traj))
        traj_mix = self._mix_properties(traj_qm, traj_ref)

        return traj_mix

    def _mix_properties(self, traj_qm, traj_ref):
        info = ''
        if self.ref_energy == 1:
            traj_qm.energy = np.copy(traj_ref.energy)

        if self.ref_grad == 1:
            traj_qm.grad = np.copy(traj_ref.grad)

        if self.ref_nac == 1:
            traj_qm.grad = np.copy(traj_ref.nac)

        if self.ref_soc == 1:
            traj_qm.soc = np.copy(traj_ref.soc)

        pot = ' '.join(['%28.16f' % x for x in traj_ref.energy])
        info += """
  &reference energy
-------------------------------------------------------
%s
-------------------------------------------------------
""" % pot

        for n in range(traj_ref.nstate):
            try:
                grad = traj_ref.grad[n]
                info += """
  &reference gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (n + 1, print_coord(np.concatenate((traj_ref.atoms, grad), axis=1)))

            except IndexError:
                info += """
  &reference gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
  Not Computed
-------------------------------------------------------------------------------
""" % (n + 1)

        for n, pair in enumerate(traj_ref.nac_coupling):
            s1, s2 = pair
            m1 = traj_ref.statemult[s1]
            m2 = traj_ref.statemult[s2]
            try:
                coupling = traj_ref.nac[n]
                info += """
  &reference nonadiabatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2, print_coord(np.concatenate((traj_ref.atoms, coupling), axis=1)))

            except IndexError:
                info += """
  &reference nonadiabatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
  Not computed
-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2)

        soc_info = ''
        for n, pair in enumerate(traj_ref.soc_coupling):
            s1, s2 = pair
            m1 = traj_ref.statemult[s1]
            m2 = traj_ref.statemult[s2]
            try:
                coupling = traj_ref.soc[n]
                soc_info += '  <H>=%10.4f            %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    coupling, s1 + 1, s2 + 1, m1, m2)

            except IndexError:
                soc_info += '  Not computed              %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    s1 + 1, s2 + 1, m1, m2)
                
        info += """
  &reference spin-orbit coupling
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % soc_info

        traj_qm.mixinfo = info

        return traj_qm
