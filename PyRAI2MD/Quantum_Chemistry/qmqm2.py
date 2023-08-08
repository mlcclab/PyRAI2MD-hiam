######################################################
#
# PyRAI2MD 2 module for QM/QM2 calculations
#
# Author Jingbai Li
# Sep 22 2022
#
######################################################

import copy
import numpy as np
from PyRAI2MD.Utils.coordinates import orca_coord
from PyRAI2MD.Utils.coordinates import string2float

class QMQM2:
    """ QMQM2 single point calculation interface

        Parameters:          Type:
            methods          list        a list of method class
            keywords         dict        keywords dict
            job_id_1         int         iteration index of trained NN, used only for adaptive sampling
            job_id_2         int         trajectory index, used only for adaptive sampling

        Attribute:           Type:
            natom            int         number of atoms.
            nstate           int         number of electronic states
            nnac             int         number of non-adiabatic couplings
            nsoc             int         number of spin-orbit couplings
            state            int         the current state
            sf_state         int         the sf_state corresponding to current state
            sf_state_list    list        a list of singlet sf_state
            activestate      int         compute gradient only for the current state
            ci               list        number of state per spin multiplicity
            mult             list        spin multiplicity
            nac_coupling     list        non-adiabatic coupling pairs
            soc_coupling     list        spin-orbit coupling pairs
            keep_tmp         int  	     keep the Molcas calculation folders (1) or not (0).
            verbose          int	     print level.
            project          str	     calculation name.
            workdir          str	     Orca calculation folder.
            dft_type         str         type of DFT calculation, dft, tddft, sf_tddft
            orca             str	     ORCA environment variable, executable folder.
            nproc            int	     number of CPUs for parallelization
            mpi              str	     path to mpi library
            use_hpc          int	     use HPC (1) for calculation or not(0), like SLURM.

        Functions:           Returns:
            train            self        fake function
            load             self        fake function
            appendix         self        fake function
            evaluate         self        run single point calculation

    """

    def __init__(self, methods=None, keywords=None, job_id_1=None, job_id_2=None):

        qm = methods[0]
        qm2 = methods[1]
        mm = methods[2]
        self.qm_high = qm(keywords=keywords, job_id=job_id_1, runtype='qm_high')
        self.qm2_high = qm2(keywords=keywords, job_id=job_id_2, runtype='qm2_high')
        self.qm2_mid = qm2(keywords=keywords, job_id=job_id_2, runtype='qm2_high_mid')

        if mm:
            self.do_mm = True
            self.mm_mid = mm(keywords=keywords, job_id=job_id_2, runtype='mm_high_mid')
            self.mm_low = mm(keywords=keywords, job_id=job_id_2, runtype='mm_high_mid_low')
        else:
            self.do_mm = False

        self.nprocs = keywords['control']['ms_ncpu']
        self.project = ''
        self.workdir = ''

    def train(self):
        ## fake function

        return self

    def load(self):
        # This should load model
        self.qm_high.load()
        self.qm2_high.load()
        self.qm2_mid.load()

        if self.do_mm:
            self.mm_mid.load()
            self.mm_low.load()

        return self

    def appendix(self, _):
        ## fake function

        return self

    def eval_wrapper(self, var):
        ifunc, traj = var
        if ifunc == 0:
            func = self.qm_high.evaluate
        elif ifunc == 1:
            func = self.qm2_high.evaluate
        else:
            func = None

        results = func(traj)

        return ifunc, results

    def evaluate(self, traj):
        natom = traj.natom
        nqmqm2 = traj.nhigh + traj.nmid
        nstate = traj.nstate
        index_high = traj.highlevel
        index_qmqm2 = traj.qmqm2_index

        # do mm calculation, mm1 for qmqm2 region, mm2 for qmqm2mm region
        if self.do_mm:
            traj_1 = copy.deepcopy(traj)
            traj_2 = copy.deepcopy(traj)
            traj_mm_mid = self.mm_mid.evaluate(traj_1)
            traj_mm_low = self.mm_low.evaluate(traj_2)
            mm_mid_completion = traj_mm_mid.status
            mm_low_completion = traj_mm_low.status
            traj.status = np.amin([traj.status, mm_mid_completion, mm_low_completion])

            energy_mm_mid = np.repeat(traj_mm_mid.energy, nstate)
            energy_mm_low = np.repeat(traj_mm_low.energy, nstate)

            grad_mm_mid = np.repeat(traj_mm_mid.grad, nstate, axis=0)
            grad_mm_low = np.repeat(traj_mm_low.grad, nstate, axis=0)

            traj.energy = energy_mm_low
            traj.grad = grad_mm_low
            traj.energy_mm1 = energy_mm_mid[0]
            traj.energy_mm2 = energy_mm_low[0]
        else:
            energy_mm_mid = np.repeat(0, nstate)
            grad_mm_mid = np.zeros((nstate, nqmqm2, 3))
            traj.energy = np.zeros(nstate)
            traj.grad = np.zeros((nstate, natom, 3))
            traj.energy_mm1 = 0
            traj.energy_mm2 = 0

        traj_3 = copy.deepcopy(traj)
        # first compute charge for high level and middle level region
        traj_qm2_mid = self.qm2_mid.evaluate(traj_3)
        traj.charges = traj_qm2_mid.charges

        # traj_results = [None, None]
        # eval_func = [[0, traj], [1, traj]]
        # pool = multiprocessing.Pool(processes=self.nprocs)
        # for val in pool.imap_unordered(self.eval_wrapper, eval_func):
        #     ifunc, results = val
        #     traj_results[ifunc] = results
        # pool.close()

        # copy traj with the updated charge
        traj_4 = copy.deepcopy(traj)
        traj_5 = copy.deepcopy(traj)
        traj_qm_high = self.qm_high.evaluate(traj_4)
        traj_qm2_high = self.qm2_high.evaluate(traj_5)

        qm_high_completion = traj_qm_high.status
        qm2_high_completion = traj_qm2_high.status
        qm2_mid_completion = traj_qm2_mid.status

        traj.err_energy = traj_qm_high.err_energy
        traj.err_grad = traj_qm_high.err_grad
        traj.err_nac = traj_qm_high.err_nac
        traj.err_soc = traj_qm_high.err_soc
        traj.status = np.amin([qm_high_completion, qm2_high_completion, qm2_mid_completion])

        traj.qm_energy = traj_qm_high.energy
        traj.qm_grad = traj_qm_high.grad
        traj.qm_nac = traj_qm_high.nac
        traj.qm_soc = traj_qm_high.soc

        traj.qm1_charge = traj_qm_high.qm1_charge
        traj.qm2_charge = traj_qm_high.qm2_charge

        # combine energy
        energy_qm_high = traj_qm_high.energy
        energy_qm2_high = np.repeat(traj_qm2_high.energy, nstate)
        energy_qm2_mid = np.repeat(traj_qm2_mid.energy, nstate)
        traj.energy += - energy_mm_mid + energy_qm2_mid  # replace the m+h region mm energy with qm2 energy
        traj.energy += - energy_qm2_high + energy_qm_high  # replace the h region qm2 energy with qm energy
        traj.energy_qm2_1 = energy_qm2_high[0]
        traj.energy_qm2_2 = energy_qm2_mid[0]

        # combine grad
        grad_qm_high = traj_qm_high.grad
        ngrad = len(grad_qm_high)
        if ngrad > 0:
            grad_qm2_high = np.repeat(traj_qm2_high.grad, nstate, axis=0)
            grad_qm2_mid = np.repeat(traj_qm2_mid.grad, nstate, axis=0)
            traj.grad[:, index_qmqm2, :] += - grad_mm_mid + grad_qm2_mid  # replace the m+h region mm grad with qm2 grad
            traj.grad[:, index_high, :] += - grad_qm2_high + grad_qm_high  # replace the h region qm2 grad with qm grad

        # combine nac
        nac_qm_high = traj_qm_high.nac
        nnac = len(nac_qm_high)
        if nnac > 0:
            traj.nac = np.zeros((nnac, natom, 3))
            traj.nac[:, index_high, :] = nac_qm_high

        # combine soc
        traj.soc = traj_qm_high.soc

        return traj

    def read_data(self, natom):
        ## function to read the PyRAI2MD logfile

        with open('%s/%s.log' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        coord = []
        energy = []
        gradient = []
        nac = []
        soc = []
        for i, line in enumerate(log):
            if '  &coordinates in Angstrom' in line:
                coord = orca_coord(log[i + 2: i + 2 + natom])

            if '  Energy state ' in line:
                e = float(line.split()[-1])
                energy.append(e)

            if '  &gradient state' in line:
                g = log[i + 2: i + 2 + natom]
                g = string2float(g)
                gradient.append(g)

            if '  &nonadiabatic coupling' in line:
                n = log[i + 2: i + 2 + natom]
                n = string2float(n)
                nac.append(n)

            if '  <H>=' in line:
                socme = float(line.split()[1])
                soc.append(socme)

        energy = np.array(energy)
        gradient = np.array(gradient)
        nac = np.array(nac)
        soc = np.array(soc)

        return coord, energy, gradient, nac, soc
