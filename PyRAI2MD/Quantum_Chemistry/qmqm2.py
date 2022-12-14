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
            job_id           int         calculation index

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

        qm1 = methods[0]
        qm2 = methods[1]
        self.qm1_high = qm1(keywords=keywords, job_id=job_id_1, runtype='qmmm')
        self.qm1_low = qm2(keywords=keywords, job_id=job_id_2, runtype='qmmm')
        self.qm2_low = qm2(keywords=keywords, job_id=job_id_2, runtype='qmmm_low')
        self.nprocs = keywords['control']['ms_ncpu']
        self.project = ''
        self.workdir = ''

    def train(self):
        ## fake function

        return self

    def load(self):
        # This should load model
        self.qm1_high.load()
        self.qm1_low.load()
        self.qm2_low.load()

        return self

    def appendix(self, _):
        ## fake function

        return self

    def eval_wrapper(self, var):
        ifunc, traj = var
        if ifunc == 0:
            func = self.qm1_high.evaluate
        elif ifunc == 1:
            func = self.qm1_low.evaluate
        else:
            func = None

        results = func(traj)

        return ifunc, results

    def evaluate(self, traj):
        index_high = traj.highlevel
        traj_1 = copy.deepcopy(traj)
        traj_2 = copy.deepcopy(traj)
        traj_3 = copy.deepcopy(traj)

        # first compute charge
        traj_qm2_low = self.qm2_low.evaluate(traj_1)
        traj.charges = traj_qm2_low.charges

        # traj_results = [None, None]
        # eval_func = [[0, traj], [1, traj]]
        # pool = multiprocessing.Pool(processes=self.nprocs)
        # for val in pool.imap_unordered(self.eval_wrapper, eval_func):
        #     ifunc, results = val
        #     traj_results[ifunc] = results
        # pool.close()

        traj_qm1_high = self.qm1_high.evaluate(traj_2)
        traj_qm1_low = self.qm1_low.evaluate(traj_3)

        qm1_high_completion = traj_qm1_high.status
        qm1_low_completion = traj_qm1_low.status
        qm2_low_completion = traj_qm2_low.status

        traj.err_energy = traj_qm1_high.err_energy
        traj.err_grad = traj_qm1_high.err_grad
        traj.err_nac = traj_qm1_high.err_nac
        traj.err_soc = traj_qm1_high.err_soc
        traj.status = np.amin([qm1_high_completion, qm1_low_completion, qm2_low_completion])

        traj.qm_energy = traj_qm1_high.energy
        traj.qm_grad = traj_qm1_high.grad
        traj.qm_nac = traj_qm1_high.nac
        traj.qm_soc = traj_qm1_high.soc

        traj.qm1_charge = traj_qm1_high.qm1_charge
        traj.qm2_charge = traj_qm1_high.qm2_charge
        
        # combine energy
        energy_qm1_high = traj_qm1_high.energy
        nstate = len(energy_qm1_high)
        energy_qm1_low = np.repeat(traj_qm1_low.energy, nstate)
        energy_qm2_low = np.repeat(traj_qm2_low.energy, nstate)
        traj.energy = energy_qm2_low - energy_qm1_low + energy_qm1_high

        # combine grad
        grad_qm1_high = traj_qm1_high.grad
        nstate = len(grad_qm1_high)
        grad_qm1_low = np.repeat(traj_qm1_low.grad, nstate, axis=0)
        grad_qm2_low = np.repeat(traj_qm2_low.grad, nstate, axis=0)
        traj.grad = grad_qm2_low
        traj.grad[:, index_high, :] = grad_qm2_low[:, index_high, :] - grad_qm1_low + grad_qm1_high

        # combine nac
        nac_qm1_high = traj_qm1_high.nac
        nnac = len(nac_qm1_high)
        natom = traj.natom
        if nnac > 0:
            traj.nac = np.zeros((nnac, natom, 3))
            traj.nac[:, index_high, :] = nac_qm1_high

        # combine soc
        traj.soc = traj_qm1_high.soc

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
