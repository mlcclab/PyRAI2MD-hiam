######################################################
#
# PyRAI2MD 2 module for ORCA 5.0.2 interface (NAC and SOC are not available)
#
# Author Jingbai Li
# Sep 16 2022
#
######################################################

import os
import sys
import subprocess
import shutil
import numpy as np

from PyRAI2MD.Utils.coordinates import reverse_string2float
from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.coordinates import print_charge

class Orca:
    """ ORCA single point calculation interface

        Parameters:          Type:
            keywords         dict         keywords dict
            job_id           int          calculation index

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

    def __init__(self, keywords=None, job_id=None, runtype='qm'):

        self.runtype = runtype
        self.nstate = 0
        self.nnac = 0
        self.nac_coupling = []
        self.state = 0
        self.sf_state = 0
        self.sf_state_list = []
        self.activestate = 0
        variables = keywords['orca']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['orca_project']
        self.workdir = variables['orca_workdir']
        self.dft_type = variables['dft_type']
        self.orca = variables['orca']
        self.mpi = variables['mpi']
        self.use_hpc = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if job_id is not None:
            self.workdir = '%s/tmp_ORCA-%s' % (self.workdir, job_id)

        elif job_id == 'Read':
            self.workdir = self.workdir

        else:
            self.workdir = '%s/tmp_ORCA' % self.workdir

        ## initialize runscript
        self.runscript = """
export ORCA_PROJECT=%s
export ORCA=%s
export MPI=%s
export ORCA_WORKDIR=%s

export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

cd $ORCA_WORKDIR
""" % (
            self.project,
            self.orca,
            self.mpi,
            self.workdir,
        )

        self.runscript += '$ORCA/orca $ORCA_WORKDIR/$ORCA_PROJECT.inp > $ORCA_WORKDIR/$ORCA_PROJECT.out\n '

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % self.project):
            with open('%s.slurm' % self.project) as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  ORCA: looking for submission file %s.slurm' % self.project)

        submission += '\n%s' % self.runscript

        with open('%s/%s.sbatch' % (self.workdir, self.project), 'w') as out:
            out.write(submission)

    def _setup_orca(self, x, q=None):
        ## make calculation folder and input file
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        ## clean calculation folder
        os.system("rm %s/*.engrad > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/*.tmp > /dev/null 2>&1" % self.workdir)

        ## write run script
        with open('%s/%s.sh' % (self.workdir, self.project), 'w') as out:
            out.write(self.runscript)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

        ## setup input according to dft_type
        if self.dft_type == 'tddft':
            self._write_tddft(x, q)
        elif self.dft_type == 'sf_tddft':
            self._write_sf_tddft(x, q, step='energy')
            self._run_orca()
            self._read_sf_tddft(len(x))
            self._write_sf_tddft(x, q, step='grad')
        else:
            self._write_dft(x, q)

    def _write_dft(self, x, q=None):
        ## write orca dft input file
        xyz = print_coord(x)
        charge = print_charge(q, 'Q')

        ## Read input template from current directory
        ## general dft ORCA template should end with '*xyz charge mult'
        with open('%s.orca' % self.project, 'r') as template:
            ld_input = template.read()
        ld_input = '!engrad\n' + ld_input
        ld_input += xyz
        ld_input += charge
        ld_input += '*\n'

        ## save xyz file
        with open('%s/%s.inp' % (self.workdir, self.project), 'w') as out:
            out.write(ld_input)

    def _write_tddft(self, x, q=None):
        ## write orca tddft input file
        xyz = print_coord(x)
        charge = print_charge(q, 'Q')

        ## Read input template from current directory
        ## general tddft ORCA template should put 'irootlist 0, 1, ...' in a single line
        ## and end with '*xyz charge mult'
        with open('%s.orca' % self.project, 'r') as template:
            ld_input = template.read().splitlines()

        si_input = ['!engrad']
        for line in ld_input:
            if 'irootlist' in line and self.activestate == 1:
                si_input.append('irootlist %s' % self.state)
            else:
                si_input.append(line)
        si_input = '\n'.join(si_input) + '\n'
        si_input += xyz
        si_input += charge
        si_input += '*\n'

        ## save xyz file
        with open('%s/%s.inp' % (self.workdir, self.project), 'w') as out:
            out.write(si_input)

    def _write_sf_tddft(self, x, q=None, step='energy'):
        ## write orca tddft input file
        xyz = print_coord(x)
        charge = print_charge(q, 'Q')

        ## Read input template from current directory
        ## general tddft ORCA template should put 'irootlist 0, 1, ...' in a single line
        ## and end with '*xyz charge mult'
        with open('%s.orca' % self.project, 'r') as template:
            ld_input = template.read().splitlines()

        ## energy step only compute sf state energy to find singlet state
        ## grad step compute the gradient of the selected sf state according to <S2>
        if step == 'energy':
            conditional_state_list = ''
            si_input = []
        else:
            si_input = ['!engrad']
            if self.activestate == 1:
                conditional_state_list = 'irootlist %s' % self.sf_state
            else:
                conditional_state_list = 'irootlist %s' % ', '.join([str(x) for x in self.sf_state_list])

        for line in ld_input:
            if 'irootlist' in line:
                si_input.append(conditional_state_list)
            else:
                si_input.append(line)
        si_input = '\n'.join(si_input) + '\n'
        si_input += xyz
        si_input += charge
        si_input += '*\n'

        ## save xyz file
        with open('%s/%s.inp' % (self.workdir, self.project), 'w') as out:
            out.write(si_input)

    def _run_orca(self):
        ## run ORCA calculation

        maindir = os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.workdir, self.project)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.workdir, self.project)])
        os.chdir(maindir)

    def _read_dft(self, natom):
        ## read ORCA output and pack data

        if not os.path.exists('%s/%s.engrad' % (self.workdir, self.project)):
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.engrad' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        ## pack ground state energy and force
        energy = []
        gradient = []
        for n, line in enumerate(log):
            if 'The current total energy in Eh' in line:
                e = float(log[n + 2])
                energy = [e]

            if 'The current gradient in Eh/bohr' in line:
                g = log[n + 2: n + 2 + natom * 3]
                gradient = [[float(x) for x in g]]

        ## no nac or soc
        energy = np.array(energy)
        gradient = np.array(gradient).reshape((1, natom, 3))
        nac = np.zeros(0)
        soc = np.zeros(0)

        return energy, gradient, nac, soc

    def _read_tddft(self, natom):
        ## read ORCA output and pack data

        if not os.path.exists('%s/%s.out' % (self.workdir, self.project)):
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.out' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        ## pack energy and force
        s0 = 0
        ex_energy = [0.0]
        gradient = []
        for n, line in enumerate(log):
            # find reference energy
            if 'E(SCF)' in line:
                s0 = float(line.split()[-2])
            # find excitation energy
            if 'STATE' in line and '<S' in line:
                e = float(line.split()[3])
                ex_energy.append(e)
            if 'CARTESIAN GRADIENT' in line:
                g = log[n + 3: n + 3 + natom]
                g = reverse_string2float(g)
                gradient.append(g)

        energy = np.array(ex_energy)[: self.nstate] + s0
        gradient = [gradient[-1]] + gradient[: -1]  # orca tddft compute ground-state gradient in the end
        gradient = np.array(gradient)
        if self.activestate == 1:
            gradall = np.zeros((self.nstate, natom, 3))
            gradall[self.state - 1] = np.array(gradient)
            gradient = gradall
        else:
            gradient = np.array(gradient)

        ## no nac or soc
        nac = np.zeros(0)
        soc = np.zeros(0)

        return energy, gradient, nac, soc

    def _read_sf_tddft(self, natom):
        ## read ORCA output and pack data

        if not os.path.exists('%s/%s.out' % (self.workdir, self.project)):
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.out' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        ## pack energy and force
        s0 = 0
        ex_energy = []
        ex_s2 = []
        gradient = []
        for n, line in enumerate(log):
            # find reference energy
            if 'E(SCF)' in line:
                s0 = float(line.split()[-2])
            # find excitation energy
            if 'STATE' in line and '<S' in line:
                e = float(line.split()[3])
                s = float(line.split()[-1])
                ex_energy.append(e)
                ex_s2.append(s)
            if 'CARTESIAN GRADIENT' in line:
                g = log[n + 3: n + 3 + natom]
                g = reverse_string2float(g)
                gradient.append(g)

        # find sf state corresponding to current and all singlet state
        energy = []
        sf_state_list = []
        for n, eig in enumerate(ex_s2):
            if eig < 0.1:
                sf_state_list.append(n + 1)
                energy.append(ex_energy[n])
        self.sf_state = sf_state_list[self.state - 1]
        self.sf_state_list = sf_state_list

        energy = np.array(ex_energy) + s0
        gradient = np.array(gradient)
        if self.activestate == 1:
            gradall = np.zeros((self.nstate, natom, 3))
            gradall[self.state - 1] = np.array(gradient)
            gradient = gradall
        else:
            gradient = np.array(gradient)

        ## no nac or soc
        nac = np.zeros(0)
        soc = np.zeros(0)

        return energy, gradient, nac, soc

    def _read_data(self, nxyz):
        ## read orca output
        if self.dft_type == 'tddft':
            return self._read_tddft(nxyz)
        elif self.dft_type == 'sf_tddft':
            return self._read_sf_tddft(nxyz)
        else:
            return self._read_dft(nxyz)

    def _qmmm(self, traj):
        ## run ORCA for QMMM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup ORCA calculation
        self._setup_orca(xyz, charge)

        ## run ORCA calculation
        self._run_orca()

        ## read ORCA output files
        energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

        return energy, gradient, nac, soc

    def _qm(self, traj):
        ## run ORCA for QM calculation

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup ORCA calculation
        self._setup_orca(xyz, charge)

        ## run ORCA calculation
        self._run_orca()

        ## read ORCA output files
        energy, gradient, nac, soc = self._read_data(nxyz)

        return energy, gradient, nac, soc

    def appendix(self, _):
        ## fake function

        return self

    def evaluate(self, traj):
        ## main function to run ORCA calculation and communicate with other PyRAI2MD modules

        ## load trajectory info
        self.nstate = traj.nstate
        self.nnac = traj.nnac
        self.nac_coupling = traj.nac_coupling
        self.state = traj.state
        self.activestate = traj.activestate

        ## compute properties
        energy = []
        gradient = []
        nac = []
        soc = []
        completion = 0

        if self.runtype == 'qm':
            energy, gradient, nac, soc = self._qm(traj)
        elif self.runtype == 'qmmm':
            energy, gradient, nac, soc = self._qmmm(traj)

        if len(energy) >= self.nstate and len(gradient) >= self.nstate and len(nac) >= self.nnac:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        # update trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.zeros(soc)
        traj.err_energy = None
        traj.err_grad = None
        traj.err_nac = None
        traj.err_soc = None
        traj.status = completion

        return traj

    def train(self):
        ## fake function

        return self

    def load(self):
        ## fake function

        return self
