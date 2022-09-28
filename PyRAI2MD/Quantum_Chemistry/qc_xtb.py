######################################################
#
# PyRAI2MD 2 module for xTB 6.5.1 interface (NAC and SOC are not available)
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

from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.coordinates import print_charge
from PyRAI2MD.Utils.coordinates import orca_coord

class Xtb:
    """ xTB single point calculation interface

        Parameters:          Type:
            keywords         dict         keywords dict
            job_id           int          calculation index

        Attribute:           Type:
            natom            int         number of atoms.
            keep_tmp         int  	     keep the Molcas calculation folders (1) or not (0).
            verbose          int	     print level.
            project          str	     calculation name.
            workdir          str	     xTB calculation folder.
            xtb              str	     xTB environment variable, executable folder.
            nproc            int	     number of CPUs for parallelization
            mpi              str	     path to mpi library
            use_hpc          int	     use HPC (1) for calculation or not(0), like SLURM.
            charges          ndarray     mulliken charges to be used for qm 2 region in multiscale calculation

        Functions:           Returns:
            train            self        fake function
            load             self        fake function
            appendix         self        fake function
            evaluate         self        run single point calculation

    """

    def __init__(self, keywords=None, job_id=None, runtype='qm'):

        self.runtype = runtype
        variables = keywords['xtb']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['xtb_project']
        self.workdir = variables['xtb_workdir']
        self.xtb = variables['xtb']
        self.nproc = variables['xtb_nproc']
        self.use_hpc = variables['use_hpc']
        self.charges = np.zeros(0)
        ## check calculation folder
        ## add index when running in adaptive sampling

        if job_id is not None:
            self.workdir = '%s/tmp_xtb-%s' % (self.workdir, job_id)

        elif job_id == 'Read':
            self.workdir = self.workdir

        else:
            self.workdir = '%s/tmp_xtb' % self.workdir

        if runtype == 'qmmm_low':
            self.workdir = '%s2' % self.workdir

        ## initialize runscript
        self.runscript = """
export XTB_PROJECT=%s
export XTBPATH=%s
export OMP_NUM_THREADS=%s
export XTB_WORKDIR=%s

cd $XTB_WORKDIR
""" % (
            self.project,
            self.xtb,
            self.nproc,
            self.workdir,
        )

        self.runscript += '$XTBPATH/bin/xtb --grad -I $XTB_WORKDIR/$XTB_PROJECT.inp $XTB_WORKDIR/$XTB_PROJECT.xyz > ' \
                          '$XTB_WORKDIR/$XTB_PROJECT.out\n '

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % self.project):
            with open('%s.slurm' % self.project) as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  xTB: looking for submission file %s.slurm' % self.project)

        submission += '\n%s' % self.runscript

        with open('%s/%s.sbatch' % (self.workdir, self.project), 'w') as out:
            out.write(submission)

    def _setup_xtb(self, x, q=None):
        ## make calculation folder and input file
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        ## clean calculation folder
        os.system("rm %s/*.engrad > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/*.out > /dev/null 2>&1" % self.workdir)

        ## write run script
        with open('%s/%s.sh' % (self.workdir, self.project), 'w') as out:
            out.write(self.runscript)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

        ## setup input according to dft_type
        self._write_xtb(x, q)

    def _write_xtb(self, x, q=None):
        ## write xtb input file
        natom = len(x)
        xyz = '%s\n\n%s' % (natom, print_coord(x))
        charge = print_charge(q)

        ## Read input template from current directory
        ## general dft xTb template should end with '*xyz charge mult'
        if os.path.exists('%s.xtb' % self.project):
            with open('%s.xtb' % self.project, 'r') as template:
                ld_input = template.read()
        else:
            ld_input = ''

        ## insert charge section
        if len(charge) > 0:
            ld_input = ld_input + '$embedding\ninput=%s.pc\n$end\n' % charge
            with open('%s/%s.pc' % (self.workdir, self.project), 'w') as out:
                out.write('%s\n%s' % (len(q), charge))

        ## save xyz and input file
        with open('%s/%s.xyz' % (self.workdir, self.project), 'w') as out:
            out.write(xyz)

        with open('%s/%s.inp' % (self.workdir, self.project), 'w') as out:
            out.write(ld_input)

    def _run_xtb(self):
        ## run xTB calculation

        maindir = os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.workdir, self.project)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.workdir, self.project)])
        os.chdir(maindir)

    def _read_data(self, natom):
        ## read xTB output and pack data
        with open('%s/%s.xyz' % (self.workdir, self.project), 'r') as inp:
            inputs = inp.read().splitlines()

        coord = orca_coord(inputs[2: 2 + natom])

        if not os.path.exists('%s/%s.engrad' % (self.workdir, self.project)):
            return coord, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        if os.path.exists('%s/charges' % self.workdir):
            with open('%s/charges' % self.workdir) as out:
                self.charges = np.loadtxt(out)

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

        return coord, energy, gradient, nac, soc

    def _qmmm(self, traj):
        ## run xTB for QMMM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup xTB calculation
        self._setup_xtb(xyz, charge)

        ## run xTB calculation
        self._run_xtb()

        ## read xTB output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

        return energy, gradient, nac, soc

    def _qm(self, traj, pc=True):
        ## run xTB for QM calculation

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)

        if pc:
            charge = traj.qm2_charge
        else:
            charge = None

        ## setup xTB calculation
        self._setup_xtb(xyz, charge)

        ## run xTb calculation
        self._run_xtb()

        ## read xTB output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        return energy, gradient, nac, soc

    def appendix(self, _):
        ## fake function

        return self

    def evaluate(self, traj):
        ## main function to run xTB calculation and communicate with other PyRAI2MD modules

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
        elif self.runtype == 'qmmm_low':
            energy, gradient, nac, soc = self._qm(traj, pc=False)
            traj.charges = np.concatenate((self.charges.reshape((-1, 1)), traj.coord), axis=1)

        if len(energy) == 1 and len(gradient) == 1:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        # update trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.array(nac)
        traj.soc = np.array(soc)
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

    def read_data(self, natom):
        ## function to read the logfile
        coord, energy, gradient, nac, soc = self._read_data(natom)

        return coord, energy, gradient, nac, soc