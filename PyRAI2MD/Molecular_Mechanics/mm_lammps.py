######################################################
#
# PyRAI2MD 2 module for LAMMPS interface
#
# Author Jingbai Li
# Apr 20 2026
#
######################################################

import os
import sys
import subprocess
import shutil
import numpy as np

from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.coordinates import lammps_charges


class LAMMPS:
    """ LAMMPS single point calculation interface

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
            static_charges   ndarray     static charge read from lammps data file
            charges          ndarray     system charges to be used for qm 2 region in multiscale calculation

        Functions:           Returns:
            train            self        fake function
            load             self        fake function
            appendix         self        fake function
            evaluate         self        run single point calculation

    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):

        self.runtype = runtype
        variables = keywords['lammps']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['lammps_project']
        self.workdir = variables['lammps_workdir']
        self.lmp = variables['lammps']
        self.nproc = variables['lammps_nproc']
        self.total_charges = variables['lammps_charges']
        self.use_hpc = variables['use_hpc']
        embedding = keywords['molecule']['embedding']
        cell = keywords['molecule']['cell']
        lattice = keywords['molecule']['lattice']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if job_id is not None:
            if job_id == 'Read':
                self.workdir = self.workdir
            else:
                self.workdir = '%s/tmp_lmp-%s' % (self.workdir, job_id)
        else:
            self.workdir = '%s/tmp_lmp' % self.workdir

        if self.runtype == 'qm2_high':
            self.workdir = '%s_qm2_h' % self.workdir
            self.tag = 'core'

        elif self.runtype == 'qm2_high_mid':
            self.workdir = '%s_qm2_m' % self.workdir
            self.tag = 'model'

        elif self.runtype == 'mm_high_mid':
            self.workdir = '%s_mm_m' % self.workdir
            self.tag = 'core'

        elif self.runtype == 'mm_high_mid_low':
            self.workdir = '%s_mm_l' % self.workdir
            self.tag = 'model'

        elif self.runtype == 'qm_high_mid_low':
            self.tag = 'model'
        
        elif self.runtype == 'qm_high':
            self.tag = 'core'

        ## initialize runscript
        self.runscript = """
export LMP_PROJECT=%s
export LMP_WORKDIR=%s
export LMP_HOME=%s
export OMP_NUM_THREADS=1

cd $LMP_WORKDIR

mpirun -np %s $LMP_HOME/lmp -in $LMP_PROJECT.in > $LMP_PROJECT.log

""" % (
            self.project,
            self.workdir,
            self.lmp,
            self.nproc,
        )

        self.static_charges = lammps_charges('%s/%s.%s.data' % (os.getcwd(), self.project, self.tag))

        ## check shake
        settings = '%s/%s.%s.in.settings' % (os.getcwd(), self.project, self.tag)
        with open(settings, 'r') as set_file:
            settings = set_file.read().splitlines()
            for line in settings:
                if 'shake' in line.lower():
                    print('LAMMPS: shake is found in settings file:')
                    print(line)
                    print('PyRAI2MD does not support shake yet, please comment shake out if it is not commented out')
                    break

        ## check keywords
        if embedding == 1 and len(cell) > 0:
            sys.exit('\n  KeywordError\n  LAMMPS: charge embedding and pbc cannot work together!')

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % self.project):
            with open('%s.slurm' % self.project) as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  LAMMPS: looking for submission file %s.slurm' % self.project)

        submission += '\n%s' % self.runscript

        with open('%s/%s.sbatch' % (self.workdir, self.project), 'w') as out:
            out.write(submission)

    def _setup_lammps(self, x, q=None, cell=None, pbc=None):
        ## make calculation folder and input file
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        ## clean calculation folder
        os.system("rm %s/%s.energy > /dev/null 2>&1" % (self.workdir, self.project))
        os.system("rm %s/%s.force > /dev/null 2>&1" % (self.workdir, self.project))
        os.system("rm %s/%s.log > /dev/null 2>&1" % (self.workdir, self.project))
        os.system("rm %s/log.lammps > /dev/null 2>&1" % self.workdir)

        ## write run script
        with open('%s/%s.sh' % (self.workdir, self.project), 'w') as out:
            out.write(self.runscript)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

        ## setup input according to dft_type
        self._write_lammps(x, q=q, cell=cell, pbc=pbc)

    def _write_lammps(self, x, q=None, cell=None, pbc=None):
        ## Read input template from current directory
        maindir = os.getcwd()
        ld_input = 'include %s/%s.%s.in.init\n' % (maindir, self.project, self.tag)
        ld_input += 'read_data %s/%s.%s.data\n' % (maindir, self.project, self.tag)
        ld_input += 'include %s/%s.%s.in.settings\n' % (maindir, self.project, self.tag)
        ld_input += 'read_dump %s.xyz 0 x y z box no replace yes format xyz\n' % self.project
        ld_input += 'thermo 100\n'
        ld_input += 'thermo_style multi\n'
        ld_input += 'thermo_modify norm yes\n'
        ld_input += 'run 0\n'
        ld_input += 'variable N equal count(all)\n'
        ld_input += 'variable E equal pe*v_N\n'
        ld_input += 'variable E format E %.18g\n'
        ld_input += 'print "${E}" file %s.energy\n' % self.project
        ld_input += 'write_dump all custom %s.coord x y z modify sort id format float %%.18g\n' % self.project
        ld_input += 'write_dump all custom %s.force fx fy fz modify sort id format float %%.18g\n' % self.project

        with open('%s/%s.in' % (self.workdir, self.project), 'w') as out:
            out.write(ld_input)

        ## prepare charge from lammps data
        # No charge for now

        ## save xyz and input file
        xyz = '%s\n\n%s' % (len(x), print_coord(x))
        xyzfile = '%s/%s.xyz' % (self.workdir, self.project)

        with open(xyzfile, 'w') as out:
            out.write(xyz)

        return 0
    
    def _run_lammps(self):
        ## run LAMMPS calculation

        maindir = os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.workdir, self.project)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.workdir, self.project)], env=os.environ.copy()) # lammps must copy environ
        os.chdir(maindir)

    def _read_data(self, natom):
        ## read LAMMPS output and pack data
        with open('%s/%s.coord' % (self.workdir, self.project), 'r') as inp:
            coord = inp.read().splitlines()

        natom = int(coord[3])
        coord = np.array([x.split()[0:3] for x in coord[9:9+natom]]).astype(float)

        if not os.path.exists('%s/%s.energy' % (self.workdir, self.project)):
            return coord, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        if not os.path.exists('%s/%s.force' % (self.workdir, self.project)):
            return coord, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.energy' % (self.workdir, self.project), 'r') as out:
            energy = [float(out.read().splitlines()[0])]

        with open('%s/%s.force' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()
            natom = int(log[3])
            gradient = np.array([x.split()[0:3] for x in log[9:9+natom]]).astype(float)

        ## no nac or soc
        energy = np.array(energy) / 627.5 # kcal/mol to hartree
        gradient = np.array(gradient).reshape((1, natom, 3)) / -627.5 * 0.529177 # kcal/mol/A to hartree/bohr and force to grad (-1)
        nac = np.zeros(0)
        soc = np.zeros(0)

        return coord, energy, gradient, nac, soc

    def _high(self, traj):
        ## run LAMMPS for high level region in QM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup LAMMPS calculation
        self._setup_lammps(xyz, q=charge)

        ## run LAMMPS calculation
        self._run_lammps()

        ## read LAMMPS output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

        return coord, energy, gradient, nac, soc

    def _high_mid(self, traj):
        ## run LAMMPS for high level region and middle level region in QM or MM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qmqm2_atoms, traj.qmqm2_coord), axis=1)
        nxyz = len(xyz)
        cell = traj.cell
        pbc = traj.pbc

        ## setup LAMMPS calculation
        self._setup_lammps(xyz, cell=cell, pbc=pbc)

        ## run LAMMPS calculation
        self._run_lammps()

        ## read LAMMPS output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        self.charges = np.zeros((traj.natom, 4))
        self.charges[traj.qmqm2_index] = np.concatenate((self.static_charges.reshape((-1, 1)), traj.qmqm2_coord), axis=1)

        ## project force and coupling
        # jacob = traj.Hcap_jacob
        # gradient = np.array([np.dot(x, jacob) for x in gradient])
        # nac = np.array([np.dot(x, jacob) for x in nac])

        return coord, energy, gradient, nac, soc

    def _high_mid_low(self, traj, ignore_charges=False):
        ## run LAMMPS for high level region, middle level region, and low level region in QM or MM calculation

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)

        cell = traj.cell
        pbc = traj.pbc

        if ignore_charges:
            charge = np.zeros(0)
        else:
            charge = traj.qm2_charge

        ## setup LAMMPS calculation
        self._setup_lammps(xyz, q=charge, cell=cell, pbc=pbc)

        ## run LAMMPS calculation
        self._run_lammps()

        ## read LAMMPS output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        return coord, energy, gradient, nac, soc

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

        if self.runtype == 'qm_high' or self.runtype == 'qm2_high':  # qm or qm2 calculation for h region
            coord, energy, gradient, nac, soc = self._high(traj)
        elif self.runtype == 'qm2_high_mid':  # qm or qm2 calculation for h + m region
            coord, energy, gradient, nac, soc = self._high_mid(traj)
            if not traj.read_charge and traj.embedding:
                traj.charges = self.charges
        elif self.runtype == 'mm_high_mid':  # mm, calculation for h + m region
            coord, energy, gradient, nac, soc = self._high_mid(traj)
        elif self.runtype == 'qm_high_mid_low':  # qm or qm2 calculation for h + m + l region
            coord, energy, gradient, nac, soc = self._high_mid_low(traj)
        elif self.runtype == 'mm_high_mid_low':  # mm calculation for h + m + l region
            coord, energy, gradient, nac, soc = self._high_mid_low(traj, ignore_charges=True)

        if len(energy) == 1 and len(gradient) == 1:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        # update coordinate if shake is applied
        if len(coord) == len(traj.coord):
            mae = np.max(np.abs(coord - traj.coord))
            if mae > 1e-6:
                traj.coord = np.copy(coord)

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

    def read_data(self, natom, ncharge):
        ## function to read the logfile
        coord, energy, gradient, nac, soc = self._read_data(natom)
        charge = np.zeros(0)
        cell = np.zeros(0)
        pbc = np.zeros(0)
        return coord, charge, cell, pbc, energy, gradient, nac, soc
