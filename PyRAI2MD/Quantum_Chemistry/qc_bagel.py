######################################################
#
# PyRAI2MD 2 module for BAGEL interface
#
# Author Jingbai Li
# Sep 20 2021
#
######################################################

import os
import sys
import subprocess
import shutil
import json
import numpy as np

from PyRAI2MD.Utils.coordinates import string2float

class Bagel:
    """ BAGEL single point calculation interface

        Parameters:          Type:
            keywords         dict         keywords dict
            job_id           int          calculation index

        Attribute:           Type:
            natom            int	      number of atoms.
            nstate           int	      number of electronic state
            nnac             int          number of non-adiabatic couplings
            nac_coupling     list         non-adiabatic coupling pairs
            state            int          current state
            activestate      int          only compute gradient for current state
            keep_tmp         int	      keep the BAGEL calculation folders (1) or not (0).
            verbose          int	      print level.
            project          str	      calculation name.
            workdir          str	      calculation folder.
            bagel            str	      BAGEL executable folder
            nproc            int	      number of CPUs for parallelization
            mpi              str	      path to mpi library
            blas             str	      path to blas library
            lapack           str	      path to lapack library
            boost            str	      path to boost library
            mkl              str	      path to mkl library
            arch             str	      CPU architecture
            threads          int	      number of threads for OMP parallelization.
            use_hpc          int	      use HPC (1) for calculation or not(0), like SLURM.
            use_mpi          int	      use MPI (1) for calculation or not(0).

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
        self.activestate = 0
        variables = keywords['bagel']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['bagel_project']
        self.workdir = variables['bagel_workdir']
        self.archive = variables['bagel_archive']
        self.bagel = variables['bagel']
        self.nproc = variables['bagel_nproc']
        self.mpi = variables['mpi']
        self.blas = variables['blas']
        self.lapack = variables['lapack']
        self.boost = variables['boost']
        self.mkl = variables['mkl']
        self.arch = variables['arch']
        self.threads = variables['omp_num_threads']
        self.use_mpi = variables['use_mpi']
        self.use_hpc = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if job_id is not None:
            self.workdir = '%s/tmp_BAGEL-%s' % (self.workdir, job_id)

        elif job_id == 'Read':
            self.workdir = self.workdir

        else:
            self.workdir = '%s/tmp_BAGEL' % self.workdir

        ## initialize runscript
        self.runscript = """
export BAGEL_PROJECT=%s
export BAGEL=%s
export BLAS=%s
export LAPACK=%s
export BOOST=%s
export MPI=%s
export BAGEL_WORKDIR=%s
export OMP_NUM_THREADS=%s
export MKL_NUM_THREADS=%s
export BAGEL_NUM_THREADS=%s
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BLAS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

source %s %s

cd $BAGEL_WORKDIR
""" % (
            self.project,
            self.bagel,
            self.blas,
            self.lapack,
            self.boost,
            self.mpi,
            self.workdir,
            self.threads,
            self.threads,
            self.threads,
            self.mkl,
            self.arch
        )

        if self.use_mpi == 0:
            self.runscript += '$BAGEL/bin/BAGEL $BAGEL_WORKDIR/$BAGEL_PROJECT.json > ' \
                              '$BAGEL_WORKDIR/$BAGEL_PROJECT.log\n '
        else:
            # TODO: intel mpi is mpiexec
            self.runscript += 'mpirun -np $SLURM_NTASKS $BAGEL/bin/BAGEL $BAGEL_WORKDIR/$BAGEL_PROJECT.json > ' \
                              '$BAGEL_WORKDIR/$BAGEL_PROJECT.log\n '

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % self.project):
            with open('%s.slurm' % self.project) as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  BAGEL: looking for submission file %s.slurm' % self.project)

        submission += '\n%s' % self.runscript

        with open('%s/%s.sbatch' % (self.workdir, self.project), 'w') as out:
            out.write(submission)

    def _setup_bagel(self, x, q=None):
        ## make calculation folder and input file
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        ## prepare .json .archive files
        self._write_coord(x, q)

        ## save .archive file
        if not os.path.exists('%s.archive' % self.project):
            sys.exit('\n  FileNotFoundError\n  BAGEL: looking for orbital %s.archive' % self.project)

        if self.archive == 'default':
            self.archive = self.project

        if not os.path.exists('%s/%s.archive' % (self.workdir, self.archive)):
            shutil.copy2('%s.archive' % self.project, '%s/%s.archive' % (self.workdir, self.archive))

        ## clean calculation folder
        os.system("rm %s/ENERGY*.out > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/FORCE_*.out > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/NACME_*.out > /dev/null 2>&1" % self.workdir)

        ## write run script
        with open('%s/%s.sh' % (self.workdir, self.project), 'w') as out:
            out.write(self.runscript)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

    def _write_coord(self, x, q=None):
        ## write coordinate file

        ## convert xyz from array to bagel format (Bohr)
        a2b = 1 / 0.529177249  # angstrom to bohr
        jxyz = []
        for n, line in enumerate(x):
            e, x, y, z = line
            jxyz.append({"atom": e, "xyz": [float(x) * a2b, float(y) * a2b, float(z) * a2b]})

        ## convert point charge from array to bagel format
        if isinstance(q, np.ndarray):
            for charge in q:
                jxyz.append(
                    {"atom": "Q", "xyz": [float(charge[1]), float(charge[2]), float(charge[3])], "charge": charge[0]}
                )

        ## Read input template from current directory
        with open('%s.bagel' % self.project, 'r') as template:
            ld_input = json.load(template)

        si_input = ld_input.copy()
        si_input['bagel'][0]['geometry'] = jxyz

        ## default is to use template force setting, replace with the current state if requested
        if self.activestate == 1:
            grads = si_input['bagel'][2]['grads']
            grads_new = [{'title': 'force', 'target': self.state - 1}]
            for grad_dict in grads:
                if grad_dict['title'] != 'force':
                    grads_new.append(grad_dict)
            si_input['bagel'][2]['grads'] = grads_new

        ## save xyz file
        with open('%s/%s.json' % (self.workdir, self.project), 'w') as out:
            json.dump(si_input, out)

    def _run_bagel(self):
        ## run BAGEL calculation

        maindir = os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.workdir, self.project)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.workdir, self.project)])
        os.chdir(maindir)

    def _read_data(self, natom):
        ## read BAGEL logfile and pack data

        if not os.path.exists('%s/%s.log' % (self.workdir, self.project)):
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.log' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        coord = []
        for line in log:
            if '"atom"' in line:
                line = line.replace(',', ' ').replace('"', ' ').split()
                coord.append([line[3], float(line[7]) * 0.529177, float(line[8]) * 0.529177, float(line[9]) * 0.529177])
        coord = coord[:natom]

        ## pack energy, only includes the requested states by self.nstate
        energy = []
        if os.path.exists('%s/ENERGY.out' % self.workdir):
            energy = np.loadtxt('%s/ENERGY.out' % self.workdir)[0: self.nstate]

        ## pack force
        gradient = []
        for i in range(self.nstate):
            if os.path.exists('%s/FORCE_%s.out' % (self.workdir, i)):
                with open('%s/FORCE_%s.out' % (self.workdir, i)) as force:
                    g = force.read().splitlines()[1: natom + 1]
                    g = string2float(g)
            else:
                g = [[0, 0, 0] for _ in range(natom)]

            gradient.append(g)

        gradient = np.array(gradient)

        ## pack nac
        nac = []
        for pair in self.nac_coupling:
            pa, pb = pair
            if os.path.exists('%s/NACME_%s_%s.out' % (self.workdir, pa, pb)):
                with open('%s/NACME_%s_%s.out' % (self.workdir, pa, pb)) as nacme:
                    n = nacme.read().splitlines()[1: natom + 1]
                    n = string2float(n)
                nac.append(n)
        nac = np.array(nac)
        soc = np.zeros(0)

        return coord, energy, gradient, nac, soc

    def _qmmm(self, traj):
        ## run BAGEL for QMMM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup BAGEL calculation
        self._setup_bagel(xyz, charge)

        ## run BAGEL calculation
        self._run_bagel()

        ## read BAGEL output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

        return energy, gradient, nac

    def _qm(self, traj):
        ## run BAGEL for QM calculation 

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup BAGEL calculation
        self._setup_bagel(xyz, charge)

        ## run BAGEL calculation
        self._run_bagel()

        ## read BAGEL output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        return energy, gradient, nac

    def appendix(self, _):
        ## fake function

        return self

    def evaluate(self, traj):
        ## main function to run BAGEL calculation and communicate with other PyRAI2MD modules

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
        completion = 0

        if self.runtype == 'qm':
            energy, gradient, nac = self._qm(traj)
        elif self.runtype == 'qmmm':
            energy, gradient, nac = self._qmmm(traj)

        if len(energy) >= self.nstate and len(gradient) >= self.nstate and len(nac) >= self.nnac:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        # update trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.zeros(0)
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
