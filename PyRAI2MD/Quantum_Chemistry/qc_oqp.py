######################################################
#
# PyRAI2MD 2 module for OQP interface (SOC are not available)
#
# Author Jingbai Li
# Apr 22 2024
#
######################################################

import os
import sys
import subprocess
import shutil
import numpy as np

from PyRAI2MD.Utils.coordinates import oqp_coord
from PyRAI2MD.Utils.coordinates import oqp_coord2list
from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.coordinates import print_charge


class OQP:
    """ OQP single point calculation interface instance class based on disk I/O

        Parameters:          Type:
            keywords         dict         keywords dict
            job_id           int          calculation index

        Attribute:           Type:
            natom            int         number of atoms.
            nstate           int         number of electronic states
            nnac             int         number of non-adiabatic couplings
            nsoc             int         number of spin-orbit couplings
            state            int         the current state
            activestate      int         compute gradient only for the current state
            ci               list        number of state per spin multiplicity
            mult             list        spin multiplicity
            nac_coupling     list        non-adiabatic coupling pairs
            soc_coupling     list        spin-orbit coupling pairs
            keep_tmp         int  	     keep the Molcas calculation folders (1) or not (0).
            verbose          int	     print level.
            project          str	     calculation name.
            workdir          str	     OQP calculation folder.
            oqp              str	     OQP_ROOT environment variable, executable folder.
            nproc            int	     number of CPUs for parallelization
            use_hpc          int	     use HPC (1) for calculation like SLURM
                                         or run calculation based on IO (0)
                                         or run calculation in memory (-1)

        Functions:           Returns:
            train            self        fake function
            load             self        fake function
            appendix         self        fake function
            evaluate         self        run single point calculation

    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):

        self.runtype = runtype
        self.nstate = 0
        self.nnac = 0
        self.nac_coupling = []
        self.state = 0
        self.activestate = 0
        self.nactype = keywords['md']['nactype']
        variables = keywords['oqp']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['oqp_project']
        self.workdir = variables['oqp_workdir']
        self.oqp = variables['oqp']
        self.nproc = variables['threads']
        self.activestate = 0
        self.pyoqp = None
        self.back_door_data = None
        use_hpc = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling
        if job_id is not None:
            self.workdir = '%s/tmp_OQP-%s' % (self.workdir, job_id)

        elif job_id == 'Read':
            self.workdir = self.workdir
            use_hpc = 1
        else:
            self.workdir = '%s/tmp_OQP' % self.workdir

        ## read oqp input template
        self.input_dict = self._read_oqp()

        ## set oqp run functions
        if use_hpc == 1:
            self._init_io()
            self._write_oqp = self._write_oqp_io
            self._run_oqp = self._run_oqp_io
            self._read_data = self._read_data_io
        else:
            self._init_mem()
            self._write_oqp = self._write_oqp_mem
            self._run_oqp = self._run_oqp_mem
            self._read_data = self._read_data_mem

    def _read_oqp(self):
        # read oqp input file as dictf
        if os.path.exists('%s.oqp' % self.project):
            with open('%s.oqp' % self.project, 'r') as template:
                ld_input = template.read()
        else:
            ld_input = ''

        ld_input = ld_input.split('[')
        input_dict = {}
        for section in ld_input:

            if len(section) == 0:
                continue

            section_name, keyvals = section.split(']')
            keyvals = keyvals.splitlines()
            input_dict[section_name] = {}

            for keyval in keyvals:

                if len(keyval) == 0:
                    continue

                key, val = keyval.split('=')
                input_dict[section_name][key] = val

        # assign requisite values
        for key in ['input', 'guess', 'properties', 'nac', 'soc']:
            if key not in input_dict.keys():
                input_dict[key] = {}

        input_dict['input']['runtype'] = 'prop'
        input_dict['input']['system'] = '%s/%s.xyz' % (self.workdir, self.project)
        input_dict['guess']['type'] = 'auto'
        input_dict['guess']['file'] = '%s/guess.json' % self.workdir
        input_dict['guess']['continue_geom'] = 'false'
        input_dict['guess']['save_mol'] = 'true'
        input_dict['properties']['export'] = 'true'

        if self.nactype == 'nacme':
            input_dict['properties']['nac'] = 'nacme'
        elif self.nactype == 'nac':
            input_dict['properties']['nac'] = 'nac'
        else:
            input_dict['properties']['nac'] = 'false'

        return input_dict

    def _init_io(self):
        ## read slurm template from .slurm files
        if os.path.exists('%s.slurm' % self.project):
            with open('%s.slurm' % self.project, 'r') as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  OQP: looking for submission file %s.slurm' % self.project)

        self.runscript = """%s

export OQP_PROJECT=%s
export OQP_WORKDIR=%s
export OQP_ROOT=%s
export OMP_NUM_THREADS=%s

cd $OQP_WORKDIR

openqp ${OQP_PROJECT}.inp

""" % (
            submission,
            self.project,
            self.workdir,
            self.oqp,
            self.nproc,
        )

        ## write hps runscript
        with open('%s/%s.sbatch' % (self.workdir, self.project), 'w') as out:
            out.write(self.runscript)

    def _init_mem(self):
        ## initialize oqp in memory for local calculation
        os.environ['OQP_ROOT'] = self.oqp
        os.environ['OMP_NUM_THREADS'] = str(self.nproc)

        from oqp.pyoqp import Runner
        self.pyoqp = Runner
        self.input_dict['properties']['back_door'] = 'true'

    def _setup_oqp(self, x, q=None):
        ## make calculation folder and input file
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        ## clean calculation folder
        os.system("rm %s/*.log > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/energies > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/grad_* > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/grad_* > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/nacme > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/dcme > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/nac_* > /dev/null 2>&1" % self.workdir)
        os.system("rm %s/soc_* > /dev/null 2>&1" % self.workdir)

        ## setup input
        self._write_oqp(x, q)

    def _write_oqp_io(self, x, q=None):
        ## write oqp input file for I/O calculation
        charge = print_charge(q, 'Q')

        if self.activestate == 1:
            self.input_dict['properties']['grad'] = '%s' % self.state
        else:
            self.input_dict['properties']['grad'] = ','.join(['%s' % (x + 1) for x in range(self.nstate)])

        if self.nactype == 'nac':
            self.input_dict['nac']['states'] = ','.join([' '.join(x) for x in self.nac_coupling])

        ## save input file
        si_input = ''
        for section, section_dict in self.input_dict.items():
            si_input += '[%s]\n' % section
            for key, val in section_dict.items():
                si_input += '%s=%s\n' % (key, val)
            si_input += '\n'

        with open('%s/%s.inp' % (self.workdir, self.project), 'w') as out:
            out.write(si_input)

        ## save xyz file
        natom = len(x)
        xyz = '%s\n\n%s' % (natom, print_coord(x))

        with open('%s/%s.xyz' % (self.workdir, self.project), 'w') as out:
            out.write(xyz)

        ## copy orbital files
        if os.path.exists('%s/%s.json' % (self.workdir, self.project)) is True:
            shutil.copy2('%s/%s.json' % (self.workdir, self.project), '%s/guess.json' % self.workdir)
        else:
            if os.path.exists('%s.json' % self.project) is True:
                shutil.copy2('%s.json' % self.project, '%s/guess.json' % self.workdir)

    def _write_oqp_mem(self, x, q=None):
        ## prepare oqp input dict for in-memory calculation
        self._write_oqp_io(x, q)
        self.runner = self.pyoqp(project=self.project,
                                 input_file='%s/%s.inp' % (self.workdir, self.project),
                                 input_dict=self.input_dict,
                                 log='%s/%s.log' % (self.workdir, self.project),
                                 silent=1)

    def _run_oqp_io(self):
        ## run oqp via I/O
        maindir = os.getcwd()
        os.chdir(self.workdir)
        subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.workdir, self.project)])
        os.chdir(maindir)

    def _run_oqp_mem(self):
        ## run oqp in memory
        maindir = os.getcwd()
        os.chdir(self.workdir)
        self.runner.back_door(self.back_door_data)
        self.runner.run()
        os.chdir(maindir)

    def _read_data_io(self, natom):
        ## read OQP output and pack data

        if not os.path.exists('%s/%s.log' % (self.workdir, self.project)):
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

        with open('%s/%s.log' % (self.workdir, self.project), 'r') as out:
            log = out.read().splitlines()

        coord = []
        for n, line in enumerate(log):
            if 'Cartesian Coordinate in Angstrom' in line:
                coord = oqp_coord(log[n + 4: n + 4 + natom])
                break

        ## pack energy, only includes the requested states by self.nstate
        energy = []
        if os.path.exists('%s/energies' % self.workdir):
            energy = np.loadtxt('%s/energies' % self.workdir).reshape(-1)[1: self.nstate + 1]  # skip the reference

        ## pack force
        gradient = []
        for i in range(self.nstate):
            if os.path.exists('%s/grad_%s' % (self.workdir, i + 1)):
                g = np.loadtxt('%s/grad_%s' % (self.workdir, i + 1))
            else:
                g = [[0, 0, 0] for _ in range(natom)]

            gradient.append(g)

        gradient = np.array(gradient)

        ## pack nac
        nac = []
        if self.nactype == 'nac':
            for pair in self.nac_coupling:
                pa, pb = pair
                if os.path.exists('%s/nac_%s_%s' % (self.workdir, pa, pb)):
                    n = np.loadtxt('%s/nac_%s_%s' % (self.workdir, pa, pb))
                    nac.append(n)
            nac = np.array(nac)

        elif self.nactype == 'nacme':
            nac = np.zeros((self.nstate, self.nstate))
            if os.path.exists('%s/nacme' % self.workdir):
                nacme = np.loadtxt('%s/nacme' % self.workdir)
                for pair in self.nac_coupling:
                    pa, pb = pair
                    nac[pa, pb] = nacme[pa, pb]
                    nac[pb, pa] = nacme[pb, pa]

        soc = np.zeros(0)

        return coord, energy, gradient, nac, soc

    def _read_data_mem(self, natom):
        results = self.runner.results()
        atoms = results['atoms']
        system = results['system']
        coord = oqp_coord2list(atoms, system)

        energy = results['energy'][1:self.nstate + 1]
        gradient = np.array(results['grad'])[1:self.nstate + 1]
        nac = np.array(results['nac'])

        if self.nactype == 'nacme':
            nacm = np.zeros((self.nstate, self.nstate))
            for pair in self.nac_coupling:
                pa, pb = pair
                nacm[pa, pb] = nac[pa, pb]
                nacm[pb, pa] = nac[pb, pa]

            nac = np.array(nacm)

        soc = np.array(results['soc'])
        self.back_door_data = (system, results['data'])

        return coord, energy, gradient, nac, soc

    def _high(self, traj):
        ## run ORCA for high level region in QM calculation

        ## create qmmm model
        traj = traj.apply_qmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup OQP calculation
        self._setup_oqp(xyz, charge)

        ## run OQP calculation
        self._run_oqp()

        ## read OQP output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

        return energy, gradient, nac, soc

    def _high_mid_low(self, traj):
        ## run ORCA for high level region, middle level region, and low level region in QM calculation

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)
        charge = traj.qm2_charge

        ## setup OQP calculation
        self._setup_oqp(xyz, charge)

        ## run OQP calculation
        self._run_oqp()

        ## read OQP output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

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

        if self.runtype == 'qm_high_mid_low':
            energy, gradient, nac, soc = self._high_mid_low(traj)
        elif self.runtype == 'qm_high':
            energy, gradient, nac, soc = self._high(traj)

        if len(energy) >= self.nstate and len(gradient) >= self.nstate and len(nac) >= self.nnac:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        # update trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
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
        coord, energy, gradient, nac, soc = self._read_data_io(natom)

        return coord, np.array(energy), np.array(gradient), np.array(nac), np.array(soc)
