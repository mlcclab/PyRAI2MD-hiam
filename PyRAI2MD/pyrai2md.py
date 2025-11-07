######################################################
#
# PyRAI2MD 2 main function
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import sys
import json
import numpy as np

if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if 'NOMPI4PY' not in os.environ:
    try:
        from mpi4py import MPI
        os.environ['MPI4PY'] = ''
    except ModuleNotFoundError:
        pass

from PyRAI2MD.Utils.extension import verify_ext
from PyRAI2MD.variables import read_input
from PyRAI2MD.methods import QM
from PyRAI2MD.Molecule.trajectory import Trajectory
from PyRAI2MD.Dynamics.aimd import AIMD
from PyRAI2MD.Dynamics.mixaimd import MIXAIMD
from PyRAI2MD.Dynamics.single_point import SinglePoint
from PyRAI2MD.Dynamics.hop_probability import HopProb
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Machine_Learning.grid_search import GridSearch
from PyRAI2MD.Machine_Learning.adaptive_sampling import AdaptiveSampling
from PyRAI2MD.Utils.coordinates import read_initcond
from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.sampling import sampling
from PyRAI2MD.Utils.logo import print_logo

class PYRAI2MD:
    """ Main PyRAI2MD interface

        Parameters:          Type:
            ld_input         str         input file name

        Attribute:           Type:
            keywords         dict        keyword dictionary

        Functions:           Returns:
            run              None        run PyRAI2MD calculation
            test             None        run PyRAI2MD testcases
    """

    def __init__(self, ld_input):
        ## check input
        version = '2.5.1'
        self.logo = print_logo(version)

        if ld_input is None:
            print(self.logo)

        else:
            if not os.path.exists(ld_input):
                sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for input file %s' % ld_input)

            input_dict = self._load_input(ld_input)
            self.keywords, input_info = read_input(input_dict)
            ## get control info
            self.keywords['version'] = self._version_info(version, input_info)
            self.title = self.keywords['control']['title']
            self.jobtype = self.keywords['control']['jobtype']
            self.qm = self.keywords['control']['qm']
            self.abinit = self.keywords['control']['abinit']

    @staticmethod
    def _version_info(version, input_info):
        info = """%s

%s

""" % (print_logo(version), input_info)

        return info

    @staticmethod
    def _load_input(ld_input):
        try:
            with open(ld_input, 'r') as ld_file:
                input_dict = json.load(ld_file)

        except json.decoder.JSONDecodeError:
            with open(ld_input, 'r') as ld_file:
                input_file = ld_file.read().splitlines()

            input_lines = []
            for line in input_file:
                code = line.split('#')[0]
                input_lines.append(code)

            input_dict = '\n'.join(input_lines).split('&')

        return input_dict

    def _machine_learning(self):
        train_data = self.keywords[self.qm[0]]['train_data']
        pred_data = self.keywords[self.qm[0]]['pred_data']
        data = Data()

        if self.jobtype == 'train':
            ## get training data
            data.load(train_data)
            data.stat()
            self.keywords[self.qm[0]]['data'] = data

            ## create model
            model = QM(self.qm, keywords=self.keywords, job_id=None)
            model.train()

        elif self.jobtype == 'prediction' or self.jobtype == 'predict':
            ## get training data and prediction data
            data.load(train_data)
            data.load(pred_data, filetype='prediction')
            data.stat()
            self.keywords[self.qm[0]]['data'] = data

            ## create model
            model = QM(self.qm, keywords=self.keywords, job_id=None)
            model.load()
            model.evaluate(None)

        return self

    def _single_point(self):
        ## create a trajectory and method model
        traj = Trajectory(self.title, keywords=self.keywords)
        method = QM(self.qm, keywords=self.keywords, job_id=None)
        method.load()

        sp = SinglePoint(trajectory=traj,
                         keywords=self.keywords,
                         qm=method,
                         job_id=None,
                         job_dir=None)
        sp.run()
        return self

    def _hop_probability(self):
        ## create a trajectory and method model
        traj = Trajectory(self.title, keywords=self.keywords)
        hop = HopProb(trajectory=traj, keywords=self.keywords)
        hop.run()
        return self

    def _dynamics(self):
        ## get md info
        md = self.keywords['md']
        initcond = md['initcond']
        ninitcond = md['ninitcond']
        method = md['method']
        ld_format = md['format']
        gl_seed = md['gl_seed']
        temp = md['temp']

        ## get molecule info
        if initcond == 0:
            mol = self.title
        else:
            ## use sampling method to generate initial condition
            mol = sampling(self.title, ninitcond, gl_seed, temp, method, ld_format)[-1]
            ## save sampled geometry and velocity
            xyz, velo = read_initcond(mol)
            initxyz_info = '%d\n%s\n%s' % (
                len(xyz),
                '%s sampled geom %s at %s K' % (method, ninitcond, temp),
                print_coord(xyz))

            with open('%s.xyz' % self.title, 'w') as initxyz:
                initxyz.write(initxyz_info)

            with open('%s.velo' % self.title, 'w') as initvelo:
                np.savetxt(initvelo, velo, fmt='%30s%30s%30s')

        ## create a trajectory and method model
        if self.qm[0] in ['nn', 'mlp', 'schnet', 'library', 'demo', 'e2n2']:
            train_data = self.keywords[self.qm[0]]['train_data']
            data = Data()
            data.load(train_data)
            data.stat()
            self.keywords[self.qm[0]]['data'] = data

        traj = Trajectory(mol, keywords=self.keywords)
        method = QM(self.qm, keywords=self.keywords, job_id=None)
        method.load()
        aimd = AIMD(trajectory=traj,
                    keywords=self.keywords,
                    qm=method,
                    job_id=None,
                    job_dir=None)
        aimd.run()

        return self

    def _hybrid_dynamics(self):
        ## get md info
        md = self.keywords['md']
        initcond = md['initcond']
        ninitcond = md['ninitcond']
        method = md['method']
        ld_format = md['format']
        gl_seed = md['gl_seed']
        temp = md['temp']

        ## get molecule info
        if initcond == 0:
            mol = self.title
        else:
            ## use sampling method to generate initial condition
            mol = sampling(self.title, ninitcond, gl_seed, temp, method, ld_format)[-1]
            ## save sampled geometry and velocity
            xyz, velo = read_initcond(mol)
            initxyz_info = '%d\n%s\n%s' % (
                len(xyz),
                '%s sampled geom %s at %s K' % (method, ninitcond, temp),
                print_coord(xyz))

            with open('%s.xyz' % self.title, 'w') as initxyz:
                initxyz.write(initxyz_info)

            with open('%s.velo' % self.title, 'w') as initvelo:
                np.savetxt(initvelo, velo, fmt='%30s%30s%30s')

        ## create a trajectory and method model
        traj = Trajectory(mol, keywords=self.keywords)
        ref = QM(self.abinit, keywords=self.keywords, job_id=None)
        ref.load()

        train_data = self.keywords[self.qm[0]]['train_data']
        data = Data()
        data.load(train_data)
        data.stat()
        self.keywords[self.qm[0]]['data'] = data
        method = QM(self.qm, keywords=self.keywords, job_id=None)
        method.load()

        mixaimd = MIXAIMD(trajectory=traj,
                          keywords=self.keywords,
                          qm=method,
                          ref=ref,
                          job_id=None,
                          job_dir=None)
        mixaimd.run()

        return self

    def _active_learning(self):
        learn_proc = AdaptiveSampling(keywords=self.keywords)
        learn_proc.search()

        return self

    def _grid_search(self):
        grid = GridSearch(keywords=self.keywords)
        grid.search()

        return self

    def run(self):
        job_func = {
            'sp': self._single_point,
            'md': self._dynamics,
            'hop': self._hop_probability,
            'hybrid': self._hybrid_dynamics,
            'adaptive': self._active_learning,
            'train': self._machine_learning,
            'prediction': self._machine_learning,
            'predict': self._machine_learning,
            'search': self._grid_search,
        }
        job_func[self.jobtype]()

        return None

    @staticmethod
    def update():
        verify_ext()

        return None

def main():
    pmd = PYRAI2MD

    if len(sys.argv) < 2:
        pmd(None)
        sys.exit("""
  PyRAI2MD input file is not set. Please use the following command
  
     pyrai2md input

""")

    else:
        if sys.argv[1] == 'update':
            pmd(None).update()
        else:
            pmd(sys.argv[1]).run()

    return None


if __name__ == '__main__':
    main()
