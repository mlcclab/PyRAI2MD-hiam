######################################################
#
# PyRAI2MD test cases
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

test_first_run = 1
test_bagel = 1
test_molcas = 1
test_molcas_tinker = 1
test_orca = 1
test_xtb = 1
test_fssh = 1
test_gsh = 1
test_nn = 1
test_pynnsmd = 1
test_grid_search = 1
test_aimd = 1
test_mixaimd = 1
test_adaptive_sampling = 1

import time
import datetime
import os

class TestCase:
    """ PyRAI2MD test cases
    1. check code completeness
        first_run

    2. test qc methods
        bagel local hpc
        molcas local hpc
        molcas_tinker local hpc
        orca local hpc
        xtb local hpc

    3. test ml method
        train and prediction
        grid_search seq, hpc

    4. test surface hopping
        fssh
        gsh nac soc

    5. test md
        aimd
        mixaimd
        ensemble

    6. test adaptive sampling
        adaptive sampling

    7. test utils
        alignment
        coordinates
        sampling

"""

    def __init__(self):
        self.register = {
            'first_run': test_first_run,
            'bagel': test_bagel,
            'molcas': test_molcas,
            'molcas_tinker': test_molcas_tinker,
            'orca': test_orca,
            'xtb': test_xtb,
            'fssh': test_fssh,
            'gsh': test_gsh,
            'neural_network': test_nn,
            'pynnsmd': test_pynnsmd,
            'grid_search': test_grid_search,
            'aimd': test_aimd,
            'mixaimd': test_mixaimd,
            'adaptive_sampling': test_adaptive_sampling,
        }

        self.test_func = {}

        if os.path.exists('./first_run/test_first_run.py'):
            from first_run.test_first_run import FirstRun
            self.test_func['first_run'] = FirstRun

        if os.path.exists('./bagel/test_bagel.py'):
            from bagel.test_bagel import TestBagel
            self.test_func['bagel'] = TestBagel

        if os.path.exists('./molcas/test_molcas.py'):
            from molcas.test_molcas import TestMolcas
            self.test_func['molcas'] = TestMolcas

        if os.path.exists('./molcas_tinker/test_molcas_tinker.py'):
            from molcas_tinker.test_molcas_tinker import TestMolcasTinker
            self.test_func['molcas_tinker'] = TestMolcasTinker

        if os.path.exists('./orca/test_orca.py'):
            from orca.test_orca import TestORCA
            self.test_func['orca'] = TestORCA

        if os.path.exists('./xtb/test_xtb.py'):
            from xtb.test_xtb import TestxTB
            self.test_func['xtb'] = TestxTB

        if os.path.exists('./neural_network/test_nn.py'):
            from neural_network.test_nn import TestNN
            self.test_func['neural_network'] = TestNN

        if os.path.exists('./pynnsmd/test_pynnsmd.py'):
            from pynnsmd.test_pynnsmd import TestPyNNsMD
            self.test_func['pynnsmd'] = TestPyNNsMD

        if os.path.exists('./grid_search/test_grid_search.py'):
            from grid_search.test_grid_search import TestGridSearch
            self.test_func['grid_search'] = TestGridSearch

        if os.path.exists('./fssh/test_fssh.py'):
            from fssh.test_fssh import TestFSSH
            self.test_func['fssh'] = TestFSSH

        if os.path.exists('./gsh/test_gsh.py'):
            from gsh.test_gsh import TestGSH
            self.test_func['gsh'] = TestGSH

        if os.path.exists('./aimd/test_aimd.py'):
            from aimd.test_aimd import TestAIMD
            self.test_func['aimd'] = TestAIMD

        if os.path.exists('./mixaimd/test_mixaimd.py'):
            from mixaimd.test_mixaimd import TestMIXAIMD
            self.test_func['mixaimd'] = TestMIXAIMD

        if os.path.exists('./adaptive_sampling/test_adaptive_sampling.py'):
            from adaptive_sampling.test_adaptive_sampling import TestAdaptiveSampling
            self.test_func['adaptive_sampling'] = TestAdaptiveSampling

    def run(self):
        heading = '''

-------------------------------------------------------
                       PyRAI2MD
               _____  ____  ____  _____   
                 |    |___  |___`   |
                 |    |___  .___|   |
-------------------------------------------------------

'''
        with open('test.log', 'w') as out:
            out.write(heading)

        print(heading)
        ntest = len(self.register)
        n = 0
        for testcase, status in self.register.items():
            n += 1
            print('Tests %3s of %3s: %-20s ...      ' % (n, ntest, testcase), end='')
            summary = ''
            if status == 0:
                summary += '\nTests %-20s Skipped\n\n' % testcase
                code = 'SKIPPED'
            else:
                start = time.time()

                summary += '==Tests==> %-20s Start:  %s\n' % (testcase, WhatIsTime())
                results, code = self.test_func[testcase]()
                summary += results
                summary += '==Tests==> %-20s End:    %s\n' % (testcase, WhatIsTime())

                end = time.time()
                walltime = HowLong(start, end)
                summary += '==Tests==> %-20sUsed:    %s\n\n' % (testcase, walltime)
            print(code)

            with open('test.log', 'a') as out:
                out.write(summary)


def WhatIsTime():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')


def HowLong(start, end):
    ## This function calculate time between start and end

    walltime = end - start
    walltime = '%5d days %5d hours %5d minutes %5d seconds' % (
        int(walltime / 86400),
        int((walltime % 86400) / 3600),
        int(((walltime % 86400) % 3600) / 60),
        int(((walltime % 86400) % 3600) % 60))
    return walltime


def main():
    test = TestCase()
    test.run()


if __name__ == '__main__':
    main()
