######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 20 2023
#
######################################################

import os
import sys
from PyRAI2MD.Utils.read_tools import ReadVal


class KeyBagel:

    def __init__(self):
        self.keywords = {
            'bagel': '',
            'bagel_nproc': 1,
            'bagel_project': None,
            'bagel_workdir': os.getcwd(),
            'bagel_archive': 'default',
            'mpi': '',
            'blas': '',
            'lapack': '',
            'boost': '',
            'mkl': '',
            'arch': '',
            'omp_num_threads': '1',
            'use_mpi': 0,
            'use_hpc': 0,
            'group': None,  # Caution! Not allow user to set.
            'keep_tmp': 1,
            'verbose': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &bagel
        keywords = self.keywords.copy()
        keyfunc = {
            'bagel': ReadVal('s'),
            'bagel_nproc': ReadVal('s'),
            'bagel_project': ReadVal('s'),
            'bagel_workdir': ReadVal('s'),
            'bagel_archive': ReadVal('s'),
            'mpi': ReadVal('s'),
            'blas': ReadVal('s'),
            'lapack': ReadVal('s'),
            'boost': ReadVal('s'),
            'mkl': ReadVal('s'),
            'arch': ReadVal('s'),
            'omp_num_threads': ReadVal('s'),
            'use_mpi': ReadVal('i'),
            'use_hpc': ReadVal('i'),
            'keep_tmp': ReadVal('i'),
            'verbose': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &bagel' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
          &bagel
        -------------------------------------------------------
          BAGEL:                    %-10s
          BAGEL_nproc:              %-10s
          BAGEL_project:            %-10s
          BAGEL_workdir:            %-10s
          BAGEL_archive:            %-10s
          MPI:                      %-10s
          BLAS:                     %-10s
          LAPACK:                   %-10s
          BOOST:                    %-10s
          MKL:                      %-10s
          Architecture:             %-10s
          Omp_num_threads:          %-10s
          Keep tmp_bagel:           %-10s
          Job distribution:         %-10s
        -------------------------------------------------------
        """ % (
            keywords['bagel'],
            keywords['bagel_nproc'],
            keywords['bagel_project'],
            keywords['bagel_workdir'],
            keywords['bagel_archive'],
            keywords['mpi'],
            keywords['blas'],
            keywords['lapack'],
            keywords['boost'],
            keywords['mkl'],
            keywords['arch'],
            keywords['omp_num_threads'],
            keywords['keep_tmp'],
            keywords['use_hpc']
        )

        return summary
