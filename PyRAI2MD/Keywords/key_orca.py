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

class KeyOrca:

    def __init__(self):
        self.keywords = {
            'orca': '',
            'orca_project': None,
            'orca_workdir': os.getcwd(),
            'dft_type': 'tddft',
            'mpi': '',
            'use_hpc': 0,
            'keep_tmp': 1,
            'verbose': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &orca
        keywords = self.keywords.copy()
        keyfunc = {
            'orca': ReadVal('s'),
            'orca_project': ReadVal('s'),
            'orca_workdir': ReadVal('s'),
            'dft_type': ReadVal('s'),
            'mpi': ReadVal('s'),
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
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &orca' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &orca
-------------------------------------------------------
  ORCA:                     %-10s
  ORCA_project:             %-10s
  ORCA_workdir:             %-10s
  DFT type:                 %-10s
  MPI:                      %-10s
  Keep tmp_orca:            %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
            keywords['orca'],
            keywords['orca_project'],
            keywords['orca_workdir'],
            keywords['dft_type'],
            keywords['mpi'],
            keywords['keep_tmp'],
            keywords['use_hpc']
        )

        return summary
