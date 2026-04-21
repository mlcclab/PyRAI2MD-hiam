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

class KeyLAMMPS:

    def __init__(self):
        self.keywords = {
            'lammps': '',
            'lammps_nproc': 1,
            'lammps_project': None,
            'lammps_workdir': os.getcwd(),
            'lammps_charges': 0,
            'use_hpc': 0,
            'keep_tmp': 1,
            'verbose': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &lammps
        keywords = self.keywords.copy()
        keyfunc = {
            'lammps': ReadVal('s'),
            'lammps_nproc': ReadVal('s'),
            'lammps_project': ReadVal('s'),
            'lammps_workdir': ReadVal('s'),
            'lammps_charges': ReadVal('i'),
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
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &lammps' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &lammps
-------------------------------------------------------
  LAMMPS:                   %-10s
  LAMMPS_nproc:             %-10s
  LAMMPS_project:           %-10s
  LAMMPS_workdir:           %-10s
  LAMMPS charges:           %-10s
  Keep tmp_lammps:          %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
            keywords['lammps'],
            keywords['lammps_nproc'],
            keywords['lammps_project'],
            keywords['lammps_workdir'],
            keywords['lammps_charges'],
            keywords['keep_tmp'],
            keywords['use_hpc']
        )

        return summary
