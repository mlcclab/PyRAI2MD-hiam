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

class KeyXtb:

    def __init__(self):
        self.keywords = {
            'xtb': '',
            'xtb_nproc': 1,
            'xtb_project': None,
            'xtb_workdir': os.getcwd(),
            'gfnver': -2,
            'mem': '1000',
            'use_hpc': 0,
            'keep_tmp': 1,
            'verbose': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &xtb
        keywords = self.keywords.copy()
        keyfunc = {
            'xtb': ReadVal('s'),
            'xtb_nproc': ReadVal('s'),
            'xtb_project': ReadVal('s'),
            'xtb_workdir': ReadVal('s'),
            'gfnver': ReadVal('i'),
            'use_hpc': ReadVal('i'),
            'mem': ReadVal('s'),
            'keep_tmp': ReadVal('i'),
            'verbose': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &xtb' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &xtb
-------------------------------------------------------
  XTB:                      %-10s
  XTB_project:              %-10s
  XTB_workdir:              %-10s
  XTB version:              %-10s
  Omp_num_threads:          %-10s
  Omp_stacksize:            %-10s
  Keep tmp_xtb:             %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
            keywords['xtb'],
            keywords['xtb_project'],
            keywords['xtb_workdir'],
            keywords['gfnver'],
            keywords['xtb_nproc'],
            keywords['mem'],
            keywords['keep_tmp'],
            keywords['use_hpc']
        )

        return summary
