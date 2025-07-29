######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Jun 5 2024
#
######################################################

import os
import sys
from PyRAI2MD.Utils.read_tools import ReadVal


class KeyOpenQP:

    def __init__(self):
        self.keywords = {
            'openqp': '',
            'openqp_project': None,
            'openqp_workdir': os.getcwd(),
            'threads': 1,
            'guess_type': 'auto',
            'align_mo': 'true',
            'method': 'tdhf',
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
            'openqp': ReadVal('s'),
            'openqp_project': ReadVal('s'),
            'openqp_workdir': ReadVal('s'),
            'threads': ReadVal('i'),
            'guess_type': ReadVal('s'),
            'align_mo': ReadVal('b'),
            'method': ReadVal('s'),
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
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &oqp' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &openqp
-------------------------------------------------------
  OpenQP:                   %-10s
  OpenQP_project:           %-10s
  OpenQP_workdir:           %-10s
  Num_omp_threads:          %-10s
  Guess orbital type:       %-10s
  Align MO:                 %-10s
  Method type:              %-10s  
  Keep tmp_OpenQP:          %-10s
  Job distribution:         %-10s
-------------------------------------------------------
    """ % (
            keywords['openqp'],
            keywords['openqp_project'],
            keywords['openqp_workdir'],
            keywords['threads'],
            keywords['guess_type'],
            keywords['align_mo'],
            keywords['method'],
            keywords['keep_tmp'],
            keywords['use_hpc']
        )

        return summary
