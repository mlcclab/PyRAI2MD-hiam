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


class KeyMolcas:

    def __init__(self):
        self.keywords = {
            'molcas': '',
            'molcas_nproc': '1',
            'molcas_mem': '2000',
            'molcas_print': '2',
            'molcas_project': None,
            'molcas_calcdir': os.getcwd(),
            'molcas_workdir': None,
            'track_phase': 0,
            'basis': 2,
            'omp_num_threads': '1',
            'use_hpc': 0,
            'group': None,  # Caution! Not allow user to set.
            'keep_tmp': 1,
            'verbose': 0,
            'tinker': '',
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &molcas
        keywords = self.keywords.copy()
        keyfunc = {
            'molcas': ReadVal('s'),
            'molcas_nproc': ReadVal('s'),
            'molcas_mem': ReadVal('s'),
            'molcas_print': ReadVal('s'),
            'molcas_project': ReadVal('s'),
            'molcas_calcdir': ReadVal('s'),
            'molcas_workdir': ReadVal('s'),
            'track_phase': ReadVal('i'),
            'basis': ReadVal('i'),
            'omp_num_threads': ReadVal('s'),
            'use_hpc': ReadVal('i'),
            'keep_tmp': ReadVal('i'),
            'verbose': ReadVal('i'),
            'tinker': ReadVal('s'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molcas' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &molcas
-------------------------------------------------------
  Molcas:                   %-10s
  Molcas_nproc:             %-10s
  Molcas_mem:               %-10s
  Molcas_print:      	    %-10s
  Molcas_project:      	    %-10s
  Molcas_workdir:      	    %-10s
  Molcas_calcdir:           %-10s
  Tinker interface:         %-10s
  Omp_num_threads:          %-10s
  Keep tmp_molcas:          %-10s
  Track phase:              %-10s
  Job distribution:         %-10s
-------------------------------------------------------
""" % (
            keywords['molcas'],
            keywords['molcas_nproc'],
            keywords['molcas_mem'],
            keywords['molcas_print'],
            keywords['molcas_project'],
            keywords['molcas_workdir'],
            keywords['molcas_calcdir'],
            keywords['tinker'],
            keywords['omp_num_threads'],
            keywords['keep_tmp'],
            keywords['track_phase'],
            keywords['use_hpc']
        )

        return summary
