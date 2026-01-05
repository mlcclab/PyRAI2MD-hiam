######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 20 2023
#
######################################################

import sys
from PyRAI2MD.Utils.read_tools import ReadVal

class KeyControl:

    def __init__(self):
        self.keywords = {
            'title': None,
            'ml_ncpu': 1,
            'qc_ncpu': 1,
            'ms_ncpu': 1,
            'gl_seed': 1,
            'remote_train': 0,
            'jobtype': 'sp',
            'qm': 'nn',
            'abinit': ['molcas'],
            'refine': 0,
            'refine_num': 4,
            'refine_start': 0,
            'refine_end': 200,
            'refine_gap': 0.3,
            'maxiter': 1,
            'maxsample': 1,
            'dynsample': 0,
            'maxdiscard': 0,
            'maxenergy': 0.05,
            'minenergy': 0.02,
            'dynenergy': 0.1,
            'inienergy': 0.3,
            'fwdenergy': 1,
            'bckenergy': 1,
            'maxgrad': 0.15,
            'mingrad': 0.06,
            'dyngrad': 0.1,
            'inigrad': 0.3,
            'fwdgrad': 1,
            'bckgrad': 1,
            'maxnac': 0.15,
            'minnac': 0.06,
            'dynnac': 0.1,
            'ininac': 0.3,
            'fwdnac': 1,
            'bcknac': 1,
            'maxsoc': 50,
            'minsoc': 20,
            'dynsoc': 0.1,
            'inisoc': 0.3,
            'fwdsoc': 1,
            'bcksoc': 1,
            'load': 1,
            'transfer': 0,
            'pop_step': 200,
            'verbose': 2,
            'silent': 1,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &control
        keywords = self.keywords.copy()
        keyfunc = {
            'title': ReadVal('s'),
            'ml_ncpu': ReadVal('i'),
            'qc_ncpu': ReadVal('i'),
            'ms_ncpu': ReadVal('i'),
            'gl_seed': ReadVal('i'),
            'remote_train': ReadVal('i'),
            'jobtype': ReadVal('s'),
            'qm': ReadVal('sl'),
            'abinit': ReadVal('sl'),
            'refine': ReadVal('i'),
            'refine_num': ReadVal('i'),
            'refine_start': ReadVal('i'),
            'refine_end': ReadVal('i'),
            'refine_gap': ReadVal('f'),
            'maxiter': ReadVal('i'),
            'maxsample': ReadVal('i'),
            'dynsample': ReadVal('i'),
            'maxdiscard': ReadVal('i'),
            'maxenergy': ReadVal('f'),
            'minenergy': ReadVal('f'),
            'dynenergy': ReadVal('f'),
            'inienergy': ReadVal('f'),
            'fwdenergy': ReadVal('i'),
            'bckenergy': ReadVal('i'),
            'maxgrad': ReadVal('f'),
            'mingrad': ReadVal('f'),
            'dyngrad': ReadVal('f'),
            'inigrad': ReadVal('f'),
            'fwdgrad': ReadVal('i'),
            'bckgrad': ReadVal('i'),
            'maxnac': ReadVal('f'),
            'minnac': ReadVal('f'),
            'dynnac': ReadVal('f'),
            'ininac': ReadVal('f'),
            'fwdnac': ReadVal('i'),
            'bcknac': ReadVal('i'),
            'maxsoc': ReadVal('f'),
            'minsoc': ReadVal('f'),
            'dynsoc': ReadVal('f'),
            'inisoc': ReadVal('f'),
            'fwdsoc': ReadVal('i'),
            'bcksoc': ReadVal('i'),
            'load': ReadVal('i'),
            'transfer': ReadVal('i'),
            'pop_step': ReadVal('i'),
            'verbose': ReadVal('i'),
            'silent': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in $control' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &control
-------------------------------------------------------
  Title:                      %-10s
  NPROCS for ML:              %-10s
  NPROCS for QC:              %-10s
  NPROCS for Multiscale:      %-10s 
  Seed:                       %-10s
  Job: 	                      %-10s
  QM:          	       	      %-10s
  Ab initio:                  %-10s
-------------------------------------------------------

        """ % (
            keywords['title'],
            keywords['ml_ncpu'],
            keywords['qc_ncpu'],
            keywords['ms_ncpu'],
            keywords['gl_seed'],
            keywords['jobtype'],
            ' '.join(keywords['qm']),
            ' '.join(keywords['abinit'])
        )

        return summary

    @staticmethod
    def info_adaptive(keywords):
        summary = """
  &adaptive sampling method
-------------------------------------------------------
  Ab initio:                  %-10s
  Load trained model:         %-10s
  Transfer learning:          %-10s
  Remote training             %-10s
  Maxiter:                    %-10s
  Sampling number per traj:   %-10s
  Use dynamical Std:          %-10s
  Max discard range           %-10s
  Refine crossing:            %-10s
  Refine points/range: 	      %-10s %-10s %-10s
  Refine gap:                 %-10s
  MaxStd  energy:             %-10s
  MinStd  energy:             %-10s
  InitStd energy:             %-10s
  Dynfctr energy:             %-10s
  Forward delay energy:       %-10s
  Backward delay energy:      %-10s
  MaxStd  gradient:           %-10s
  MinStd  gradient:           %-10s
  InitStd gradient:           %-10s
  Dynfctr gradient:           %-10s
  Forward delay	gradient:     %-10s
  Backward delay gradient:    %-10s
  MaxStd  nac:                %-10s
  MinStd  nac:                %-10s
  InitStd nac:                %-10s
  Dynfctr nac:                %-10s
  Forward delay	nac:          %-10s
  Backward delay nac:         %-10s
  MaxStd  soc:                %-10s
  MinStd  soc:                %-10s
  InitStd soc:                %-10s
  Dynfctr soc:                %-10s
  Forward delay	soc:   	      %-10s
  Backward delay soc:  	      %-10s
-------------------------------------------------------

        """ % (
            ' '.join(keywords['abinit']),
            keywords['load'],
            keywords['transfer'],
            keywords['remote_train'],
            keywords['maxiter'],
            keywords['maxsample'],
            keywords['dynsample'],
            keywords['maxdiscard'],
            keywords['refine'],
            keywords['refine_num'],
            keywords['refine_start'],
            keywords['refine_end'],
            keywords['refine_gap'],
            keywords['maxenergy'],
            keywords['minenergy'],
            keywords['inienergy'],
            keywords['dynenergy'],
            keywords['fwdenergy'],
            keywords['bckenergy'],
            keywords['maxgrad'],
            keywords['mingrad'],
            keywords['inigrad'],
            keywords['dyngrad'],
            keywords['fwdgrad'],
            keywords['bckgrad'],
            keywords['maxnac'],
            keywords['minnac'],
            keywords['ininac'],
            keywords['dynnac'],
            keywords['fwdnac'],
            keywords['bcknac'],
            keywords['maxsoc'],
            keywords['minsoc'],
            keywords['inisoc'],
            keywords['dynsoc'],
            keywords['fwdsoc'],
            keywords['bcksoc']
        )

        return summary
