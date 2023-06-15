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


class KeyMD:

    def __init__(self):
        self.keywords = {
            'gl_seed': 1,  # Caution! Not allow user to set.
            'initcond': 0,
            'excess': 0,
            'scale': 1,
            'target': 0,
            'graddesc': 0,
            'reset': 0,
            'resetstep': 0,
            'ninitcond': 20,
            'method': 'wigner',
            'format': 'molden',
            'randvelo': 0,
            'temp': 300,
            'step': 10,
            'size': 20.67,
            'root': 1,
            'activestate': 0,
            'sfhp': 'nosh',
            'nactype': 'ktdc',
            'phasecheck': 1,
            'gap': 0.5,
            'gapsoc': 0.5,
            'substep': 20,
            'integrate': 0,
            'deco': '0.1',
            'adjust': 1,
            'reflect': 1,
            'maxh': 10,
            'dosoc': 0,
            'thermo': 'off',
            'thermodelay': 200,
            'silent': 1,
            'verbose': 0,
            'direct': 2000,
            'buffer': 500,
            'record': 'whole',
            'record_step': 0,
            'checkpoint': 0,
            'restart': 0,
            'addstep': 0,
            'group': None,  # Caution! Not allow user to set.
            'ref_energy': 0,
            'ref_grad': 0,
            'ref_nac': 0,
            'ref_soc': 0,
            'datapath': None,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &md
        keywords = self.keywords.copy()
        keyfunc = {
            'initcond': ReadVal('i'),
            'excess': ReadVal('f'),
            'scale': ReadVal('f'),
            'target': ReadVal('f'),
            'graddesc': ReadVal('i'),
            'reset': ReadVal('i'),
            'resetstep': ReadVal('i'),
            'ninitcond': ReadVal('i'),
            'method': ReadVal('s'),
            'format': ReadVal('s'),
            'randvelo': ReadVal('i'),
            'temp': ReadVal('f'),
            'step': ReadVal('i'),
            'size': ReadVal('f'),
            'root': ReadVal('i'),
            'activestate': ReadVal('i'),
            'sfhp': ReadVal('s'),
            'nactype': ReadVal('s'),
            'phasecheck': ReadVal('i'),
            'gap': ReadVal('f'),
            'gapsoc': ReadVal('f'),
            'substep': ReadVal('i'),
            'integrate': ReadVal('i'),
            'deco': ReadVal('s'),
            'adjust': ReadVal('i'),
            'reflect': ReadVal('i'),
            'maxh': ReadVal('i'),
            'dosoc': ReadVal('i'),
            'thermo': ReadVal('s'),
            'thermodelay': ReadVal('i'),
            'silent': ReadVal('i'),
            'verbose': ReadVal('i'),
            'direct': ReadVal('i'),
            'buffer': ReadVal('i'),
            'record': ReadVal('s'),
            'record_step': ReadVal('i'),
            'checkpoint': ReadVal('i'),
            'restart': ReadVal('i'),
            'addstep': ReadVal('i'),
            'ref_energy': ReadVal('i'),
            'ref_grad': ReadVal('i'),
            'ref_nac': ReadVal('i'),
            'ref_soc': ReadVal('i'),
            'datapath': ReadVal('s'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &md' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &initial condition
-------------------------------------------------------
  Generate initial condition: %-10s
  Number:                     %-10s
  Method:                     %-10s 
  Format:                     %-10s
-------------------------------------------------------

        """ % (
            keywords['initcond'],
            keywords['ninitcond'],
            keywords['method'],
            keywords['format']
        )

        summary += """
  &md
-------------------------------------------------------
  Initial state:              %-10s
  Initialize random velocity  %-10s
  Temperature (K):            %-10s
  Step:                       %-10s
  Dt (au):                    %-10s
  Only active state grad      %-10s
  Surface hopping:            %-10s
  NAC type:                   %-10s
  Phase correction            %-10s
  Substep:                    %-10s
  Integrate probability       %-10s
  Decoherence:                %-10s
  Adjust velocity:            %-10s
  Reflect velocity:           %-10s
  Maxhop:                     %-10s
  IC hopping gap threshold    %-10s
  ISC hopping gap threshold   %-10s
  Thermodynamic:              %-10s
  Thermodynamic delay:        %-10s
  Print level:                %-10s
  Direct output:              %-10s
  Buffer output:              %-10s
  Record MD data:             %-10s
  Record MD steps:            %-10s
  Checkpoint steps:           %-10s 
  Restart function:           %-10s
  Additional steps:           %-10s
-------------------------------------------------------

        """ % (
            keywords['root'],
            keywords['randvelo'],
            keywords['temp'],
            keywords['step'],
            keywords['size'],
            keywords['activestate'],
            keywords['sfhp'],
            keywords['nactype'],
            keywords['phasecheck'],
            keywords['substep'],
            keywords['integrate'],
            keywords['deco'],
            keywords['adjust'],
            keywords['reflect'],
            keywords['maxh'],
            keywords['gap'],
            keywords['gapsoc'],
            keywords['thermo'],
            keywords['thermodelay'],
            keywords['verbose'],
            keywords['direct'],
            keywords['buffer'],
            keywords['record'],
            keywords['record_step'],
            keywords['checkpoint'],
            keywords['restart'],
            keywords['addstep']
        )

        summary += """
  &md velocity control
-------------------------------------------------------
  Excess kinetic energy       %-10s
  Scale kinetic energy        %-10s
  Target kinetic energy       %-10s
  Gradient descent path       %-10s
  Reset velocity:             %-10s
  Reset step:                 %-10s
-------------------------------------------------------

        """ % (
            keywords['excess'],
            keywords['scale'],
            keywords['target'],
            keywords['graddesc'],
            keywords['reset'],
            keywords['resetstep']
        )

        return summary

    @staticmethod
    def info_hybrid(keywords):
        summary = """
  &hybrid namd
-------------------------------------------------------
  Mix Energy                  %-10s
  Mix Gradient                %-10s
  Mix NAC                     %-10s
  Mix SOC                     %-10s
-------------------------------------------------------

        """ % (
            keywords['ref_energy'],
            keywords['ref_grad'],
            keywords['ref_nac'],
            keywords['ref_soc']
        )

        return summary
