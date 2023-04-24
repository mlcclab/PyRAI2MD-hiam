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

class KeyDimeNet:

    def __init__(self, key_type='nac'):
        eg = None

        nac = {
            'model_type': ' ',
            'batch_size': 30,
            'val_size': 2000,
            'hidden_channels': 256,
            'blocks': 6,
            'bilinear': 8,
            'spherical': 7,
            'radial': 6,
            'lr': 0.001,
            'epo': 200,
        }

        soc = None

        keywords = {
            'eg': eg,
            'nac': nac,
            'soc': soc,
        }

        self.keywords = keywords[key_type]
        self.key_type = key_type

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &dime_eg,&dime_nac,&dime_soc
        keywords = self.keywords.copy()
        keyfunc = {
            'model_type': ReadVal('s'),
            'batch_size': ReadVal('i'),
            'val_size': ReadVal('i'),
            'hidden_channels': ReadVal('i'),
            'blocks': ReadVal('i'),
            'bilinear': ReadVal('i'),
            'spherical': ReadVal('i'),
            'radial': ReadVal('i'),
            'lr': ReadVal('f'),
            'epo': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit(
                    '\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &dime_%s' % (key, self.key_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(nac):
        summary = """

  DimeNet (NAC only)

  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  Model type:                 DimeNet%-13s DimeNet%-13s DimeNet%-13s
  Batch size:                 %-20s %-20s %-20s
  Validation size:            %-20s %-20s %-20s 
  Number of hidden channels:  %-20s %-20s %-20s
  Number of blocks:           %-20s %-20s %-20s
  Number of bilinear:         %-20s %-20s %-20s
  Number of spherical:        %-20s %-20s %-20s
  Number of radial:           %-20s %-20s %-20s
  Learning rate:              %-20s %-20s %-20s
  Epochs:                     %-20s %-20s %-20s
----------------------------------------------------------------------------------------------
        """ % (
            '',
            nac['model_type'],
            '',
            'n/a',
            nac['batch_size'],
            'n/a',
            'n/a',
            nac['val_size'],
            'n/a',
            'n/a',
            nac['hidden_channels'],
            'n/a',
            'n/a',
            nac['blocks'],
            'n/a',
            'n/a',
            nac['bilinear'],
            'n/a',
            'n/a',
            nac['spherical'],
            'n/a',
            'n/a',
            nac['radial'],
            'n/a',
            'n/a',
            nac['lr'],
            'n/a',
            'n/a',
            nac['epo'],
            'n/a',
        )

        return summary
