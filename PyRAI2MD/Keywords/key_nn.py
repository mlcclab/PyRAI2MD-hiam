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
from PyRAI2MD.Utils.read_tools import ReadIndex


class KeyNN:

    def __init__(self, nn_type='nn'):
        self.keywords = {
            'train_mode': 'training',
            'train_data': None,
            'pred_data': None,
            'modeldir': None,
            'silent': 1,
            'nsplits': 10,
            'nn_eg_type': 1,
            'nn_nac_type': 0,
            'nn_soc_type': 0,
            'multiscale': [],
            'shuffle': False,
            'eg_unit': 'si',
            'nac_unit': 'si',
            'soc_unit': 'si',
            'ml_seed': 1,  # Caution! Not allow user to set.
            'permute_map': 'No',
            'gpu': 0,
        }

        self.nn_type = nn_type

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &{nn_type}
        keywords = self.keywords.copy()
        keyfunc = {
            'train_mode': ReadVal('s'),
            'train_data': ReadVal('s'),
            'pred_data': ReadVal('s'),
            'modeldir': ReadVal('s'),
            'nsplits': ReadVal('i'),
            'nn_eg_type': ReadVal('i'),
            'nn_nac_type': ReadVal('i'),
            'nn_soc_type': ReadVal('i'),
            'multiscale': ReadIndex('g'),
            'shuffle': ReadVal('b'),
            'eg_unit': ReadVal('s'),
            'nac_unit': ReadVal('s'),
            'soc_unit': ReadVal('s'),
            'permute_map': ReadVal('s'),
            'gpu': ReadVal('i'),
            'silent': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &%s' % (key, self.nn_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    def info(self, keywords):
        summary = """
          &%s
        -------------------------------------------------------
          Train data:                 %-10s
          Prediction data:            %-10s
          Train mode:                 %-10s
          Silent mode:                %-10s
          Data splits:                %-10s
          NN EG type:                 %-10s
          NN NAC type:                %-10s
          NN SOC type:                %-10s
          Multiscale:                 %-10s
          Shuffle data:               %-10s
          EG unit:                    %-10s
          NAC unit:                   %-10s
          Data permutation            %-10s
        -------------------------------------------------------

        """ % (
            self.nn_type,
            keywords['train_data'],
            keywords['pred_data'],
            keywords['train_mode'],
            keywords['silent'],
            keywords['nsplits'],
            keywords['nn_eg_type'],
            keywords['nn_nac_type'],
            keywords['nn_soc_type'],
            [[x[: 5], '...'] for x in keywords['multiscale']],
            keywords['shuffle'],
            keywords['eg_unit'],
            keywords['nac_unit'],
            keywords['permute_map']
        )

        return summary
