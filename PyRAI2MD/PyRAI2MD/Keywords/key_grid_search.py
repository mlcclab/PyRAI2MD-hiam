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


class KeySearch:

    def __init__(self):
        self.keywords = {
            'depth': [1],
            'nn_size': [20],
            'batch_size': [32],
            'reg_l1': [1e-8],
            'reg_l2': [1e-8],
            'dropout': [0.005],
            'n_features': [16],
            'n_blocks': [3],
            'l_max': [1],
            'n_rbf': [8],
            'rbf_layers': [2],
            'rbf_neurons': [32],
            'use_hpc': 1,
            'retrieve': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables form &search
        keywords = self.keywords.copy()
        keyfunc = {
            'depth': ReadVal('il'),
            'nn_size': ReadVal('il'),
            'batch_size': ReadVal('il'),
            'reg_l1': ReadVal('fl'),
            'reg_l2': ReadVal('fl'),
            'dropout': ReadVal('fl'),
            'node_features': ReadVal('il'),
            'n_features': ReadVal('il'),
            'n_blocks': ReadVal('il'),
            'l_max': ReadVal('il'),
            'n_rbf': ReadVal('il'),
            'rbf_layers': ReadVal('il'),
            'rbf_neurons': ReadVal('il'),
            'use_hpc': ReadVal('i'),
            'retrieve': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &search' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &grid search
-------------------------------------------------------
  (nn/demo)
  Layers:                     %-10s
  Neurons/layer::             %-10s
  Batch:                      %-10s
  L1:                         %-10s
  L2:                         %-10s
  Dropout:                    %-10s
  (library)
  n_features                  %-10s
  n_blocks                    %-10s
  l_max                       %-10s
  n_rbf                       %-10s
  rbf_layers                  %-10s
  rbf_neurons                 %-10s
  Job distribution            %-10s
  Retrieve data               %-10s
-------------------------------------------------------

""" % (
            keywords['depth'],
            keywords['nn_size'],
            keywords['batch_size'],
            keywords['reg_l1'],
            keywords['reg_l2'],
            keywords['dropout'],
            keywords['n_features'],
            keywords['n_blocks'],
            keywords['l_max'],
            keywords['n_rbf'],
            keywords['rbf_layers'],
            keywords['rbf_neurons'],
            keywords['use_hpc'],
            keywords['retrieve'],
        )

        return summary
