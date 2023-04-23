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

class KeyMolecule:

    def __init__(self):
        self.keywords = {
            'qmmm_key': None,
            'qmmm_xyz': 'Input',
            'ci': [1],
            'spin': [0],
            'coupling': [],
            'highlevel': [],
            'midlevel': [],
            'embedding': 1,
            'boundary': [],
            'freeze': [],
            'constrain': [],
            'shape': 'ellipsoid',
            'factor': 40,
            'cavity': [],
            'center': [],
            'compress': [],
            'track_type': None,
            'track_index': [],
            'track_thrhd': [],
            'primitive': [],
            'lattice': [],
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &molecule
        keywords = self.keywords.copy()
        keyfunc = {
            'ci': ReadVal('il'),
            'spin': ReadVal('il'),
            'coupling': ReadIndex('g'),
            'qmmm_key': ReadVal('s'),
            'qmmm_xyz': ReadVal('s'),
            'highlevel': ReadIndex('s', start=1),
            'midlevel': ReadIndex('s', start=1),
            'embedding': ReadVal('i'),
            'boundary': ReadIndex('g'),
            'freeze': ReadIndex('s', start=1),
            'constrain': ReadIndex('s', start=1),
            'shape': ReadVal('s'),
            'factor': ReadVal('i'),
            'cavity': ReadVal('fl'),
            'center': ReadIndex('s', start=1),
            'compress': ReadVal('fl'),
            'track_type': ReadVal('s'),
            'track_index': ReadIndex('g', start=1),
            'track_thrhd': ReadVal('fl'),
            'primitive': ReadIndex('g'),
            'lattice': ReadIndex('s'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molecule' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &molecule
-------------------------------------------------------
  States:                     %-10s
  Spin:                       %-10s
  Interstates:                %-10s
  QMMM keyfile:               %-10s
  QMMM xyzfile:               %-10s
  High level region:          %-10s ...
  Middel level region:        %-10s ...
  Boundary:                   %-10s ...
  Embedding charges:          %-10s
  Frozen atoms:               %-10s
  Constrained atoms:          %-10s
  External potential shape:   %-10s
  External potential factor:  %-10s
  External potential radius:  %-10s
  External potential center:  %-10s
  Compress potential shape    %-10s
  Track geometry type         %-10s
  Track indices               %-10s
  Track threshold             %-10s                  
  Primitive vectors:          %-10s
  Lattice constant:           %-10s
-------------------------------------------------------

        """ % (
            keywords['ci'],
            keywords['spin'],
            keywords['coupling'],
            keywords['qmmm_key'],
            keywords['qmmm_xyz'],
            keywords['highlevel'][0:10],
            keywords['midlevel'][0:10],
            keywords['boundary'][0:5],
            keywords['embedding'],
            keywords['freeze'],
            keywords['constrain'],
            keywords['shape'],
            keywords['factor'],
            keywords['cavity'],
            keywords['center'],
            keywords['compress'],
            keywords['track_type'],
            keywords['track_index'],
            keywords['track_thrhd'],
            keywords['primitive'],
            keywords['lattice']
        )

        return summary
