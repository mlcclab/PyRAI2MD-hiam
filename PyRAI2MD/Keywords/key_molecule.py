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
            'embedding': False,
            'read_charge': False,
            'boundary': [],
            'freeze': [],
            'constrain': [],
            'cbond': [],
            'cangle': [],
            'cdihedral': [],
            'tbond': [],
            'tangle': [],
            'tdihedral': [],
            'fbond': 10.0,
            'fangle': 0.005,
            'fdihedral': 1e-6,
            'shape': 'ellipsoid',
            'factor': [40],
            'scale': [1.0],
            'cavity': [],
            'center': [],
            'center_type': 'xyz',
            'groups': [],
            'compress': [],
            'track_type': None,
            'track_index': [],
            'track_thrhd': [],
            'track_stop': 0,
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
            'embedding': ReadVal('b'),
            'read_charge': ReadVal('b'),
            'boundary': ReadIndex('g'),
            'freeze': ReadIndex('s', start=1),
            'constrain': ReadIndex('s', start=1),
            'cbond': ReadIndex('g', start=1, sort=False),
            'cangle': ReadIndex('g', start=1, sort=False),
            'cdihedral': ReadIndex('g', start=1, sort=False),
            'tbond': ReadVal('fl'),
            'tangle': ReadVal('fl'),
            'tdihedral': ReadVal('fl'),
            'fbond': ReadVal('f'),
            'fangle': ReadVal('f'),
            'fdihedral': ReadVal('f'),
            'shape': ReadVal('s'),
            'factor': ReadVal('il'),
            'scale': ReadVal('fl'),
            'cavity': ReadVal('fl'),
            'center': ReadIndex('s', start=1),
            'center_type': ReadVal('s'),
            'groups': ReadIndex('g', start=0, sort=False),
            'compress': ReadVal('fl'),
            'track_type': ReadVal('s'),
            'track_index': ReadIndex('g', start=1),
            'track_thrhd': ReadVal('fl'),
            'track_stop': ReadVal('i'),
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
  Reading charge:             %-10s
  Frozen atoms:               %-10s
  Constrained atoms:          %-10s
  Restrained bonds:           %-10s
  Restrained angles:          %-10s
  Restrained dihedrals:       %-10s
  Restrained bond value:      %-10s
  Restrained angle value:     %-10s
  Restrained dihedral value:  %-10s
  Bond potential scale:       %-10s
  Angle potential scale:      %-10s
  Dihedral potential scale:   %-10s
  External potential shape:   %-10s
  External potential radius:  %-10s
  External potential factor:  %-10s
  External potential scale:   %-10s  
  Compress potential volume:  %-10s
  Potential center:           %-10s
  Potential center type:      %-10s
  Constrained groups          %-10s
  Track geometry type:        %-10s
  Track indices:              %-10s
  Track threshold:            %-10s
  Track stop task:            %-10s
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
            keywords['read_charge'],
            keywords['freeze'],
            keywords['constrain'],
            keywords['cbond'],
            keywords['cangle'],
            keywords['cdihedral'],
            keywords['tbond'],
            keywords['tangle'],
            keywords['tdihedral'],
            keywords['fbond'],
            keywords['fangle'],
            keywords['fdihedral'],
            keywords['shape'],
            keywords['cavity'],
            keywords['factor'],
            keywords['scale'],
            keywords['compress'],
            keywords['center'],
            keywords['center_type'],
            keywords['groups'],
            keywords['track_type'],
            keywords['track_index'],
            keywords['track_thrhd'],
            keywords['track_stop'],
            keywords['primitive'],
            keywords['lattice']
        )

        return summary
