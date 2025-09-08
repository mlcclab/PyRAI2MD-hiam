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


class KeyReadFile:

    def __init__(self):
        self.keywords = {
            'natom': 0,
            'ncharge': 0,
            'file': None,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables form &file
        keywords = self.keywords.copy()
        keyfunc = {
            'natom': ReadVal('i'),
            'ncharge': ReadVal('i'),
            'file': ReadVal('s'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &file' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &orca
-------------------------------------------------------
  Number of atoms:          %-10s
  Number of charges:        %-10s
  List file                 %-10s
-------------------------------------------------------\
        """ % (
            keywords['natom'],
            keywords['ncharge'],
            keywords['file'],
        )

        return summary
