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

class KeyTempl:

    def __init__(self):
        self.keywords = {
            'key': 'value'
        }

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &xtb
        keywords = self.keywords.copy()
        keyfunc = {
            'key1': ReadVal('s'),
            'key2': ReadIndex('gl'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &templ' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &templ
-------------------------------------------------------
  Key:                      %-10s
-------------------------------------------------------
    """ % (
            keywords['key'],
        )

        return summary
