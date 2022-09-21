
######################################################
#
# PyRAI2MD test first run
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

import os
from first_run.code_info import register
from first_run.code_info import review

try:
    import PyRAI2MD
    pyrai2mddir = os.path.dirname(PyRAI2MD.__file__)

except ModuleNotFoundError:
    pyrai2mddir = ''

def FirstRun():

    """ first run test

    1. check code completeness
    2. review code structure

    """

    code = 'PASSED'
    totline = 0
    totfile = 0
    length = {}
    summary = """
 *---------------------------------------------------*
 |                                                   |
 |             Check Code Completeness               |
 |                                                   |
 *---------------------------------------------------*

"""
    for name, location in register.items():
        mod = '%s/%s' % (pyrai2mddir, location)
        status = os.path.exists(mod)
        if status:
            totfile += 1
            with open(mod, 'r') as file:
                n = len(file.readlines())
            totline += n
            length[name] = n
            mark = 'Found:'
        else:
            length[name] = 0
            mark = 'Missing:'
            code = 'FAILED(incomplete code)'
        summary += '%-10s %s\n' % (mark, mod)
    summary += """
 *---------------------------------------------------*
 |                                                   |
 |                 Code Structure                    |
 |                                                   |
 *---------------------------------------------------*
"""
    summary += review(length, totline, totfile)

    return summary, code
