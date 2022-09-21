######################################################
#
# PyRAI2MD test GFN-xTB
#
# Author Jingbai Li
# Sep 20 2022
#
######################################################

import os
import shutil
import subprocess

try:
    import PyRAI2MD

    pyrai2mddir = os.path.dirname(PyRAI2MD.__file__)

except ModuleNotFoundError:
    pyrai2mddir = ''


def TestxTB():
    """ xtb test

    1. xtb energy and gradient

    """

    testdir = '%s/results/xtb' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'XTB': 'VariableNotFound',
    }

    coordpath = './xtb/xtb_data/test.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    if 'XTB' in os.environ:
        record['XTB'] = os.environ['XTB']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |             GFN-xTB Test Calculation              |
 |                                                   |
 *---------------------------------------------------*

 Some environment variables are needed for this test:

    export XTB=/path

 Check files and settings:
-------------------------------------------------------
"""
    for key, location in record.items():
        summary += ' %-10s %s\n' % (key, location)

    for key, location in record.items():
        if location == 'FileNotFound':
            summary += '\n Test files are incomplete, please download it again, skip test\n\n'
            return summary, 'FAILED(test file unavailable)'
        if location == 'VariableNotFound':
            summary += '\n Environment variables are not set, cannot find program, skip test\n\n'
            return summary, 'FAILED(environment variable missing)'

    CopyInput(record, testdir)
    Setup(record, testdir)

    summary += """
 Copy files:
 %-10s --> %s/test.xyz

 Run GFN-xTB:
""" % ('coord', testdir)

    results, code = RunxTB(testdir)

    summary += """
-------------------------------------------------------
                  GFN-xTB OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    return summary, code


def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/test.xyz' % testdir)

def Setup(record, testdir):
    ld_input = """&CONTROL
title         test
qc_ncpu       2
jobtype       sp
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

""" % (record['XTB'],
       testdir)

    with open('%s/test_inp' % testdir, 'w') as out:
        out.write(ld_input)


def Collect(testdir):
    with open('%s/test.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunxTB(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp > stdout 2>&1', shell=True)
    os.chdir(maindir)
    results = Collect(testdir)
    if len(results.splitlines()) < 13:
        code = 'FAILED(GFN-xTB runtime error)'
    else:
        code = 'PASSED'
    return results, code
