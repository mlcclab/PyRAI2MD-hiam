######################################################
#
# PyRAI2MD test OpenQP
#
# Author Jingbai Li
# Aug 16 2024
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


def TestOpenQP():
    """ OpenQP test

    1. Open QP mrsf-tddft energy and gradient

    """

    testdir = '%s/results/openqp' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'mrsf_tddft': 'FileNotFound',
        'OPENQP': 'VariableNotFound',
    }

    coordpath = './openqp/openqp_data/c6h6.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    coordpath = './openqp/openqp_data/mrsf_tddft.inp'
    if os.path.exists(coordpath):
        record['mrsf_tddft'] = coordpath

    if 'OPENQP' in os.environ:
        record['OPENQP'] = os.environ['OPENQP']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |              OpenQP Test Calculation              |
 |                                                   |
 *---------------------------------------------------*

 Some environment variables are needed for this test:

    export OPENQP=/path

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
 %-10s --> %s/mrsf_tddft.xyz
 %-10s --> %s/mrsf_tddft.openqp

 Run OpenQP:
""" % ('coord', testdir,
       'mrsf_tddft', testdir,
       )

    results, code = RunOpenQP_mrsf_tddft(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
             OpenQP MRSF-TDDFT OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
 OpenQP MRSF-TDDFT failed, stop here
"""

    return summary, code

def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/mrsf_tddft.xyz' % testdir)
    shutil.copy2(record['mrsf_tddft'], '%s/mrsf_tddft.openqp' % testdir)

def Setup(record, testdir):
    mrsf_tddft_input = """&CONTROL
title         mrsf_tddft
qc_ncpu       1
jobtype       sp
qm            openqp

&MOLECULE
ci   3
spin 0

&openqp
openqp        %s
openqp_workdir  %s

""" % (
        record['OPENQP'],
        testdir
    )

    with open('%s/mrsf_tddft_inp' % testdir, 'w') as out:
        out.write(mrsf_tddft_input)


def Collect(testdir, name):
    with open('%s/%s.log' % (testdir, name), 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunOpenQP_mrsf_tddft(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md mrsf_tddft_inp', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'mrsf_tddft')
    if len(results.splitlines()) < 14:
        code = 'FAILED(OpenQP runtime error)'
    else:
        code = 'PASSED'
    return results, code
