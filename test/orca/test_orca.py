######################################################
#
# PyRAI2MD test ORCA
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


def TestORCA():
    """ orca test

    1. orca energy and gradient

    """

    testdir = '%s/results/orca' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'dft': 'FileNotFound',
        'tddft': 'FileNotFound',
        'sf_tddft': 'FileNotFound',
        'ORCA': 'VariableNotFound',
    }

    coordpath = './orca/orca_data/c6h6.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    coordpath = './orca/orca_data/dft.inp'
    if os.path.exists(coordpath):
        record['dft'] = coordpath

    coordpath = './orca/orca_data/tddft.inp'
    if os.path.exists(coordpath):
        record['tddft'] = coordpath

    coordpath = './orca/orca_data/sf_tddft.inp'
    if os.path.exists(coordpath):
        record['sf_tddft'] = coordpath

    if 'ORCA' in os.environ:
        record['ORCA'] = os.environ['ORCA']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |               ORCA Test Calculation               |
 |                                                   |
 *---------------------------------------------------*

 Some environment variables are needed for this test:

    export ORCA=/path

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
 %-10s --> %s/dft.xyz
 %-10s --> %s/tddft.xyz
 %-10s --> %s/sf_tddft.xyz
 %-10s --> %s/dft.orca
 %-10s --> %s/tddft.orca
 %-10s --> %s/sf_tddft.orca

 Run ORCA:
""" % ('coord', testdir,
       'coord', testdir,
       'coord', testdir,
       'dft', testdir,
       'tddft', testdir,
       'sf_tddft', testdir)

    results, code = RunORCA_dft(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                  ORCA DFT OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
    """ % results
    else:
        summary += """
 ORCA DFT failed, stop here
"""
        return summary, code

    results, code = RunORCA_tddft(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                  ORCA TDDFT OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
 ORCA TDDFT failed, stop here
"""
        return summary, code

    results, code = RunORCA_sf_tddft(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
             ORCA Spin-Flip TDDFT OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
 ORCA Spin-Flip TDDFT failed, stop here
"""

    return summary, code

def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/dft.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/tddft.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/sf_tddft.xyz' % testdir)
    shutil.copy2(record['dft'], '%s/dft.orca' % testdir)
    shutil.copy2(record['tddft'], '%s/tddft.orca' % testdir)
    shutil.copy2(record['sf_tddft'], '%s/sf_tddft.orca' % testdir)

def Setup(record, testdir):
    dft_input = """&CONTROL
title         dft
qc_ncpu       1
jobtype       sp
qm            orca

&ORCA
orca          %s
orca_workdir  %s
dft_type      dft
""" % (
        record['ORCA'],
        testdir
    )

    tddft_input = """&CONTROL
title         tddft
qc_ncpu       1
jobtype       sp
qm            orca

&MOLECULE
ci   3
spin 0

&ORCA
orca          %s
orca_workdir  %s
dft_type      tddft
""" % (
        record['ORCA'],
        testdir
    )

    sf_tddft_input = """&CONTROL
title         sf_tddft
qc_ncpu       1
jobtype       sp
qm            orca

&MOLECULE
ci   2
spin 0

&ORCA
orca          %s
orca_workdir  %s
dft_type      sf_tddft
""" % (
        record['ORCA'],
        testdir
    )

    with open('%s/dft_inp' % testdir, 'w') as out:
        out.write(dft_input)

    with open('%s/tddft_inp' % testdir, 'w') as out:
        out.write(tddft_input)

    with open('%s/sf_tddft_inp' % testdir, 'w') as out:
        out.write(sf_tddft_input)


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

def RunORCA_dft(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md dft_inp', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'dft')
    if len(results.splitlines()) < 13:
        code = 'FAILED(ORCA runtime error)'
    else:
        code = 'PASSED'
    return results, code

def RunORCA_tddft(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md tddft_inp', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'tddft')
    if len(results.splitlines()) < 13:
        code = 'FAILED(ORCA runtime error)'
    else:
        code = 'PASSED'
    return results, code

def RunORCA_sf_tddft(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md sf_tddft_inp', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'sf_tddft')
    if len(results.splitlines()) < 13:
        code = 'FAILED(ORCA runtime error)'
    else:
        code = 'PASSED'
    return results, code
