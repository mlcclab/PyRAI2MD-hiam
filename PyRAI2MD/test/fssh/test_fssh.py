######################################################
#
# PyRAI2MD test FSSH
#
# Author Jingbai Li
# Oct 1 2021
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


def TestFSSH():
    """ molcas test

    1. FSSH calculation

    """

    testdir = '%s/results/fssh' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'energy': 'FileNotFound',
        'energy1': 'FileNotFound',
        'energy2': 'FileNotFound',
        'energy3': 'FileNotfound',
        'kinetic1': 'FileNotFound',
        'kinetic2': 'FileNotFound',
        'velo1': 'FileNotFound',
        'velo2': 'FileNotFound',
        'nac1': 'FileNotFound',
        'nac2': 'FileNotFound',
        'soc1': 'FileNotFound',
        'soc2': 'FileNotFound',
        'pop2': 'FileNotFound',
    }

    filepath = './fssh/fssh_data/c3h2o.xyz'
    if os.path.exists(filepath):
        record['coord'] = filepath

    filepath = './fssh/fssh_data/c3h2o.energy'
    if os.path.exists(filepath):
        record['energy'] = filepath

    filepath = './fssh/fssh_data/c3h2o.energy.1'
    if os.path.exists(filepath):
        record['energy1'] = filepath

    filepath = './fssh/fssh_data/c3h2o.energy.2'
    if os.path.exists(filepath):
        record['energy2'] = filepath

    filepath = './fssh/fssh_data/c3h2o.energy.3'
    if os.path.exists(filepath):
        record['energy3'] = filepath

    filepath = './fssh/fssh_data/c3h2o.kinetic.1'
    if os.path.exists(filepath):
        record['kinetic1'] = filepath

    filepath = './fssh/fssh_data/c3h2o.kinetic.2'
    if os.path.exists(filepath):
        record['kinetic2'] = filepath

    filepath = './fssh/fssh_data/c3h2o.nac.1'
    if os.path.exists(filepath):
        record['nac1'] = filepath

    filepath = './fssh/fssh_data/c3h2o.nac.2'
    if os.path.exists(filepath):
        record['nac2'] = filepath

    filepath = './fssh/fssh_data/c3h2o.soc.1'
    if os.path.exists(filepath):
        record['soc1'] = filepath

    filepath = './fssh/fssh_data/c3h2o.soc.2'
    if os.path.exists(filepath):
        record['soc2'] = filepath

    filepath = './fssh/fssh_data/c3h2o.velo.1'
    if os.path.exists(filepath):
        record['velo1'] = filepath

    filepath = './fssh/fssh_data/c3h2o.velo.2'
    if os.path.exists(filepath):
        record['velo2'] = filepath

    filepath = './fssh/fssh_data/c3h2o.pop.2'
    if os.path.exists(filepath):
        record['pop2'] = filepath

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |              FSSH Test Calculation                |
 |                                                   |
 *---------------------------------------------------*

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
    Setup(testdir)

    summary += """
 Copy files:
 %-10s --> %s/c3h2o.xyz
 %-10s --> %s/c3h2o.energy
 %-10s --> %s/c3h2o.energy.1
 %-10s --> %s/c3h2o.energy.2
 %-10s --> %s/c3h2o.energy.3
 %-10s --> %s/c3h2o.kinetic
 %-10s --> %s/c3h2o.kinetic.1
 %-10s --> %s/c3h2o.nac
 %-10s --> %s/c3h2o.nac.1
 %-10s --> %s/c3h2o.soc
 %-10s --> %s/c3h2o.soc.1
 %-10s --> %s/c3h2o.velo
 %-10s --> %s/c3h2o.velo.1
 %-10s --> %s/c3h2o.pop.1

 Run FSSH Calculation:
""" % ('coord', testdir,
       'energy1', testdir,
       'energy2', testdir,
       'energy', testdir,
       'energy3', testdir,
       'kinetic1', testdir,
       'kinetic2', testdir,
       'nac1', testdir,
       'nac2', testdir,
       'soc1', testdir,
       'soc2', testdir,
       'velo1', testdir,
       'velo2', testdir,
       'pop2', testdir)

    results, code = RunFSSH(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                     FSSH OUTPUT (NAC)
-------------------------------------------------------
%s
-------------------------------------------------------

 nactype == nac test done, entering nactype == ktdc test
""" % results
    else:
        summary += """
 nactype == test failed, stop here
"""
        return summary, code

    results, code = RunFSSH2(testdir)

    summary += """
-------------------------------------------------------
                     FSSH OUTPUT (kTDC)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    return summary, code


def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/c3h2o.xyz' % testdir)
    shutil.copy2(record['energy'], '%s/c3h2o.energy.2' % testdir)
    shutil.copy2(record['energy1'], '%s/c3h2o.energy' % testdir)
    shutil.copy2(record['energy2'], '%s/c3h2o.energy.1' % testdir)
    shutil.copy2(record['energy3'], '%s/c3h2o.energy.3' % testdir)
    shutil.copy2(record['kinetic1'], '%s/c3h2o.kinetic' % testdir)
    shutil.copy2(record['kinetic2'], '%s/c3h2o.kinetic.1' % testdir)
    shutil.copy2(record['nac1'], '%s/c3h2o.nac' % testdir)
    shutil.copy2(record['nac2'], '%s/c3h2o.nac.1' % testdir)
    shutil.copy2(record['soc1'], '%s/c3h2o.soc' % testdir)
    shutil.copy2(record['soc2'], '%s/c3h2o.soc.1' % testdir)
    shutil.copy2(record['velo1'], '%s/c3h2o.velo' % testdir)
    shutil.copy2(record['velo2'], '%s/c3h2o.velo.1' % testdir)
    shutil.copy2(record['pop2'], '%s/c3h2o.pop.1' % testdir)


def Setup(testdir):
    ld_input = """&CONTROL
title         c3h2o
qc_ncpu       2
jobtype       hop
qm            molcas

&Molecule
ci       2 1
spin     0 1
coupling 1 2, 2 3

&MD
root 2
sfhp fssh
nactype nac
datapath %s
verbose 0
""" % testdir

    input2 = """&CONTROL
title         c3h2o
qc_ncpu       2
jobtype       hop
qm            molcas

&Molecule
ci	 2 1
spin     0 1
coupling 1 2, 2 3

&MD
root 2
sfhp fssh
nactype ktdc
datapath %s
verbose 0
""" % testdir

    with open('%s/test_inp' % testdir, 'w') as out:
        out.write(ld_input)

    with open('%s/test_inp2' % testdir, 'w') as out:
        out.write(input2)


def Collect(testdir):
    with open('%s/c3h2o.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results


def RunFSSH(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp', shell=True)
    os.chdir(maindir)
    results = Collect(testdir)
    if len(results.splitlines()) < 10:
        code = 'FAILED(fssh nac runtime error)'
    else:
        code = 'PASSED'

    subprocess.run('mv %s/c3h2o.log %s/c3h2o.log.1' % (testdir, testdir), shell=True)

    return results, code


def RunFSSH2(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp2', shell=True)
    os.chdir(maindir)
    results = Collect(testdir)
    if len(results.splitlines()) < 10:
        code = 'FAILED(fssh ktdc runtime error)'
    else:
        code = 'PASSED'

    return results, code
