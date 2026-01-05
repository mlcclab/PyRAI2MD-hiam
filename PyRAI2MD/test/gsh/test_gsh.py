######################################################
#
# PyRAI2MD test GSH
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


def TestGSH():
    """ molcas test

    1. GSH internal conversion
    2. GSH intersystem crossing

    """

    testdir = '%s/results/gsh' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'coord1': 'FileNotFound',
        'coord2': 'FileNotfound',
        'energy': 'FileNotfound',
        'energy1': 'FileNotFound',
        'energy2': 'FileNotFound',
        'grad': 'FileNotFound',
        'grad1': 'FileNotFound',
        'grad2': 'FileNotFound',
        'kinetic': 'FileNotfound',
        'kinetic1': 'FileNotFound',
        'kinetic2': 'FileNotFound',
        'velo': 'FileNotFound',
        'velo1': 'FileNotFound',
        'velo2': 'FileNotFound',
        'soc': 'FileNotFound',
        'soc1': 'FileNotFound',
        'soc2': 'FileNotFound',
    }
    record2 = {
        'coord': 'FileNotFound',
        'coord1': 'FileNotFound',
        'coord2': 'FileNotfound',
        'energy': 'FileNotfound',
        'energy1': 'FileNotFound',
        'energy2': 'FileNotFound',
        'grad': 'FileNotFound',
        'grad1': 'FileNotFound',
        'grad2': 'FileNotFound',
        'kinetic': 'FileNotfound',
        'kinetic1': 'FileNotFound',
        'kinetic2': 'FileNotFound',
        'velo': 'FileNotFound',
        'velo1': 'FileNotFound',
        'velo2': 'FileNotFound',
        'soc': 'FileNotFound',
        'soc1': 'FileNotFound',
        'soc2': 'FileNotFound',
    }

    filepath = './gsh/gsh_data/c3h2o.xyz'
    if os.path.exists(filepath):
        record['coord'] = filepath

    filepath = './gsh/gsh_data/c3h2o.xyz.1'
    if os.path.exists(filepath):
        record['coord1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.xyz.2'
    if os.path.exists(filepath):
        record['coord2'] = filepath

    filepath = './gsh/gsh_data/c3h2o.energy'
    if os.path.exists(filepath):
        record['energy'] = filepath

    filepath = './gsh/gsh_data/c3h2o.energy.1'
    if os.path.exists(filepath):
        record['energy1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.energy.2'
    if os.path.exists(filepath):
        record['energy2'] = filepath

    filepath = './gsh/gsh_data/c3h2o.grad'
    if os.path.exists(filepath):
        record['grad'] = filepath

    filepath = './gsh/gsh_data/c3h2o.grad.1'
    if os.path.exists(filepath):
        record['grad1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.grad.2'
    if os.path.exists(filepath):
        record['grad2'] = filepath

    filepath = './gsh/gsh_data/c3h2o.kinetic'
    if os.path.exists(filepath):
        record['kinetic'] = filepath

    filepath = './gsh/gsh_data/c3h2o.kinetic.1'
    if os.path.exists(filepath):
        record['kinetic1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.kinetic.2'
    if os.path.exists(filepath):
        record['kinetic2'] = filepath

    filepath = './gsh/gsh_data/c3h2o.velo'
    if os.path.exists(filepath):
        record['velo'] = filepath

    filepath = './gsh/gsh_data/c3h2o.velo.1'
    if os.path.exists(filepath):
        record['velo1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.velo.2'
    if os.path.exists(filepath):
        record['velo2'] = filepath

    filepath = './gsh/gsh_data/c3h2o.soc'
    if os.path.exists(filepath):
        record['soc'] = filepath

    filepath = './gsh/gsh_data/c3h2o.soc.1'
    if os.path.exists(filepath):
        record['soc1'] = filepath

    filepath = './gsh/gsh_data/c3h2o.soc.2'
    if os.path.exists(filepath):
        record['soc2'] = filepath
    ## -----------------------------------------------------------
    filepath = './gsh/gsh_data/c2h4.xyz'
    if os.path.exists(filepath):
        record2['coord'] = filepath

    filepath = './gsh/gsh_data/c2h4.xyz.1'
    if os.path.exists(filepath):
        record2['coord1'] = filepath

    filepath = './gsh/gsh_data/c2h4.xyz.2'
    if os.path.exists(filepath):
        record2['coord2'] = filepath

    filepath = './gsh/gsh_data/c2h4.energy'
    if os.path.exists(filepath):
        record2['energy'] = filepath

    filepath = './gsh/gsh_data/c2h4.energy.1'
    if os.path.exists(filepath):
        record2['energy1'] = filepath

    filepath = './gsh/gsh_data/c2h4.energy.2'
    if os.path.exists(filepath):
        record2['energy2'] = filepath

    filepath = './gsh/gsh_data/c2h4.grad'
    if os.path.exists(filepath):
        record2['grad'] = filepath

    filepath = './gsh/gsh_data/c2h4.grad.1'
    if os.path.exists(filepath):
        record2['grad1'] = filepath

    filepath = './gsh/gsh_data/c2h4.grad.2'
    if os.path.exists(filepath):
        record2['grad2'] = filepath

    filepath = './gsh/gsh_data/c2h4.kinetic'
    if os.path.exists(filepath):
        record2['kinetic'] = filepath

    filepath = './gsh/gsh_data/c2h4.kinetic.1'
    if os.path.exists(filepath):
        record2['kinetic1'] = filepath

    filepath = './gsh/gsh_data/c2h4.kinetic.2'
    if os.path.exists(filepath):
        record2['kinetic2'] = filepath

    filepath = './gsh/gsh_data/c2h4.velo'
    if os.path.exists(filepath):
        record2['velo'] = filepath

    filepath = './gsh/gsh_data/c2h4.velo.1'
    if os.path.exists(filepath):
        record2['velo1'] = filepath

    filepath = './gsh/gsh_data/c2h4.velo.2'
    if os.path.exists(filepath):
        record2['velo2'] = filepath

    filepath = './gsh/gsh_data/c2h4.soc'
    if os.path.exists(filepath):
        record2['soc'] = filepath

    filepath = './gsh/gsh_data/c2h4.soc.1'
    if os.path.exists(filepath):
        record2['soc1'] = filepath

    filepath = './gsh/gsh_data/c2h4.soc.2'
    if os.path.exists(filepath):
        record2['soc2'] = filepath

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |               GSH Test Calculation                |
 |                                                   |
 *---------------------------------------------------*

  These data were modified only for functionality test

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

    CopyInput(record, record2, testdir)
    Setup(testdir)

    summary += """
Copy internal conversion test files
 %-10s --> %s/c3h2o.xyz
 %-10s --> %s/c3h2o.xyz.1
 %-10s --> %s/c3h2o.xyz.2
 %-10s --> %s/c3h2o.energy
 %-10s --> %s/c3h2o.energy.1
 %-10s --> %s/c3h2o.energy.2
 %-10s --> %s/c3h2o.grad
 %-10s --> %s/c3h2o.grad.1
 %-10s --> %s/c3h2o.grad.2
 %-10s --> %s/c3h2o.kinetic
 %-10s --> %s/c3h2o.kinetic.1
 %-10s --> %s/c3h2o.kinetic.2
 %-10s --> %s/c3h2o.velo
 %-10s --> %s/c3h2o.velo.1
 %-10s --> %s/c3h2o.velo.2
 %-10s --> %s/c3h2o.soc
 %-10s --> %s/c3h2o.soc.1
 %-10s --> %s/c3h2o.soc.2

Copy intersystem crossing test files
 %-10s --> %s/c2h4.xyz
 %-10s --> %s/c2h4.xyz.1
 %-10s --> %s/c2h4.xyz.2
 %-10s --> %s/c2h4.energy
 %-10s --> %s/c2h4.energy.1
 %-10s --> %s/c2h4.energy.2
 %-10s --> %s/c2h4.grad
 %-10s --> %s/c2h4.grad.1
 %-10s --> %s/c2h4.grad.2
 %-10s --> %s/c2h4.kinetic
 %-10s --> %s/c2h4.kinetic.1
 %-10s --> %s/c2h4.kinetic.2
 %-10s --> %s/c2h4.velo
 %-10s --> %s/c2h4.velo.1
 %-10s --> %s/c2h4.velo.2
 %-10s --> %s/c2h4.soc
 %-10s --> %s/c2h4.soc.1
 %-10s --> %s/c2h4.soc.2

 Run GSH Calculation:
""" % ('coord', testdir,
       'coord1', testdir,
       'coord2', testdir,
       'energy', testdir,
       'energy1', testdir,
       'energy2', testdir,
       'grad', testdir,
       'grad1', testdir,
       'grad2', testdir,
       'kinetic', testdir,
       'kinetic1', testdir,
       'kinetic2', testdir,
       'velo', testdir,
       'velo1', testdir,
       'velo2', testdir,
       'soc', testdir,
       'soc1', testdir,
       'soc2', testdir,
       'coord', testdir,
       'coord1', testdir,
       'coord2', testdir,
       'energy', testdir,
       'energy1', testdir,
       'energy2', testdir,
       'grad', testdir,
       'grad1', testdir,
       'grad2', testdir,
       'kinetic', testdir,
       'kinetic1', testdir,
       'kinetic2', testdir,
       'velo', testdir,
       'velo1', testdir,
       'velo2', testdir,
       'soc', testdir,
       'soc1', testdir,
       'soc2', testdir)

    results, code = RunGSHIC(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                     GSH OUTPUT (internal conversion)
-------------------------------------------------------
%s
-------------------------------------------------------

 Internal conversion test done, entering intersystem crossing test
""" % results
    else:
        summary += """
 Internal conversion test failed, stop here
"""
        return summary, code

    results, code = RunGSHISC(testdir)

    summary += """
-------------------------------------------------------
                     GSH OUTPUT (intersystem crossing)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results

    return summary, code


def CopyInput(record, record2, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/c3h2o.xyz' % testdir)
    shutil.copy2(record['coord1'], '%s/c3h2o.xyz.1' % testdir)
    shutil.copy2(record['coord2'], '%s/c3h2o.xyz.2' % testdir)
    shutil.copy2(record['energy'], '%s/c3h2o.energy' % testdir)
    shutil.copy2(record['energy1'], '%s/c3h2o.energy.1' % testdir)
    shutil.copy2(record['energy2'], '%s/c3h2o.energy.2' % testdir)
    shutil.copy2(record['grad'], '%s/c3h2o.grad' % testdir)
    shutil.copy2(record['grad1'], '%s/c3h2o.grad.1' % testdir)
    shutil.copy2(record['grad2'], '%s/c3h2o.grad.2' % testdir)
    shutil.copy2(record['kinetic'], '%s/c3h2o.kinetic' % testdir)
    shutil.copy2(record['kinetic1'], '%s/c3h2o.kinetic.1' % testdir)
    shutil.copy2(record['kinetic2'], '%s/c3h2o.kinetic.2' % testdir)
    shutil.copy2(record['velo'], '%s/c3h2o.velo' % testdir)
    shutil.copy2(record['velo1'], '%s/c3h2o.velo.1' % testdir)
    shutil.copy2(record['velo2'], '%s/c3h2o.velo.2' % testdir)
    shutil.copy2(record['soc'], '%s/c3h2o.soc' % testdir)
    shutil.copy2(record['soc1'], '%s/c3h2o.soc.1' % testdir)
    shutil.copy2(record['soc2'], '%s/c3h2o.soc.2' % testdir)

    shutil.copy2(record2['coord'], '%s/c2h4.xyz' % testdir)
    shutil.copy2(record2['coord1'], '%s/c2h4.xyz.1' % testdir)
    shutil.copy2(record2['coord2'], '%s/c2h4.xyz.2' % testdir)
    shutil.copy2(record2['energy'], '%s/c2h4.energy' % testdir)
    shutil.copy2(record2['energy1'], '%s/c2h4.energy.1' % testdir)
    shutil.copy2(record2['energy2'], '%s/c2h4.energy.2' % testdir)
    shutil.copy2(record2['grad'], '%s/c2h4.grad' % testdir)
    shutil.copy2(record2['grad1'], '%s/c2h4.grad.1' % testdir)
    shutil.copy2(record2['grad2'], '%s/c2h4.grad.2' % testdir)
    shutil.copy2(record2['kinetic'], '%s/c2h4.kinetic' % testdir)
    shutil.copy2(record2['kinetic1'], '%s/c2h4.kinetic.1' % testdir)
    shutil.copy2(record2['kinetic2'], '%s/c2h4.kinetic.2' % testdir)
    shutil.copy2(record2['velo'], '%s/c2h4.velo' % testdir)
    shutil.copy2(record2['velo1'], '%s/c2h4.velo.1' % testdir)
    shutil.copy2(record2['velo2'], '%s/c2h4.velo.2' % testdir)
    shutil.copy2(record2['soc'], '%s/c2h4.soc' % testdir)
    shutil.copy2(record2['soc1'], '%s/c2h4.soc.1' % testdir)
    shutil.copy2(record2['soc2'], '%s/c2h4.soc.2' % testdir)


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
sfhp gsh
datapath %s
verbose 1
""" % testdir
    with open('%s/test_ic' % testdir, 'w') as out:
        out.write(ld_input)

    ld_input = """&CONTROL
title         c2h4
qc_ncpu       2
jobtype       hop
qm            molcas

&Molecule
ci       3 2
spin     0 1
coupling 1 4

&MD
root 4
sfhp gsh
dosoc 1
datapath %s
verbose 1
""" % testdir
    with open('%s/test_isc' % testdir, 'w') as out:
        out.write(ld_input)


def Collect(title, testdir):
    with open('%s/%s.log' % (testdir, title), 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results


def RunGSHIC(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_ic', shell=True)
    os.chdir(maindir)
    results = Collect('c3h2o', testdir)
    if len(results.splitlines()) < 10:
        code = 'FAILED(gsh ic runtime error)'
    else:
        code = 'PASSED'
    return results, code


def RunGSHISC(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_isc', shell=True)
    os.chdir(maindir)
    results = Collect('c2h4', testdir)
    if len(results.splitlines()) < 10:
        code = 'FAILED(gsh isc runtime error)'
    else:
        code = 'PASSED'
    return results, code
