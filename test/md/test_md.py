######################################################
#
# PyRAI2MD test MD with constraints
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


def TestMD():
    """ constrained md test

    1. freeze water atom 1 2 3
    2. apply ellipsoid potential on water dimer
    3. apply ellipsoid potential centered on water atom 4 5 6
    4. apply cuboid potential on water dimer
    5. apply cuboid potential on water atom 4 5 6

    """

    testdir = '%s/results/md' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'XTB': 'VariableNotFound',
    }

    coordpath = './md/md_data/h2o.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    if 'XTB' in os.environ:
        record['XTB'] = os.environ['XTB']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |         Constrained MD Test Calculation           |
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
 %-10s --> %s/frozen_water.xyz
 %-10s --> %s/ellipsoid_water_dimer.xyz
 %-10s --> %s/ellipsoid_water_center.xyz
 %-10s --> %s/cuboid_water_dimer.xyz
 %-10s --> %s/cuboid_water_center.xyz

 Run GFN-xTB:
""" % ('coord', testdir,
       'coord', testdir,
       'coord', testdir,
       'coord', testdir,
       'coord', testdir,
       )

    results, code = RunMD1(testdir)
    if code == 'PASSED':
        summary += """
-------------------------------------------------------
      MD OUTPUT (frozen water)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
-------------------------------------------------------
  frozen water MD failed, stop here
-------------------------------------------------------
"""
        return summary, code

    results, code = RunMD2(testdir)
    if code == 'PASSED':
        summary += """
-------------------------------------------------------
      MD OUTPUT (ellipsoid potential on water dimer)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
-------------------------------------------------------
  ellipsoid potential on water dimer MD failed, stop here
-------------------------------------------------------
"""
        return summary, code

    results, code = RunMD3(testdir)
    if code == 'PASSED':
        summary += """
-------------------------------------------------------
      MD OUTPUT (ellipsoid potential on water center)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
-------------------------------------------------------
  ellipsoid potential on water center MD failed, stop here
-------------------------------------------------------
"""
        return summary, code

    results, code = RunMD4(testdir)
    if code == 'PASSED':
        summary += """
-------------------------------------------------------
      MD OUTPUT (cuboid potential on water dimer)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
-------------------------------------------------------
  cuboid potential on water dimer MD failed, stop here
-------------------------------------------------------
"""
        return summary, code

    results, code = RunMD5(testdir)
    if code == 'PASSED':
        summary += """
-------------------------------------------------------
      MD OUTPUT (cuboid potential on water center)
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
-------------------------------------------------------
  cuboid potential on water center MD failed, stop here
-------------------------------------------------------
"""
        return summary, code

    return summary, code

def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/frozen_water.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/ellipsoid_water_dimer.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/ellipsoid_water_center.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/cuboid_water_dimer.xyz' % testdir)
    shutil.copy2(record['coord'], '%s/cuboid_water_center.xyz' % testdir)

    velo = '0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n'

    with open('%s/frozen_water.velo' % testdir, 'w') as out:
        out.write(velo)
    with open('%s/ellipsoid_water_dimer.velo' % testdir, 'w') as out:
        out.write(velo)
    with open('%s/ellipsoid_water_center.velo' % testdir, 'w') as out:
        out.write(velo)
    with open('%s/cuboid_water_dimer.velo' % testdir, 'w') as out:
        out.write(velo)
    with open('%s/cuboid_water_center.velo' % testdir, 'w') as out:
        out.write(velo)

def Setup(record, testdir):
    input1 = """&CONTROL
title         frozen_water
qc_ncpu       1
jobtype       md
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

&molecule
ci 1
spin 0
freeze 1-3

&md
step 10
size 20.67
root 1
""" % (record['XTB'],
       testdir)

    input2 = """&CONTROL
title         ellipsoid_water_dimer
qc_ncpu       1
jobtype       md
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

&molecule
ci 1
spin 0
shape ellipsoid
cavity 2.5 2.5 2.5

&md
step 10
size 20.67
root 1
    """ % (record['XTB'],
           testdir)

    input3 = """&CONTROL
title         ellipsoid_water_center
qc_ncpu       1
jobtype       md
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

&molecule
ci 1
spin 0
shape ellipsoid
center 4-6
cavity 3.5 3.5 3.5

&md
step 10
size 20.67
root 1
    """ % (record['XTB'],
           testdir)

    input4 = """&CONTROL
title         cuboid_water_dimer
qc_ncpu       1
jobtype       md
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

&molecule
ci 1
spin 0
constrain 1-3
shape cuboid
cavity 2.5 2.5 2.5

&md
step 10
size 20.67
root 1
    """ % (record['XTB'],
           testdir)

    input5 = """&CONTROL
title         cuboid_water_center
qc_ncpu       1
jobtype       md
qm            xtb

&xtb
xtb           %s
xtb_workdir   %s

&molecule
ci 1
spin 0
constrain 1-3
shape cuboid
center 4-6
cavity 3.5 3.5 3.5

&md
step 10
size 20.67
root 1
    """ % (record['XTB'],
           testdir)

    with open('%s/test_inp1' % testdir, 'w') as out:
        out.write(input1)

    with open('%s/test_inp2' % testdir, 'w') as out:
        out.write(input2)

    with open('%s/test_inp3' % testdir, 'w') as out:
        out.write(input3)

    with open('%s/test_inp4' % testdir, 'w') as out:
        out.write(input4)

    with open('%s/test_inp5' % testdir, 'w') as out:
        out.write(input5)

def Collect(testdir, title):
    with open('%s/%s.log' % (testdir, title), 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunMD1(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp1 > stdout 2>&1', shell=True)
    os.chdir(maindir)

    results = Collect(testdir, 'frozen_water')
    if len(results.splitlines()) < 120:
        code = 'FAILED(MD frozen water runtime error)'
    else:
        code = 'PASSED'

    return results, code


def RunMD2(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp2 >> stdout 2>&1', shell=True)
    os.chdir(maindir)

    results = Collect(testdir, 'ellipsoid_water_dimer')
    if len(results.splitlines()) < 120:
        code = 'FAILED(MD ellipsoid potential on water dimer runtime error)'
    else:
        code = 'PASSED'

    return results, code


def RunMD3(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp3 >> stdout 2>&1', shell=True)
    os.chdir(maindir)

    results = Collect(testdir, 'ellipsoid_water_center')
    if len(results.splitlines()) < 120:
        code = 'FAILED(MD ellipsoid potential on water center runtime error)'
    else:
        code = 'PASSED'

    return results, code


def RunMD4(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp4 >> stdout 2>&1', shell=True)
    os.chdir(maindir)

    results = Collect(testdir, 'cuboid_water_dimer')
    if len(results.splitlines()) < 120:
        code = 'FAILED(MD cuboid potential on water dimer runtime error)'
    else:
        code = 'PASSED'

    return results, code


def RunMD5(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp5 >> stdout 2>&1', shell=True)
    os.chdir(maindir)

    results = Collect(testdir, 'cuboid_water_center')
    if len(results.splitlines()) < 120:
        code = 'FAILED(MD cuboid potential on water center runtime error)'
    else:
        code = 'PASSED'

    return results, code
