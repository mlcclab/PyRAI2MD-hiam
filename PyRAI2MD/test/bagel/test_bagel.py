######################################################
#
# PyRAI2MD test bagel
#
# Author Jingbai Li
# Sep 30 2021
#
######################################################

import os
import shutil
import json
import subprocess

try:
    import PyRAI2MD

    pyrai2mddir = os.path.dirname(PyRAI2MD.__file__)

except ModuleNotFoundError:
    pyrai2mddir = ''


def TestBagel():
    """ bagel test

    1. CASSCF energy and orbital
    2. XMS-CASPT2 energy, gradient and NAC

    """

    testdir = '%s/results/bagel' % (os.getcwd())
    record = {
        'casscf': 'FileNotFound',
        'caspt2': 'FileNotFound',
        'coord': 'FileNotFound',
        'BAGEL': 'VariableNotFound',
        'BLAS': 'VariableNotFound',
        'LAPACK': 'VariableNotFound',
        'BOOST': 'VariableNotFound',
        'MPI': 'VariableNotFound',
        'MKL': 'VariableNotFound',
        'ARCH': 'VariableNotFound',
    }

    casscfpath = './bagel/bagel_data/c2h4-casscf.json'
    if os.path.exists(casscfpath):
        record['casscf'] = casscfpath

    caspt2path = './bagel/bagel_data/c2h4-xms-caspt2.json'
    if os.path.exists(caspt2path):
        record['caspt2'] = caspt2path

    coordpath = './bagel/bagel_data/c2h4.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    if 'BAGEL' in os.environ:
        record['BAGEL'] = os.environ['BAGEL']

    if 'BLAS' in os.environ:
        record['BLAS'] = os.environ['BLAS']

    if 'LAPACK' in os.environ:
        record['LAPACK'] = os.environ['LAPACK']

    if 'BOOST' in os.environ:
        record['BOOST'] = os.environ['BOOST']

    if 'MKL' in os.environ:
        record['MKL'] = os.environ['MKL']

    if 'ARCH' in os.environ:
        record['ARCH'] = os.environ['ARCH']

    if 'MPI' in os.environ:
        record['MPI'] = os.environ['MPI']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |             BAGEL Test Calculation                |
 |                                                   |
 *---------------------------------------------------*

 Some environment variables are needed for this test:

    export BAGEL=/path
    export BLAS=/path
    export LAPACK=/path
    export BOOST=/path
    export MKL=/path
    export ARCH='CPU_architecture'
    export MPI=/path

 LD_LIBRARY_PATH will be automatically set up 

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
 %-10s --> %s/c2h4-casscf.json
 %-10s --> %s/c2h4-xms-caspt2.json (renamed to c2h4.bagel)
 %-10s --> %s/c2h4.xyz

 Run BAGEL CASSCF:
""" % ('casscf', testdir, 'caspt2', testdir, 'coord', testdir)

    code = RunCASSCF(testdir)

    if code == 'PASSED':
        summary += """
 CASSCF done, entering XMS-CASPT2

"""
    else:
        summary += """
 CASSCF failed, stop here
"""
        return summary, code

    results, code = RunCASPT2(testdir)

    summary += """
-------------------------------------------------------
                     BAGEL OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    return summary, code


def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    with open(record['caspt2'], 'r') as template:
        ld_input = json.load(template)
        si_input = ld_input.copy()
        si_input['bagel'][0]['basis'] = '%s/share/cc-pvdz.json' % os.environ['BAGEL']
        si_input['bagel'][0]['df_basis'] = '%s/share/cc-pvdz-jkfit.json' % os.environ['BAGEL']

    with open('%s/c2h4.bagel' % testdir, 'w') as out:
        json.dump(si_input, out)

    shutil.copy2(record['casscf'], '%s/c2h4-casscf.json' % testdir)
    shutil.copy2(record['coord'], '%s/c2h4.xyz' % testdir)


def Setup(record, testdir):
    ld_input = """&CONTROL
title         c2h4
qc_ncpu       2
jobtype       sp
qm            bagel

&Bagel
bagel         %s
blas          %s
lapack        %s
boost         %s
mpi           %s
mkl           %s
arch          %s
bagel_workdir %s

&Molecule
ci       3
spin     0
coupling 1 2, 2 3

&MD
root 1
activestate 1
""" % (record['BAGEL'],
       record['BLAS'],
       record['LAPACK'],
       record['BOOST'],
       record['MPI'],
       record['MKL'],
       record['ARCH'],
       testdir)

    runscript = """
export INPUT=c2h4-casscf
export BAGEL=%s
export MPI=%s
export BLAS=%s
export LAPACK=%s
export BOOST=%s
export MKL=%s
export ARCH=%s
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BAGEL_NUM_THREADS=1
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BLAS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH
source $MKL $ARCH

$BAGEL/bin/BAGEL $INPUT.json > $INPUT.log

""" % (record['BAGEL'],
       record['MPI'],
       record['BLAS'],
       record['LAPACK'],
       record['BOOST'],
       record['MKL'],
       record['ARCH'])

    with open('%s/test_inp' % testdir, 'w') as out:
        out.write(ld_input)

    with open('%s/bagelcasscf.sh' % testdir, 'w') as out:
        out.write(runscript)


def Collect(testdir):
    with open('%s/c2h4.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results


def RunCASSCF(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('bash bagelcasscf.sh > stdout 2>&1', shell=True)
    os.chdir(maindir)
    with open('%s/c2h4-casscf.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()
    code = 'FAILED(casscf runtime error)'
    for line in log[-10:]:
        if 'METHOD:' in line:
            code = 'PASSED'
    return code


def RunCASPT2(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md test_inp >> stdout 2>&1', shell=True)
    os.chdir(maindir)
    results = Collect(testdir)
    if len(results.splitlines()) < 14:
        code = 'FAILED(xms-caspt2 runtime error)'
    else:
        code = 'PASSED'
    return results, code
