######################################################
#
# PyRAI2MD test QMQM2
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


def TestQMQM2():
    """ xtb test

    1. ORCA/xtb energy and gradient
    2. BAGEL/xtb energy and gradient
    3. Molcas/xtb energy and gradient

    """

    testdir = '%s/results/qmqm2' % (os.getcwd())
    record = {
        'coord': 'FileNotFound',
        'coord2': 'FileNotFound',
        'orca': 'FileNotFound',
        'bagel': 'FileNotFound',
        'molcas': 'FileNotFound',
        'XTB': 'VariableNotFound',
        'ORCA': 'VariableNotFound',
        'BAGEL': 'VariableNotFound',
        'BLAS': 'VariableNotFound',
        'LAPACK': 'VariableNotFound',
        'BOOST': 'VariableNotFound',
        'MPI': 'VariableNotFound',
        'MKL': 'VariableNotFound',
        'ARCH': 'VariableNotFound',
        'MOLCAS': 'VariableNotFound',
    }

    coordpath = './qmqm2/qmqm2_data/c2h4.xyz'
    if os.path.exists(coordpath):
        record['coord'] = coordpath

    coordpath = './qmqm2/qmqm2_data/c2h4_4w.xyz'
    if os.path.exists(coordpath):
        record['coord2'] = coordpath

    coordpath = './qmqm2/qmqm2_data/orca.inp'
    if os.path.exists(coordpath):
        record['orca'] = coordpath

    coordpath = './qmqm2/qmqm2_data/bagel_prep.json'
    if os.path.exists(coordpath):
        record['bagel_prep'] = coordpath

    coordpath = './qmqm2/qmqm2_data/bagel.json'
    if os.path.exists(coordpath):
        record['bagel'] = coordpath

    coordpath = './qmqm2/qmqm2_data/molcas_prep.inp'
    if os.path.exists(coordpath):
        record['molcas_prep'] = coordpath

    coordpath = './qmqm2/qmqm2_data/molcas.inp'
    if os.path.exists(coordpath):
        record['molcas'] = coordpath

    if 'XTB' in os.environ:
        record['XTB'] = os.environ['XTB']

    if 'ORCA' in os.environ:
        record['ORCA'] = os.environ['ORCA']

    if 'ORCA' in os.environ:
        record['ORCA'] = os.environ['ORCA']

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

    if 'MOLCAS' in os.environ:
        record['MOLCAS'] = os.environ['MOLCAS']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |              QMQM2 Test Calculation               |
 |                                                   |
 *---------------------------------------------------*

 Some environment variables are needed for this test:

    export XTB=/path
    export ORCA=/path
    export BAGEL=/path
    export BLAS=/path
    export LAPACK=/path
    export BOOST=/path
    export MKL=/path
    export ARCH='CPU_architecture'
    export MPI=/path
    export MOLCAS=/path
    
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
 %-10s --> %s/c2h4.xyz (renamed to bagel_prep.xyz)
 %-10s --> %s/c2h4.xyz (renamed to molcas_prep.xyz)
 %-10s --> %s/c2h4_4w.xyz (renamed to qmqm2_orca.xyz)
 %-10s --> %s/c2h4_4w.xyz (renamed to qmqm2_bagel.xyz)
 %-10s --> %s/c2h4_4w.xyz (renamed to qmqm2_molcas.xyz)
 %-10s --> %s/orca.inp (renamed to qmqm2_orca.orca)
 %-10s --> %s/bagel_prep.json
 %-10s --> %s/bagel.json (renamed to qmqm2_bagel.bagel)
 %-10s --> %s/molcas_prep.inp
 %-10s --> %s/molcas.inp (renamed to qmqm2_molcas.molcas)
 
 Run QMQM2:
""" % ('coord', testdir,
       'coord', testdir,
       'coord2', testdir,
       'coord2', testdir,
       'coord2', testdir,
       'orca', testdir,
       'bagel_prep', testdir,
       'bagel', testdir,
       'molcas_prep', testdir,
       'molcas', testdir
       )

    results, code = RunORCA_tddft(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                  ORCA/xTB OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
    """ % results
    else:
        summary += """
 ORCA TDDFT failed, stop here
"""
        return summary, code

    code = RunBAGEL_CASSCF(testdir)

    if code == 'PASSED':
        summary += """
 BAGEL casscf done, entering QMQM2 calculation 
        """
    else:
        summary += """
 BAGEL casscf failed, stop here
"""
        return summary, code

    results, code = RunBAGEL_CASPT2(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                  BAGEL/xTB OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
 BAGEL QM1 failed, stop here
"""
        return summary, code

    code = RunMOLCAS_CASSCF(testdir)

    if code == 'PASSED':
        summary += """
 MOLCAS casscf done, entering QMQM2 calculation 
        """
    else:
        summary += """
 MOLCAS casscf failed, stop here
"""
        return summary, code

    results, code = RunMOLCAS_CASPT2(testdir)

    if code == 'PASSED':
        summary += """
-------------------------------------------------------
                  MOLCAS/xTb OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    else:
        summary += """
 MOLCAS QM1 failed, stop here
"""

    return summary, code

def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['coord'], '%s/c2h4.xyz' % testdir)
    shutil.copy2(record['coord2'], '%s/qmqm2_orca.xyz' % testdir)
    shutil.copy2(record['coord2'], '%s/qmqm2_bagel.xyz' % testdir)
    shutil.copy2(record['coord2'], '%s/qmqm2_molcas.xyz' % testdir)
    shutil.copy2(record['orca'], '%s/qmqm2_orca.orca' % testdir)
    shutil.copy2(record['bagel'], '%s/qmqm2_bagel.bagel' % testdir)
    shutil.copy2(record['bagel_prep'], '%s/bagel_prep.json' % testdir)
    shutil.copy2(record['molcas'], '%s/qmqm2_molcas.molcas' % testdir)
    shutil.copy2(record['molcas_prep'], '%s/molcas_prep.inp' % testdir)

def Setup(record, testdir):
    orca_xtb_input = """&CONTROL
title         qmqm2_orca
qc_ncpu       1
ms_ncpu       2
jobtype       sp
qm            orca xtb

&MOLECULE
ci   3
spin 0
highlevel     1-6

&ORCA
orca          %s
orca_workdir  %s
dft_type      tddft

&XTB
xtb           %s
xtb_workdir   %s

&MD
root 1
activestate 0
""" % (
        record['ORCA'],
        testdir,
        record['XTB'],
        testdir
    )

    bagel_xtb_input = """&CONTROL
title         qmqm2_bagel
qc_ncpu       2
ms_ncpu       2
jobtype       sp
qm            bagel xtb

&Bagel
bagel         %s
blas          %s
lapack        %s
boost         %s
mpi           %s
mkl           %s
arch          %s
bagel_workdir %s

&XTB
xtb           %s
xtb_workdir   %s

&Molecule
ci       3
spin     0
coupling 1 2, 2 3
highlevel     1-6

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
       testdir,
       record['XTB'],
       testdir)

    molcas_xtb_input = """&CONTROL
title         qmqm2_molcas
qc_ncpu       1
jobtype       sp
qm            molcas xtb

&Molcas
molcas         %s
molcas_calcdir %s

&XTB
xtb           %s
xtb_workdir   %s

&Molecule
ci       3 2
spin     0 1
coupling 1 2, 2 3, 4 5, 2 4, 2 5 
highlevel     1-6

&MD
root 4
activestate 1
""" % (record['MOLCAS'],
       testdir,
        record['XTB'],
        testdir)

    bagel_runscript = """
export INPUT=bagel_prep
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

    molcas_runscript = """
export INPUT=molcas_prep
export MOLCAS=%s
export MOLCAS_PROJECT=$INPUT
export MOLCAS_WORKDIR=$PWD

$MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
rm -r $MOLCAS_PROJECT
""" % record['MOLCAS']

    with open('%s/qmqm2_orca' % testdir, 'w') as out:
        out.write(orca_xtb_input)

    with open('%s/qmqm2_bagel' % testdir, 'w') as out:
        out.write(bagel_xtb_input)

    with open('%s/qmqm2_molcas' % testdir, 'w') as out:
        out.write(molcas_xtb_input)

    with open('%s/bagel_prep.sh' % testdir, 'w') as out:
        out.write(bagel_runscript)

    with open('%s/molcas_prep.sh' % testdir, 'w') as out:
        out.write(molcas_runscript)

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

def RunORCA_tddft(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md qmqm2_orca > stdout 2>&1', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'qmqm2_orca')
    if len(results.splitlines()) < 13:
        code = 'FAILED(ORCA QM1 runtime error)'
    else:
        code = 'PASSED'
    return results, code

def RunBAGEL_CASSCF(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('bash bagel_prep.sh >> stdout 2>&1', shell=True)
    os.chdir(maindir)
    with open('%s/bagel_prep.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()
    code = 'FAILED(BAGEL prep runtime error)'
    for line in log[-10:]:
        if 'METHOD:' in line:
            code = 'PASSED'
    return code


def RunBAGEL_CASPT2(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md qmqm2_bagel >> stdout 2>&1', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'qmqm2_bagel')
    if len(results.splitlines()) < 13:
        code = 'FAILED(BAGEL QM1 runtime error)'
    else:
        code = 'PASSED'
    return results, code

def RunMOLCAS_CASSCF(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('bash molcas_prep.sh >> stdout 2>&1', shell=True)
    os.chdir(maindir)
    with open('%s/molcas_prep.log' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()
    code = 'FAILED(Molcas prep runtime error)'
    for line in log[-10:]:
        if 'Happy' in line:
            code = 'PASSED'
            shutil.copy2('%s/molcas_prep.RasOrb' % testdir, '%s/qmqm2_molcas.StrOrb' % testdir)
    return code


def RunMOLCAS_CASPT2(testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('pyrai2md qmqm2_molcas >> stdout 2>&1', shell=True)
    os.chdir(maindir)
    results = Collect(testdir, 'qmqm2_molcas')
    if len(results.splitlines()) < 13:
        code = 'FAILED(Molcas QM1 runtime error)'
    else:
        code = 'PASSED'
    return results, code
