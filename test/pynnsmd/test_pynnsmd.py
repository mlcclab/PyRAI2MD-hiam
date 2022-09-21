######################################################
#
# PyRAI2MD test pyNNsMD neural network
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


def TestPyNNsMD():
    """ neural network test

    1. MLP energy grad soc training and prediction
    2. SchNet energy grad soc training and prediction

    """

    testdir = '%s/results/pynnsmd' % (os.getcwd())
    record = {
        'egs': 'FileNotFound',
        'egs_train': 'FileNotFound',
        'egs_predict': 'FileNotFound',
        'egs_train_sch': 'FileNotFound',
        'egs_predict_sch': 'FileNotFound',
    }

    filepath = './pynnsmd/train_data/egs.json'
    if os.path.exists(filepath):
        record['egs'] = filepath

    filepath = './pynnsmd/train_data/egs_train'
    if os.path.exists(filepath):
        record['egs_train'] = filepath

    filepath = './pynnsmd/train_data/egs_predict'
    if os.path.exists(filepath):
        record['egs_predict'] = filepath

    filepath = './pynnsmd/train_data/egs_train_sch'
    if os.path.exists(filepath):
        record['egs_train_sch'] = filepath

    filepath = './pynnsmd/train_data/egs_predict_sch'
    if os.path.exists(filepath):
        record['egs_predict_sch'] = filepath

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |           Neural Network Test Calculation         |
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

    summary += """
 Copy files:
 %-10s --> %s/egs.json
 %-10s --> %s/egs_train
 %-10s --> %s/egs_predict
 %-10s --> %s/egn_train_sch
 %-10s --> %s/egn_predict_sch


 Run MOLCAS CASSCF:
""" % ('egs', testdir,
       'egn_train', testdir,
       'egn_predict', testdir,
       'egs_train_sch', testdir,
       'egs_predict_sch', testdir)

    results, code = RunNN(testdir)
    summary += """
-------------------------------------------------------
                  PyNNsMD OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results

    return summary, code

def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['egs'], '%s/egs.json' % testdir)
    shutil.copy2(record['egs_train'], '%s/egs_train' % testdir)
    shutil.copy2(record['egs_predict'], '%s/egs_predict' % testdir)
    shutil.copy2(record['egs_train_sch'], '%s/egs_train_sch' % testdir)
    shutil.copy2(record['egs_predict_sch'], '%s/egs_predict_sch' % testdir)

def Collect(testdir, title):
    with open('%s/NN-%s.log' % (testdir, title), 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """ Number of atoms:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def Check(testdir):
    with open('%s/max_abs_dev.txt' % testdir, 'r') as logfile:
        log = logfile.read().splitlines()

    results = """%s
                    ...
%s
""" % ('\n'.join(log[:10]), '\n'.join(log[-10:]))

    return results

def RunNN(testdir):
    maindir = os.getcwd()
    results = ''

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_train > stdout_egs 2>&1', shell=True)
    os.chdir(maindir)
    tmp = Collect(testdir, 'egs')
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(MLP training runtime error)'
        return results, code
    else:
        results += ' MLP training done, entering MLP prediction...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_predict >> stdout_egs 2>&1', shell=True)
    os.chdir(maindir)
    tmp = Check(testdir)
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(MLP prediction runtime error)'
        return results, code
    else:
        results += ' MLP prediction done, entering SchNet training...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_train_sch > stdout_egs_sch 2>&1', shell=True)
    os.chdir(maindir)
    tmp = Collect(testdir, 'egs_sch')
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(SchNet training runtime error)'
        return results, code
    else:
        results += ' SchNet training done, entering SchNet prediction...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_predict_sch >> stdout_egs_sch 2>&1', shell=True)
    os.chdir(maindir)
    tmp = Check(testdir)
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(SchNet prediction runtime error)'
        return results, code
    else:
        code = 'PASSED'
        results += ' SchNet prediction done\n'

    return results, code
