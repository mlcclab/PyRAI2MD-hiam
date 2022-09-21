######################################################
#
# PyRAI2MD test neural network
#
# Author Jingbai Li
# Oct 11 2021
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


def TestNN():
    """ neural network test

    1. energy grad nac training and prediction
    2. energy grad soc training and prediction

    """

    testdir = '%s/results/neural_network' % (os.getcwd())
    record = {
        'egn': 'FileNotFound',
        'egs': 'FileNotFound',
        'permute': 'FileNotFound',
        'invd': 'FileNotFound',
        'egn_train': 'FileNotFound',
        'egn_predict': 'FileNotFound',
        'egs_train': 'FileNotFound',
        'egs_predict': 'FileNotFound',
    }

    filepath = './neural_network/train_data/egn.json'
    if os.path.exists(filepath):
        record['egn'] = filepath

    filepath = './neural_network/train_data/egs.json'
    if os.path.exists(filepath):
        record['egs'] = filepath

    filepath = './neural_network/train_data/allpath'
    if os.path.exists(filepath):
        record['permute'] = filepath

    filepath = './neural_network/train_data/invd'
    if os.path.exists(filepath):
        record['invd'] = filepath

    filepath = './neural_network/train_data/egn_train'
    if os.path.exists(filepath):
        record['egn_train'] = filepath

    filepath = './neural_network/train_data/egn_predict'
    if os.path.exists(filepath):
        record['egn_predict'] = filepath

    filepath = './neural_network/train_data/egs_train'
    if os.path.exists(filepath):
        record['egs_train'] = filepath

    filepath = './neural_network/train_data/egs_predict'
    if os.path.exists(filepath):
        record['egs_predict'] = filepath

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
 %-10s --> %s/egn.json
 %-10s --> %s/egs.json
 %-10s --> %s/allpath
 %-10s --> %s/invd
 %-10s --> %s/egn_train
 %-10s --> %s/egn_predict
 %-10s --> %s/egs_train
 %-10s --> %s/egs_predict

 Run MOLCAS CASSCF:
""" % ('egn', testdir,
       'egs', testdir,
       'permute', testdir,
       'invd', testdir,
       'egn_train', testdir,
       'egn_predict', testdir,
       'egs_train', testdir,
       'egs_predict', testdir)

    results, code = RunNN(testdir)

    summary += """
-------------------------------------------------------
                Neural Networks OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    return summary, code


def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['egn'], '%s/egn.json' % testdir)
    shutil.copy2(record['egs'], '%s/egs.json' % testdir)
    shutil.copy2(record['permute'], '%s/allpath' % testdir)
    shutil.copy2(record['invd'], '%s/invd' % testdir)
    shutil.copy2(record['egn_train'], '%s/egn_train' % testdir)
    shutil.copy2(record['egn_predict'], '%s/egn_predict' % testdir)
    shutil.copy2(record['egs_train'], '%s/egs_train' % testdir)
    shutil.copy2(record['egs_predict'], '%s/egs_predict' % testdir)


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
    subprocess.run('pyrai2md egn_train > stdout_egn', shell=True)
    os.chdir(maindir)
    tmp = Collect(testdir, 'egn')
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(egn training runtime error)'
        return results, code
    else:
        results += ' egn training done, entering egn prediction...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egn_predict >> stdout_egn', shell=True)
    os.chdir(maindir)
    tmp = Check(testdir)
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(egn prediction runtime error)'
        return results, code
    else:
        results += ' egn prediction done, entering egs training...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_train > stdout_egs', shell=True)
    os.chdir(maindir)
    tmp = Collect(testdir, 'egs')
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(egn training runtime error)'
        return results, code
    else:
        results += ' egs training done, entering egs prediction...\n'

    os.chdir(testdir)
    subprocess.run('pyrai2md egs_predict >> stdout_egs', shell=True)
    os.chdir(maindir)
    tmp = Check(testdir)
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(egs prediction runtime error)'
        return results, code
    else:
        code = 'PASSED'
        results += ' egs prediction done\n'

    return results, code
