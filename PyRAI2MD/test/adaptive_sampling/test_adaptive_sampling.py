######################################################
#
# PyRAI2MD test adaptive sampling
#
# Author Jingbai Li
# Oct 19 2021
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


def TestAdaptiveSampling():
    """ adaptive sampling test

    1. adaptive sampling with energy grad soc training and prediction

    """

    testdir = '%s/results/adaptive_sampling' % (os.getcwd())
    record = {
        'energy': 'FileNotFound',
        'orbital': 'FileNotFound',
        'freq': 'FileNotFound',
        'data': 'FileNotFound',
        'input': 'FileNotFound',
        'model': 'FileNotFound',
        'MOLCAS': 'VariableNotFound',
    }

    filepath = './adaptive_sampling/data/atod.inp'
    if os.path.exists(filepath):
        record['energy'] = filepath

    filepath = './adaptive_sampling/data/atod.StrOrb'
    if os.path.exists(filepath):
        record['orbital'] = filepath

    filepath = './adaptive_sampling/data/atod.freq.molden'
    if os.path.exists(filepath):
        record['freq'] = filepath

    filepath = './adaptive_sampling/data/atod.json'
    if os.path.exists(filepath):
        record['data'] = filepath

    filepath = './adaptive_sampling/data/input'
    if os.path.exists(filepath):
        record['input'] = filepath

    filepath = './adaptive_sampling/data/NN-atod'
    if os.path.exists(filepath):
        record['model'] = filepath

    if 'MOLCAS' in os.environ:
        record['MOLCAS'] = os.environ['MOLCAS']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |         Adaptive Sampling Test Calculation        |
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
 %-10s --> %s/atod.inp (renamed to atod.molcas)
 %-10s --> %s/atod.StrOrb
 %-10s --> %s/atod.freq.molden
 %-10s --> %s/atod.json
 %-10s --> %s/NN-atod
 %-10s --> %s/input

 Run Adaptive sampling:
""" % ('energy', testdir,
       'orbital', testdir,
       'freq', testdir,
       'data', testdir,
       'model', testdir,
       'input', testdir)

    results, code = RunSampling(testdir)

    summary += """
-------------------------------------------------------
                Adaptive Sampling OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % results
    return summary, code


def CopyInput(record, testdir):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    shutil.copy2(record['energy'], '%s/atod.molcas' % testdir)
    shutil.copy2(record['orbital'], '%s/atod.StrOrb' % testdir)
    shutil.copy2(record['freq'], '%s/atod.freq.molden' % testdir)
    shutil.copy2(record['data'], '%s/atod.json' % testdir)

    if os.path.exists('%s/NN-atod' % testdir):
        shutil.rmtree('%s/NN-atod' % testdir)
    shutil.copytree(record['model'], '%s/NN-atod' % testdir)

    with open(record['input'], 'r') as infile:
        file = infile.read().splitlines()
    ld_input = ''
    for line in file:
        if len(line.split()) > 0:
            if 'molcas' in line.split()[0]:
                ld_input += 'molcas  %s\n' % (record['MOLCAS'])
            else:
                ld_input += '%s\n' % line
        else:
            ld_input += '%s\n' % line

    with open('%s/input' % testdir, 'w') as out:
        out.write(ld_input)


def Collect(testdir, title):
    with open('%s/%s.log' % (testdir, title), 'r') as logfile:
        log = logfile.read().splitlines()

    results = []
    for n, line in enumerate(log):
        if """ Number of iterations:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results


def RunSampling(testdir):
    maindir = os.getcwd()
    results = ''

    os.chdir(testdir)
    subprocess.run('pyrai2md input > stdout', shell=True)
    os.chdir(maindir)
    tmp = Collect(testdir, 'atod')
    results += tmp

    if len(tmp.splitlines()) < 10:
        code = 'FAILED(adaptive sampling runtime error)'
        return results, code
    else:
        code = 'PASSED'
        results += ' adaptive sampling done\n'

    return results, code
