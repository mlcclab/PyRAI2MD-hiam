######################################################
#
# PyRAI2MD 2 module for packing training data
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

import os
import sys
import json
import multiprocessing
import numpy as np
from optparse import OptionParser

from PyRAI2MD.variables import read_input
from PyRAI2MD.methods import QM


def main():
    usage = """
    PyRAI2MD training data tool

    Usage:
        python3 training_data_tool.py [options]

    """

    description = ''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-i', dest='input', type=str, nargs=1, help='input file name.', default='input')
    parser.add_option('-n', dest='ncpu', type=int, nargs=1, help='number of cpus.', default=1)

    (options, args) = parser.parse_args()
    inputs = options.input
    ncpu = options.ncpu

    if not os.path.exists(inputs):
        sys.exit('\n  FileNotFoundError\n PyRAI2MD: looking for input file %s' % inputs)

    with open(inputs) as infile:
        input_dict = infile.read().split('&')

    keywords, _ = read_input(input_dict)

    file = keywords['file']['file']

    if not os.path.exists(file):
        sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for list file %s' % file)

    with open(file, 'r') as infile:
        file_list = infile.read().splitlines()

    key = PrepKey(keywords)
    natom = key['natom']
    ncharge = key['ncharge']
    nstate = key['nstate']
    nnac = key['nnac']
    nsoc = key['nsoc']

    wrapper = [[n, f, key] for n, f in enumerate(file_list)]
    nfile = len(wrapper)
    ncpu = np.amin([nfile, ncpu])
    pool = multiprocessing.Pool(processes=ncpu)

    xyz_list = [[] for _ in range(nfile)]
    cell_list = [[] for _ in range(nfile)]
    pbc_list = [[] for _ in range(nfile)]
    charge_list = [[] for _ in range(nfile)]
    energy_list = [[] for _ in range(nfile)]
    grad_list = [[] for _ in range(nfile)]
    nac_list = [[] for _ in range(nfile)]
    soc_list = [[] for _ in range(nfile)]

    n = 0
    for val in pool.imap_unordered(ReadData, wrapper):
        n += 1
        id, xyz, cell, pbc, charge, energy, grad, nac, soc = val
        xyz_list[id] = xyz
        cell_list[id] = cell
        pbc_list[id] = pbc
        charge_list[id] = charge
        energy_list[id] = energy
        grad_list[id] = grad
        nac_list[id] = nac
        soc_list[id] = soc

        sys.stdout.write('CPU: %3d Extracting %6d/%-6d\r' % (ncpu, n, nfile))

    pool.close()

    dataset = {
        'natom': natom,
        'ncharge': ncharge,
        'nstate': nstate,
        'nnac': nnac,
        'nsoc': nsoc,
        'xyz': xyz_list,
        'cell': cell_list,
        'pbc': pbc_list,
        'charge': charge_list,
        'energy': energy_list,
        'grad': grad_list,
        'nac': nac_list,
        'soc': soc_list,
    }

    print('\n    --- Summary ---')
    print('natom:  %5d' % natom)
    print('ncharge:%5d' % ncharge)
    print('nstate: %5d' % nstate)
    print('nnac:   %5d' % nnac)
    print('nsoc:   %5d' % nsoc)
    print('    --- Data shape ---')
    print('xyz:   %30s' % (str(np.array(xyz_list).shape)))
    print('cell:  %30s' % (str(np.array(cell_list).shape)))
    print('pbc:   %30s' % (str(np.array(pbc_list).shape)))
    print('charge:%30s' % (str(np.array(charge_list).shape)))
    print('energy:%30s' % (str(np.array(energy_list).shape)))
    print('grad:  %30s' % (str(np.array(grad_list).shape)))
    print('nac:   %30s' % (str(np.array(nac_list).shape)))
    print('soc:   %30s' % (str(np.array(soc_list).shape)))

    with open('data.json', 'w') as outdata:
        json.dump(dataset, outdata)


def PrepKey(key):
    qm = key['control']['qm']
    natom = key['file']['natom']
    ncharge = key['file']['ncharge']
    ci = key['molecule']['ci']
    nstate = int(np.sum(ci))
    spin = key['molecule']['spin']
    coupling = key['molecule']['coupling']

    mult = []
    statemult = []
    for n, s in enumerate(ci):
        mt = int(spin[n] * 2 + 1)
        mult.append(mt)
        for _ in range(s):
            statemult.append(mt)

    nac_coupling = []
    soc_coupling = []
    for n, pair in enumerate(coupling):
        s1, s2 = pair
        s1 -= 1
        s2 -= 1
        if statemult[s1] != statemult[s2]:
            soc_coupling.append(sorted([s1, s2]))
        else:
            nac_coupling.append(sorted([s1, s2]))

    nnac = len(nac_coupling)
    nsoc = len(soc_coupling)

    keywords = {
        'qm': qm,
        'natom': natom,
        'ncharge': ncharge,
        'ci': ci,
        'nstate': nstate,
        'mult': mult,
        'statemult': statemult,
        'nac_coupling': nac_coupling,
        'soc_coupling': soc_coupling,
        'nnac': nnac,
        'nsoc': nsoc,
        'key': key,
    }

    return keywords


def ReadData(var):
    id, f, keywords = var

    qm = keywords['qm']
    ci = keywords['ci']
    natom = keywords['natom']
    ncharge = keywords['ncharge']
    nstate = keywords['nstate']
    mult = keywords['mult']
    nac_coupling = keywords['nac_coupling']
    soc_coupling = keywords['soc_coupling']
    nnac = keywords['nnac']
    nsoc = keywords['nsoc']
    key = keywords['key']

    data = QM(qm, keywords=key, job_id='Read').get_method()
    data.project = f.split('/')[-1]
    data.workdir = f
    data.calcdir = f
    data.ci = ci
    data.nstate = nstate
    data.mult = mult
    data.nac_coupling = nac_coupling
    data.soc_coupling = soc_coupling
    data.nnac = nnac
    data.nsoc = nsoc

    xyz, charge, cell, pbc, energy, grad, nac, soc = data.read_data(natom, ncharge)

    cell = cell.tolist()
    pbc = pbc.tolist()
    charge = charge.tolist()
    energy = energy.tolist()
    grad = grad.tolist()
    nac = nac.tolist()
    soc = soc.tolist()

    return id, xyz, cell, pbc, charge, energy, grad, nac, soc


if __name__ == '__main__':
    main()
