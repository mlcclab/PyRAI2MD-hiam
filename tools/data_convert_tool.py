######################################################
#
# PyRAI2MD 2 module for training data version convertor
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

import sys
import json

def main():
    usage = """
    PyRAI2MD training data shuffle tool

    Usage:
        python3 data_shuffle_tool.py [input_file]

    """

    if len(sys.argv) <= 1:
        exit(usage)

    title = sys.argv[1].split('.')[0]
    with open(sys.argv[1], 'r') as indata:
        dataset = json.load(indata)

    natom, nstate, xyz, invrset, energy, grad, nac, civec, movecset = dataset

    size = len(xyz)
    soc = [[] for _ in range(size)]

    newset = {
        'natom': natom,
        'nstate': nstate,
        'nnac': 1,
        'nsoc': 0,
        'xyz': xyz,
        'energy': energy,
        'grad': grad,
        'nac': nac,
        'soc': soc,
    }

    with open('%s-new.json' % title, 'w') as outdata:
        json.dump(newset, outdata)


if __name__ == '__main__':
    main()
