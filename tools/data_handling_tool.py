######################################################
#
# PyRAI2MD 2 module for handling training data
#
# Author Jingbai Li
# Jun 14 2023
#
######################################################
import os.path

import sys
import json
import numpy as np
from optparse import OptionParser


def main():
    usage = """
    PyRAI2MD training data handling tool

    Usage:
        python data_handling_tool.py [input_data] [options]

    ----------------------------------------------------------------------------------------------
    Example 1. Shuffling data
    ----------------------------------------------------------------------------------------------
        This is the default operation. Use -s to set a seed for random number generator., e.g.,
    
        python data_handling_tool.py data.json
        python data_handling_tool.py data.json -s 1234
    ----------------------------------------------------------------------------------------------
    Example 2. Shuffling data for two groups separately
    ----------------------------------------------------------------------------------------------
        Use -g to set a number for the data size of the first group for shuffling. e.g.,
    
        python data_handling_tool.py data.json -g 200
    
        Both group will be shuffled.
        The first 200 data will be shuffled independently from the rest of data. 
    ----------------------------------------------------------------------------------------------
    Example 3. Splitting data into two set
    ----------------------------------------------------------------------------------------------
        use -p to set a number for the data size of the first group for splitting, e.g.,
        
        python data_handling_tool.py data.json -p 200
        
        The first set has 200 data and the second set has the rest of the data.
        The input data will be shuffled before splitting in default. Use -r to skip shuffling, e.g.,
        
        python data_handling_tool.py data.json -p 200 -r 0
    ----------------------------------------------------------------------------------------------
    Example 4. Merging a data into the input data
    ----------------------------------------------------------------------------------------------
        use -m to specify the filename of the data to merge, e.g.,
        
        python data_handling_tool.py data.json -m data2.json

        Data splitting does not affect the data merging, only the input data will be used in merging.
        The input data will be shuffled before merging in default. Use -r to skip shuffling, e.g.,

        python data_handling_tool.py data.json -m data2.json -r 0
    ----------------------------------------------------------------------------------------------
    """

    description = ''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-s', dest='seed', type=int, nargs=1, help='random number seed', default=1234)
    parser.add_option('-r', dest='shuffle', type=int, nargs=1, help='shuffle input data', default=1)
    parser.add_option('-g', dest='group', type=int, nargs=1, help='set a number to shuffle data separately', default=0)
    parser.add_option('-p', dest='split', type=int, nargs=1, help='set a number to split data', default=0)
    parser.add_option('-m', dest='merge', type=str, nargs=1, help='merge a given data to input data', default='')

    (options, args) = parser.parse_args()
    seed = options.seed
    shuffle = options.shuffle
    group = options.group
    split = options.split
    merge = options.merge

    if len(sys.argv) < 2:
        exit(usage)
    title = sys.argv[1].split('.')[0]

    print('\n    Load the input data\n')
    data = check_data(sys.argv[1])

    if shuffle == 1:
        np.random.seed(seed)
        data = shuffle_data(data, group)
        print('    Save shuffled data: %s-shuffled.json' % title)
        with open('%s-shuffled.json' % title, 'w') as outdata:
            json.dump(data, outdata)

    if split > 0:
        if shuffle == 1:
            print('\n    Split dataset after shuffling data\n')
        else:
            print('\n    Split dataset without shuffling data\n')

        set_1, set_2 = split_data(data, split)

        print('    Save set 1 data: %s-set-1.json' % title)
        print('    Save set 2 data: %s-set-2.json' % title)

        with open('%s-set-1.json' % title, 'w') as outdata:
            json.dump(set_1, outdata)

        with open('%s-set-2.json' % title, 'w') as outdata:
            json.dump(set_2, outdata)

    if os.path.exists(merge):
        print('\n    Load data to merge\n')
        data = merge_data(data, merge)
        print('    Save merged data: %s-merged.json' % title)
        with open('%s-merged.json' % title, 'w') as outdata:
            json.dump(data, outdata)

    print('\n    Complete\n')
def check_data(file):
    with open(file, 'r') as indata:
        data = json.load(indata)

    natom = data['natom']
    nstate = data['nstate']
    nnac = data['nnac']
    nsoc = data['nsoc']
    xyz = data['xyz']
    energy = data['energy']
    grad = data['grad']
    nac = data['nac']
    soc = data['soc']
    size = len(xyz)

    log_info = """
    Data info
  ---------------------------
    data size: %s
    natom:     %s
    nstate:    %s
    nnac:      %s
    nsoc:      %s
    xyz:       %s
    energy:    %s
    grad:      %s
    nac:       %s
    soc:       %s
  ---------------------------
      """ % (size,
             natom,
             nstate,
             nnac,
             nsoc,
             np.array(xyz).shape,
             np.array(energy).shape,
             np.array(grad).shape,
             np.array(nac).shape,
             np.array(soc).shape,
             )

    print(log_info)

    return data

def shuffle_data(data, group):
    natom = data['natom']
    nstate = data['nstate']
    nnac = data['nnac']
    nsoc = data['nsoc']
    xyz = data['xyz']
    energy = data['energy']
    grad = data['grad']
    nac = data['nac']
    soc = data['soc']
    size = len(xyz)
    index = np.arange(size)

    if size > group > 0:
        print('    Shuffle data in each group separately')
        print('    Group 1 size:  %s' % group)
        print('    Group 2 size:  %s' % (size - group))
        group_idx1 = index[:group]
        group_idx2 = index[group:]
        np.random.shuffle(group_idx1)
        np.random.shuffle(group_idx2)
        index = np.concatenate((group_idx1, group_idx2))
    else:
        print('    Shuffle all data')
        np.random.shuffle(index)

    xyz = np.array(xyz)[index].tolist()
    energy = np.array(energy)[index].tolist()
    grad = np.array(grad)[index].tolist()
    nac = np.array(nac)[index].tolist()
    soc = np.array(soc)[index].tolist()

    newset = {
        'natom': natom,
        'nstate': nstate,
        'nnac': nnac,
        'nsoc': nsoc,
        'xyz': xyz,
        'energy': energy,
        'grad': grad,
        'nac': nac,
        'soc': soc,
    }

    return newset

def split_data(data, split):
    natom = data['natom']
    nstate = data['nstate']
    nnac = data['nnac']
    nsoc = data['nsoc']
    xyz = data['xyz']
    size = len(xyz)

    if size > split:
        set_2 = size - split
        print('    Split data into two sets')
        print('    Set 1 size: %s' % split)
        print('    Set 2 size: %s' % set_2)
    else:
        exit('\n    Error: Split number is larger than the total number of data\n')

    xyz_1 = data['xyz'][: split]
    energy_1 = data['energy'][: split]
    grad_1 = data['grad'][: split]
    nac_1 = data['nac'][: split]
    soc_1 = data['soc'][: split]

    xyz_2 = data['xyz'][split:]
    energy_2 = data['energy'][split:]
    grad_2 = data['grad'][split:]
    nac_2 = data['nac'][split:]
    soc_2 = data['soc'][split:]

    newset_1 = {
        'natom': natom,
        'nstate': nstate,
        'nnac': nnac,
        'nsoc': nsoc,
        'xyz': xyz_1,
        'energy': energy_1,
        'grad': grad_1,
        'nac': nac_1,
        'soc': soc_1,
    }

    newset_2 = {
        'natom': natom,
        'nstate': nstate,
        'nnac': nnac,
        'nsoc': nsoc,
        'xyz': xyz_2,
        'energy': energy_2,
        'grad': grad_2,
        'nac': nac_2,
        'soc': soc_2,
    }

    return newset_1, newset_2

def merge_data(data, merge):
    natom = data['natom']
    nstate = data['nstate']
    nnac = data['nnac']
    nsoc = data['nsoc']
    xyz = data['xyz']
    energy = data['energy']
    grad = data['grad']
    nac = data['nac']
    soc = data['soc']

    merge = check_data(merge)
    natom2 = merge['natom']
    nstate2 = merge['nstate']
    nnac2 = merge['nnac']
    nsoc2 = merge['nsoc']
    xyz2 = merge['xyz']
    energy2 = merge['energy']
    grad2 = merge['grad']
    nac2 = merge['nac']
    soc2 = merge['soc']

    check_natom = match_int(natom, natom2)
    check_nstate = match_int(nstate, nstate2)
    check_nnac = match_int(nnac, nnac2)
    check_nsoc = match_int(nsoc, nsoc2)
    check_xyz = match_array(xyz, xyz2)
    check_energy = match_array(energy, energy2)
    check_grad = match_array(grad, grad2)
    check_nac = match_array(nac, nac2)
    check_soc = match_array(soc, soc2)

    merge_info = """
    Check data shape
  ---------------------------
    natom:     %s
    nstate:    %s
    nnac:      %s
    nsoc:      %s
    xyz:       %s
    energy:    %s
    grad:      %s
    nac:       %s
    soc:       %s
  ---------------------------
    """ % (
        check_natom,
        check_nstate,
        check_nnac,
        check_nsoc,
        check_xyz,
        check_energy,
        check_grad,
        check_nac,
        check_soc
    )

    print(merge_info)
    m = [check_natom, check_nstate, check_nnac, check_nsoc, check_xyz, check_energy, check_grad, check_nac, check_soc]

    if 'not match' in m:
        exit('\n    Error: data sets do not match\n')

    newset = {
        'natom': natom,
        'nstate': nstate,
        'nnac': nnac,
        'nsoc': nsoc,
        'xyz': xyz + xyz2,
        'energy': energy + energy2,
        'grad': grad + grad2,
        'nac': nac + nac2,
        'soc': soc + soc2,
    }

    return newset

def match_int(a, b):
    if a == b:
        result = 'match'
    else:
        result = 'not match'

    return result

def match_array(a, b):
    shape_1 = np.array(a).shape
    shape_2 = np.array(b).shape
    result = 'not match'

    if len(shape_1) != len(shape_2) != 2:
        return result

    if len(shape_1) == 2:
        if shape_1[1] == shape_2[1]:
            result = 'match'

    elif len(shape_1) == 3:
        if shape_1[1] == shape_2[1] and shape_1[2] == shape_2[2]:
            result = 'match'

    elif len(shape_1) == 4:
        if shape_1[1] == shape_2[1] and shape_1[2] == shape_2[2] and shape_1[3] == shape_2[3]:
            result = 'match'

    return result


if __name__ == '__main__':
    main()
