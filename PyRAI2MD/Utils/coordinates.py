######################################################
#
# PyRAI2MD 2 module for utility tools - coordinates formatting
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import sys
import numpy as np

from PyRAI2MD.Molecule.atom import Atom

def molcas_coord(xyz):
    ## This function convert Molcas coordinates to list

    coord = []
    for line in xyz:
        index, a, x, y, z = line.split()[0:5]
        coord.append([a.split(index)[0], float(x), float(y), float(z)])

    return coord

def orca_coord(xyz):
    ## This function convert orca coordinates to list

    coord = []
    for line in xyz:
        a, x, y, z = line.split()[0:4]
        coord.append([a, float(x), float(y), float(z)])

    return coord

def oqp_coord(xyz):
    ## This function convert oqp coordinates to list

    coord = []
    for line in xyz:
        idx, a, x, y, z = line.split()[0:5]
        coord.append([Atom(int(a)).name, float(x), float(y), float(z)])

    return coord

def oqp_coord2list(atoms, xyz):
    ## This function convert oqp coordinates to list
    xyz = xyz.reshape((len(atoms), 3)) * 0.52917721090299996
    coord = []
    for n, line in enumerate(xyz):
        a = atoms[n]
        x, y, z = line
        coord.append([Atom(int(a)).name, float(x), float(y), float(z)])

    return coord

def string2float(x):
    ## This function convert 1D string (e,x,y,z) list to 2D float array

    x = [[float(i) for i in row.split()[1: 4]] for row in x]
    return x

def reverse_string2float(x):
    ## This function convert 1D string (e,x,y,z) list to 2D float array

    x = [[float(i) for i in row.split()[-3:]] for row in x]
    return x

def complex2string(x):
    ## This function convert 2D complex array to 2D string array

    x = [[str(i) for i in row] for row in x]
    return x

def string2complex(x):
    ## This function convert 2D string array back to 2D complex array

    x = [[complex(i) for i in row] for row in x]
    return x

def verify_xyz(mol):
    ## This function determine the coordinate file type

    if isinstance(mol, str):

        if os.path.exists('%s.xyz' % mol):
            with open('%s.xyz' % mol, 'r') as ld_input:
                xyzfile = ld_input.read().splitlines()
            flag = xyzfile[2].split()[1]

            try:
                float(flag)
                xyztype = 'xyz'

            except ValueError:
                xyztype = 'tinker'

        else:
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for coordinate file %s' % mol)

    elif isinstance(mol, np.ndarray):
        xyztype = 'array'

    elif isinstance(mol, list):
        xyztype = 'array'

    elif isinstance(mol, dict):
        xyztype = 'dict'
    else:
        xyztype = 'unknown'

    return xyztype

def read_coord(mol):
    ## This function read xyz and velo from files
    with open('%s.xyz' % mol) as xyzfile:
        file = xyzfile.read().splitlines()

    natom = int(file[0])
    atoms = []
    coord = []
    for i, line in enumerate(file[2: 2 + natom]):
        e, x, y, z = line.split()[0:4]
        atoms.append(e)
        coord.append([x, y, z])

    if os.path.exists('%s.velo' % mol):
        velo = read_float_from_text('%s.velo' % mol)

    elif os.path.exists('%s.velocity.xyz' % mol):
        velo = read_float_from_text('%s.velocity.xyz' % mol)

    else:
        velo = np.zeros([natom, 3])

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo

def read_charge(mol):
    ## This function read point charges from files
    if not os.path.exists('%s.charge' % mol):
        sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for charge file %s.charge' % mol)

    with open('%s.charge' % mol) as charge_file:
        file = charge_file.read().splitlines()

    ncharge = int(file[0])
    charges = []
    for i, line in enumerate(file[2: 2 + ncharge]):
        q, x, y, z = line.split()[0:4]
        charges.append([q, x, y, z])

    charges = np.array(charges).astype(float)

    return charges

def read_initcond(mol):
    ## This function read xyz and velo from initial condition list
    natom = len(mol)
    atoms = []
    coord = []
    velo = np.zeros((natom, 3))
    for i, line in enumerate(mol):

        if len(line) >= 9:
            e, x, y, z, vx, vy, vz, m, chrg = line[0:9]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = float(vx), float(vy), float(vz)
        elif 9 > len(line) >= 7:
            e, x, y, z, vx, vy, vz = line[0:7]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = float(vx), float(vy), float(vz)
        else:
            e, x, y, z = line[0:4]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = 0.0, 0.0, 0.0

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo

def print_coord(xyz):
    ## This function convert a numpy array of coordinates to a formatted string

    coord = ''
    for line in xyz:
        e, x, y, z = line
        coord += '%-5s%24.16f%24.16f%24.16f\n' % (e, float(x), float(y), float(z))

    return coord

def print_charge(charge, charge_name='', unit='Angstrom'):
    ## This function convert a numpy array of coordinates to a formatted string
    coord = ''
    if not isinstance(charge, np.ndarray):
        return coord

    if unit == 'Angstrom':
        f = 1
    elif unit == 'Bohr':
        f = 1 / 0.529177249
    else:
        f = 1

    for line in charge:
        q, x, y, z = line
        coord += '%-5s%24.16f%24.16f%24.16f%24.16f\n' % (
            charge_name, float(q), float(x) * f, float(y) * f, float(z) * f
        )

    return coord

def print_matrix(mat):
    ## This function convert a numpy array to a formatted string
    matrix = ''
    for row in mat:
        matrix += ' '.join(['%24.16f' % x for x in row]) + '\n'

    return matrix

def mark_atom(xyz, marks):
    ## This function marks atoms for different basis set specification of Molcas

    new_xyz = []

    for n, line in enumerate(xyz):
        e, x, y, z = line
        e = marks[n].split()[0]
        new_xyz.append([e, x, y, z])

    return new_xyz

def read_tinker_key(xyz, key, dtype):
    ## This function read tinker key and txyz file

    ## read key
    if not os.path.exists(key):
        sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for qmmm key file %s' % key)

    with open(key, 'r') as keyfile:
        key = keyfile.read().splitlines()

    ## read xyz and velo
    if dtype == 'file':
        if not os.path.exists(xyz):
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for qmmm xyz file %s' % xyz)

        with open(xyz, 'r') as txyzfile:
            txyz = txyzfile.read().splitlines()

        title = xyz.split('.xyz')[0]
        if os.path.exists('%s.velo' % title):
            velo = read_float_from_text('%s.velo' % title)

        elif os.path.exists('%s.velocity.xyz' % title):
            velo = read_float_from_text('%s.velocity.xyz' % title)

        else:
            velo = np.zeros(0)

    elif dtype == 'dict':
        txyz = xyz['txyz']
        velo = xyz['velo']
        velo = np.array([x.replace('D', 'e').split() for x in velo]).astype(float)

    else:
        sys.exit('\n  TypeError\n  PyRAI2MD: xyz and velo dtype should be either file or dict, found %s' % dtype)

    ## check key
    highlevel = []
    active = []
    nola = []
    la = []
    nact = -1
    atomtype = {}
    for line in key[1:]:
        line = line.split()
        if len(line) > 1:
            atype = line[0].upper()

            if atype == 'QM':
                nact += 1
                highlevel.append(nact)
                nola.append(nact)
                active.append(int(line[1]) - 1)
                atomtype[int(line[1]) - 1] = nact

            elif atype == 'MM':
                nact += 1
                nola.append(nact)
                active.append(int(line[1]) - 1)
                atomtype[int(line[1]) - 1] = nact

            elif atype == 'LA':
                la.append(int(line[1]) - 1)

    ## check velocity
    if len(velo) == 0:
        velo = np.zeros([len(nola), 3])
    else:
        velo = velo[nola]

    ## read txyz
    info = [txyz[0]]
    atoms = []
    coord = []
    inactive = []
    boundary = []
    for line in txyz[1:]:
        if len(line) > 0:
            line = line.split()
            info.append(line)
            index = int(line[0]) - 1
            if index in active:
                atoms.append(lookup_amber(line[1]))
                coord.append([float(line[2]), float(line[3]), float(line[4])])

            elif index in la:
                o = int(line[6]) - 1
                t = int(line[7]) - 1
                boundary.append([atomtype[o], atomtype[t]])

            else:
                inactive.append(index - 1)

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord)

    mol_info = {
        'atoms': atoms,
        'coord': coord,
        'velo': velo,
        'inact': inactive,
        'active': active,
        'link': la,
        'boundary': boundary,
        'highlevel': highlevel,
        'txyz': info,
    }

    return mol_info

def read_float_from_text(txt):
    ## This function read float from text
    with open(txt, 'r') as ld_input:
        ftxt = ld_input.read().splitlines()

    flist = []
    for fx in ftxt:
        fx = fx.replace('D', 'e')
        flist.append(fx.split())

    farray = np.array(flist).astype(float)

    return farray

def lookup_amber(name):
    ## This function find the atom number for amber atom type

    amber_dict = {
        'C': 'C',
        'CT': 'C',
        'CA': 'C',
        'CM': 'C',
        'CC': 'C',
        'CV': 'C',
        'CW': 'C',
        'CR': 'C',
        'CB': 'C',
        'C*': 'C',
        'CN': 'C',
        'CK': 'C',
        'CQ': 'C',
        'C2R': 'C',
        'C3R': 'C',
        'N': 'N',
        'NA': 'N',
        'NB': 'N',
        'NC': 'N',
        'N*': 'N',
        'N2': 'N',
        'N3': 'N',
        'O': 'O',
        'OW': 'O',
        'OH': 'O',
        'OS': 'O',
        'OT': 'O',
        'O2': 'O',
        'S': 'S',
        'SH': 'S',
        'P': 'P',
        'H': 'H',
        'HW': 'H',
        'HO': 'H',
        'HR': 'H',
        'HS': 'H',
        'HT': 'H',
        'HA': 'H',
        'HC': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'H4': 'H',
        'H5': 'H',
        'HP': 'H',
        'LAH': 'H',
        'Cl-': 'Cl',
        'F': 'F'
    }

    if name in amber_dict.keys():
        atom = amber_dict[name]
    else:
        atom = name[0]
        print('Do not find atom type %s, use %s instead' % (name, atom))

    return atom

def atomic_number(x):
    z = [Atom(atom).name for atom in x]

    return z

def numerize_xyz(xyz):
    new_xyz = []
    for mol in xyz:
        new_mol = []
        for atom in mol:
            name, cx, cy, cz = atom
            z = Atom(name).name
            new_mol.append([int(z), float(cx), float(cy), float(cz)])
        new_xyz.append(new_mol)

    return new_xyz
