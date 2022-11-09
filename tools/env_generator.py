## ----------------------
## Chemical environment generator - a script to automatically create solvent or aggregate environment
## ----------------------
##
## New version Nov 3 2022 Jingbai Li

import os
import sys
import subprocess
import multiprocessing
import numpy as np

## import initial condition sampling module
try:
    from PyRAI2MD.Utils.sampling import sampling
    has_sampling = True
except ModuleNotFoundError:
    exit('PyRAI2MD is not installed, sampling is disabled')
    has_sampling = False

def main(argv):
    ##  This is the main function
    ##  It read all options from a creation file and

    usage = """

    Chemical Environment Generator

    Usage:
      python3 env_generator.py creation or
      python3 env_generator for help

    a creation file contains the following parameters

      title         name of calculation
      cpus          1 # number of CPUs for merging and reading initial conditions
      mode          create  # run mode, create, merge, or read
      env_type      solvent  # type of environment, solvent or aggregate 
      solute        solute.xyz  # solute molecule xyz file
      solvent       solvent.xyz  # solvent molecule xyz file
      nshell        2  # number of solvent shell
      density       1  # solvent density in g/cm**-3
      mass          18  # molar mass of solvent in g/mol
      packmol       /path/to/packmol  # the path to packmol
      cell          cell.xyz  # unit cell molecule xyz file
      nmol          2  # number of molecules in the unit cell
      center        1  # the index of the molecule in the unit cell that will be used as the center in the aggregate
      a             1  # a distance
      b             1  # b distance
      c             1  # c distance
      alpha         90  # alpha angle 
      beta          90  # beta angle
      gamma         90  # gamma angle
      na            3  # number of translation in a direction, the order is 0, +1, -1, +2, -2, +3, -3,...
      nb            3  # number of translation in b direction
      nc            3  # number of translation in c direction
      radius        14  # cutoff radius to build aggregate
      method        wigner  # initial condition sampling method
      ninitcond     1  # number of sampled initial condition
      seed          1  # random seed for sampling
      format        xyz  # frequency file format
      temp          298.15  # sampling temperature
      read          list.txt  # a list of path to read the finally equilibrated conditions
      skip          10  # number of MD step to be skipped before reading conditions
      freq          1  # frequency of reading conditions in each trajectory from the last snapshot  
      combine       yes  # combine the initial velocity with the corresponding atoms in the final condition

    Running this script will print more information about the requisite keywords

    """

    ## defaults parameters
    title = None
    cpus = 1
    mode = 'create'
    env_type = None

    solute = None
    solvent = None
    nshell = None
    density = None
    mass = None
    packmol = None

    cell = None
    nmol = None
    center = None
    a = None
    b = None
    c = None
    alpha = None
    beta = None
    gamma = None
    na = None
    nb = None
    nc = None
    radius = None

    dist = 'wigner'
    ninitcond = 0
    iseed = 1
    iformat = 'molden'
    temp = 273.15
    read = None
    skip = 0
    freq = 1
    combine = 'yes'

    if len(argv) <= 1:
        exit(usage)
    else:
        print('\n\n Chemical Environment Generator \n\n')

    with open(argv[1]) as inp:
        inputfile = inp.read().splitlines()

    for line in inputfile:
        if len(line.split()) < 2:
            continue
        key = line.split()[0].lower()

        if 'title' == key:
            title = line.split()[1].lower()
        elif 'cpus' == key:
            cpus = int(line.split()[1])
        elif 'mode' == key:
            mode = line.split()[1].lower()
        elif 'env_type' == key:
            env_type = line.split()[1].lower()
        elif 'solute' == key:
            solute = line.split()[1]
        elif 'solvent' == key:
            solvent = line.split()[1]
        elif 'nshell' == key:
            nshell = int(line.split()[1])
        elif 'density' == key:
            density = float(line.split()[1])
        elif 'mass' == key:
            mass = float(line.split()[1])
        elif 'packmol' == key:
            packmol = line.split()[1]
        elif 'cell' == key:
            cell = line.split()[1]
        elif 'nmol' == key:
            nmol = int(line.split()[1])
        elif 'center' == key:
            center = int(line.split()[1])
        elif 'a' == key:
            a = float(line.split()[1])
        elif 'b' == key:
            b = float(line.split()[1])
        elif 'c' == key:
            c = float(line.split()[1])
        elif 'alpha' == key:
            alpha = float(line.split()[1])
        elif 'beta' == key:
            beta = float(line.split()[1])
        elif 'gamma' == key:
            gamma = float(line.split()[1])
        elif 'na' == key:
            na = int(line.split()[1])
        elif 'nb' == key:
            nb = int(line.split()[1])
        elif 'nc' == key:
            nc = int(line.split()[1])
        elif 'radius' == key:
            radius = float(line.split()[1])
        elif 'method' == key:
            dist = line.split()[1].lower()
        elif 'ninitcond' == key:
            ninitcond = int(line.split()[1])
        elif 'seed' == key:
            iseed = int(line.split()[1])
        elif 'format' == key:
            iformat = line.split()[1].lower()
        elif 'temp' == key:
            temp = float(line.split()[1])
        elif 'read' == key:
            read = line.split()[1]
        elif 'skip' == key:
            skip = int(line.split()[1])
        elif 'freq' == key:
            freq = int(line.split()[1])
        elif 'combine' == key:
            combine = line.split()[1].lower()

    key_dict = {
        'title': title,
        'cpus': cpus,
        'mode': mode,
        'env_type': env_type,
        'solute': solute,
        'solvent': solvent,
        'nshell': nshell,
        'density': density,
        'mass': mass,
        'packmol': packmol,
        'cell': cell,
        'nmol': nmol,
        'center': center,
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'na': na,
        'nb': nb,
        'nc': nc,
        'radius': radius,
        'method': dist,
        'ninitcond': ninitcond,
        'iseed': iseed,
        'iformat': iformat,
        'temp': temp,
        'read': read,
        'skip': skip,
        'freq': freq,
        'combine': combine,
    }

    if mode == 'create':
        create_env(key_dict)
    elif mode == 'merge':
        merge_env(key_dict)
    elif mode == 'read':
        read_final_cond(key_dict)
    else:
        exit('\n KeywordError: unrecognized mode %s\n' % mode)


def create_env(key_dict):
    env_type = key_dict['env_type']
    if env_type == 'solvent':
        create_solvent(key_dict)
    elif env_type == 'aggregate':
        create_aggregate(key_dict)
    else:
        exit('\n KeywordError: unrecognized env_type %s\n' % env_type)

    print('''
 HINTS: 
    you might want to merge the environment with the initial conditions
    to do so, change mode to merge
    ''')

    return None

def create_solvent(key_dict):
    print('''
 Tips for creating solvent shell using packmol
    the following keyword must be set for creating solvent environment

    solute    solute.xyz  # solute molecule xyz file
    solvent   solvent.xyz  # solvent molecule xyz file
    nshell    2  # number of solvent shell
    density   1  # solvent density in g/cm**-3
    mass      18  # molar mass of solvent in g/mol
    packmol   /path/to/packmol  # the path to packmol

    ''')

    in_dict = {
        'solute': key_dict['solute'],
        'solvent': key_dict['solvent'],
        'nshell': key_dict['nshell'],
        'density': key_dict['density'],
        'mass': key_dict['mass'],
        'packmol': key_dict['packmol'],
    }

    for key in in_dict.keys():
        if in_dict[key] is None:
            exit('\n KeyError: missing keyword in env_file: %s\n' % key)

    solute = in_dict['solute']
    solvent = in_dict['solvent']
    nshell = in_dict['nshell']
    density = in_dict['density']
    mass = in_dict['mass']
    packmol = in_dict['packmol']

    with open(solute, 'r') as inxyz:
        geom_c = inxyz.read().splitlines()

    _, geom_c = read_xyz(geom_c)
    rad_in = find_rad_in(geom_c)

    cm_to_a = 1e8  # angstrom to cm
    avg = 6.022 * 1e23  # avogadro's number
    fsphere = 4 / 3 * 3.1415926  # spherical volume factor

    unit_den = density / cm_to_a ** 3 / mass * avg
    unit_rad = (1 / unit_den / fsphere) ** (1 / 3)
    rad_out = rad_in + unit_rad * nshell
    vol = fsphere * (rad_out ** 3 - rad_in ** 3)
    num = vol * unit_den

    print(' computing inner radius %8.2f' % rad_in)
    print(' computing solvent unit radius %8.2f' % unit_rad)
    print(' creating %s layer of spherical shell from %8.2f to %8.2f Angstrom' % (nshell, rad_in, rad_out))
    print(' total volume is %16.8f' % vol)
    print(' total number of solvent molecule is %8.0f' % num)
    print(' writing packmol input > sol.pkm')
    write_packmol(solute, solvent, num, rad_in, rad_out)
    print(' running packmol\n')
    subprocess.run('%s/bin/packmol < sol.pkm > sol.log' % packmol, shell=True)
    print(' writing environment > env.xyz')
    print(' COMPLETE')

    return None


def read_xyz(xyz):
    natom = int(xyz[0])
    atom = np.array([x.split()[0] for x in xyz[2: 2 + natom]])
    coord = np.array([x.split()[1: 4] for x in xyz[2: 2 + natom]]).astype(float)

    return atom, coord

def find_rad_in(xyz):
    # compute the radius of the inner core for solute molecule
    natom = len(xyz)
    src = []
    dst = []
    for n in range(natom):
        src += [n for _ in range(n + 1, natom)]
        dst += [x for x in range(n + 1, natom)]

    v = xyz[src] - xyz[dst]
    d = np.sum(v ** 2, axis=1) ** 0.5
    rad = np.amax(d) / 2

    return rad

def write_packmol(core, sol, num, rad_in, rad_out):
    output = """tolerance 2.5
output env.xyz
filetype xyz

structure %s
number 1
fixed 0. 0. 0. 0. 0. 0.
end structure

structure %s
number %8.0f
inside sphere 0. 0. 0. %8.2f
outside sphere 0. 0. 0. %8.2f
end structure
""" % (core, sol, num, rad_out, rad_in)

    with open('sol.pkm', 'w') as out:
        out.write(output)

    return None

def create_aggregate(key_dict):
    print('''
 Tips for creating aggregates from supercell
    the following keyword must be set for creating aggregate environment

    cell      cell.xyz  # unit cell molecule xyz file
    nmol      2  # number of molecules in the unit cell
    center    1  # the index of the molecule in the unit cell that will be used as the center in the aggregate
    a         1  # a distance
    b         1  # b distance
    c         1  # c distance
    alpha     90  # alpha angle 
    beta      90  # beta angle
    gamma     90  # gamma angle
    na        3  # number of translation in a direction, the order is 0, +1, -1, +2, -2, +3, -3,...
    nb        3  # number of translation in b direction
    nc        3  # number of translation in c direction
    radius    14  # cutoff radius to build aggregate

    ''')
    in_dict = {
        'cell': key_dict['cell'],
        'nmol': key_dict['nmol'],
        'center': key_dict['center'],
        'a': key_dict['a'],
        'b': key_dict['b'],
        'c': key_dict['c'],
        'alpha': key_dict['alpha'],
        'beta': key_dict['beta'],
        'gamma': key_dict['gamma'],
        'na': key_dict['na'],
        'nb': key_dict['nb'],
        'nc': key_dict['nc'],
        'radius': key_dict['radius'],
    }

    for key in in_dict.keys():
        if in_dict[key] is None:
            exit('\n KeyError: missing keyword in env_file: %s\n' % key)

    cell = in_dict['cell']
    nmol = in_dict['nmol']
    center = in_dict['center']
    a = in_dict['a']
    b = in_dict['b']
    c = in_dict['c']
    alpha = in_dict['alpha']
    beta = in_dict['beta']
    gamma = in_dict['gamma']
    na = in_dict['na']
    nb = in_dict['nb']
    nc = in_dict['nc']
    radius = in_dict['radius']

    ## compute unit vectors
    alpha = alpha / 180 * np.pi
    beta = beta / 180 * np.pi
    gamma = gamma / 180 * np.pi
    pa = np.array([1, 0, 0])
    pb = np.array([np.cos(gamma), np.sin(gamma), 0])
    pc_x = np.cos(beta)
    pc_y = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    pc_z = (1 - pc_x ** 2 - pc_y ** 2) ** 0.5
    pc = np.array([pc_x, pc_y, pc_z])

    va = pa * a
    vb = pb * b
    vc = pc * c
    print(' computing lattice vectors')
    print(' A ', pa)
    print(' B ', pb)
    print(' C ', pc)

    with open(cell, 'r') as incell:
        cell = incell.read().splitlines()

    atom, coord = read_cell(cell)
    natom = int(len(atom) / nmol)
    atom = atom[: natom]
    coord = coord.reshape((nmol, natom, 3))

    order = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]
    if na > len(order):
        exit('\n ValueError: na %s is too larger, the maximum value is 13' % na)
    if nb > len(order):
        exit('\n ValueError: nb %s is too larger, the maximum value is 13' % nb)
    if na > len(order):
        exit('\n ValueError: nc %s is too larger, the maximum value is 13' % nc)

    ta = order[0: na]
    tb = order[0: nb]
    tc = order[0: nc]

    supercell = []
    for pos_a in ta:
        sa = va * pos_a
        for pos_b in tb:
            sb = vb * pos_b
            for pos_c in tc:
                sc = vc * pos_c
                for mol in coord:
                    supercell.append(mol + sa + sb + sc)

    supercell = cut_cell(supercell, center, radius)
    maxrad = np.amax([np.sum(x ** 2, axis=1) ** 0.5 for x in supercell])
    core = write_core(atom, supercell[0])
    output = write_supercell(atom, supercell)

    with open('mol.xyz', 'w') as out:
        out.write(core)

    with open('env.xyz', 'w') as out:
        out.write(output)

    print(' building supercell')
    print(' cutting aggregate')
    print(' computing aggregate max radius ', maxrad)
    print(' writing center molecule > mol.xyz')
    print(' writing environment > env.xyz')
    print(' COMPLETE')

    return None

def read_cell(xyz):

    natom = int(xyz[0])
    coord = np.array([x.split()[0: 4] for x in xyz[2: 2 + natom]])
    atom = coord[:, 0]
    coord = coord[:, 1: 4].astype(float)

    return atom, coord

def cut_cell(supercell, nref, radius):
    ref = supercell[nref - 1]
    center = np.mean(ref, axis=0)
    cell = [ref - center]
    for n, mol in enumerate(supercell):
        if n == nref - 1:
            continue
        com = np.mean(mol, axis=0)
        d = np.sum((com - center) ** 2) ** 0.5
        if d <= radius:
            cell.append(mol - center)

    return cell

def write_core(atom, xyz):
    natom = len(xyz)
    output = '%s\n\n' % natom
    for n, coord in enumerate(xyz):
        output += '%-5s %24.15f %24.16f %24.16f\n' % (atom[n], coord[0], coord[1], coord[2])

    return output

def write_supercell(atom, supercell):
    nmol = len(supercell)
    natom = len(supercell[0])
    output = '%s\n%s mol %s atom\n' % (natom * nmol, nmol, natom)
    for cell in supercell:
        for n, coord in enumerate(cell):
            output += '%-5s %24.15f %24.16f %24.16f\n' % (atom[n], coord[0], coord[1], coord[2])

    return output

def merge_env(key_dict):
    iformat = key_dict['iformat']
    cpus = key_dict['cpus']
    if iformat == 'xyz':
        atom, initcond = read_initcond(key_dict)
    elif iformat == 'no':
        print(' skip sampling initial condition')
        print(' the whole environment is considered as initial condition')
        initcond = []
    else:
        atom, initcond = sample_initcond(key_dict)

    print(' reading environment file env.xyz')
    if not os.path.exists('env.xyz'):
        exit('\n FileNotFoundError: cannot find env.xyz file')

    with open('env.xyz', 'r') as inxyz:
        geom_e = inxyz.read().splitlines()

    env_atom, env = read_xyz(geom_e)

    if len(initcond) > 0:
        output = ['' for _ in initcond]
        diff = [[] for _ in initcond]
        variables_wrapper = [(n, env_atom, env, cond) for n, cond in enumerate(initcond)]
        cpus = np.amin([cpus, len(initcond)])
        pool = multiprocessing.Pool(processes=cpus)
        n = 0
        for val in pool.imap_unordered(merge_wrapper, variables_wrapper):
            n += 1
            pos, out, rmsd = val
            output[pos] = out
            diff[pos] = rmsd
            sys.stdout.write('CPU: %3d merging data: %d/%d\r' % (cpus, n, len(initcond)))
        pool.close()
        output = '\n'.join(output) + '\n'
        print(' maximum root-mean-square-deviation ', np.amax(diff))
    else:
        cm = np.mean(env[:, 0: 3], axis=0)
        env[:, 0: 3] = env[:, 0: 3] - cm
        output = '%s' % write_init(0, env_atom, env)

    with open('merged.init', 'w') as out:
        out.write(output)

    print(' merging %s initial condition with environment' % (len(initcond)))
    print(' moving system center to 0 0 0')
    print(' writing initial condition with environment > merged.init')
    print(' COMPLETE')
    print('''
 HINTS: 
       you might want to equilibrate the environment with fixed initial conditions
       traj_generator.py can read merged.init to setup PyRAI2MD calculations
       ''')

    return None

def read_initcond(key_dict):
    print('''
 Tips for reading initial condition from .init.xyz file   
    the .init.xyz file can be generated by traj_generator.py after sampling initial conditions
    use the keyword title to specify the name of the .init.xyz file. The title should not include .init.xyz
        
    ''')
    title = key_dict['title']
    if os.path.exists('%s.init.xyz' % title):
        print(' reading initial condition from %s.init.xyz' % title)
    else:
        exit('\n ValueError: cannot find %s.init.xyz' % title)

    with open('%s.init.xyz' % title, 'r') as indata:
        cond = indata.read().splitlines()

    natom = int(cond[0].split()[2])
    atom = []
    initcond = []
    for n, line in enumerate(cond):
        if 'Init' in line:
            coord = np.array([x.split()[0:7] for x in cond[n + 1: n + 1 + natom]])
            atom = coord[:, 0]
            xyz = coord[:, 1: 7].astype(float)
            initcond.append(xyz)

    return atom, initcond

def sample_initcond(key_dict):
    print('''
 Tips for sampling initial condition from frequency file
    sampling initial conditions uses PyRAI2MD's module, make sure you have installed it first
    the following keyword must be set for sampling initial conditions
    
    method        wigner  # initial condition sampling method
    ninitcond     1  # number of sampled initial condition
    seed          1  # random seed for sampling
    format        xyz  # frequency file format
    temp          298.15  # sampling temperature
    
    ''')

    if not has_sampling:
        exit('\n ModuleNotFoundError: PyRAI2MD is not installed')

    in_dict = {
        'method': key_dict['method'],
        'ninitcond': key_dict['ninitcond'],
        'iseed': key_dict['iseed'],
        'iformat': key_dict['iformat'],
        'temp': key_dict['temp'],
    }

    for key in in_dict.keys():
        if in_dict[key] is None:
            exit('\n KeyError: missing keyword in env_file: %s\n' % key)

    title = key_dict['title']
    method = in_dict['method']
    ninitcond = in_dict['ninitcond']
    iseed = in_dict['iseed']
    iformat = in_dict['iformat']
    temp = in_dict['temp']
    ensemble = sampling(title, ninitcond, iseed, temp, method, iformat)
    atom = []
    initcond = []
    for cond in ensemble:
        atom = cond[:, 0]
        initcond.append(cond[:, 1: 7])

    return atom, initcond

def merge_wrapper(var):
    n, env_atom, env, cond = var
    cond, rmsd = merge(env, cond)
    out = '%s' % write_init(n, env_atom, cond)

    return n, out, rmsd

def merge(env, xyz):
    shell = env[len(xyz):]
    shell = np.concatenate((shell, np.zeros((len(shell), 3))), axis=1)
    cond = np.concatenate((xyz, shell), axis=0).astype(float)
    cm = np.mean(cond[:, 0: 3], axis=0)
    cond[:, 0: 3] = cond[:, 0: 3] - cm
    rmsd = np.mean((cond[:len(xyz), 0: 3] - xyz[:, 0: 3].astype(float)) ** 2) ** 0.5
    return cond, rmsd

def write_init(idx, atom, cond):
    natom = len(atom)
    ndata = len(cond[0])
    output = 'Init %s %s X(A) Y(A) Z(A) Vx(au) Vy(au) Vz(au) g/mol e\n' % (idx + 1, natom)
    for n in range(natom):
        name = atom[n]
        x = cond[n][0]
        y = cond[n][1]
        z = cond[n][2]
        if ndata >= 6:
            vx = cond[n][3]
            vy = cond[n][4]
            vz = cond[n][5]
        else:
            vx = 0
            vy = 0
            vz = 0
        output += '%-5s %24.16f %24.16f %24.16f %24.16f %24.16f %24.16f 0 0\n' % (name, x, y, z, vx, vy, vz)

    return output

def read_final_cond(key_dict):
    print('''
 Tips for reading final condition from equilibrated trajectories   
    the following keyword must be set for reading trajectories

    read          list.txt  # a list of path to read the finally equilibrated conditions
    skip          10  # number of MD step to be skipped before reading conditions
    freq          1  # frequency of reading conditions in each trajectory from the last snapshot  
    combine       yes  # combine the initial velocity with the corresponding atoms in the final condition
    
    ''')

    in_dict = {
        'read': key_dict['read'],
        'skip': key_dict['skip'],
        'freq': key_dict['freq'],
        'combine': key_dict['combine'],
    }

    for key in in_dict.keys():
        if in_dict[key] is None:
            exit('\n KeyError: missing keyword in env_file: %s\n' % key)

    cpus = key_dict['cpus']
    read = in_dict['read']
    skip = in_dict['skip']
    freq = in_dict['freq']
    combine = in_dict['combine']

    with open(read, 'r') as infile:
        file = infile.read().splitlines()

    file_list = []
    for f in file:
        if len(f) > 0:
            file_list.append(f)

    if combine == 'no':
        initcond = [[] for _ in file_list]
    else:
        _, initcond = read_initcond(key_dict)

    nfile = len(file_list)
    ncond = len(initcond)
    ntask = np.amin([nfile, ncond])

    variables_wrapper = [(n, file_list[n], initcond[n], skip, freq) for n in range(ntask)]
    output = ['' for _ in range(ntask)]
    diff = [[] for _ in range(ntask)]
    cpus = np.amin([cpus, ntask])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for val in pool.imap_unordered(read_wrapper, variables_wrapper):
        n += 1
        pos, out, rmsd = val
        output[pos] = out
        diff[pos] = rmsd
        sys.stdout.write('CPU: %3d combining data: %d/%d\r' % (cpus, n, ntask))
    pool.close()
    output = '\n'.join(output) + '\n'

    with open('final.init', 'w') as out:
        out.write(output)

    print(' reading %s initial conditions' % ncond)
    print(' reading %s trajectories' % nfile)
    print(' working on the first %s of them' % ntask)
    print(' select %s snapshots in one trajectory' % freq)
    print(' maximum root-mean-square-deviation ', np.amax(diff))
    print(' combining initial velocity %s' % combine)
    print(' write final condition > final.init')
    print(' COMPLETE')

    return None

def read_wrapper(var):
    n, file, cond, skip, freq = var
    out = ''
    atom, traj = read_traj(file)
    select = np.arange(len(traj))[skip:]
    select = select[::-1]
    marker = (np.arange(freq) / freq * len(select)).astype(int)
    select = select[marker]
    rmsd = []
    index = n * freq
    for snapshot in traj[select]:
        index += 1
        if len(cond) > 0:
            natom = len(cond)
            snapshot[: natom, 3: 6] = cond[:, 3: 6]
            rmsd.append(np.mean((snapshot[: natom, 3: 6] - cond[:, 3: 6]) ** 2) ** 0.5)
        else:
            rmsd.append(0)

        out += '%s' % write_init(index - 1, atom, snapshot)

    return n, out, rmsd

def read_traj(file):
    filename = file.split('/')[-1]
    with open('%s/%s.md.xyz' % (file, filename), 'r') as infile:
        xyz = infile.read().splitlines()

    with open('%s/%s.md.velo' % (file, filename), 'r') as infile:
        velo = infile.read().splitlines()

    atom = []
    traj = []
    natom = int(xyz[0])
    for n in range(len(xyz)):
        if (n + 1) % (natom + 2) == 0:
            coord = xyz[n + 1 - natom: n + 1]
            v = velo[n + 1 - natom: n + 1]
            atom = [x.split()[0] for x in coord]
            coord = np.array([x.split()[1: 4] for x in coord]).astype(float)
            v = np.array([x.split()[1: 4] for x in v]).astype(float)
            traj.append(np.concatenate((coord, v), axis=1))
    traj = np.array(traj)

    return atom, traj


if __name__ == '__main__':
    main(sys.argv)
