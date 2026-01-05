######################################################
#
# PyRAI2MD 2 module for generating training data structures
#
# Author Jingbai Li
# Jue 14 2023
#
######################################################

import os
import sys
import subprocess
import multiprocessing
import numpy as np

try:
    from PyRAI2MD.Utils.sampling import sampling
except ModuleNotFoundError:
    exit('PyRAI2MD is not installed, stop script')

try:
    import geodesic_interpolate
except ModuleNotFoundError:
    exit('Geodesic_interpolate is not installed, stop script')

def main(argv):
    usage = """
    PyRAI2MD structure sampling tool

    Usage:
        python3 structure_sampling_tool.py sampling or
        python3 structure_sampling_tool.py for help

    a sampling file contains the following parameters

      cpus          1 number of cpu for parallel generation
      wigner        reac.freq.molden meci.freq.molden prod.freq.molden
      seed          1
      temp          298.15  
      nw            10
      scale         1
      refxyz        reac.xyz
      interp        reac.xyz meci.xyz prod.xyz
      ni            10
      skip_wigner   0
      skip_first    0
      skip_last     0
    """

    if len(argv) <= 1:
        exit(usage)

    cpus = 1
    wigner = []
    seed = 1
    temp = 298.15
    nw = 10
    scale = 1
    refxyz = ''
    interp = []
    ni = 10
    skip_wigner = 0
    skip_first = 0
    skip_last = 0

    with open(argv[1]) as inp:
        inputfile = inp.read().splitlines()

    for line in inputfile:
        if len(line.split()) < 2:
            continue
        key = line.split()[0].lower()

        if 'cpus' == key:
            cpus = int(line.split()[1])
        elif 'wigner' == key:
            wigner = line.split()[1:4]
        elif 'seed' == key:
            seed = float(line.split()[1])
        elif 'temp' == key:
            temp = float(line.split()[1])
        elif 'nw' == key:
            nw = int(line.split()[1])
        elif 'scale' == key:
            scale = float(line.split()[1])
        elif 'refxyz' == key:
            refxyz = line.split()[1]
        elif 'interp' == key:
            interp = line.split()[1:4]
        elif 'ni' == key:
            ni = int(line.split()[1])
        elif 'skip_wigner' == key:
            skip_wigner = int(line.split()[1])
        elif 'skip_first' == key:
            skip_first = int(line.split()[1])
        elif 'skip_last' == key:
            skip_last = int(line.split()[1])

    wigner_list = []
    for n, file in enumerate(wigner):
        if os.path.exists(file):
            wigner_list.append(file)

    interp_list = []
    for n, file in enumerate(interp):
        if os.path.exists(file):
            interp_list.append(file)

    key_dict = {
        'cpus': cpus,
        'wigner': wigner_list,
        'seed': seed,
        'temp': temp,
        'nw': nw,
        'scale': scale,
        'refxyz': refxyz,
        'interp': interp_list,
        'ni': ni,
        'skip_wigner': skip_wigner,
        'skip_first': skip_first,
        'skip_last': skip_last,
    }

    jobtype = find_jobtype(key_dict)

    job_func = {
        'interp': quick_interp,
        'wigner': wigner_sampling,
        'wigner_plus_interp': wigner_plus_interp,
        'interp_wigner': interp_wigner,
    }

    if jobtype != 'None':
        job_func[jobtype](key_dict)
    print('\n\n  Complete\n')

def find_jobtype(key_dict):
    nw = len(key_dict['wigner'])
    ni = len(key_dict['interp'])

    if nw == 0 and ni < 2:
        jobtype = 'None'
    elif nw == 0 and ni >= 2:
        jobtype = 'interp'
    elif nw == 1 and ni < 1:
        jobtype = 'wigner'
    elif nw == 1 and ni >= 1:
        jobtype = 'wigner_plus_interp'
    elif nw > 1:
        jobtype = 'interp_wigner'
    else:
        jobtype = 'None'

    job_info = """
    
    Structure sampling setting
    
  ------------------------------------------------------------------------------------------------------
      Number     Number of
    of Wigner  Interpolation        Job type
      files        files
  ------------------------------------------------------------------------------------------------------
        0           <2         No job will be done
        
        0          >=2         Two or three points interpolation
        
        1            0         One point Wigner sampling
        
        1            1         One point Wigner sampling and read interpolated structures
                               and add Wigner sampled atomic displacement to interpolated structures

        1          >=2         One point Wigner sampling and two or three points interpolation 
                               and add Wigner sampled atomic displacement to interpolated structures
                               
      >=2          >=0         Two or three point Wigner sampling 
                               and interpolation between the Wigner sampled structures
  ------------------------------------------------------------------------------------------------------
    Checking job info ...
    Found %s Wigner sampling input files
    Found %s Interpolation input files
    Detected job type is: %s
    """ % (nw, ni, jobtype)

    print(job_info)

    return jobtype

def quick_interp(key_dict):
    interp = key_dict['interp']
    ni = key_dict['ni']
    skip_first = key_dict['skip_first']
    skip_last = key_dict['skip_last']
    n_interp = len(interp)

    job_info = """
    
    Geodesic interpolation    
  ------------------------------------------------------------------------------------------------------
    Number of the interpolation paths:         %s
    Number of the interpolation steps:         %s
    Skip the first interpolated structure:     %s
    Skip the last interpolated structures:     %s
    Save the interpolated structure:           interp.xyz
  ------------------------------------------------------------------------------------------------------
    """ % (n_interp - 1, ni, skip_first, skip_last)
    print(job_info)

    if n_interp > 2:
        n_path1 = ni + skip_first
        n_path2 = ni + 1 + skip_last
        atoms, coord_1 = read_xyz(interp[0])
        atoms, coord_2 = read_xyz(interp[1])
        atoms, coord_3 = read_xyz(interp[2])
        start = skip_first
        end = ni * 2 + skip_first
    else:
        n_path1 = ni + skip_first + skip_last
        n_path2 = 0
        atoms, coord_1 = read_xyz(interp[0])
        atoms, coord_2 = read_xyz(interp[1])
        coord_3 = None
        start = skip_first
        end = ni + skip_first

    coord_list = interpolation([atoms, coord_1, coord_2, coord_3, n_path1, n_path2])[start:end]

    output = ''
    for n, coord in enumerate(coord_list):
        output += write_xyz(atoms, coord, n + 1)

    with open('interp.xyz', 'w') as out:
        out.write(output)

    return atoms, coord_list

def interpolation(var):
    atoms, coord_1, coord_2, coord_3, n_path1, n_path2 = var
    path_1 = geodesic_interpolation(atoms, coord_1, coord_2, n_path1, '1')
    interp_path = align_path(path_1, coord_1)
    if n_path2 > 0:
        path_2 = geodesic_interpolation(atoms, coord_2, coord_3, n_path2, '2')
        path_2 = align_path(path_2, interp_path[-1])
        interp_path = interp_path + path_2[1:]

    return interp_path

def align_path(coord_path, ref):
    mol = coord_path[0]
    p = mol.copy()
    q = ref.copy()
    pc = p.mean(axis=0)
    qc = q.mean(axis=0)
    p -= pc
    q -= qc
    c = np.dot(np.transpose(p), q)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:  # ensure right-hand system
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)

    aligned_coord = []
    for mol in coord_path:
        coord = np.dot(mol - pc, u) + qc
        aligned_coord.append(coord)

    return aligned_coord

def geodesic_interpolation(atoms, coord_1, coord_2, ngeom, jobid=''):
    output = ''
    output += write_xyz(atoms, coord_1, 1)
    output += write_xyz(atoms, coord_2, 2)

    with open('geom-%s.tmp.xyz' % jobid, 'w') as out:
        out.write(output)

    subprocess.run(
        'geodesic_interpolate geom-%s.tmp.xyz --output output-%s.tmp.xyz --nimages %s > stdout 2>&1' % (
            jobid,
            jobid,
            ngeom
        ),
        shell=True
    )

    with open('output-%s.tmp.xyz' % jobid, 'r') as infile:
        file = infile.read().splitlines()

    natom = int(file[0])

    coord_list = []
    for n, line in enumerate(file):
        if 'Frame' in line:
            coord = file[n + 1: n + 1 + natom]
            coord = np.array([x.split()[1:4] for x in coord]).astype(float)
            coord_list.append(coord)

    return coord_list

def wigner_sampling(key_dict):
    wigner = key_dict['wigner']
    seed = key_dict['seed']
    temp = key_dict['temp']
    nw = key_dict['nw']
    scale = key_dict['scale']
    refxyz = key_dict['refxyz']
    skip_wigner = key_dict['skip_wigner']
    n_wigner = len(wigner)
    atoms = []

    callsample = ['molden', 'bagel', 'g16', 'orca']

    wigner_info = ''
    for n, mol in enumerate(wigner):
        iformat = mol.split('.')[-1]
        title = mol.split('.')[0]
        wigner_info += '    Wigner sampling point                      %s\n' % (n + 1)
        if iformat in callsample:
            wigner_info += '    Save Wigner sampling structures:           wigner-%s-%s.xyz\n' % (title, temp)
            wigner_info += '    Save Wigner sampling initconds:            %s.init\n' % title
        else:
            wigner_info += '    Load Wigner sampling initconds:            %s.init.xyz\n' % title

    if n_wigner == 1 and os.path.exists(refxyz) and scale != 1:
        scale_wigner = 1
        scale_info = """    Scale Wigner sampling structures:          Yes
    Reference structure file:                  %s
    Scale factor:                              %s
    Save the scaled structure:                 wigner-scaled-%s.xyz
    Save the scaled initial conditions:        wigner-scaled-%s.init.xyz""" % (refxyz, scale, scale, scale)
    else:
        scale_wigner = 0
        scale_info = '    Scale Wigner sampling structures:          No'

    job_info = """

    Wigner sampling    
  ------------------------------------------------------------------------------------------------------
    Random number seed:                        %s
    Temperature:                               %s
    Number of the Wigner sampling points:      %s
    Number of the Wigner sampling structures:  %s
    Skip Wigner sampling structures:           %s
%s%s
  ------------------------------------------------------------------------------------------------------
  """ % (seed, temp, n_wigner, nw, skip_wigner, wigner_info, scale_info)
    print(job_info)

    coord_sup_list = []
    for mol in wigner:
        iformat = mol.split('.')[-1]
        title = mol.split('.')[0]
        coord_list = sampling(title, skip_wigner + nw, seed, temp, 'wigner', iformat)[skip_wigner:]
        atoms = [x[0] for x in coord_list[0]]
        coord_list = np.array(coord_list)[:, :, 1:].astype(float)
        coord_sup_list.append(coord_list)

    if scale_wigner:
        atoms, ref = read_xyz(refxyz)
        scaled_coord_list = []
        out_xyz = ''
        out_initxyz = ''
        for n, xyz in enumerate(coord_sup_list[0]):
            xyz[:, 0: 3] = (xyz[:, 0: 3] - ref) * scale + ref
            xyz[:, 3: 6] = xyz[:, 3: 6] * scale
            scaled_coord_list.append(xyz)
            out_xyz += write_xyz(atoms, xyz[:, 0: 3], n + 1)
            out_initxyz += write_init_xyz(atoms, xyz, n + 1)
        coord_sup_list = [scaled_coord_list]

        with open('wigner-scaled-%s.xyz' % scale, 'w') as out:
            out.write(out_xyz)

        with open('wigner-scaled-%s.init.xyz' % scale, 'w') as out:
            out.write(out_initxyz)

    return atoms, coord_sup_list

def wigner_plus_interp(key_dict):
    nw = key_dict['nw']
    ni = key_dict['ni']
    n_interp = len(key_dict['interp'])
    atoms, wigner_list = wigner_sampling(key_dict)
    if n_interp == 1:
        interp_list = read_interp(key_dict)
    else:
        atoms, interp_list = quick_interp(key_dict)

    wigner_list = wigner_list[0]
    wigner_1 = wigner_list[0][:, 0:3]
    interp_1 = key_dict['interp'][0]
    atoms, interp_1 = read_xyz(interp_1)
    refxyz = key_dict['refxyz']

    if os.path.exists(refxyz):
        atom, ref = read_xyz(refxyz)
        wigner_rmsd = np.mean((wigner_1 - ref) ** 2) ** 0.5
        interp_rmsd = np.mean((interp_1 - ref) ** 2) ** 0.5
        ref_info = """    Wigner reference structure:                %s
    RMSD to the first Wigner sampling:         %-8.2f
    RMSD to the first point of interpolation:  %-8.2f""" % (refxyz, wigner_rmsd, interp_rmsd)
    else:
        ref = interp_1
        wigner_rmsd = np.mean((wigner_1 - ref) ** 2) ** 0.5
        ref_info = """    No reference structure is given
    Use the first point of interpolation as reference
    RMSD to the first Wigner sampling:         %-8.2f""" % wigner_rmsd

    job_info = """

    Adding Wigner sampled displacements on interpolated structures   
  ------------------------------------------------------------------------------------------------------
    Check reference structure and the RMSD to Wigner sampling and interpolation
    Large RMSD indicate inconsistent structures used in Wigner sampling and interpolation
%s
    Number of the generated structures:        %s
    Save generated structures:                 wigner-interp-merged.xyz
    Save generated conditions:                 wigner-interp-merged.init.xyz
  ------------------------------------------------------------------------------------------------------
      """ % (ref_info, int(nw * ni))
    print(job_info)

    coord_list = []
    for wigner in wigner_list:
        for interp in interp_list:
            coord = wigner[:, 0:3] - ref + interp
            coord_list.append(coord)

    output = ''
    output_init = ''
    for n, coord in enumerate(coord_list):
        output += write_xyz(atoms, coord, n + 1)
        output_init += write_init_xyz2(atoms, coord, n + 1)

    with open('wigner-interp-merged.xyz', 'w') as out:
        out.write(output)

    with open('wigner-interp-merged.init.xyz', 'w') as out:
        out.write(output_init)

    return None


def read_interp(key_dict):
    interp = key_dict['interp'][0]
    skip_first = key_dict['skip_first']
    skip_last = key_dict['skip_last']
    ni = key_dict['ni']

    coord_list = []
    with open(interp, 'r') as infile:
        file = infile.read().splitlines()

    natom = int(file[0])
    for n, _ in enumerate(file):
        if (n + 1) % (natom + 2) == 0:  # at the last line of each coordinates
            coord = file[n - natom + 1:n + 1]
            coord = np.array([x.split()[1: 4] for x in coord]).astype(float)
            coord_list.append(coord)

    n_interp = len(coord_list)

    if skip_first + ni + skip_last > n_interp:
        interp_info = '    Not enough interpolated structures'
        stop = 1
    else:
        interp_info = '    Select interpolated structure:             %s - %s' % (
            skip_first + 1, skip_first + ni + skip_last
        )
        stop = 0

    job_info = """

    Geodesic interpolation    
  ------------------------------------------------------------------------------------------------------
    Number of the interpolation steps:         %s
    Read interpolated structures:              %s
    Available interpolated structures:         %s
    Skip the first interpolated structure:     %s
    Skip the last interpolated structures:     %s
%s
  ------------------------------------------------------------------------------------------------------
    """ % (ni, interp, n_interp, skip_first, skip_last, interp_info)

    print(job_info)

    if stop:
        exit('\n\n  Error\n')

    return coord_list[skip_first:ni]

def interp_wigner(key_dict):
    cpus = key_dict['cpus']
    nw = key_dict['nw']
    ni = key_dict['ni']
    skip_wigner = key_dict['skip_wigner']
    skip_first = key_dict['skip_first']
    skip_last = key_dict['skip_last']

    atoms, wigner_sup_list = wigner_sampling(key_dict)
    n_wigner = len(wigner_sup_list)

    if n_wigner > 2:
        wigner_1 = len(wigner_sup_list[0])
        wigner_2 = len(wigner_sup_list[1])
        wigner_3 = len(wigner_sup_list[2])
        n_path1 = ni + skip_first
        n_path2 = ni + 1 + skip_last
        wigner_info = """    Interpolating between 3 points
    Available structure of Wigner sampling 1:  %s
    Available structure of Wigner sampling 2:  %s
    Available structure of Wigner sampling 3:  %s""" % (wigner_1, wigner_2, wigner_3)
    else:
        wigner_1 = len(wigner_sup_list[0])
        wigner_2 = len(wigner_sup_list[1])
        wigner_3 = nw
        n_path1 = ni + skip_first + skip_last
        n_path2 = 0
        wigner_info = """    Interpolating between 2 points
    Available structure of Wigner sampling 1:  %s
    Available structure of Wigner sampling 2:  %s""" % (wigner_1, wigner_2)

    if wigner_1 < nw or wigner_2 < nw or wigner_3 < nw:
        check_info = '    Not enough Wigner sampled structures for interpolation'
        stop = 1
    else:
        check_info = """    Number of the generated structures:        %d
    Save generated structures:                 interpolated-wigner.xyz
    Save generated conditions:                 interpolated-wigner.init.xyz""" % (nw * ni * (n_wigner - 1))
        stop = 0

    job_info = """

    Interpolating pathways between the Wigner sampled structures    
  ------------------------------------------------------------------------------------------------------
    Number of the Wigner sampling points:      %s
    Skip Wigner sampling structures:           %s
    Number of the interpolation points:        %s
    Skip the first interpolated structure:     %s
    Skip the last interpolated structures:     %s
    Check Wigner sampling structures
%s
%s
  ------------------------------------------------------------------------------------------------------
    """ % (nw, skip_wigner, ni, skip_first, skip_last, wigner_info, check_info)

    print(job_info)

    if stop:
        exit('\n\n  Error\n')

    coord_list = [[] for _ in range(nw)]

    if n_wigner > 2:
        variables_wrapper = [[
            x,
            atoms,
            wigner_sup_list[0][x],
            wigner_sup_list[1][x],
            wigner_sup_list[2][x],
            n_path1,
            n_path2
            ] for x in range(nw)]
        start = skip_first
        end = ni * 2 + skip_first
    else:
        variables_wrapper = [[
            x,
            atoms,
            wigner_sup_list[0][x],
            wigner_sup_list[1][x],
            None,
            n_path1,
            n_path2
            ] for x in range(nw)]
        start = skip_first
        end = ni + skip_first

    ncpus = min([cpus, nw])
    pool = multiprocessing.Pool(processes=ncpus)
    n = 0
    sys.stdout.write('CPU: %3d interpolating: 0/%d\r' % (ncpus, nw))
    for val in pool.imap_unordered(interpolation_wrapper, variables_wrapper):
        n += 1
        idx, coord = val
        coord_list[idx] = coord[start:end]
        sys.stdout.write('CPU: %3d interpolating: %d/%d\r' % (ncpus, n, nw))
    pool.close()

    coord_list = np.array(coord_list).reshape((-1, len(atoms), 3))
    output = ''
    output_init = ''
    for n, coord in enumerate(coord_list):
        output += write_xyz(atoms, coord, n + 1)
        output_init += write_init_xyz2(atoms, coord, n + 1)

    with open('interpolated-wigner.xyz', 'w') as out:
        out.write(output)

    with open('interpolated-wigner.init.xyz', 'w') as out:
        out.write(output_init)

    return None

def interpolation_wrapper(var):
    idx, atoms, coord_1, coord_2, coord_3, n_path1, n_path2 = var
    coord_1 = coord_1[:, 0: 3]
    coord_2 = coord_2[:, 0: 3]
    path_1 = geodesic_interpolation(atoms, coord_1, coord_2, n_path1, str(idx + 1))
    interp_path = align_path(path_1, coord_1)
    if n_path2 > 0:
        coord_3 = coord_3[:, 0: 3]
        path_2 = geodesic_interpolation(atoms, coord_2, coord_3, n_path2, str(idx + 1) + '-2')
        path_2 = align_path(path_2, interp_path[-1])
        interp_path = interp_path + path_2[1:]

    return idx, interp_path

def read_xyz(coord):
    with open(coord, 'r') as infile:
        file = infile.read().splitlines()
    natom = int(file[0])
    atoms = [x.split()[0] for x in file[2: 2 + natom]]
    xyz = np.array([x.split()[1: 4] for x in file[2: 2 + natom]]).astype(float)

    return atoms, xyz

def write_xyz(atoms, coord, idx):
    natom = len(atoms)
    output = '%s\nGeom %s\n' % (natom, idx)
    for n, xyz in enumerate(coord):
        a = atoms[n]
        x, y, z = xyz
        output += '%-5s%24.16f%24.16f%24.16f\n' % (a, x, y, z)

    return output

def write_init_xyz(atoms, coord, idx):
    natom = len(atoms)
    output = 'Init %5d %5s\n' % (idx, natom)
    for n in range(natom):
        a = atoms[n]
        x, y, z, vx, vy, vz, m, q = coord[n]
        output += '%-5s%30s%30s%30s%30s%30s%30s%16s%6s\n' % (a, x, y, z, vx, vy, vz, m, q)

    return output

def write_init_xyz2(atoms, coord, idx):
    natom = len(atoms)
    output = 'Init %5d %5s\n' % (idx, natom)
    vx, vy, vz, m, q = 0, 0, 0, 0, 0
    for n in range(natom):
        a = atoms[n]
        x, y, z = coord[n][0:3]
        output += '%-5s%30s%30s%30s%30s%30s%30s%16s%6s\n' % (a, x, y, z, vx, vy, vz, m, q)

    return output


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    main(sys.argv)
