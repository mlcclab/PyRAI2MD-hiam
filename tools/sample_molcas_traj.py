## ----------------------
## Sample snapshots of molcas trajectory
## ----------------------

import sys
import multiprocessing
import numpy as np

def align(interp, ref, pick):
    p = interp.copy()[pick]
    q = ref.copy()[pick]
    pc = p.mean(axis=0)
    qc = q.mean(axis=0)
    p -= pc
    q -= qc
    c = np.dot(np.transpose(p), q)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:                    # ensure right-hand system
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)

    coord = np.dot(interp - pc, u) + qc

    return coord

def read_xyz(file):
    natom = int(file[0])
    atoms = np.array([x.split()[0] for x in file[2: 2 + natom]])
    coord = np.array([x.split()[1: 4] for x in file[2: 2 + natom]]).astype(float)

    return atoms, coord

def write_initxyz(idx, atoms, coord):
    natom = len(atoms)
    output = 'Init %s %s\n' % (idx + 1, natom)
    for n, xyz in enumerate(coord):
        output += '%-5s%24.16f%24.16f%24.16f 0 0 0 0 0\n' % (atoms[n], xyz[0], xyz[1], xyz[2])

    return output

def write_xyz(idx, atoms, coord, cmmt):
    natom = len(atoms)
    output = '%s\ngeom %s %s\n' % (idx + 1, cmmt, natom)
    for n, xyz in enumerate(coord):
        output += '%-5s%24.16f%24.16f%24.16f\n' % (atoms[n], xyz[0], xyz[1], xyz[2])

    return output

def sample_traj(val):
    idx, filepath, nsample, ref, pick = val
    filename = filepath.split('/')[-1]

    with open('%s/%s.md.xyz' % (filepath, filename), 'r') as infile:
        traj = infile.read().splitlines()

    with open('%s/%s.md.energies' % (filepath, filename), 'r') as infile:
        energy = infile.read().replace('D', 'E').splitlines()

    energy = np.array([x.split()[4: 6] for x in energy[1:]]).astype(float)
    gap = np.abs(energy[:, 0] - energy[:, 1])
    select = np.argsort(gap)[0: nsample - 1]
    select = [0] + select.tolist()
    natom = int(traj[0])
    coord = []
    for n in select:
        xyz = traj[int(n * (natom + 2)) + 2: int((n + 1) * (natom + 2))]
        xyz = np.array([x.split()[1: 4] for x in xyz]).astype(float)
        xyz = align(xyz, ref, pick)
        coord.append(xyz)

    return idx, coord, gap[select]

def main():
    with open(sys.argv[1], 'r') as infile:
        filelist = infile.read().splitlines()

    with open(sys.argv[2], 'r') as infile:
        refxyz = infile.read().splitlines()

    title = 'psb6'
    ntraj = len(filelist)
    nsample = 5  # 1 initial 4 minimum energy gap
    ncpu = 20
    pick = [6, 7, 8, 9, 10, 11]
    pick = [x - 1 for x in pick]
    atoms, ref = read_xyz(refxyz)
    natom = len(atoms)

    coord_pool = [[] for _ in filelist]
    gap_pool = [[] for _ in filelist]
    variables_wrapper = [[n, x, nsample, ref, pick] for n, x in enumerate(filelist)]
    pool = multiprocessing.Pool(processes=ncpu)
    n = 0
    for val in pool.imap_unordered(sample_traj, variables_wrapper):
        n += 1
        idx, coord, gap = val
        coord_pool[idx] = coord
        gap_pool[idx] = gap
        sys.stdout.write('CPU: %3d Traj: %5s/%s\r' % (ncpu, n, ntraj))
    pool.close()

    coord_pool = np.array(coord_pool).reshape((ntraj * nsample, natom, 3))
    gap_pool = np.array(gap_pool).reshape(-1)
    initxyz = ''
    for n, coord in enumerate(coord_pool):
        initxyz += write_initxyz(n, atoms, coord)
    xyz = ''
    for n, coord in enumerate(coord_pool):
        xyz += write_xyz(n, atoms, coord, gap_pool[n])

    with open('%s.init.xyz' % title, 'w') as out:
        out.write(initxyz)

    with open('%s.xyz' % title, 'w') as out:
        out.write(xyz)

    print('COMPLETE')


if __name__ == '__main__':
    main()
