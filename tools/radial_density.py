import sys
import multiprocessing
import numpy as np

mass_dict = {
    'H': 1.008,
    'C': 12.001,
    'F': 18.998,
}

def radial_density(atoms, coord, maxrad, interval):
    block_list = np.zeros(int(maxrad / interval))
    den_list = np.zeros(int(maxrad / interval))
    com = np.mean(coord, axis=0)
    coord -= com
    dist = np.sum(coord ** 2, axis=1) ** 0.5
    for n, d in enumerate(dist):
        mass = mass_dict[atoms[n]]
        idx = int(dist[n] / interval)
        block_list[idx] += mass

    for n, m in enumerate(block_list):
        r1 = (n + 1) * interval
        r2 = n * interval
        vol = 4 / 3 * np.pi * (r1 ** 3 - r2 ** 3)
        den = m / (6.022 * 1e23) / (vol / 10e24)
        den_list[n] = den

    return den_list, np.amax(dist)

def radial_density_wrapper(var):
    idx, title, maxrad, interval, skip = var
    filename = title.split('/')[-1]
    with open('%s/%s.md.xyz' % (title, filename), 'r') as infile:
        file = infile.read().splitlines()

    natom = int(file[0])
    coord_list = []
    atoms = []
    for n, line in enumerate(file):
        if 'coord' in line:
            coord = file[n + 1: n + 1 + natom]
            atoms = [x.split()[0] for x in coord][skip:]
            xyz = np.array([x.split()[1:4] for x in coord]).astype(float)[skip:]
            coord_list.append(xyz)

    den_list_1, max1 = radial_density(atoms, coord_list[0], maxrad, interval)
    den_list_2, max2 = radial_density(atoms, coord_list[10], maxrad, interval)
    den_list_3, max3 = radial_density(atoms, coord_list[-1], maxrad, interval)

    return idx, den_list_1, den_list_2, den_list_3, max1, max2, max3

def main():
    with open(sys.argv[1], 'r') as infile:
        file = infile.read().splitlines()

    ncpus = 20
    maxrad = float(sys.argv[2])
    skip = int(sys.argv[3])
    interval = 0.1
    den_list_1 = [[] for _ in file]
    den_list_2 = [[] for _ in file]
    den_list_3 = [[] for _ in file]
    r_max_1 = [[] for _ in file]
    r_max_2 = [[] for _ in file]
    r_max_3 = [[] for _ in file]

    ntraj = len(file)
    variables_wrapper = [[n, x, maxrad, interval, skip] for n, x in enumerate(file)]
    pool = multiprocessing.Pool(processes=ncpus)
    n = 0
    sys.stdout.write('CPU: %3d Reading: 0/%d\r' % (ncpus, ntraj))
    for val in pool.imap_unordered(radial_density_wrapper, variables_wrapper):
        n += 1
        idx, d1, d2, d3, r1, r2, r3 = val
        den_list_1[idx] = d1
        den_list_2[idx] = d2
        den_list_3[idx] = d3
        r_max_1[idx] = r1
        r_max_2[idx] = r2
        r_max_3[idx] = r3

        sys.stdout.write('CPU: %3d Reading: %d/%d\r' % (ncpus, n, ntraj))
    pool.close()

    den_1 = np.mean(den_list_1, axis=0) * (10/6.022)
    den_2 = np.mean(den_list_2, axis=0) * (10/6.022)
    den_3 = np.mean(den_list_3, axis=0) * (10/6.022)
    r1 = np.mean(r_max_1)
    r2 = np.mean(r_max_2)
    r3 = np.mean(r_max_3)

    print('D1 =', den_1.tolist())
    print('D2 =', den_2.tolist())
    print('D3 =', den_3.tolist())
    print('R1 =', r1)
    print('R2 =', r2)
    print('R3 =', r3)


if __name__ == '__main__':
    main()
