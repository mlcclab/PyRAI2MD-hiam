import sys
import copy
import numpy as np
from numpy import linalg as la


# define the x and y length of the box
xhi = 103.30944
yhi = 101.14144
zhi = 60

# define the unit cell of perovskite
na = 16
nb = 16
ra = xhi/na
rb = yhi/na

# define the condition to set the iodine plane to compute the z distance
paxis = 1  # 0 for x, 1 for y
pcond = yhi / 2  # half of the y box
pz1 = 0   # z value to find the iodine on the first plane
pz2 = -3  # z value to find the iodine on the second plane
pthresh = 2  # threshold to find the iodine plane

# define the maximum value of z to identify the adsorbed molecules
zads = 5

# define the maximum value of z above the pvk layers, and the interval value for distribution analysis
zmax = 30
zint = 0.5

# define the maximum value of x for the pvk surface (y=x), and the interval value for coverage analysis
xmax = 104
xint = 2

# define the maximum value of x to identify if the surface is covered by an atom
xrad = 2

# define the maximum value and interval for radial distance distribution
rmax = 100
rint = 0.5

# define the maximum value and interval for angular distance distribution
agmax = 180
agint = 10
zshift = 100

# total number of Pb
npb = 16 * 16 * 3 * 1

# total number of I
ni = 16 * 16 * 3 * 3

# total number of FA
nfa = 16 * 16 * 3

# number of atom in FA
natom_fa = 8

# number of atom in molecule 1
natom_1 = 88

# number of molecules in the component 1
nmol_1 = 120

# number of atom in molecule 2
natom_2 = 15

# number of molecules in the component 2
nmol_2 = 120

# selected atoms for distance analysis, starting from 0
c60 = [x for x in range(60)]  # pcbm c60
ben = [x + 61 for x in range(6)]  # benzene ring in the small molecule
mol = [x for x in range(6)]  # benzene ring in the small molecule
oc = [81]  # C=O's carbon in pcbm
co = [82]  # C=O's oxygen in pcbm
oh = [14]  # O-H's hydrogen in the COOH group of the small molecule
co2 = [12]  # C=O's oxygen in the COOH group of the small molecule
oc2 = [13]  # C=O's carbon in the COOH group of the small molecule
coo = [12, 13]  # C=O-O's oxygen in the COOH group of the small molecule
i = [11]  # C-I's iodine in the small molecule
c6 = [x for x in range(6)]  # benzene's carbon in the small molecule
fah = [3, 4, 5, 6, 7]  # FA's hydrogen

# selected atoms for angle analysis, starting from 0
a_c60 = [51, 0]  # C60 direction in pcbm
a_ben = [66, 61]  # ben direction in pcbm
a_co = [82, 81]  # C=O in pcbm
a_r = [81, 72]  # OMe chain in pcbm
a_i = [11, 5]  # C-I in the small molecule

# define the threshold to find close interactions
dthresh = 3.5
ithresh = 5.0

npvk = npb + ni + nfa * natom_fa

def complete_octa(coord):
    new_coord = copy.deepcopy(coord)

    x1 = coord[coord[:, :, 0] < 0.75 * ra] + np.array([na * ra, 0, 0])
    x2 = coord[coord[:, :, 0] > (na - 0.75) * ra] - np.array([na * ra, 0, 0])
    for crd in [x1, x2]:
        if len(crd) > 0:
            new_coord = np.concatenate((new_coord, crd.reshape((-1, 1, 3))), axis=0)

    y1 = new_coord[new_coord[:, :, 1] < 0.75 * rb] + np.array([0, nb * rb, 0])
    y2 = new_coord[new_coord[:, :, 1] > (nb - 0.75) * rb] - np.array([0, nb * rb, 0])

    for crd in [y1, y2]:
        if len(crd) > 0:
            new_coord = np.concatenate((new_coord, crd.reshape((-1, 1, 3))), axis=0)

    return new_coord

def check_neighbor(box, xyz):
    # xyz in [nmol, natom, 3]

    dist = box/2
    f = np.abs(xyz) > dist
    s = np.sign(xyz)
    xyz -= f * s * box

    return xyz

def check_connectivity(box, xyz):
    # xyz in [nmol, natom, 3]

    a0 = xyz[:, 0:1]
    a1 = xyz
    dist = box/2
    f = np.abs(a1 - a0) > dist
    s = np.sign(a1 - a0)
    xyz -= f * s * box

    return xyz

def read_xyz(group, natom, nmol):
    atoms = [x.split()[0] for x in group[:natom]]
    coord = np.array([x.split()[1: 4] for x in group]).reshape((nmol, natom, 3)).astype(float)
    coord = check_connectivity(np.array([xhi, yhi, zhi]), coord)
    return atoms, coord

def compute_iodine_plane(coord, guess):
    zc = coord[:, :, 2]
    dz = np.abs(zc - guess)
    zp = np.mean(zc[dz < pthresh])

    return zp

def coverage(coord, px, pif, zs1, zs2, title):
    height = np.zeros(int(zmax / zint))
    a = np.arange(0, xmax + 1e-6, xint).reshape((1, -1))
    x, y = np.meshgrid(a, a.T)
    points = np.stack((x.reshape(-1), y.reshape(-1))).T
    l_points = points[points[:, 1] <= (yhi / 2)]
    r_points = points[points[:, 1] > (yhi / 2)]
    l_nads = 0
    r_nads = 0
    check_coord = []
    record = []
    dist = []
    # check z distance
    for crd in copy.deepcopy(coord):
        # find the position on the plane
        p = np.mean(crd[:, px])
        check_crd = copy.deepcopy(crd)
        # shift the coord to the iodine plane
        if p < pif:
            check_crd[:, 2] -= zs1
        else:
            check_crd[:, 2] -= zs2

        z = np.amin(check_crd[:, 2])
        zc = np.mean(check_crd[:, 2])   # record z of geometry center
        dist.append(z)
        if zc >= zmax:
            continue

        z_idx = int(zc / zint)
        height[z_idx] += 1

        if z < zads:
            cy = np.mean(crd[:, 1])
            if cy > (yhi / 2):
                r_nads += 1
            else:
                l_nads += 1

            check_coord.append(crd[:, 0:2])  # record x y for coverage analysis
            record.append(crd)

    nads = l_nads + r_nads
    check_coord = np.array(check_coord)
    record = np.array(record)

    # compute coverage
    l_covered = np.zeros(len(l_points))
    for n, p in enumerate(l_points):
        if len(check_coord) == 0:
            continue
        d = np.sum((check_coord - p) ** 2, axis=2) ** 0.5
        check_p = np.sum(d < xrad)
        if check_p > 0:
            l_covered[n] += 1

    r_covered = np.zeros(len(r_points))
    for n, p in enumerate(r_points):
        if len(check_coord) == 0:
            continue
        d = np.sum((check_coord - p) ** 2, axis=2) ** 0.5
        check_p = np.sum(d < xrad)
        if check_p > 0:
            r_covered[n] += 1

    covered = np.concatenate((l_covered, r_covered))
    r_area = np.sum(r_covered) / len(r_points)
    l_area = np.sum(l_covered) / len(l_points)
    area = np.sum(covered)/len(points)

    np.savetxt('dist-%s.txt' % title, np.array(dist))

    return nads, r_nads, l_nads, height, area, r_area, l_area, covered, r_covered, l_covered, record

def compute_rdf(crd_1, crd_2):
    rdf = []
    c1 = np.mean(crd_1, axis=1)
    c2 = np.mean(crd_2, axis=1)

    for c in c1:
        rdf1 = np.zeros(int(rmax / rint))
        dv = c2 - c
        dv = check_neighbor(np.array([xhi, yhi, zhi]), dv)
        d = np.sum(dv ** 2, axis=1) ** 0.5
        for dd in d[d > 0]:

            if dd >= rmax:
                continue

            d_idx = int(dd / rint)
            rdf1[d_idx] += 1
        rdf.append(rdf1)

    rdf = np.array(rdf)

    return rdf

def compute_mindist(crd_1, crd_2):
    min_list = []
    for c1 in crd_1:
        dist = []
        for c in c1:
            dv = crd_2 - c
            dv = check_neighbor(np.array([xhi, yhi, zhi]), dv)
            d = np.sum(dv ** 2, axis=2) ** 0.5
            dist.append(d)

        min_list.append(np.amin(dist))
    min_list = np.array(min_list)

    return min_list

def compute_angle(crd, b1=45, b2=135, title=''):
    up = 0
    para = 0
    down = 0
    a_list = []
    angles = np.zeros(int(agmax/agint))
    for c in crd:
        angle = project_z(c)
        a_idx = int(angle/agint)
        angles[a_idx] += 1
        a_list.append(angle)
        if 0 <= angle < b1:
            down += 1
        elif b1 <= angle <= b2:
            para += 1
        else:
            up += 1

    angles = np.array(angles)

    np.savetxt('angles-%s.txt' % title, np.array(a_list))

    return angles, up, para, down

def project_z(var):
    a, b = var
    c = np.array([b[0], b[1], b[2] - zshift])

    v1 = a - b
    v2 = c - b
    v1 = v1 / la.norm(v1)
    v2 = v2 / la.norm(v2)
    cosa = np.dot(v1, v2)
    alpha = np.arccos(cosa) * 57.2958

    return alpha

def compute_orient(crd):
    up = 0
    para = 0
    down = 0

    for c in crd:
        a, b = c
        d = b[2] - a[2]
        if d < 0:
            up += 1
        elif 0 <= d <= 3.5:
            para += 1
        else:
            down += 1

    return up, para, down

def write_xyz(atm_1, crd_1, atm_2=None, crd_2=None, atm_3=None, crd_3=None, atm_4=None, crd_4=None):
    if len(crd_1) == 0:
        return ''
    natom = len(atm_1) * len(crd_1)

    if atm_2:
        natom += len(atm_2) * len(crd_2)

    if atm_3:
        natom += len(atm_3) * len(crd_3)

    if atm_4:
        natom += len(atm_4) * len(crd_4)

    out_coord = '%s\nGeom \n' % natom

    for crd in crd_1:
        for n, line in enumerate(crd):
            a = atm_1[n]
            x, y, z = line
            out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    if atm_2:
        for crd in crd_2:
            for n, line in enumerate(crd):
                a = atm_2[n]
                x, y, z = line
                out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    if atm_3:
        for crd in crd_3:
            for n, line in enumerate(crd):
                a = atm_3[n]
                x, y, z = line
                out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    if atm_4:
        for crd in crd_4:
            for n, line in enumerate(crd):
                a = atm_4[n]
                x, y, z = line
                out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    return out_coord


with open(sys.argv[1], 'r') as infile:
    file = infile.read().splitlines()

print('Reading file %s' % sys.argv[1])
pb_group = file[2: 2 + npb]
i_group = file[2 + npb: 2 + npb + ni]
fa_group = file[2 + npb + ni: 2 + npvk]
group_1 = file[2 + npvk: 2 + npvk + nmol_1 * natom_1]
group_2 = file[2 + npvk + nmol_1 * natom_1: 2 + npvk + nmol_1 * natom_1 + nmol_2 * natom_2]

print('Preparing coordinates ')
atoms_pb, coord_pb = read_xyz(pb_group, 1, npb)
atoms_i, coord_i = read_xyz(i_group, 1, ni)
atoms_fa, coord_fa = read_xyz(fa_group, natom_fa, nfa)
atoms_1, coord_1 = read_xyz(group_1, natom_1, nmol_1)
atoms_2, coord_2 = read_xyz(group_2, natom_2, nmol_2)

output = write_xyz(atoms_1, coord_1)
with open('layer_pcbm.xyz', 'w') as out:
    out.write(output)

output = write_xyz(atoms_2, coord_2)
with open('layer_mol.xyz', 'w') as out:
    out.write(output)

new_coord_pb = complete_octa(coord_pb)
new_coord_i = complete_octa(coord_i)
output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa)

with open('surface.xyz', 'w') as out:
    out.write(output)

output = write_xyz(atoms_fa, coord_fa)
with open('layer_fa.xyz', 'w') as out:
    out.write(output)

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i)
with open('layer_pbi.xyz', 'w') as out:
    out.write(output)

print('Coverage analysis ')
# compute iodine plane
z1 = compute_iodine_plane(coord_i, pz1)
z2 = compute_iodine_plane(coord_i, pz2)

# compute coverage of molecules
nads_1, r_nads_1, l_nads_1, height_1, area_1, r_area_1, l_area_1, covered_1, r_covered_1, l_covered_1, record_1 = coverage(coord_1, paxis, pcond, z1, z2, 'pcbm-all')
nads_2, r_nads_2, l_nads_2, height_2, area_2, r_area_2, l_area_2, covered_2, r_covered_2, l_covered_2, record_2 = coverage(coord_2, paxis, pcond, z1, z2, 'fiba')
nads_3, r_nads_3, l_nads_3, height_3, area_3, r_area_3, l_area_3, covered_3, r_covered_3, l_covered_3, record_3 = coverage(record_1[:, c60], paxis, pcond, z1, z2, 'pcbm-c60')
r_area_t = np.sum((r_covered_1 + r_covered_2) > 0) / len(r_covered_1)
l_area_t = np.sum((l_covered_1 + l_covered_2) > 0) / len(l_covered_1)
area_t = np.sum((covered_1 + covered_2) > 0) / len(covered_1)

output = write_xyz(atoms_1, record_1, atoms_2, record_2)

print('Adsorption threshold: %s' % zads)
print('Reference surface 1: %8.2f' % z1)
print('Reference surface 2: %8.2f' % z2)
print('Adsorbed 1 right: %s left: %s all: %s' % (r_nads_1, l_nads_1, nads_1, ))
print('Adsorbed 2 right: %s left: %s all: %s' % (r_nads_2, l_nads_2, nads_2))
print('Coverage 1 right: %8.2f left: %8.2f all: %8.2f' % (r_area_1, l_area_1, area_1))
print('Coverage 2 right: %8.2f left: %8.2f all: %8.2f' % (r_area_2, l_area_2, area_2))
print('Coverage all right: %8.2f left: %8.2f all: %8.2f' % (r_area_t, l_area_t, area_t))

np.savetxt('z_distrib_1.txt', height_1)
np.savetxt('z_distrib_2.txt', height_2)
np.savetxt('z_distrib_3.txt', height_3)

with open('adsorbed.xyz', 'w') as out:
    out.write(output)

print('Radial distance analysis ')
# compute intermolecular distances
rdf_11 = compute_rdf(coord_1[:, c60], coord_1[:, c60])
rdf_12 = compute_rdf(coord_1[:, c60], coord_2[:, mol])
rdf_21 = compute_rdf(coord_1[:, ben], coord_2[:, mol])
rdf_22 = compute_rdf(coord_2[:, mol], coord_2[:, mol])

rdf_11_co = compute_rdf(coord_1[:, co], coord_1[:, co])
rdf_12_h = compute_rdf(coord_1[:, co], coord_2[:, oh])
rdf_22_h = compute_rdf(coord_2[:, oc2], coord_2[:, oc2])

np.savetxt('rdf_pcbm_pcbm.txt', rdf_11)
np.savetxt('rdf_pcbm_mol.txt', rdf_12)
np.savetxt('rdf_ben_mol.txt', rdf_21)
np.savetxt('rdf_mol_mol.txt', rdf_22)
np.savetxt('rdf_co_co.txt', rdf_11_co)
np.savetxt('rdf_co_cooh.txt', rdf_12_h)
np.savetxt('rdf_cooh_cooh.txt', rdf_22_h)

print('Angle distance analysis ')
# compute intramolecular angle
t_ang_c60, t_up_c60, t_para_c60, t_down_c60 = compute_angle(coord_1[:, a_c60], title='pcbm-c60')
t_ang_co, t_up_co, t_para_co, t_down_co = compute_angle(coord_1[:, a_co], title='pcbm-co')
t_ang_r, t_up_r, t_para_r, t_down_r = compute_angle(coord_1[:, a_r], title='pcbm-come')
t_ang_i, t_up_i, t_para_i, t_down_i = compute_angle(coord_2[:, a_i], title='fiba')

ang_c60, up_c60, para_c60, down_c60 = compute_angle(record_1[:, a_c60], title='ads-pcbm-c60')
ang_co, up_co, para_co, down_co = compute_angle(record_1[:, a_co], title='ads-pcbm-co')
ang_r, up_r, para_r, down_r = compute_angle(record_1[:, a_r], title='ads-pcbm-come')
ang_i, up_i, para_i, down_i = compute_angle(record_2[:, a_i], title='ads-fiba')

print('All up/parallel/down C60: %s %s %s' % (t_up_c60, t_para_c60, t_down_c60))
print('Ads up/parallel/down C60: %s %s %s' % (up_c60, para_c60, down_c60))

print('All up/parallel/down ROMe: %s %s %s' % (t_up_r, t_para_r, t_down_r))
print('Ads up/parallel/down ROMe: %s %s %s' % (up_r, para_r, down_r))

print('All up/parallel/down C=O: %s %s %s' % (t_up_co, t_para_co, t_down_co))
print('Ads up/parallel/down C=O: %s %s %s' % (up_co, para_co, down_co))

print('All up/parallel/down C-I: %s %s %s' % (t_up_i, t_para_i, t_down_i))
print('Ads up/parallel/down C-I: %s %s %s' % (up_i, para_i, down_i))

t_ang_ben, t_up_ben, t_para_ben, t_down_ben = compute_angle(coord_1[:, a_ben], b1=60, b2=120, title='pcbm-ben')
ang_ben, up_ben, para_ben, down_ben = compute_angle(record_1[:, a_ben], b1=60, b2=120, title='ads-pcbm-ben')

print('All up/parallel/down PCBM Ben: %s %s %s' % (t_up_ben, t_para_ben, t_down_ben))
print('Ads up/parallel/down PCBM Ben: %s %s %s' % (up_ben, para_ben, down_ben))

np.savetxt('tot_ang_pc60.txt', t_ang_c60)
np.savetxt('tot_ang_pcr.txt', t_ang_r)
np.savetxt('tot_ang_pcbm.txt', t_ang_co)
np.savetxt('tot_ang_mol.txt', t_ang_i)

np.savetxt('ads_ang_pc60.txt', ang_c60)
np.savetxt('ads_ang_pcr.txt', t_ang_r)
np.savetxt('ads_ang_pcbm.txt', ang_co)
np.savetxt('ads_ang_mol.txt', ang_i)

c60_com = np.mean(coord_1[:, c60], axis=1, keepdims=True)
pcbm = np.concatenate((c60_com, coord_1[:, co]), axis=1)
t_up_pcbm, t_para_pcbm, t_down_pcbm = compute_orient(pcbm)
c60_com = np.mean(record_1[:, c60], axis=1, keepdims=True)
pcbm = np.concatenate((c60_com, record_1[:, co]), axis=1)
up_pcbm, para_pcbm, down_pcbm = compute_orient(pcbm)
print('All up/parallel/down PCBM: %s %s %s' % (t_up_pcbm, t_para_pcbm, t_down_pcbm))
print('Ads up/parallel/down PCBM: %s %s %s' % (up_pcbm, para_pcbm, down_pcbm))

pcbm_co_pb = compute_mindist(coord_1[:, co], coord_pb)
print('Searching PCBM C=O-Pb interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(pcbm_co_pb), dthresh, np.sum(pcbm_co_pb <= dthresh)))

pcbm_co_fa = compute_mindist(coord_1[:, co], coord_fa[:, fah])
print('Searching PCBM C=O-FA interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(pcbm_co_fa), dthresh, np.sum(pcbm_co_fa <= dthresh)))

mol_co_pb = compute_mindist(coord_2[:, coo], coord_pb)
print('Searching Mol C=O-Pb interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(mol_co_pb), dthresh, np.sum(mol_co_pb <= dthresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_co_pb <= dthresh])
with open('mol-co-pb.xyz', 'w') as out:
    out.write(output)

mol_co_fa = compute_mindist(coord_2[:, coo], coord_fa[:, fah])
print('Searching Mol C=O-FA interactions')
print('Shortest: %8.2f Found %8.2f A: %s' % (np.amin(mol_co_fa), dthresh, np.sum(mol_co_fa <= dthresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_co_fa <= dthresh])
with open('mol-co-fa.xyz', 'w') as out:
    out.write(output)

mol_oh_i = compute_mindist(coord_2[:, oh], coord_i)
print('Searching Mol COOH-I interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(mol_oh_i), dthresh, np.sum(mol_oh_i <= dthresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_oh_i <= dthresh])
with open('mol-oh-i.xyz', 'w') as out:
    out.write(output)

mol_ci_i = compute_mindist(coord_2[:, i], coord_i)
print('Searching Mol C-I-I interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(mol_ci_i), dthresh, np.sum(mol_ci_i <= ithresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_ci_i <= ithresh])
with open('mol-i-i.xyz', 'w') as out:
    out.write(output)

mol_ben_i = compute_mindist(np.mean(coord_2[:, c6], axis=1, keepdims=True), coord_i)
print('Searching Mol Ben-I interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(mol_ben_i), dthresh, np.sum(mol_ben_i <= dthresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_ben_i <= dthresh])
with open('mol-ben-i.xyz', 'w') as out:
    out.write(output)

mol_ben_fa = compute_mindist(np.mean(coord_2[:, c6], axis=1, keepdims=True), coord_fa[:, fah])
print('Searching Mol Ben-FA interactions')
print('Shortest: %8.2f Found in %8.2f A: %s' % (np.amin(mol_ben_fa), dthresh, np.sum(mol_ben_fa <= dthresh)))

output = write_xyz(atoms_pb, new_coord_pb, atoms_i, new_coord_i, atoms_fa, coord_fa, atoms_2, coord_2[mol_ben_fa <= dthresh])
with open('mol-ben-fa.xyz', 'w') as out:
    out.write(output)

dist_data = np.array([pcbm_co_pb, pcbm_co_fa, mol_co_pb, mol_co_fa, mol_oh_i, mol_ci_i, mol_ben_i, mol_ben_fa])
np.savetxt('dist_data.txt', dist_data)
print('')


print('Done')
