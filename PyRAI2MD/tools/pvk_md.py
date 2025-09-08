import sys
import copy
import numpy as np
from numpy import linalg as la

"""
==============================

  System definition section

==============================
"""

# save coordinate to xyz for selected snapshots
save_snapshots = [1]

# define system unit, order does matter
system_name = [
    'pvk_pb',
    'pvk_i',
    'pvk_fa',
    'nio_ni',
    'nio_o',
    'ito_in',
    'ito_o',
    'fto_sn',
    'fto_o',
    'sam',
    'mol',
]

# define number of atom in each unit
system_atoms = {
    'pvk_pb': 1,
    'pvk_i': 1,
    'pvk_fa': 8,
    'sam': 154,
    'mol': 27,
    'nio_ni': 1,
    'nio_o': 1,
    'ito_in': 1,
    'ito_o': 1,
    'fto_sn': 1,
    'fto_o': 1,
}

# define number of unit
system_number = {
    'pvk_pb': 512,
    'pvk_i': 1280,
    'pvk_fa': 256,
    'nio_ni': 0,
    'nio_o': 0,
    'ito_in': 0,
    'ito_o': 0,
    'fto_sn': 1920,
    'fto_o': 3840,
    'sam': 60,
    'mol': 0,
}

# define system layer
system_layer = {
    'pbi': ['pvk_pb', 'pvk_i'],
    'fa': ['pvk_fa'],
    # 'ito': ['ito_in', 'ito_o'],
    'fto': ['fto_sn', 'fto_o'],
    # 'nio': ['nio_ni', 'nio_o'],
    'sam': ['sam'],
    # 'mol': ['mol'],
}

# define number of cells in each layer
system_cells = {
    'pbi': [16, 16],  # na, nb
    'fa': [0, 0],  # 0 will be considered as single molecule
    'ito': [7, 4],
    'fto': [32, 15],
    'nio': [24, 24],
    'sam': [0, 0],
    'mol': [0, 0],
}

# define number of unique position in a and b of the cell
system_unique = {
    'sam': [0, 0],
    'mol': [0, 0],
    'pbi': [1, 1],
    'fa': [0, 0],
    'nio': [2, 2],
    'fto': [2, 2],
    'ito': [4, 8],
}

"""
==============================

  End of System definition

==============================
"""

"""
==============================

  Analysis settings

==============================
"""
# define the condition for partial surface analysis
system_coverage = {
    'skip': False,  # skip coverage analysis
    'paxis': 1,  # 0 for x, 1 for y
    'pcond': 0.5,  # 0.5 for half of the y box, could be other fraction numbers
    'adsorb': ['sam'],  # a list of unit layer to compute coverage
    'surf1': 'fto',  # name of first part of the surface, should be available in system_layer
    'surf2': 'fto',  # name of second part of the surface, should be available in system_layer
    'pz1': 420,  # use given number of the largest z value of the atoms to guess the position of the surface
    'pz2': 420,  # use given number of the largest z value of the atoms to guess the position of the surface
    'pthresh': 2,  # threshold to find the surface plane
    'zshift': 0,  # shift the surface for coverage calculation
    'xrad': 2,  # the maximum radius to identify if the surface is covered by an atom
    'xint': 2,  # the interval value for coverage analysis
    'zads': 2,  # maximum value of z to identify the adsorbed molecules
}

# define the settings for radial distance distribution
system_rdf = {
    'skip': False,  # skip rdf analysis
    'center': [
        ['sam', []],
    ],  # a list of name of the unit and the indices of the selected atoms for rdf center
    'neighbor': [
        ['sam', []],
    ],  # a list of name of the unit and the indices of the selected atoms for rdf neighbor
    'rmax': 100,  # rdf max
    'rint': 0.5,  # rdf interval
    'oligomer': [
        ['sam', []],
    ],  # select the unit to check oligomer
    'rad': 2,  # radius to search oligomer
}

# define the settings for angular distance distribution
system_angle = {
    'skip': False,  # skip angle analysis
    'angle': [
        ['sam', [77, 0, 81]],
        ['sam', [0, 81]],
        # ['mol', [0, 18]],
        #   ['pvk_pb', ['pvk_i']],
    ],  # a list of name of the unit and the indices of the selected atoms,
    # three indices for user defined angle
    # two indices for z projection of the vector
    # one string loop over all units to search angles
    'agmax': 180,  # adf max
    'agint': 10,  # adf interval
    'zlevel': [64, 128],  # search angles for the given number of atoms in the selected units with the largest z value
    'zskip': [0, 0],  # skip the given number of atoms in searching angle
    'zlimit': 4,  # search angels with in the given radius
    'b1': 45,  # threshold for down angle
    'b2': 135,  # threshold for up angle
}

# selected atoms for distance analysis, starting from 0
system_mindist = {
    'skip': False,  # skip minimum distance analysis
    'dist': [
        ['sam', [78], 'fto_sn', [0]],
        ['sam', [83], 'fto_sn', [0]],
        ['sam', [88], 'fto_sn', [0]],
        # ['sam', [31], 'nio_ni', [0]],
        # ['sam', [31], 'nio_o', [0]],
        # ['sam', [26], 'nio_ni', [0]],
        # ['sam', [26], 'nio_o', [0]],
        # ['mol', [12], 'nio_ni', [0]],
        # ['mol', [12], 'nio_o', [0]],
        # ['mol', [24], 'nio_ni', [0]],
        # ['mol', [24], 'nio_o', [0]],
        # ['sam', [31], 'ito_in', [0]],
        # ['sam', [31], 'ito_o', [0]],
        # ['sam', [26], 'ito_in', [0]],
        # ['sam', [26], 'ito_o', [0]],
        # ['mol', [12], 'ito_in', [0]],
        # ['mol', [12], 'ito_o', [0]],
        # ['mol', [24], 'ito_in', [0]],
        # ['mol', [24], 'ito_o', [0]],
    ],  # a list of name of the unit and the indices of the selected atoms,
    # the indices are used to compute the average point of the selected atoms
    'dthresh': [4.5],  # define the threshold to find close interactions
    # 'save_surface': 'nio',  # save selected molecules with a surface in system_layer
    # 'save_surface': 'ito',
    'save_surface': 'fto',  # save selected molecules with a surface in system_layer

}

"""
==============================

  End of System definition

==============================
"""

system_info = {
    'name': system_name,
    'atoms': system_atoms,
    'number': system_number,
    'layer': system_layer,
    'cell': system_cells,
    'unique': system_unique,
    'coverage': system_coverage,
    'rdf': system_rdf,
    'angle': system_angle,
    'mindist': system_mindist,
    'save': True,
}


def complete_edge(coord, ra, rb, na, nb, ua, ub):
    """
    ra/rb   a/b length of system
    na/nb   number of cell in a/b direction
    ua/ub   number of unique atoms in a/b of unit cell
    """
    new_coord = copy.deepcopy(coord)

    ta = ra / na / (ua + 1)
    tb = rb / nb / (ub + 1)

    x1 = coord[coord[:, :, 0] < ta] + np.array([ra, 0, 0])
    x2 = coord[coord[:, :, 0] > (ra - ta)] - np.array([ra, 0, 0])
    for crd in [x1, x2]:
        if len(crd) > 0:
            new_coord = np.concatenate((new_coord, crd.reshape((-1, 1, 3))), axis=0)

    y1 = new_coord[new_coord[:, :, 1] < tb] + np.array([0, rb, 0])
    y2 = new_coord[new_coord[:, :, 1] > (rb - tb)] - np.array([0, rb, 0])

    for crd in [y1, y2]:
        if len(crd) > 0:
            new_coord = np.concatenate((new_coord, crd.reshape((-1, 1, 3))), axis=0)

    return new_coord


def check_neighbor(box, xyz):
    # xyz in [natom, 3]
    dist = box / 2
    f = np.abs(xyz) > dist
    s = np.sign(xyz)
    xyz[:, 0: 2] -= np.array(f * s * box)[:, 0: 2]

    return xyz


def check_connectivity(box, xyz):
    # xyz in [nmol, natom, 3]

    a0 = xyz[:, 0:1]
    a1 = xyz
    dist = box / 2
    f = np.abs(a1 - a0) > dist
    s = np.sign(a1 - a0)
    xyz[:, :, 0: 2] -= np.array(f * s * box)[:, :, 0: 2]

    com = np.mean(xyz, axis=1, keepdims=True)
    p = (com - box) > 0
    n = com < 0
    xyz[:, :, 0: 2] -= np.array(p * box)[:, :, 0: 2]
    xyz[:, :, 0: 2] += np.array(n * box)[:, :, 0: 2]
    return xyz


def read_xyz(group, box, natom, nmol):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])
    atoms = [x.split()[0] for x in group[:natom]]
    coord = np.array([x.split()[1: 4] for x in group]).reshape((nmol, natom, 3)).astype(float)
    coord = check_connectivity(np.array([xhi, yhi, zhi]), coord)
    return atoms, coord


def compute_surf_plane(coord, guess, pthresh):
    zc = coord[:, :, 2]
    zg = np.mean(np.sort(zc.reshape(-1))[-guess:])
    dz = np.abs(zc - zg)
    zp = np.mean(zc[dz < pthresh])

    return zp


def coverage(idx, symbol, coord, box, info):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])
    skip = info['coverage']['skip']
    paxis = info['coverage']['paxis']
    pcond = info['coverage']['pcond']
    adsorb = info['coverage']['adsorb']
    surf1 = info['coverage']['surf1']
    surf2 = info['coverage']['surf2']
    pz1 = info['coverage']['pz1']
    pz2 = info['coverage']['pz2']
    pthresh = info['coverage']['pthresh']
    zshift = info['coverage']['zshift']
    xrad = info['coverage']['xrad']
    xint = info['coverage']['xint']
    zads = info['coverage']['zads']
    layer = info['layer']
    save = info['save']

    if skip:
        return '', {}

    output = 'Coverage step %8s ' % idx
    # compute surface plane
    coord1 = []
    for k in layer[surf1]:
        coord1 += coord[k].tolist()

    coord2 = []
    for k in layer[surf2]:
        coord2 += coord[k].tolist()

    z1 = compute_surf_plane(np.array(coord1), pz1, pthresh)
    z2 = compute_surf_plane(np.array(coord2), pz2, pthresh)

    print('Adsorption threshold: %s' % zads)
    print('Reference surface 1: %8.2f' % z1)
    print('Reference surface 2: %8.2f' % z2)
    print('Shifting surface by: %8.2f' % zshift)

    a = np.arange(0, xhi + 1e-6, xint).reshape((1, -1))
    b = np.arange(0, yhi + 1e-6, xint).reshape((1, -1))
    x, y = np.meshgrid(a, b.T)
    points = np.stack((x.reshape(-1), y.reshape(-1))).T

    l_points = points[points[:, paxis] <= ([xhi, yhi][paxis] * pcond)]
    r_points = points[points[:, paxis] > ([xhi, yhi][paxis] * pcond)]

    l_nads_t = 0
    r_nads_t = 0
    nads_t = 0
    l_covered_t = 0
    r_covered_t = 0
    covered_t = 0
    atoms_t = []
    record_t = []
    record_coord = {}

    # check z distance
    pc = [xhi, yhi][paxis] * pcond
    for k in adsorb:
        l_nads = 0
        r_nads = 0
        check_coord = []
        record = []
        atoms = symbol[k]
        for crd in copy.deepcopy(coord[k]):
            # find the position on the plane
            p = np.mean(crd[:, paxis])
            check_crd = copy.deepcopy(crd)
            # shift the coord to the iodine plane
            if p < pc:
                check_crd[:, 2] -= z1 + zshift
            else:
                check_crd[:, 2] -= z2 + zshift

            z_min = np.amin(check_crd[:, 2])
            z_max = np.amax(check_crd[:, 2])

            if 0 < z_min <= zads or z_min <= 0 <= z_max or -zads <= z_max < 0:
                if p < pc:
                    l_nads += 1
                else:
                    r_nads += 1

                check_coord.append(check_crd)  # record x y z for coverage analysis
                record.append(crd)

        nads = l_nads + r_nads
        l_nads_t += l_nads
        r_nads_t += r_nads
        nads_t += nads
        check_coord = np.array(check_coord)
        record = np.array(record)
        record_coord[k] = copy.deepcopy(record)

        # compute coverage
        check_coord = check_coord[np.abs(check_coord[:, :, 2]) < zads]
        check_coord = check_coord[:, 0: 2]  # pick x and y
        l_covered = np.zeros(len(l_points))
        for n, p in enumerate(l_points):
            if len(check_coord) == 0:
                continue
            d = np.sum((check_coord - p) ** 2, axis=1) ** 0.5
            check_p = np.sum(d < xrad)
            if check_p > 0:
                l_covered[n] += 1

        r_covered = np.zeros(len(r_points))
        for n, p in enumerate(r_points):
            if len(check_coord) == 0:
                continue
            d = np.sum((check_coord - p) ** 2, axis=1) ** 0.5
            check_p = np.sum(d < xrad)
            if check_p > 0:
                r_covered[n] += 1

        covered = np.concatenate((l_covered, r_covered))
        r_area = np.sum(r_covered) / len(r_points)
        l_area = np.sum(l_covered) / len(l_points)
        area = np.sum(covered) / len(points)

        r_covered_t += r_covered
        l_covered_t += l_covered
        covered_t += covered
        atoms_t.append(atoms)
        record_t.append(record)
        output_coord = write_xyz([atoms], [record])

        print('Unit: %s' % k)
        print('Adsorbed on surf1: %s' % r_nads)
        print('Adsorbed on surf2: %s' % l_nads)
        print('Adsorbed on both:  %s' % nads)
        print('Coverage of surf1: %8.2f' % r_area)
        print('Coverage of surf2: %8.2f' % l_area)
        print('Coverage of both:  %8.2f' % area)

        output += 'Unit %8s %8s %8s %8s %8.2f %8.2f %8.2f \n' % (k, r_nads, l_nads, nads, r_area, l_area, area)
        if save:
            with open('adsorbed_%s_%s.xyz' % (k, idx), 'w') as out:
                out.write(output_coord)

    r_area_t = np.sum(r_covered_t > 0) / len(r_points)
    l_area_t = np.sum(l_covered_t > 0) / len(l_points)
    area_t = np.sum(covered_t > 0) / len(points)

    print('Total')
    print('Adsorbed on surf1: %s' % r_nads_t)
    print('Adsorbed on surf2: %s' % l_nads_t)
    print('Adsorbed on both:  %s' % nads_t)
    print('Coverage of surf1: %8.2f' % r_area_t)
    print('Coverage of surf2: %8.2f' % l_area_t)
    print('Coverage of both:  %8.2f' % area_t)

    output += 'Coverage step %8s Total %8s %8s %8s %8.2f %8.2f %8.2f \n' % (
        idx, r_nads_t, l_nads_t, nads_t, r_area_t, l_area_t, area_t
    )

    out_coord = write_xyz(atoms_t, record_t)

    if save:
        with open('adsorbed_all_%s.xyz' % idx, 'w') as out:
            out.write(out_coord)

    return output, record_coord


def compute_rdf(idx, coord, box, info):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    skip = info['rdf']['skip']
    list_1 = info['rdf']['center']
    list_2 = info['rdf']['neighbor']
    rmax = info['rdf']['rmax']
    rint = info['rdf']['rint']

    if skip:
        return ''

    output = ''
    for n, _ in enumerate(list_1):
        crd_1, sel_1 = list_1[n]
        crd_2, sel_2 = list_2[n]
        rdf = []

        if len(sel_1) == 0:
            sel_1 = np.arange(len(coord[crd_1][0]))
        if len(sel_2) == 0:
            sel_2 = np.arange(len(coord[crd_2][0]))

        c1 = np.mean(coord[crd_1][:, sel_1], axis=1)
        c2 = np.mean(coord[crd_2][:, sel_2], axis=1)

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

        rdf = np.mean(rdf, axis=0)

        output += 'RDF %s step %s from %s to %s %s\n' % (n + 1, idx, crd_1, crd_2, ' '.join(['%8s' % x for x in rdf]))

    return output


def compute_olig(idx, symbol, coord, box, info):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    skip = info['rdf']['skip']
    olig_list = info['rdf']['oligomer']
    rad = info['rdf']['rad']

    if len(olig_list) == 0:
        return ''

    print('Search radius: %8.2f ' % rad)
    if skip:
        return ''

    output = ''
    crd_names = ''
    all_atom = []
    all_coord = []
    all_select = []
    for olig_info in olig_list:
        olig, sel = olig_info
        if len(sel) == 0:
            sel = np.arange(len(symbol[olig]))
        all_atom += [symbol[olig] for _ in coord[olig]]  # collect all atom symbols
        all_coord += coord[olig].tolist()  # collect all coordinates
        all_select += [sel for _ in coord[olig]]  # collect all selected index
        crd_names += ' %s' % olig

    conn = np.zeros((len(all_coord), len(all_coord)))
    for n, c1 in enumerate(all_coord):
        c1 = np.array(c1)
        sel1 = all_select[n]
        c1_dim = len(c1[sel1])
        for m, c2 in enumerate(all_coord[n + 1:]):
            c2 = np.array(c2)
            sel2 = all_select[n + m + 1]
            c2_dim = len(c2[sel2])
            c12 = np.repeat(c1[sel1], c2_dim, axis=0)  # [q1, q2, q3] --> [q1, q1, ..., q2, q2, ..., q3, q3, ...]
            c21 = np.tile(c2[sel2], (c1_dim, 1))  # [q1, q2, q3] --> [q1, q2, q3, q1, q2, q3, ..., ..., ...]
            dv = c12 - c21
            dv = check_neighbor(np.array([xhi, yhi, zhi]), dv)
            d = np.sum(dv ** 2, axis=1) ** 0.5
            dmin = np.amin(d)
            if dmin <= rad:
                conn[n, n + m + 1] = 1
                conn[n + m + 1, n] = 1

    conn_list = recursive_search(conn)
    # for rw in conn:
    #    print(rw.tolist())
    # print(conn_list)
    conn_count = {}
    conn_coord = {}
    for cn in conn_list:
        ct = len(cn)
        conn_atom = []
        conn_crd = []

        for index in cn:
            atom = all_atom[index]
            coord = all_coord[index]
            conn_atom.append(atom)
            conn_crd.append(coord)

        if ct in conn_count:
            conn_count[ct] += 1
            conn_coord[ct].append([conn_atom, conn_crd])
        else:
            conn_count[ct] = 1
            conn_coord[ct] = [[conn_atom, conn_crd]]

    info = ''
    for key, val in conn_count.items():
        info += 'size: %s  num: %s || ' % (key, val)
        output_coord = ''
        for og in conn_coord[key]:
            atoms, coord = og
            output_coord += write_individual_xyz(atoms, coord, np.array([xhi, yhi, zhi]))

        with open('olig_x%s_%s.xyz' % (key, idx), 'w') as out:
            out.write(output_coord)

    print('OLIG step %s including %s %s\n' % (idx, crd_names, info))
    output += 'OLIG step %s including %s %s\n' % (idx, crd_names, info)

    return output


def recursive_search(m):
    m_list = []
    for n, row in enumerate(m):
        found = 0
        for ml in m_list:
            if n in ml:
                found = 1
                break

        if found == 1:
            continue

        mem_list = find_mem(n, np.array(m), [])
        m_list.append(mem_list)

    return m_list


def find_mem(r, m, mem_list):
    mem_list.append(int(r))

    cn = np.where(m[r] > 0)[0]
    for n in cn:
        if n not in mem_list:
            mem_list = find_mem(n, m.T, mem_list)

    return list(set(mem_list))


def compute_mindist(idx, symbol, coord, box, info):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    skip = info['mindist']['skip']
    dist = info['mindist']['dist']
    dthresh = info['mindist']['dthresh']
    surf = info['mindist']['save_surface']
    layer = info['layer']
    save = info['save']

    if skip:
        return ''

    ndist = len(dist)
    nthresh = len(dthresh)
    pad = ndist - nthresh
    if pad > 0:
        dthresh += [dthresh[-1] for _ in range(pad)]
    else:
        dthresh = dthresh[:ndist]

    output = ''
    for n, dst in enumerate(dist):
        k1, sel1, k2, sel2 = dst
        crd_1 = np.mean(coord[k1][:, sel1], axis=1)
        crd_2 = np.mean(coord[k2][:, sel2], axis=1)

        nfound = 0
        f1 = []
        f2 = []
        min_list = []
        for m, c1 in enumerate(crd_1):
            dv = crd_2 - c1
            dv = check_neighbor(np.array([xhi, yhi, zhi]), dv)
            d = np.sum(dv ** 2, axis=1) ** 0.5
            min_dist = np.min(d)
            min_list.append(min_dist)

            if 0 < min_dist <= dthresh[n]:
                f1.append(m)
                f2.append(np.argmin(d))
                nfound += 1
        print('MinDist from %s to %s in %8.2f A found %s' % (k1, k2, dthresh[n], nfound))

        output += 'MinDist %s step %8s from %s to %s in %8.2f found %8s %s\n' % (
            n + 1, idx, k1, k2, dthresh[n], nfound, ' '.join(['%8.2f' % x for x in min_list])
        )

        if save:
            out_sym = []
            out_crd = []

            if surf:
                for k in layer[surf]:
                    out_sym.append(symbol[k])
                    out_crd.append(coord[k])
                title = '%s_%s_%s' % (k1, k2, surf)
            else:
                title = '%s_%s' % (k1, k2)

            out_sym.append(symbol[k1])
            out_sym.append(symbol[k2])
            out_crd.append(coord[k1][f1])
            out_crd.append(coord[k2][f2])

            output_coord = write_xyz(out_sym, out_crd)

            with open('mindist_%s_%s.xyz' % (title, idx), 'w') as out:
                out.write(output_coord)

    return output


def compute_angle(idx, coord, record_coord, box, info):
    skip = info['angle']['skip']
    angle = info['angle']['angle']
    agmax = info['angle']['agmax']
    agint = info['angle']['agint']
    zlevel = info['angle']['zlevel']
    zskip = info['angle']['zskip']
    zlimit = info['angle']['zlimit']
    b1 = info['angle']['b1']
    b2 = info['angle']['b2']

    if skip:
        return ''

    output = ''
    for n, agl in enumerate(angle):
        k, sel = agl

        up = 0
        para = 0
        down = 0
        a_list = []
        angles = np.zeros(int(agmax / agint))
        up2 = 0
        para2 = 0
        down2 = 0
        a_list2 = []
        angles2 = np.zeros(int(agmax / agint))

        if len(sel) > 3:
            sel = sel[:3]

        if len(sel) >= 2:
            crd = coord[k][:, sel]
            if k in record_coord.keys():
                crd2 = record_coord[k][:, sel]
            else:
                crd2 = []
        else:
            crd = search_angle(coord[k], coord[str(sel[0])], box, zlevel, zskip, zlimit)
            if k in record_coord.keys():
                crd2 = search_angle(record_coord[k], record_coord[str(sel[0])], box, zlevel, zskip, zlimit)
            else:
                crd2 = []

        for c in crd:
            ag = project_angle(c, box)
            a_idx = int(ag / agint)
            angles[a_idx] += 1
            a_list.append(ag)
            if 0 <= ag < b1:
                down += 1
            elif b1 <= ag <= b2:
                para += 1
            else:
                up += 1

        for c in crd2:
            ag = project_angle(c, box)
            a_idx = int(ag / agint)
            angles2[a_idx] += 1
            a_list2.append(ag)
            if 0 <= ag < b1:
                down2 += 1
            elif b1 <= ag <= b2:
                para2 += 1
            else:
                up2 += 1

        print('Unit %s all up/parallel/down: %s %s %s' % (k, up, para, down))
        print('Unit %s ads up/parallel/down: %s %s %s' % (k, up2, para2, down2))

        output += 'ADF %s step %8s unit %s all up/para/down %8s %8s %8s %s \n' % (
            n + 1, idx, k, up, para, down, ' '.join(['%8.2f' % x for x in a_list])
        )
        output += 'ADF %s step %8s unit %s ads up/para/down %8s %8s %8s %s \n' % (
            n + 1, idx, k, up2, para2, down2, ' '.join(['%8.2f' % x for x in a_list2])
        )

    return output


def search_angle(crd1, crd2, box, zlevel, zskip, zlimit):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    coord = []
    z1, z2 = zlevel
    s1, s2 = zskip
    sel1 = np.argsort(crd1[s1:, 0, 2])[-z1:]
    sel2 = np.argsort(crd2[s2:, 0, 2])[-z2:]

    for c1 in crd1[s1:][sel1]:
        for c2 in crd2[s2:][sel2]:
            dc = (c1 - c2).reshape((1, 3))
            dc = check_neighbor(np.array([xhi, yhi, zhi]), dc)
            d = np.sum(dc ** 2, axis=1) ** 0.5
            if d <= zlimit:
                coord.append([c1, c2])

    coord = np.array(coord).reshape((-1, 2, 3))

    return coord


def project_angle(var, box):
    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    if len(var) == 3:
        a, b, c = var
    else:
        a, b = var
        c = np.array([b[0], b[1], b[2] - 100])

    v1 = a - b
    v2 = c - b
    v1 = check_neighbor(np.array([xhi, yhi, zhi]), v1.reshape((1, 3)))
    v2 = check_neighbor(np.array([xhi, yhi, zhi]), v2.reshape((1, 3)))
    v1 = v1.reshape(-1) / la.norm(v1)
    v2 = v2.reshape(-1) / la.norm(v2)
    cosa = np.dot(v1, v2)
    alpha = np.arccos(cosa) * 57.2958

    return alpha


def write_xyz(atm, crd):
    """
    atm     a list of atoms, shape -> N_type_of_unit * N_atoms_per_unit
    crd     a list of coord, shape -> N_type_of_unit * N_unit * N_atom_per_unit * 3
    """
    out_coord = ''
    natom = 0
    for n, _ in enumerate(atm):
        natom += len(atm[n]) * len(crd[n])
        for coord in crd[n]:
            for m, line in enumerate(coord):
                a = atm[n][m]
                x, y, z = line
                out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    out_coord = '%s\nGeom \n' % natom + out_coord

    return out_coord

def write_individual_xyz(atm, crd, box):
    """
    atm     a list of atoms, shape -> N_unit * N_atoms_per_unit
    crd     a list of coord, shape -> N_unit * N_atom_per_unit * 3
    """

    natom = 0
    ref = np.mean(crd[0], axis=0)
    out_coord = ''
    for n, at in enumerate(atm):
        natom += len(at)
        coord = np.array(crd[n])
        center = np.mean(coord, axis=0, keepdims=True)
        coord += check_neighbor(box, center - ref) - center
        for m, cr in enumerate(coord):
            a = at[m]
            x, y, z = cr
            out_coord += '%-5s %24.16f %24.16f %24.16f\n' % (a, x, y, z)

    out_coord = '%s\nGeom \n' % natom + out_coord + '\n'

    return out_coord

def read_traj(file):
    natom = int(file[0])
    traj = []
    for n, line in enumerate(file):
        if 'Time' in line:
            coord = file[n + 1: n + 1 + natom]
            traj.append(coord)

    return traj


def read_box(file, nstep):
    box = []
    for n, line in enumerate(file):
        if 'BOX' in line:
            box_size = file[n + 1: n + 4]
            box_size = np.array([x.split()[0:2] for x in box_size]).astype(float)
            box.append(box_size)

    dstep = nstep - len(box)

    if dstep > 0:
        box += [box[-1] for _ in range(dstep)]
    else:
        box = box[:nstep]

    return box


def main():
    with open(sys.argv[1], 'r') as infile:
        file = infile.read().splitlines()

    with open(sys.argv[1].replace('.xyz', '.box'), 'r') as infile:
        box_file = infile.read().splitlines()

    print('Reading file %s' % sys.argv[1])

    traj = read_traj(file)
    box = read_box(box_file, len(traj))
    output = ''
    for n, trj in enumerate(traj):
        if n + 1 in save_snapshots:
            system_info['save'] = True
        else:
            system_info['save'] = False

        results = traj_analysis(n + 1, trj, box[n], system_info)
        output += results

    with open('log', 'w') as out:
        out.write(output)

    print('Done')


def prepare_coord(idx, file, box, info):
    """
    file        the xyz of a snapshot
    box         the box size
    info        system info
    """
    name = info['name']
    atoms = info['atoms']
    number = info['number']
    layer = info['layer']
    cell = info['cell']
    uniq = info['unique']
    save = info['save']

    xhi, yhi, zhi = np.abs(box[:, 1] - box[:, 0])

    symbol = {}
    coord = {}
    count = 0
    for key in name:
        natom = atoms[key]
        nmol = number[key]
        sym, crd = read_xyz(file[count: count + natom * nmol], box, natom, nmol)
        symbol[key] = sym
        coord[key] = crd
        count += natom * nmol
        print('Unit: %10s number: %10s atoms: %10s' % (key, nmol, natom))

    if save:
        print('Saving layers ')
        for key, val in layer.items():
            layer_sym = []
            layer_crd = []
            for k in val:
                na, nb = cell[key]
                ua, ub = uniq[key]
                sym = symbol[k]
                crd = coord[k]

                if ua > 0:
                    crd = complete_edge(crd, xhi, yhi, na, nb, ua, ub)

                layer_sym += [sym]
                layer_crd += [crd]

            output = write_xyz(layer_sym, layer_crd)
            with open('layer_%s_%s.xyz' % (key, idx), 'w') as out:
                out.write(output)

    return symbol, coord


def traj_analysis(idx, file, box, info):
    print('\nSnapshot %s' % idx)
    results = ''
    print('\nPreparing coordinates\n')
    symbol, coord = prepare_coord(idx, file, box, info)

    print('\nCoverage analysis\n')
    # compute coverage of molecules
    output, record_coord = coverage(idx, symbol, coord, box, info)
    results += output

    print('\nRadial distance analysis\n')
    results += compute_rdf(idx, coord, box, info)

    print('\nOligomer analysis\n')
    results += compute_olig(idx, symbol, coord, box, info)

    print('\nDistance analysis\n')
    results += compute_mindist(idx, symbol, coord, box, info)

    print('\nAngle analysis\n')
    results += compute_angle(idx, coord, record_coord, box, info)

    return results


if __name__ == '__main__':
    main()
