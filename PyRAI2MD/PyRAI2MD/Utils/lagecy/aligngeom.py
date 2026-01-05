######################################################
#
# PyRAI2MD 2 module for aligning molecular structures
#
# Author Jingbai Li
# May 21 2021
#
######################################################
"""
import numpy as np
from numpy import linalg as la
# from scipy.optimize import linear_sum_assignment

def compute_rmsd(atoms, xyz, ref):
    ## This function calculate RMSD between product and reference
    ## This function call kabsch to reduce RMSD between product and reference
    ## This function call hungarian to align product and reference

    xyz_atoms = [x for x in atoms]
    ref_atoms = [x for x in atoms]
    xyz -= xyz.mean(axis=0)  # translate to the centroid
    ref -= ref.mean(axis=0)  # translate to the centroid

    swap = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]])

    reflection = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1]])

    order = []
    rmsd = []
    for s in swap:
        for r in reflection:
            tri_atoms = [x for x in ref_atoms]
            tri = np.array([x for x in ref])
            tri = tri[:, s]
            tri = np.dot(tri, np.diag(r))
            tri -= tri.mean(axis=0)
            int_xyz = inertia(xyz_atoms, xyz)
            int_tri = inertia(tri_atoms, tri)
            rot1 = rotate(int_xyz, int_tri)
            rot2 = rotate(int_xyz, -int_tri)
            tri1 = np.dot(tri, rot1)
            tri2 = np.dot(tri, rot2)
            order1 = hungarian(xyz_atoms, tri_atoms, xyz, tri1)
            order2 = hungarian(xyz_atoms, tri_atoms, xyz, tri2)
            rmsd1 = kabsch(xyz, tri[order1])
            rmsd2 = kabsch(xyz, tri[order2])
            order += [order1, order2]
            rmsd += [rmsd1, rmsd2]
    pick = np.argmin(rmsd)
    rmsd = rmsd[pick]
    # order = order[pick]
    # ref = ref[order]

    return rmsd

def kabsch(p, q):
    ## This function use Kabsch algorithm to reduce RMSD by rotation

    c = np.dot(np.transpose(p), q)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:  # ensure right-hand system
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    p = np.dot(p, u)
    diff = p - q
    n = len(p)
    return np.sqrt((diff * diff).sum() / n)

def inertia(xyz, mass):
    ## This function calculate principal axis

    xyz = np.array([i for i in xyz])  # copy the array to avoid changing it
    mass = np.array(mass).reshape(-1)
    xyz -= np.average(xyz, weights=mass, axis=0)
    xx = 0.0
    yy = 0.0
    zz = 0.0
    xy = 0.0
    xz = 0.0
    yz = 0.0
    for n, i in enumerate(xyz):
        xx += mass[n] * (i[1] ** 2 + i[2] ** 2)
        yy += mass[n] * (i[0] ** 2 + i[2] ** 2)
        zz += mass[n] * (i[0] ** 2 + i[1] ** 2)
        xy += -mass[n] * i[0] * i[1]
        xz += -mass[n] * i[0] * i[2]
        yz += -mass[n] * i[1] * i[2]

    i = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    eigval, eigvec = np.linalg.eig(i)

    return eigvec[np.argmax(eigval)]

def rotate(p: np.ndarray, q: np.ndarray):
    ## This function calculate the matrix rotate p onto q

    if (p == q).all():
        return np.eye(3)
    elif (p == -q).all():
        # return a rotation of pi around the y-axis
        return np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    else:
        v = np.cross(p, q)
        s = np.linalg.norm(v)
        c = np.vdot(p, q)
        vx = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        return np.eye(3) + vx + np.dot(vx, vx) * ((1. - c) / (s * s))


def hungarian(xyz_atoms, ref_atoms, xyz, ref):
    ## This function use hungarian algorithm to align xyz onto ref
    ## This function call linear_sum_assignment from scipy to solve hungarian problem
    ## This function call inertia to find principal axis
    ## This function call rotate xyz onto aligned ref

    unique_atoms = np.unique(xyz_atoms)

    reorder = np.zeros(len(ref_atoms), dtype=int)
    for atom in unique_atoms:
        xyz_idx = []
        ref_idx = []

        for n, p in enumerate(xyz_atoms):
            if p == atom:
                xyz_idx.append(n)
        for m, q in enumerate(ref_atoms):
            if q == atom:
                ref_idx.append(m)

        xyz_idx = np.array(xyz_idx)
        ref_idx = np.array(ref_idx)
        a = xyz[xyz_idx]
        b = ref[ref_idx]
        ab = np.array([[la.norm(val_a - val_b) for val_b in b] for val_a in a])
        a_idx, b_idx = linear_sum_assignment(ab)
        reorder[xyz_idx] = ref_idx[b_idx]

    return reorder


def align_geom(x, geom_pool):
    ## This function align a geometry with all train data geometries to find most similar one 
    atoms = np.array(x)[:, 0].astype(str)
    xyz = np.array(x)[:, 1: 4].astype(float)
    similar = [compute_rmsd(atoms, xyz, np.array(geom)[:, 1: 4].astype(float)) for geom in geom_pool]

    return np.argmin(similar), np.amin(similar)
"""