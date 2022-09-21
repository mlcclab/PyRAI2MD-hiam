######################################################
#
# PyRAI2MD 2 module for translation and rotation velocity removal
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

import numpy as np
from numpy import linalg as la

def reset_velo(traj):
    """ Removing translation and rotation velocity

        Parameters:          Type:
            traj             class    trajectory class

        Return:              Type:
            traj             class    trajectory class

    """

    itr = traj.itr  # current MD step
    xyz = traj.coord  # cartesian coordinate in angstrom (Nx3)
    velo = traj.velo  # velocity in Eh/Bohr (Nx3)
    gd = traj.graddesc  # gradient descent
    m = traj.mass  # mass matrix in ams unit (Nx1)
    test = 0  # debug mode

    ## in gradient descent, do not reset velocity since they are zero
    if gd == 1:
        return traj

    ## find center of mass and momentum of inertia as principal axis
    ## then project velocity and position vector to principal axis system
    com, paxis, caxis = inertia(xyz, m)
    pvelo = np.dot(velo, paxis)
    pxyz = np.dot(xyz - com, paxis)

    ## find the translation and angular velocity at center of mass
    vcom = get_vcom(velo, m)
    wcom = get_wcom(pxyz, pvelo, m)

    ## first remove the translation with un-projected velocity
    velo1 = remove_vcom(velo, vcom)
    vcom1 = get_vcom(velo1, m)

    ## then project the new velocity to principal axis system
    ## find the angular velocity then remove it
    pvel1 = np.dot(velo1, paxis)
    wcom1 = get_wcom(pxyz, pvel1, m)
    velo2 = remove_wcom(pxyz, pvel1, wcom1, caxis)

    ## compute kinetic energy for original, translation removed, and translation/rotation removed velocity
    k1 = 0.5 * np.sum(m * velo ** 2)
    k2 = 0.5 * np.sum(m * velo1 ** 2)
    k3 = 0.5 * np.sum(m * velo2 ** 2)

    ## scale the new velocity to conserve kinetic energy
    velo_notr = velo2 * (k1 / k3) ** 0.5

    if test == 1:
        vcom2 = get_vcom(velo2, m)
        pvel2 = np.dot(velo2, paxis)
        wcom2 = get_wcom(pxyz, pvel2, m)

        # print('Original')
        # print('Principle axis')
        # print(paxis)
        # print('Cartesian axis')
        # print(caxis)
        print('Iter: ', itr)
        print('Original: VCOM ', vcom, 'WCOM ', wcom, 'K ', k1)
        print('Rm Trans: VCOM ', vcom1, 'WCOM ', wcom1, 'K ', k2)
        print('Rm Tr/Rr: VCOM ', vcom2, 'WCOM ', wcom2, 'K ', k3)
        print('E_Tr ', k1 - k2, 'E_Rr ', k2 - k3)

    traj.velo = np.copy(velo_notr)

    return traj


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


def check_mirror(coord, i):
    coord1 = np.dot(coord, i)
    coord2 = np.dot(coord, -i)
    rmsd1 = kabsch(coord, coord1)
    rmsd2 = kabsch(coord, coord2)

    if rmsd1 <= rmsd2:
        im = i
    else:
        im = -i

    return im


def inertia(xyz, m):
    ## this function compute momentum of inertia as principal axis

    com = np.sum(m * xyz, axis=0) / np.sum(m)
    body = xyz - com

    ## initialize momentum of inertia (3x3)
    i = np.zeros([3, 3])

    ## compute momentum of inertia
    for n, i in enumerate(body):
        i += m[n][0] * (np.sum(i ** 2) * np.diag(np.ones(3)) - np.outer(i, i))

    ## compute principal axis
    eigval, eigvec = np.linalg.eig(i)
    prin_axis = check_mirror(xyz, eigvec)
    cart_axis = la.inv(prin_axis)

    return com, prin_axis, cart_axis


def get_com(xyz, m):
    ## This function compute center of mass

    com = np.sum(m * xyz, axis=0) / np.sum(m)

    return com


def get_vcom(velo, m):
    ## This function compute velocity at center of mass

    vcom = np.sum(m * velo, axis=0) / np.sum(m)

    return vcom


def remove_vcom(velo, vcom):
    ## This function remove velocity at center of mass from velocity on each atom

    new_velo = velo - vcom

    return new_velo


def get_wcom(xyz, velo, m):
    ## This function compute angular velocity about momentum of inertia as principal axis
    ## xyz and velo are projected to principal axis

    ## initial average angular velocity matrix and average momentum of inertia
    wcom = np.zeros(3)
    jt = np.zeros([3, 3])

    ## compute angular velocity and momentum of inertia
    for n, i in enumerate(xyz):
        w = np.cross(i, velo[n]) / np.sum(i ** 2)
        j = m[n][0] * (np.sum(i ** 2) * np.diag(np.ones(3)) - np.outer(i, i))
        wcom += np.dot(j, w)
        jt += j

    wcom = np.dot(la.inv(jt), wcom)

    return wcom


def remove_wcom(xyz, velo, wcom, cart_axis):
    ## This function removes angular velocity about momentum of inertia as principal axis from velocity on each atom
    ## xyz and velo are	projected to principal axis
    ## new_velo is projected back to reference axis

    new_velo = []
    for n, i in enumerate(velo):
        linear = np.cross(wcom, xyz[n])
        radial = i - linear
        new_velo.append(radial)

    new_velo = np.array(new_velo)
    new_velo = np.dot(new_velo, cart_axis)

    return new_velo
