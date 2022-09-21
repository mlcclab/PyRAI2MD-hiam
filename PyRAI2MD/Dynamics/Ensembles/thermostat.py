######################################################
#
# PyRAI2MD 2 module for thermostat in NVT ensemble
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import numpy as np


def nose_hoover(traj):
    """ Velocity scaling function in NVT ensemble (Nose Hoover thermostat)

        Parameters:          Type:
            traj             class       trajectory class

        Attribute:           Type:
            natom            int         number of atoms
            temp             float       temperature
            kinetic          float	 kinetic energy
            Vs               list        additional velocity information
            kb               float       Boltzmann's constant
            fs_to_au         float       unit conversion fs to au of time

    """

    natom = traj.natom
    kinetic = traj.kinetic
    temp = traj.temp
    size = traj.size
    vs = traj.vs
    kb = 3.16881 * 10 ** -6
    fs_to_au = 2.4188843265857 * 10 ** -2

    if len(vs) == 0:
        freq = 1 / (22 / fs_to_au)  # 22 fs to au Hz
        q1 = 3 * natom * temp * kb / freq ** 2
        q2 = temp * kb / freq ** 2
        traj.vs = [q1, q2, 0, 0]

    else:
        q1, q2, v1, v2 = vs
        g2 = (q1 * v1 ** 2 - temp * kb) / q2
        v2 += g2 * size / 4
        v1 *= np.exp(-v2 * size / 8)
        g1 = (2 * kinetic - 3 * natom * temp * kb) / q1
        v1 += g1 * size / 4
        v1 *= np.exp(-v2 * size / 8)
        s = np.exp(-v1 * size / 2)

        traj.kinetic *= s ** 2
        traj.velo *= s

        v1 *= np.exp(-v2 * size / 8)
        g1 = (2 * kinetic - 3 * natom * temp * kb) / q1
        v1 += g1 * size / 4
        v1 *= np.exp(-v2 * size / 8)
        g2 = (q1 * v1 ** 2 - temp * kb) / q2
        v2 += g2 * size / 4

        traj.vs = [q1, q2, v1, v2]

    return traj
