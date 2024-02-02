######################################################
#
# PyRAI2MD 2 module for constraining molecule
#
# Author Jingbai Li
# Sep 27 2022
#
######################################################

import numpy as np


class Constraint:
    """ Molecular constraint class

            Parameters:          Type:
                keywords         dict        trajectory keyword list
                natom            int         number of atoms

            Attribute:           Type:
                alpha            int         exponential
                ellipsoid        list        a list of radius along x, y, and z-axis
                groups           list        a list of grouped atoms in sequential,
                                             [[4,3],[5,6]] means four groups of 3 atoms and 5 groups of 6 atoms

            Function:            Returns:
                apply_potential  self        apply external potential then update energy and gradients
                freeze_atom      self        zero out gradients of the frozen atoms

        """

    def __init__(self, keywords=None, natom=0, mass=None):

        constrained_atoms = keywords['molecule']['constrain']
        frozen_atoms = keywords['molecule']['freeze']
        cavity = keywords['molecule']['cavity']
        center_frag = keywords['molecule']['center']
        compressor = keywords['molecule']['compress']
        groups = keywords['molecule']['groups']

        self.alpha = np.amax([keywords['molecule']['factor'], 2])
        self.shape = keywords['molecule']['shape']
        self.mass = mass
        self.center_type = keywords['molecule']['center_type']

        if len(frozen_atoms) > 0:
            self.has_frozen = True
            self.frozen_atoms = np.array(frozen_atoms)
        else:
            self.has_frozen = False
            self.frozen_atoms = np.zeros(0)

        if len(center_frag) > 0:
            self.has_center = True
            self.center_frag = np.array(center_frag)
        else:
            self.has_center = False
            self.center_frag = np.zeros(0)

        if len(constrained_atoms) > 0 and self.has_center:
            self.constrained_atoms = np.array(constrained_atoms)
        else:
            # if constrain or center is not defined, put all atoms in constrain
            self.constrained_atoms = np.arange(natom)

        if len(groups) == 0:
            # if groups is not defined, put each constrained atom in a single group
            self.groups = np.array([[len(self.constrained_atoms), 1]]).astype(int)
        else:
            self.groups = np.array(groups).astype(int)

        self.group_map, self.group_idx = self._gen_group_map()
        self.group_mass = np.zeros_like(mass)
        np.add.at(self.group_mass, self.group_map, self.mass[self.group_idx])
        self.group_reduced_mass = self.mass / self.group_mass

        if len(cavity) == 1:
            self.has_potential = True
            self.cavity = np.array([cavity[0], cavity[0], cavity[0]]).reshape((1, 3))

        elif len(cavity) == 2:
            self.has_potential = True
            self.cavity = np.array([cavity[0], cavity[1], cavity[1]]).reshape((1, 3))

        elif len(cavity) >= 3:
            self.has_potential = True
            self.cavity = np.array([cavity[0], cavity[1], cavity[2]]).reshape((1, 3))

        else:
            self.has_potential = False
            self.cavity = np.ones(3)

        if len(compressor) >= 2:
            self.has_compressor = True
            self.dr = (compressor[0] - 1) / compressor[1]
            self.pos = compressor[1]
        else:
            self.has_compressor = False
            self.dr = 0
            self.pos = 0

    def _gen_group_map(self):
        # generate a map for fast summation of molecular coordinates and mass

        num_constrained_atoms = len(self.constrained_atoms)
        groups_natom = np.cumsum(self.groups[:, 0] * self.groups[:, 1])
        num_grouped_atoms = groups_natom[-1]

        if num_constrained_atoms != num_grouped_atoms:
            # stop if the number of atoms does not match
            exit('\n  ValueError\n  PyRAI2MD: %s constrained atoms does not match %s grouped atoms!' % (
                num_constrained_atoms, num_grouped_atoms
            ))

        # find the starting index of each group
        starting_index = np.concatenate((np.array([0]), groups_natom[:-1]))

        # expand the atom map
        # e.g. [[0,1,2], [3,4,5]] -> [0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5]
        group_map = np.zeros(0).astype(int)
        # expand the atom index
        # e.g. [[0,1,2], [3,4,5]] -> [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
        group_idx = np.zeros(0).astype(int)
        for n, _ in enumerate(self.groups):
            start = starting_index[n]
            end = groups_natom[n]
            nmol, natom = self.groups[n]
            g_map = self.constrained_atoms[start:end].reshape((nmol, natom)).repeat(natom, axis=0).reshape(-1)
            g_idx = self.constrained_atoms[start:end].repeat(natom)
            group_map = np.concatenate((group_map, g_map))
            group_idx = np.concatenate((group_idx, g_idx))

        return group_map, group_idx

    def _polynomial_potential(self, coord, center, itr):
        # V_i = F_i ** (a / 2)
        # F_i = (X_i - X_0) ** 2/ R ** 2
        # X_i = sum(x_i * m_i)/sum(m_i)
        # dX_i/dx_i = m_i/sum(m_i)  ==> group_reduced_mass
        # dF_i/dX_i = 2 * (X_i - X_0) / R ** 2
        # dV_i/dF_i = (a / 2) * F_i ** (a / 2 - 1)
        # dV_i/dx_i = (a / 2) * F_i ** (a / 2 - 1) * 2 * (X_i - X_0) / R ** 2 * m_i/sum(m_i)

        # compute X_i = sum(x_i * m_i)/sum(m_i)
        x = np.zeros_like(coord)
        np.add.at(x, self.group_map, coord[self.group_idx])

        # compute X_i - X_0
        x -= center

        # compute cavity R
        cavity = np.ones_like(coord) * self.cavity

        if self.has_compressor:
            if itr < self.pos:
                cavity *= 1 + self.dr * itr
            else:
                cavity *= 1 + self.dr * self.pos

        # compute F_i
        if self.shape == 'ellipsoid':
            r_over_r0 = np.sum(x ** 2 / cavity ** 2, axis=1, keepdims=True)  # elementwise divide then atom-wise sum
        else:  # cuboid
            r_over_r0 = coord ** 2 / cavity ** 2  # elementwise divide

        # compute V = sum(V_i)
        energy = np.sum(r_over_r0 ** (self.alpha / 2))
        scale = self.alpha * r_over_r0 ** (self.alpha / 2 - 1)
        vec = x * self.group_reduced_mass[self.constrained_atoms] / cavity ** 2  # element-wise divide

        grad = scale * vec

        return energy, grad

    def apply_potential(self, traj):
        if not self.has_potential:
            return traj

        if traj.record_center:
            center = traj.center
        else:
            if self.has_center:
                center_frag_coord = traj.coord[self.center_frag]
                center_mass = self.mass[self.center_frag]
            else:
                center_frag_coord = traj.coord
                center_mass = self.mass

            if self.center_type == 'mass':
                center = np.sum(center_frag_coord * center_mass, axis=0) / np.sum(center_mass)
            else:
                center = np.mean(center_frag_coord, axis=0)

            traj.center = np.copy(center)
            traj.record_center = True

        coord = traj.coord[self.constrained_atoms] * self.group_reduced_mass[self.constrained_atoms]
        ext_energy, ext_grad = self._polynomial_potential(coord, center, traj.itr)
        traj.energy += ext_energy
        traj.grad[:, self.constrained_atoms, :] += ext_grad
        traj.ext_pot = ext_energy

        return traj

    def freeze_atom(self, traj):
        if self.has_frozen:
            traj.grad[:, self.frozen_atoms, :] = np.zeros_like(traj.grad[:, self.frozen_atoms, :])

        return traj


class GeomTracker:
    """ Geometry tracker class

            Parameters:          Type:
                keywords         dict        trajectory keyword list

            Attribute:           Type:
                track_type       str         type of geometric parameters
                track_index      list        atom indices of a list of distances or fragment
                track_thrhd      list        a list of threshold of the tracking parameters

            Function:            Returns:
                apply_potential  self        apply external potential then update energy and gradients
                freeze_atom      self        zero out gradients of the frozen atoms

        """

    def __init__(self, keywords=None):
        self.track_type = keywords['molecule']['track_type']
        self.track_index = np.array(keywords['molecule']['track_index'])
        self.track_thrhd = keywords['molecule']['track_thrhd']

        diff = len(self.track_index) - len(self.track_thrhd)

        if diff > 0:
            add = [self.track_thrhd[-1] for _ in range(diff)]
            self.track_thrhd = self.track_thrhd + add
        else:
            self.track_thrhd = self.track_thrhd[:len(self.track_index)]

    @staticmethod
    def _check_param(coord, src, dst, thrhd):
        a = np.mean(coord[src], axis=0)
        b = np.mean(coord[dst], axis=0)
        d = np.sum((a - b) ** 2) ** 0.5

        if d > thrhd:
            return True, d

        return False, d

    def check(self, traj):
        # track_index [[a, b, c, ...],[d, e, f, ...], ...]
        stop = False

        if self.track_type == 'frag':

            if len(self.track_index) < 2:
                exit('\n  ValueError\n  PyRAI2MD: track_index requires to list of index but found one %s' %
                     self.track_index
                     )

            stop, d = self._check_param(
                coord=traj.coord,
                src=self.track_index[0],
                dst=self.track_index[1],
                thrhd=self.track_thrhd[0]
            )

            info = '  Fragment distance: %8.4f %8.4f %s\n' % (d, self.track_thrhd[0], stop)

        elif self.track_type == 'dist':
            info = ''
            status = []
            for n, indx in enumerate(self.track_index):

                if len(indx) < 2:
                    exit('\n  ValueError\n  PyRAI2MD: track_index requires two indices but found one %s' % indx)

                stop, d = self._check_param(
                    coord=traj.coord,
                    src=[indx[0]],
                    dst=[indx[1]],
                    thrhd=self.track_thrhd[n],
                )

                info += '  %-5s %5s %5s %8.4f %8.4f %s\n' % (
                    n + 1, indx[0] + 1, indx[1] + 1, d, self.track_thrhd[n], stop
                )

                status.append(stop)
            stop = (True in status)

        else:
            stop = False
            info = None

        return stop, info
