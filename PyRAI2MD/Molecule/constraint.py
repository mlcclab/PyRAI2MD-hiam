######################################################
#
# PyRAI2MD 2 module for constraining molecule
#
# Author Jingbai Li
# Sep 27 2022
#
######################################################

import numpy as np

from PyRAI2MD.Utils.geom_tools import BND
from PyRAI2MD.Utils.geom_tools import AGL
from PyRAI2MD.Utils.geom_tools import DHD


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

        tbond = keywords['molecule']['tbond']
        tangle = keywords['molecule']['tangle']
        tdihedral = keywords['molecule']['tdihedral']

        constrained_atoms = keywords['molecule']['constrain']
        frozen_atoms = keywords['molecule']['freeze']
        cavity = keywords['molecule']['cavity']
        center_frag = keywords['molecule']['center']
        compressor = keywords['molecule']['compress']
        groups = keywords['molecule']['groups']

        # constraining potential on bond, angle, and dihedral
        self.cbond = keywords['molecule']['cbond']
        self.cangle = keywords['molecule']['cangle']
        self.cdihedral = keywords['molecule']['cdihedral']
        self.fbond = keywords['molecule']['fbond']
        self.fangle = keywords['molecule']['fangle']
        self.fdihedral = keywords['molecule']['fdihedral']
        self.target_bond = np.zeros(0)
        self.target_angle = np.zeros(0)
        self.target_dihedral = np.zeros(0)

        if len(self.cbond) > 0:
            if len(tbond) > 0:
                self.target_bond = self._match_list(self.cbond, tbond)
            self.has_cbond = True
        else:
            self.has_cbond = False

        if len(self.cangle) > 0:
            if len(tangle) > 0:
                self.target_angle = self._match_list(self.cangle, tangle)
            self.has_cangle = True
        else:
            self.has_cangle = False

        if len(self.cdihedral) > 0:
            if len(tdihedral) > 0:
                self.target_dihedral = self._match_list(self.cdihedral, tdihedral)
            self.has_cdihedral = True
        else:
            self.has_cdihedral = False

        # constraining potential on wall
        factor = keywords['molecule']['factor']
        scale = keywords['molecule']['scale']
        shape = keywords['molecule']['shape']
        self.alpha = [np.amax([x, 2]) for x in factor]
        self.pre_factor = self._match_list(self.alpha, scale)
        self.shape = self._match_list(self.alpha, shape)
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
            self.has_cavity = True
            self.cavity = np.array([cavity[0], cavity[0], cavity[0]]).reshape((1, 3))

        elif len(cavity) == 2:
            self.has_cavity = True
            self.cavity = np.array([cavity[0], cavity[1], cavity[1]]).reshape((1, 3))

        elif len(cavity) >= 3:
            self.has_cavity = True
            self.cavity = np.array([cavity[0], cavity[1], cavity[2]]).reshape((1, 3))

        else:
            self.has_cavity = False
            self.cavity = np.ones(3)

        if len(compressor) >= 2:
            self.has_compressor = True
            self.dr = (compressor[0] - 1) / compressor[1]
            self.pos = compressor[1]
        else:
            self.has_compressor = False
            self.dr = 0
            self.pos = 0

    @staticmethod
    def _match_list(tar_list, chk_list):
        num_tar = len(tar_list)
        num_chk = len(chk_list)

        if num_tar > num_chk:
            add_list = np.repeat(chk_list[-1], num_tar - num_chk).tolist()
            out_list = np.array(chk_list + add_list)
        else:
            out_list = np.array(chk_list)[:num_tar]

        return out_list

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

        # loop over potentials
        energy = np.zeros(0)
        grad = np.zeros(0)
        for n, alpha in enumerate(self.alpha):
            pre_factor = self.pre_factor[n]

            # compute F_i
            if self.shape == 'ellipsoid':
                r_over_r0 = np.sum(x ** 2 / cavity ** 2, axis=1, keepdims=True)  # elementwise divide then atom-wise sum
            else:  # cuboid
                r_over_r0 = x ** 2 / cavity ** 2  # elementwise divide

            # compute V = sum(V_i)
            energy += pre_factor * np.sum(r_over_r0 ** (alpha / 2))
            scale = pre_factor * alpha * r_over_r0 ** (alpha / 2 - 1)
            vec = x * self.group_reduced_mass[self.constrained_atoms] / cavity ** 2  # element-wise divide
            grad += scale * vec

        return energy, grad

    def apply_potential(self, traj):
        if self.has_cavity:
            traj = self.apply_cavity(traj)

        if self.has_cbond:
            traj = self.apply_cbond(traj)

        if self.has_cangle:
            traj = self.apply_cangle(traj)

        if self.has_cdihedral:
            traj = self.apply_cdihedral(traj)

        return traj

    def apply_cbond(self, traj):
        if len(traj.target_bond) > 0:
            target_bond = traj.target_bond
        else:
            if len(self.target_bond) > 0:
                target_bond = self.target_bond
            else:
                target_bond = np.array([BND(traj.coord, idx) for idx in self.cbond])

            traj.target_bond = target_bond

        coord = traj.coord
        bond_energy = 0
        bond_grad = np.zeros_like(coord)
        record_bond = []
        for n, idx in enumerate(self.cbond):
            r, g = BND(coord, idx, grad=True)
            record_bond.append(r)
            dr = (r - target_bond[n])
            bond_energy += self.fbond * dr ** 2
            bond_grad += 2 * self.fbond * dr * g

        traj.energy += bond_energy
        traj.grad += bond_grad * 0.529177249
        traj.bond_pot = bond_energy
        traj.record_bond = record_bond

        return traj

    def apply_cangle(self, traj):
        if len(traj.target_angle) > 0:
            target_angle = traj.target_angle
        else:
            if len(self.target_angle) > 0:
                target_angle = self.target_angle
            else:
                target_angle = np.array([AGL(traj.coord, idx) for idx in self.cangle])

            traj.target_angle = target_angle

        coord = traj.coord
        angle_energy = 0
        angle_grad = np.zeros_like(coord)
        record_angle = []
        for n, idx in enumerate(self.cangle):
            r, g = AGL(coord, idx, grad=True)
            record_angle.append(r)
            dr = (r - target_angle[n])
            angle_energy += self.fangle * dr ** 2
            angle_grad += 2 * self.fangle * dr * g

        traj.energy += angle_energy
        traj.grad += angle_grad * 0.529177249
        traj.angle_pot = angle_energy
        traj.record_angle = record_angle

        return traj

    def apply_cdihedral(self, traj):
        if len(traj.target_dihedral) > 0:
            target_dihedral = traj.target_dihedral
        else:
            if len(self.target_dihedral) > 0:
                target_dihedral = self.target_dihedral
            else:
                target_dihedral = np.array([DHD(traj.coord, idx) for idx in self.cdihedral])

            traj.target_dihedral = target_dihedral

        coord = traj.coord
        dihedral_energy = 0
        dihedral_grad = np.zeros_like(coord)
        record_dihedral = []
        for n, idx in enumerate(self.cdihedral):
            r, g = DHD(coord, idx, grad=True)
            record_dihedral.append(r)
            dr = (r - target_dihedral[n])
            if np.abs(dr) > 180:
                dr = (360 - np.abs(dr)) * np.sign(target_dihedral[n])
            dihedral_energy += self.fdihedral * dr ** 2
            dihedral_grad += 2 * self.fdihedral * dr * g

        traj.energy += dihedral_energy
        traj.grad += dihedral_grad * 0.529177249
        traj.dihedral_pot = dihedral_energy
        traj.record_dihedral = record_dihedral

        return traj

    def apply_cavity(self, traj):
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
            elif self.center_type == 'origin':
                center = np.array([0., 0., 0.])
            else:
                center = np.mean(center_frag_coord, axis=0)

            traj.center = np.copy(center)
            traj.record_center = True

        coord = traj.coord[self.constrained_atoms] * self.group_reduced_mass[self.constrained_atoms]
        ext_energy, ext_grad = self._polynomial_potential(coord, center, traj.itr)
        traj.energy += ext_energy
        traj.grad[:, self.constrained_atoms, :] += ext_grad * 0.529177249
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
