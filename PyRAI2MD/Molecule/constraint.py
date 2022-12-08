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
    """ Molecular property class

            Parameters:          Type:
                keywords         dict        trajectory keyword list

            Attribute:           Type:
                alpha            int         exponential
                ellipsoid        list        a list of radius along x, y, and z-axis

            Function:            Returns:
                apply_potential  self        apply external potential then update energy and gradients
                freeze_atom      self        zero out gradients of the frozen atoms

        """

    def __init__(self, keywords=None):

        constrained_atoms = keywords['molecule']['constrain']
        frozen_atoms = keywords['molecule']['freeze']
        cavity = keywords['molecule']['cavity']
        center_frag = keywords['molecule']['center']
        compressor = keywords['molecule']['compress']

        self.alpha = np.amax([keywords['molecule']['factor'], 2])
        self.shape = keywords['molecule']['shape']

        if len(frozen_atoms) > 0:
            self.has_frozen = True
            self.frozen_atoms = np.array(frozen_atoms) - 1
        else:
            self.has_frozen = False
            self.frozen_atoms = np.zeros(0)

        if len(center_frag) > 0:
            self.has_center = True
            self.center_frag = np.array(center_frag) - 1
        else:
            self.has_center = False
            self.center_frag = np.zeros(0)

        if len(constrained_atoms) > 0 and self.has_center:
            self.has_constrained = True
            self.constrained_atoms = np.array(constrained_atoms) - 1
        else:
            self.has_constrained = False
            self.constrained_atoms = np.zeros(0)

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

    def _polynomial_potential(self, coord, itr):
        cavity = np.ones_like(coord) * self.cavity

        if self.has_compressor:
            if itr < self.pos:
                cavity *= 1 + self.dr * itr
            else:
                cavity *= 1 + self.dr * self.pos

        if self.shape == 'ellipsoid':
            r_over_r0 = np.sum(coord ** 2 / cavity ** 2, axis=1, keepdims=True)  # elementwise divide then atom-wise sum
        else:  # cuboid
            r_over_r0 = coord ** 2 / cavity ** 2  # elementwise divide

        energy = np.sum(r_over_r0 ** (self.alpha / 2))
        vec = self.alpha * coord / cavity ** 2  # element-wise divide
        scale = r_over_r0 ** (self.alpha / 2 - 1)
        grad = vec * scale

        return energy, grad

    def apply_potential(self, traj):
        if not self.has_potential:
            return traj

        if traj.record_center:
            center = traj.center
        else:
            if self.has_center:
                center_frag_coord = traj.coord[self.center_frag]
            else:
                center_frag_coord = traj.coord

            center = np.mean(center_frag_coord, axis=0)
            traj.center = np.copy(center)
            traj.record_center = True

        shifted_coord = traj.coord - center

        if self.has_constrained:
            ext_energy, ext_grad = self._polynomial_potential(shifted_coord[self.constrained_atoms], traj.itr)
            traj.energy += ext_energy  # numpy can automatic broad cast ext_energy
            traj.grad[:, self.constrained_atoms, :] += ext_grad  # numpy can automatic broad cast ext_grad
        else:
            ext_energy, ext_grad = self._polynomial_potential(shifted_coord, traj.itr)
            traj.energy += ext_energy
            traj.grad += ext_grad

        return traj

    def freeze_atom(self, traj):
        if self.has_frozen:
            traj.grad[:, self.frozen_atoms, :] = np.zeros_like(traj.grad[:, self.frozen_atoms, :])

        return traj
