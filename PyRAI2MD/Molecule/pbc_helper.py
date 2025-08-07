######################################################
#
# PyRAI2MD 2 module for periodic boundary condition check
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

import numpy as np


def compute_cell(lattice):
    """ Compute primitive vectors from lattice constant

        Parameters:          Type:
            lattice          ndarray    lattice constant

        Return:              Type:
            primitives       ndarray    atom index at high and low level boundary

    """
    a, b, c, alpha, beta, gamma = lattice
    alpha /= 180/np.pi
    beta /= 180/np.pi
    gamma /= 180/np.pi

    x = np.array([1, 0, 0]) * a
    y = np.array([np.cos(gamma), np.sin(gamma), 0]) * b
    z = np.array([
        np.cos(beta),
        (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        (1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
         - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2) ** 0.5 / np.sin(gamma)
    ]) * c
    cell = np.array([x, y, z])

    return cell

def apply_pbc(coord, cell, pbc):
    """ Check periodic boundary condition and adjust coordinates

        Parameters:          Type:
            coord            ndarray    atomic coordinate
            cell             ndarray    lattice vector
            pbc              ndarray    periodic boundary condition

        Return:              Type:
            coord            ndarray    nuclear coordinates

    """
    inv_cell = np.linalg.inv(cell)
    frac = np.dot(coord, inv_cell)
    pbc_axis = np.where(pbc != 0)
    frac[:, pbc_axis] %= 1
    cart = np.dot(frac, cell)

    return cart
