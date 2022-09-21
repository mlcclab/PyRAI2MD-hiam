######################################################
#
# PyRAI2MD 2 module for periodic boundary condition check
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

"""
primitive_vector
[[1,3,4],[1,2,3],[1,2,3]]


lattice_constant
[a,b,c,alpha,beta,gamma]
"""

def compute_primitives(lattice):
    """ Compute primitive vectors from lattice constant

        Parameters:          Type:
            lattice          ndarray    lattice constant

        Return:              Type:
            primitives       ndarray    atom index at high and low level boundary

    """
    primitives = lattice

    ## under construction

    return primitives

def apply_pbc(coord, pbc):
    """ Check periodic boundary condition and adjust coordinates

        Parameters:          Type:
            coord            ndarray    nuclear coordinates
            pbc              ndarray    periodic boundary condition

        Return:              Type:
            coord            ndarray    nuclear coordinates

    """

    ## under construction

    return coord, pbc
