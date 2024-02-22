######################################################
#
# PyRAI2MD 2 module for utility tools - geometrical derivatives
#
# Author Jingbai Li
# Feb 22 2024
#
######################################################

import numpy as np

def BND(xyz, var, grad=False):
    ## This function calculate distance
    ## a<->b

    a, b = var[0:2]
    v1 = xyz[a]
    v2 = xyz[b]
    v = v1 - v2
    r = np.linalg.norm(v)

    if not grad:
        return r
    else:
        g = np.zeros_like(xyz)
        dv = v / r

        g[a] = dv
        g[b] = -dv
        return r, g

def AGL(xyz, var, grad=False):
    ## This function calculate angle
    ## a<-b->c

    a, b, c = var[0:3]
    v1 = xyz[a]
    v2 = xyz[b]
    v3 = xyz[c]
    r1 = v1 - v2
    r2 = v3 - v2
    l1 = np.linalg.norm(r1)
    l2 = np.linalg.norm(r2)
    d = np.dot(r1, r2) / l1 / l2
    alpha = np.arccos(d) * 57.2958

    if not grad:
        return alpha
    else:
        g = np.zeros_like(xyz)
        f = -57.2958 / (1 - d ** 2) ** 0.5 / l1 / l2
        dv1 = f * (r2 - r1 * d * l2 / l1)
        dv3 = f * (r1 - r2 * d * l1 / l2)
        dv2 = -dv1 - dv3

        g[a] = dv1
        g[b] = dv2
        g[c] = dv3
        return alpha, g


def DHD(xyz, var, grad=False):
    ## This function calculate dihedral angle
    ##   n1    n2
    ##    |    |
    ## a<-b-><-c->d

    a, b, c, d = var[0:4]
    v1 = xyz[a]
    v2 = xyz[b]
    v3 = xyz[c]
    v4 = xyz[d]
    r1 = v1 - v2
    r2 = v3 - v2
    r3 = v4 - v3
    n1 = np.cross(r1, r2)
    n2 = np.cross(r2, r3)
    l1 = np.linalg.norm(n1)
    l2 = np.linalg.norm(n2)
    dd = np.dot(n1, n2) / l1 / l2
    beta = np.arccos(dd) * 57.2958
    sigma = np.sign(np.dot(n1, r3))
    beta *= sigma

    if not grad:
        return beta
    else:
        g = np.zeros_like(xyz)
        f = sigma * -57.2958 / (1 - dd ** 2) ** 0.5 / l1 / l2
        dn1 = f * (n2 - n1 * dd * l2 / l1)
        dn2 = f * (n1 - n2 * dd * l1 / l2)

        r4 = v1 - v3
        r5 = -r1
        r6 = v2 - v4

        g[a] = np.cross(r2, dn1)
        g[b] = np.cross(r4, dn1) + np.cross(r3, dn2)
        g[c] = np.cross(r5, dn1) + np.cross(r6, dn2)
        g[d] = np.cross(r2, dn2)

        return beta, g
