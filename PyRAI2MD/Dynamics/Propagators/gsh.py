######################################################
#
# PyRAI2MD 2 module for global (generalized) surface hopping
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import sys
import numpy as np

from PyRAI2MD.Dynamics.Propagators.tsh_helper import adjust_velo


def gsh(traj):
    """ Computing the fewest switches surface hopping
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            traj             class       trajectory class

        Return:              Type:
            At               ndarray     the present state density matrix
            Ht               ndarray     the present energy matrix (model Hamiltonian)
            Dt               ndarray     the present nonadiabatic matrix
            Vt               ndarray     the adjusted velocity after surface hopping
            hoped            int         surface hopping decision
            old_state        int         the last state
            state            int         the new state

    """

    itr = traj.itr
    nstate = traj.nstate
    state = traj.state
    verbose = traj.verbose
    v = traj.velo
    m = traj.mass
    e = traj.energy
    statemult = traj.statemult
    maxhop = traj.maxh
    adjust = traj.adjust
    reflect = traj.reflect

    # random number
    z = np.random.uniform(0, 1)

    # initialize return values
    old_state = state
    new_state = state
    hoped = 0
    vt = v

    # initialize state index and order
    stateindex = np.argsort(e)
    stateorder = np.argsort(e).argsort()

    # compute surface hopping probability
    if itr > 2:

        # array of approximate NAC matrix for the same spin multiplicity, unity for different spin
        n = np.ones([nstate, v.shape[0], v.shape[1]])

        # array of hopping probability
        g = np.zeros(nstate)

        # accumulated probability
        gsum = 0

        target_spin = statemult[state - 1]

        for i in range(nstate):

            # skip the present state
            if i == state - 1:
                continue

            state_spin = statemult[i]

            if state_spin == target_spin:
                p, n[i] = internal_conversion_prob(i, traj)
            else:
                p = intersystem_crossing_prob(i, traj)

            g[i] += p

        event = 0
        for j in range(nstate):
            gsum += g[stateindex[j]]
            nhop = np.abs(stateindex[j] - state + 1)
            if gsum > z and 0 < nhop <= maxhop:
                new_state = stateindex[j] + 1
                event = 1
                break

        # if surface hopping event has occurred
        if event == 1:
            # Velocity must be adjusted because hop has occurred
            vt, frustrated = adjust_velo(e[old_state - 1], e[state - 1], v, m, n[state - 1], adjust, reflect)

            # if hop is frustrated, revert the new state to old state
            if frustrated == 1:
                state = old_state
                hoped = 2
            else:
                state = new_state
                hoped = 1

        summary = ''
        for n in range(nstate):
            summary += '    %-5s %-5s %-5s %12.8f\n' % (n + 1, statemult[n], stateorder[n] + 1, g[n])

        info = """
    Random number:           %12.8f
    Accumulated probability: %12.8f
    state mult  level   probability
%s
    """ % (z, gsum, summary)

    else:
        info = '  No surface hopping is performed'

    # allocate zeros vector for population state density
    at = np.zeros([nstate, nstate])

    # assign state density at current state to 1
    at[new_state - 1, new_state - 1] = 1

    # Current energy matrix
    ht = np.diag(e)

    # Current non-adiabatic matrix
    dt = np.zeros([nstate, nstate])

    if itr > 2 and verbose >= 2:
        print(info)

    return at, ht, dt, vt, hoped, old_state, state, info


def internal_conversion_prob(i, traj):
    """ Computing the probability of internal conversion
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            i                int         computing state
            traj             class       trajectory class

        Return:              Type:
            P                float       surface hopping probability
            N                ndarray     approximate non-adiabatic coupling vectors

    """

    state = traj.state
    v = traj.velo
    m = traj.mass
    e = traj.energy
    ep = traj.energy1
    epp = traj.energy2
    g = traj.grad
    gpp = traj.grad2
    r = traj.coord
    rp = traj.coord1
    rpp = traj.coord2
    ekinp = traj.kinetic1
    gap = traj.gap
    test = 0

    # determine the energy gap by taking absolute value
    del_e = np.abs([e[i] - e[state - 1], ep[i] - ep[state - 1], epp[i] - epp[state - 1]])

    # total energy in the system at time t2 (t)
    etotp = ep[state - 1] + ekinp

    # average energy in the system over time period
    ex = (ep[i] + ep[state - 1]) / 2

    # early stop if it does not satisfy surface hopping condition
    if np.argmin(del_e) != 1 or del_e[1] > gap / 27.211396132 or etotp - ex < 0:
        p = 0
        nac = np.zeros(v.shape)
        return p, nac

    de = del_e[1]
    # Implementation of EQ 7
    begin_term = (-1 / (r - rpp))

    arg_min = np.argmin([i, state - 1])
    arg_max = np.argmax([i, state - 1])

    f1_grad_manip_1 = (g[arg_min]) * (rp - rpp)
    f1_grad_manip_2 = (gpp[arg_max]) * (rp - r)

    f_ia_1 = begin_term * (f1_grad_manip_1 - f1_grad_manip_2)

    if test == 1:
        print('IC  EQ 7 R & Rpp: %s %s' % (r, rpp))
        print('IC  EQ 7 begin term: %s' % begin_term)
        print('IC  EQ 7 arg_max/min: %s %s' % (arg_max, arg_min))
        print('IC  EQ 7 f1_1/f1_2: %s %s' % (f1_grad_manip_1, f1_grad_manip_1))
        print('IC  EQ 7 done, F_1a_1: %s' % f_ia_1)

    # Implementation of EQ 8
    f2_grad_manip_1 = (g[arg_max]) * (rp - rpp)
    f2_grad_manip_2 = (gpp[arg_min]) * (rp - r)
    f_ia_2 = begin_term * (f2_grad_manip_1 - f2_grad_manip_2)

    if test == 1:
        print('IC  EQ 8 done, F_1a_2: %s' % f_ia_2)

    # approximate nonadiabatic (vibronic) couplings, which are
    # left out in BO approximation
    nac = (f_ia_2 - f_ia_1) / (m ** 0.5)
    nac = nac / (np.sum(nac ** 2) ** 0.5)

    if test == 1:
        print('IC  Approximate NAC done: %s' % nac)

    # EQ 4, EQ 5
    # F_A = ((F_ia_2 - F_ia_1) / mu)**0.5
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / m) ** 0.5

    # F_B = (abs(F_ia_2 * F_ia_1) / mu**0.5)
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1) / m)) ** 0.5

    if test == 1:
        print('IC  EQ 4 done, F_A: %s' % f_a)
        print('IC  EQ 5 done, F_B: %s' % f_b)

    # compute a**2 and b**2 from EQ 1 and EQ 2
    # ---- note: dE = 2Vx AND h_bar**2 = 1 in Hartree atomic unit
    a_squared = (f_a * f_b) / (2 * de ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * de))
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))

    if test == 1:
        print('IC  EQ 1 & 2 done, a^2, b^2: %s %s' % (a_squared, b_squared))
        print('IC  Compute F sign done: %s' % sign)

    # sign of slope determines computation of surface
    # hopping probability P (eq 3)
    pi_over_four_term = -(np.pi / (4 * a_squared ** 0.5))
    b_in_denom_term = (2 / (b_squared + (np.abs(b_squared ** 2 + sign)) ** 0.5))
    p = np.exp(pi_over_four_term * b_in_denom_term ** 0.5)

    if test == 1:
        print('IC  P numerator done: %s' % pi_over_four_term)
        print('IC  P denominator done: %s' % b_in_denom_term)
        print('IC  P done: %s' % p)

    return p, nac


def intersystem_crossing_prob(i, traj):
    """ Computing the probability of intersystem crossing
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2020,22, 11440-11451
        The equations are adapted from C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            i                int         computing state
            traj             class       trajectory class

        Return:              Type:
            P                float       surface hopping probability

    """

    state = traj.state
    soc_coupling = traj.soc_coupling
    soc = traj.last_soc
    m = traj.mass
    e = traj.energy
    ep = traj.energy1
    epp = traj.energy2
    gp = traj.grad1
    ekinp = traj.kinetic1
    gap = traj.gapsoc
    test = 0

    # determine the energy gap and type of crossing
    del_e = [e[i] - e[state - 1], ep[i] - ep[state - 1], epp[i] - epp[state - 1]]
    # parallel = np.sign(delE[0]* delE[2])
    parallel = -1  # assume non-parallel PESs

    # total energy in the system at time t2 (t)
    etotp = ep[state - 1] + ekinp

    # set hopping point energy to target state
    ex = ep[i]

    # early stop if it does not satisfy surface hopping condition
    if np.argmin(np.abs(del_e)) != 1 or np.abs(del_e[1]) > gap / 27.211396132 or etotp - ex < 0:
        p = 0
        return p

    # early stop if it soc was not computed (ignored)
    soc_pair = sorted([state - 1, i])
    if soc_pair not in soc_coupling:
        p = 0
        return p

    # get soc coupling
    soc_pos = soc_coupling.index(soc_pair)
    if len(soc) >= soc_pos + 1:
        soclength = soc[soc_pos]
    else:
        sys.exit(
            '\n  DataNotFoundError\n  PyRAI2MD: looking for spin-orbit coupling between %s and %s' % (state, i + 1))

    v12x2 = 2 * soclength / 219474.6  # convert cm-1 to hartree

    # Implementation of EQ 7
    f_ia_1 = gp[state - 1]
    f_ia_2 = gp[i]

    if test == 1:
        print('ISC EQ 7 done: %s' % f_ia_1)
        print('ISC EQ 8 done: %s' % f_ia_2)

    # EQ 4, EQ 5
    f_a = np.sum((f_ia_2 - f_ia_1) ** 2 / m) ** 0.5
    f_b = np.abs(np.sum((f_ia_2 * f_ia_1) / m)) ** 0.5

    if test == 1:
        print('ISC EQ 4 done, F_A: %s' % f_a)
        print('ISC EQ 5 done, F_B: %s' % f_b)

    # compute a**2 and b**2 from EQ 1 and EQ 2
    # ---- note: V12x2 = 2 * SOC AND h_bar**2 = 1 in Hartree atomic unit
    a_squared = (f_a * f_b) / (2 * v12x2 ** 3)
    b_squared = (etotp - ex) * (f_a / (f_b * v12x2))

    if test == 1:
        print('ISC EQ 1 & 2 done: %s %s' % (a_squared, b_squared))

    # GOAL: determine sign in denominator of improved Landau Zener formula for switching
    # probability at crossing region
    sign = np.sign(np.sum(f_ia_1 * f_ia_2))

    if test == 1:
        print('ISC Compute F sign done: %s' % sign)

    # hopping probability P (eq 3)
    pi_over_four_term = -(np.pi / (4 * a_squared ** 0.5))
    b_in_denom_term = (2 / (b_squared + (np.abs(b_squared ** 2 + sign)) ** 0.5))
    p = np.exp(pi_over_four_term * b_in_denom_term ** 0.5)

    if test == 1:
        print('LZ-P numerator done: %s' % pi_over_four_term)
        print('LZ-P denominator done: %s' % b_in_denom_term)
        print('LZ-P done: %s' % p)
        print("""parallel crossing: %s
 1 - P / (P + 1) = %s
 1 - P           = %s
""" % (parallel, 1 - p / (p + 1), 1 - p))

    if parallel == 1:
        p = 1 - p / (p + 1)
    else:
        p = 1 - p

    return p
