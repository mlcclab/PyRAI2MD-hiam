######################################################
#
# PyRAI2MD 2 module for setting ensemble
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

from PyRAI2MD.Dynamics.Ensembles.microcanonical import nve
from PyRAI2MD.Dynamics.Ensembles.thermostat import nose_hoover

def ensemble(traj):
    """ Setting trajectory ensemble

        Parameters:          Type:
            traj             class       trajectory class

        Attribute:           Type:
            thermo           str         choose an ensemble to apply thermostat or not
            thermodelay      int         delay time step for applying thermostat
            state            int         the present state

        Return:              Type:
            traj             class       trajectory class

    """

    thermo = traj.thermo
    thermodelay = traj.thermodelay
    state = traj.state

    if thermo == '-1' or thermo.lower() == 'off':
        return traj
    if thermo == '0' or thermo.lower() == 'nve':
        traj = nve(traj)
    elif thermo == '1' or thermo.lower() == 'nvt':
        traj = nose_hoover(traj)
    ## NVE for excited-state, NoseHoover for ground-state after a certain amount of time
    elif thermo == '2' or thermo.lower() == 'nve_nvt':
        if state > 1:
            traj.iter_x = traj.iter
        delay = traj.iter - traj.iter_x
        if state == 1 and delay >= thermodelay:
            traj = nose_hoover(traj)
        else:
            traj = nve(traj)
    ## NVE for excited-state without scaling, NoseHoover for ground-state after a certain amount of time
    elif thermo == '3' or thermo.lower() == 'mixednvt':
        if state > 1:
            traj.iter_x = traj.iter
        delay = traj.iter - traj.iter_x
        if state == 1 and delay >= thermodelay:
            traj = nose_hoover(traj)
        else:
            return traj

    ## TODO add barostat

    return traj
