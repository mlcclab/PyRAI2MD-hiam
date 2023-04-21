######################################################
#
# PyRAI2MD 2 module for ab initio molecular dynamics
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import sys
import time
import pickle
import numpy as np

from PyRAI2MD.Dynamics.Propagators.surface_hopping import surfhop
from PyRAI2MD.Dynamics.Ensembles.ensemble import ensemble
from PyRAI2MD.Dynamics.verlet import verlet_i, verlet_ii
from PyRAI2MD.Dynamics.reset_velocity import reset_velo
from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long
from PyRAI2MD.Utils.coordinates import print_coord
from PyRAI2MD.Utils.coordinates import print_charge
from PyRAI2MD.Molecule.constraint import Constraint
from PyRAI2MD.Molecule.constraint import GeomTracker


class AIMD:
    """ Ab initial molecular dynamics class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            job_id           int         trajectory id index
            job_dir          boolean     create a subdirectory

        Attributes:          Type:
            version          str         version information header
            title            str         calculation title
            maxerr_energy    float       maximum energy error threshold
            maxerr_grad      float       maximum gradient error threshold
            maxerr_nac       float       maximum nac error threshold
            maxerr_soc       float       maximum soc error threshold
            silent           int         silent mode for screen output
            verbose          int         verbose level of output information
            direct           int         number of steps to record output
            buffer           int         number of steps to skip output
            record           int         record the history of trajectory
            checkpoint       int         trajectory checkpoint frequency
            restart          int         restart calculation
            addstep          int         number of steps that will be added in restarted calculation
            stop             int         trajectory termination signal
            skipstep         int         number of steps being skipped to write output
            skiptraj         int         number of steps being skipped to save trajectory

        Functions:           Returns:
            run              class       run molecular dynamics simulation

    """

    def __init__(self, trajectory=None, keywords=None, qm=None, job_id=None, job_dir=None):
        self.timing = 0  # I use this to test calculation time

        ## initialize variables
        self.version = keywords['version']
        self.title = keywords['control']['title']
        self.maxerr_energy = keywords['control']['maxenergy']
        self.maxerr_grad = keywords['control']['maxgrad']
        self.maxerr_nac = keywords['control']['maxnac']
        self.maxerr_soc = keywords['control']['maxsoc']
        self.randvelo = keywords['md']['randvelo']
        self.silent = keywords['md']['silent']
        self.verbose = keywords['md']['verbose']
        self.direct = keywords['md']['direct']
        self.buffer = keywords['md']['buffer']
        self.record = keywords['md']['record']
        self.checkpoint = keywords['md']['checkpoint']
        self.restart = keywords['md']['restart']
        self.addstep = keywords['md']['addstep']
        self.stop = 0
        self.skipstep = 0
        self.skiptraj = 0

        ## update calculation title if the id is available
        if job_id is not None:
            self.title = '%s-%s' % (self.title, job_id)

        ## setup molecular dynamics calculation path
        if job_dir is not None:
            self.logpath = '%s/%s' % (os.getcwd(), self.title)
            if not os.path.exists(self.logpath):
                os.makedirs(self.logpath)
        else:
            self.logpath = os.getcwd()

        ## create a constraint object
        self.ext_pot = Constraint(keywords=keywords)
        ## create a geometry tracker object
        self.geom_tracker = GeomTracker(keywords=keywords)
        ## create a trajectory object
        self.traj = trajectory

        ## create an electronic method object
        self.QM = qm

        ## check if it is a restart calculation and if the previous check point pkl file exists
        if self.restart == 1:
            check_f1 = os.path.exists('%s/%s.pkl' % (self.logpath, self.title))
            check_f2 = os.path.exists('%s/%s.log' % (self.logpath, self.title))
            check_f3 = os.path.exists('%s/%s.md.energies' % (self.logpath, self.title))
            check_f4 = os.path.exists('%s/%s.md.xyz' % (self.logpath, self.title))
            check_f5 = os.path.exists('%s/%s.md.velo' % (self.logpath, self.title))
            check_f6 = os.path.exists('%s/%s.sh.energies' % (self.logpath, self.title))
            check_f7 = os.path.exists('%s/%s.sh.xyz' % (self.logpath, self.title))
            check_f8 = os.path.exists('%s/%s.sh.velo' % (self.logpath, self.title))
            checksignal = int(check_f1) + int(check_f2) + int(check_f3) + int(check_f4) + int(check_f5) + \
                          int(check_f6) + int(check_f7) + int(check_f8)

            if checksignal == 8:
                with open('%s/%s.pkl' % (self.logpath, self.title), 'rb') as mdinfo:
                    self.traj = pickle.load(mdinfo)
            else:
                sys.exit("""\n PyRAI2MD: Checkpoint files are incomplete.
  Cannot restart, please consider start it over again, sorry!

                File          Found
                pkl           %s
       	       	log           %s
       	       	md.energies   %s
       	       	md.xyz        %s
       	       	md.velo       %s
       	       	sh.energies   %s
       	       	sh.xyz        %s
                sh.velo       %s
                """ % (check_f1, check_f2, check_f3, check_f4, check_f5, check_f6, check_f7, check_f8))

        ## check if it is a freshly new calculation then create output files
        ## otherwise, the new results will be appended to the existing log in a restart calculation
        elif self.restart == 0 or os.path.exists('%s/%s.log' % (self.logpath, self.title)) is False:
            log = open('%s/%s.log' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.md.energies' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.sh.energies' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.md.xyz' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.sh.xyz' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.md.velo' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.sh.velo' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.md.charge' % (self.logpath, self.title), 'w')
            log.close()
            log = open('%s/%s.sh.charge' % (self.logpath, self.title), 'w')
            log.close()

    def _propagate(self):
        #
        # -----------------------------------------
        #  ---- Initial Kinetic Energy Scaling
        # -----------------------------------------
        #
        if self.traj.itr == 1:
            ## initialize random velocity according to temperature
            if self.randvelo == 1:
                self.traj.random_velo()

            f = 1
            ## add excess kinetic energy in the first step if requested
            if self.traj.excess != 0:
                k0 = self._kinetic_energy(self.traj)
                f = ((k0 + self.traj.excess) / k0) ** 0.5

            ## scale kinetic energy in the first step if requested
            if self.traj.scale != 1:
                f = self.traj.scale ** 0.5

            ## scale kinetic energy to target value in the first step if requested
            if self.traj.target != 0:
                k0 = self._kinetic_energy(self.traj)
                f = (self.traj.target / k0) ** 0.5

            self.traj.velo *= f

        #
        # -----------------------------------------
        #  ---- Trajectory Propagation
        # -----------------------------------------
        #
        #  update previous-previous and previous nuclear properties
        self.traj.update_nu()

        ## update current kinetic energies, coordinates, and gradient
        self.traj = verlet_i(self.traj)

        if self.timing == 1:
            print('verlet', time.time())

        # compute potential energy surface
        self.traj = self._potential_energies(self.traj)

        # apply phase nac correction if requested
        self.traj.phase_correction()

        # apply external potential energy if requested
        self.traj = self.ext_pot.apply_potential(self.traj)

        # freeze selected atom if requested
        self.traj = self.ext_pot.freeze_atom(self.traj)

        if self.timing == 1:
            print('compute_egn', time.time())

        ## update current velocity
        self.traj = verlet_ii(self.traj)

        if self.timing == 1:
            print('verlet_2', time.time())

        self.traj = self._kinetic_energy(self.traj)

        #
        # -----------------------------------------
        #  ---- Velocity Adjustment
        # -----------------------------------------
        #
        #  reset velocity to avoid flying ice cube
        #  end function early if velocity reset is not requested
        if self.traj.reset != 1:
            return None

        ## end function early if	velocity reset step is 0 but iteration is more than 1
        if self.traj.resetstep == 0 and self.traj.itr > 1:
            return None

        ## end function early if velocity reset step is not 0 but iteration is not the multiple of it 
        if self.traj.resetstep != 0:
            if self.traj.itr % self.traj.resetstep != 0:
                return None

        ## finally reset velocity here
        self.traj = reset_velo(self.traj)

    def _potential_energies(self, traj):
        traj = self.QM.evaluate(traj)

        return traj

    @staticmethod
    def _kinetic_energy(traj):
        traj.kinetic = np.sum(0.5 * (traj.mass * traj.velo ** 2))

        return traj

    def _thermodynamic(self):
        self.traj = ensemble(self.traj)

        return self.traj

    def _surfacehop(self):
        ## update previous population, energy matrix, and non-adiabatic coupling matrix
        self.traj.update_el()

        # update current population, energy matrix, and non-adiabatic coupling matrix
        self.traj = surfhop(self.traj)

        return self.traj

    def _reactor(self):
        ## TODO periodically adjust velocity to push reactants toward center

        return self

    def _heading(self):
        state_info = ''.join(['%4d' % (x + 1) for x in range(len(self.traj.statemult))])
        mult_info = ''.join(['%4d' % x for x in self.traj.statemult])

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*


  State order:      %s
  Multiplicity:     %s

  QMMM key:         %s
  QMMM xyz          %s
  Active atoms:     %s
  Inactive atoms:   %s
  Link atoms:       %s
  Highlevel atoms:  %s
  Midlevel atoms:   %s
  Lowlevel atoms:   %s

""" % (
            self.version,
            state_info,
            mult_info,
            self.traj.qmmm_key,
            self.traj.qmmm_xyz,
            self.traj.natom,
            self.traj.ninac,
            self.traj.nlink,
            self.traj.nhigh,
            self.traj.nmid,
            self.traj.nlow
        )

        return headline

    def _chkerror(self):
        ## This function check the errors in energy, force, and NAC
        ## This function stop MD if the errors exceed the threshold
        ## This function stop MD if the qm calculation failed

        if self.traj.err_energy is not None and \
                self.traj.err_grad is not None and \
                self.traj.err_nac is not None and \
                self.traj.err_soc is not None:
            if self.traj.err_energy > self.maxerr_energy or \
                    self.traj.err_grad > self.maxerr_grad or \
                    self.traj.err_nac > self.maxerr_nac or \
                    self.traj.err_soc > self.maxerr_soc:
                self.stop = 1
        elif self.traj.status == 0:
            self.stop = 2

    def _chkpopulation(self):
        ## This function check the state population
        ## This function stop MD if the state population is nan
        ## This function stop MD if the state population exceed 0.01 or below - 0.01

        for p in np.diag(np.real(self.traj.a)):
            if np.isnan(p):
                self.stop = 3
                return self

        if np.amax(np.diag(np.real(self.traj.a))) > 1.01:
            self.stop = 4

        if np.amin(np.diag(np.real(self.traj.a))) < -0.01:
            self.stop = 4

        return self

    def _chkgeom(self):
        ## This function check the geometry
        ## This function stop MD if the geometry satisfies the requirement

        stop, info = self.geom_tracker.check(self.traj)
        self.traj.tracker = info

        if stop:
            self.stop = 5

        return self

    def _chkpoint(self):
        ## record the last a few MD step
        self.traj.record()

        ## prepare a comment line for xyz file
        cmmt = '%s coord %d state %d' % (self.title, self.traj.itr, self.traj.last_state)

        ## prepare the surface hopping section using Molcas output format
        ## add surface hopping information to xyz comment line
        if self.traj.hoped == 0:
            hop_info = ' A surface hopping is not allowed\n  **\n At state: %3d\n' % self.traj.state

        elif self.traj.hoped == 1:
            hop_info = ' A surface hopping event happened\n  **\n From state: %3d to state: %3d *\n' % (
                self.traj.last_state, self.traj.state)
            cmmt += ' to %d CI' % self.traj.state

        elif self.traj.hoped == 2:
            hop_info = ' A surface hopping is frustrated\n  **\n At state: %3d\n' % self.traj.state

        else:
            hop_info = ' A surface hopping is not allowed\n  **\n At state: %3d\n' % self.traj.state

        ## prepare population and potential energy info
        pop = ' '.join(['%28.16f' % x for x in np.diag(np.real(self.traj.a))])
        pot = ' '.join(['%28.16f' % x for x in self.traj.energy])

        ## prepare xyz, velo, and energy info
        xyz_info = '%d\n%s\n%s' % (
            self.traj.natom, cmmt, print_coord(np.concatenate((self.traj.atoms, self.traj.coord), axis=1)))
        velo_info = '%d\n%s\n%s' % (
            self.traj.natom, cmmt, print_coord(np.concatenate((self.traj.atoms, self.traj.velo), axis=1)))
        energy_info = '%20.2f%28.16f%28.16f%28.16f%s\n' % (
            self.traj.itr * self.traj.size,
            self.traj.energy[self.traj.last_state - 1],
            self.traj.kinetic,
            self.traj.energy[self.traj.last_state - 1] + self.traj.kinetic,
            pot)
        if len(self.traj.qm2_charge) > 0:
            charge_info = '%d\n%s\n%s' % (len(self.traj.qm2_charge), cmmt, print_charge(self.traj.qm2_charge))
        else:
            charge_info = ''

        ## prepare logfile info
        log_info = \
            ' Iter: %8d  Ekin = %28.16f au T = %8.2f K dt = %10d CI: %3d\n Root chosen for geometry opt %3d\n' % (
                self.traj.itr,
                self.traj.kinetic,
                self.traj.temp,
                self.traj.size,
                self.traj.nstate,
                self.traj.last_state
            )

        if self.traj.energy_qm2_1 != 0:
            log_info += """
  &multiscale energy
-------------------------------------------------------
  QM2(high) %16.8f 
  QM2(mid)  %16.8f
  MM(mid)   %16.8f 
  MM(low)   %16.8f
-------------------------------------------------------
""" % (
                self.traj.energy_qm2_1,
                self.traj.energy_qm2_2,
                self.traj.energy_mm1,
                self.traj.energy_mm2,
            )

        if self.traj.ext_pot != 0:
            log_info += ' constraining potential energy: %16.8f\n' % self.traj.ext_pot

        if self.traj.tracker:
            log_info += ''

        log_info += '\n Gnuplot: %s %s %28.16f\n  **\n  **\n  **\n%s\n' % (
            pop,
            pot,
            self.traj.energy[self.traj.last_state - 1],
            hop_info)

        ## add verbose info
        log_info += self._verbose_log_info(self.verbose)

        ## add surface hopping info
        log_info += """
  &surface hopping information
-------------------------------------------------------
%s
-------------------------------------------------------

""" % self.traj.shinfo

        if self.traj.tracker:
            log_info += """
  &geometric parameter monitor
-------------------------------------------------------
%s-------------------------------------------------------
""" % self.traj.tracker

        ## add error info
        if self.traj.err_energy is not None and \
                self.traj.err_grad is not None and \
                self.traj.err_nac is not None and \
                self.traj.err_soc is not None:
            log_info += """  
  &error iter %-10s
-------------------------------------------------------
  Energy   MaxStDev:          %-10.4f
  Gradient MaxStDev:          %-10.4f
  Nac      MaxStDev:          %-10.4f
  Soc      MaxStDev:          %-10.4f
-------------------------------------------------------

""" % (
                self.traj.itr,
                self.traj.err_energy,
                self.traj.err_grad,
                self.traj.err_nac,
                self.traj.err_soc
            )

        ## print log on screen
        if self.silent == 0:
            print(log_info)

        ## always record surface hopping event 
        if self.traj.hoped == 1:
            self._record_surface_hopping(
                self.logpath,
                self.title,
                energy_info,
                xyz_info,
                velo_info,
                charge_info,
            )

        ## checkpoint trajectory class to pkl
        if self.checkpoint > 0:
            self.skiptraj = self._step_counter(self.skiptraj, self.checkpoint)
            self.skiptraj = self._force_output(self.skiptraj)
        else:
            self.skiptraj = 1
        if self.skiptraj == 0:
            with open('%s.pkl' % self.title, 'wb') as mdtraj:
                pickle.dump(self.traj, mdtraj)

        ## write logfile to disk
        if self.traj.itr > self.direct:
            self.skipstep = self._step_counter(self.skipstep, self.buffer)
            self.skipstep = self._force_output(self.skipstep)
        if self.skipstep == 0:
            self._dump_to_disk(self.logpath,
                               self.title,
                               log_info,
                               energy_info,
                               xyz_info,
                               velo_info,
                               charge_info,
                               )

    def _step_counter(self, counter, step):
        counter += 1

        ## reset counter to 0 at printing step or end point or stopped point
        if counter == step or counter == self.traj.step or self.stop == 1:
            counter = 0

        return counter

    def _force_output(self, counter):
        if self.traj.itr == self.traj.step or self.stop != 0:
            counter = 0
        return counter

    def _verbose_log_info(self, verbose):
        log_info = ''

        if verbose == 0:
            return log_info

        log_info += """
  &coordinates in Angstrom
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (print_coord(np.concatenate((self.traj.atoms, self.traj.coord), axis=1)))

        log_info += """
  &velocities in Bohr/au
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (print_coord(np.concatenate((self.traj.atoms, self.traj.velo), axis=1)))

        if len(self.traj.qm2_charge) > 0:
            log_info += """
  &external charges
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (print_charge(self.traj.qm2_charge))

        for n in range(self.traj.nstate):
            try:
                grad = self.traj.grad[n]
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (n + 1, print_coord(np.concatenate((self.traj.atoms, grad), axis=1)))

            except IndexError:
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
  Not Computed
-------------------------------------------------------------------------------
""" % (n + 1)

        for n, pair in enumerate(self.traj.nac_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.nac[n]
                log_info += """
  &nonadiabatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2, print_coord(np.concatenate((self.traj.atoms, coupling), axis=1)))

            except IndexError:
                log_info += """
  &nonadiabatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
  Not computed
-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2)

        soc_info = ''
        for n, pair in enumerate(self.traj.soc_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.soc[n]
                soc_info += '  <H>=%10.4f            %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    coupling, s1 + 1, s2 + 1, m1, m2)

            except IndexError:
                soc_info += '  Not computed              %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    s1 + 1, s2 + 1, m1, m2)

        if len(self.traj.soc_coupling) > 0:
            log_info += """
  &spin-orbit coupling
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % soc_info

        return log_info

    @staticmethod
    def _dump_to_disk(logpath, title, log_info, energy_info, xyz_info, velo_info, charge_info):
        ## output data to disk
        with open('%s/%s.log' % (logpath, title), 'a') as log:
            log.write(log_info)

        with open('%s/%s.md.energies' % (logpath, title), 'a') as log:
            log.write(energy_info)

        with open('%s/%s.md.xyz' % (logpath, title), 'a') as log:
            log.write(xyz_info)

        with open('%s/%s.md.velo' % (logpath, title), 'a') as log:
            log.write(velo_info)

        with open('%s/%s.md.charge' % (logpath, title), 'a') as log:
            log.write(charge_info)

    @staticmethod
    def _record_surface_hopping(logpath, title, energy_info, xyz_info, velo_info, charge_info):
        ## output data for surface hopping event to disk
        with open('%s/%s.sh.energies' % (logpath, title), 'a') as log:
            log.write(energy_info)

        with open('%s/%s.sh.xyz' % (logpath, title), 'a') as log:
            log.write(xyz_info)

        with open('%s/%s.sh.velo' % (logpath, title), 'a') as log:
            log.write(velo_info)

        with open('%s/%s.sh.charge' % (logpath, title), 'a') as log:
            log.write(charge_info)

    def run(self):
        warning = ''
        start = time.time()

        ## add heading to new output files
        heading = 'Nonadiabatic Molecular Dynamics Start: %20s\n%s' % (what_is_time(), self._heading())

        if self.silent == 0:
            print(heading)

        if self.restart == 0:
            with open('%s/%s.log' % (self.logpath, self.title), 'a') as log:
                log.write(heading)

            mdhead = '%20s%28s%28s%28s%28s\n' % ('time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...')
            with open('%s/%s.md.energies' % (self.logpath, self.title), 'a') as log:
                log.write(mdhead)

            with open('%s/%s.sh.energies' % (self.logpath, self.title), 'a') as log:
                log.write(mdhead)

        ## loop over molecular dynamics steps
        self.traj.step += self.addstep
        for itr in range(self.traj.step - self.traj.itr):
            self.traj.itr += 1

            if self.timing == 1:
                print('start', time.time())

            ## propagate nuclear positions (E,G,N,R,V,Ekin)
            self._propagate()

            if self.timing == 1:
                print('propagate', time.time())

            ## adjust kinetics (Ekin,V,thermostat)
            self._thermodynamic()

            if self.timing == 1:
                print('thermostat', time.time())

            ## detect surface hopping
            self._surfacehop()  # update A,H,D,V,state

            if self.timing == 1:
                print('surfacehop', time.time())

            ## check errors and checkpointing
            self._chkgeom()
            self._chkerror()
            self._chkpopulation()
            self._chkpoint()

            if self.timing == 1:
                print('save', time.time())

            ## terminate trajectory
            if self.stop == 1:
                warning = 'Trajectory terminated because the NN prediction differences are larger than thresholds.'
                break
            elif self.stop == 2:
                warning = 'Trajectory terminated because the QM calculation failed.'
                break
            elif self.stop == 3:
                warning = 'Trajectory terminated because the state population is nan.'
                break
            elif self.stop == 4:
                warning = 'Trajectory terminated because the state population exceed 0–1.'
                break
            elif self.stop == 5:
                warning = 'Trajectory terminated because it meets the geometry requirement.'
                break

        end = time.time()
        walltime = how_long(start, end)
        tailing = '%s\nNonadiabatic Molecular Dynamics End: %20s Total: %20s\n' % (warning, what_is_time(), walltime)

        if self.silent == 0:
            print(tailing)

        with open('%s/%s.log' % (self.logpath, self.title), 'a') as log:
            log.write(tailing)

        return self.traj
