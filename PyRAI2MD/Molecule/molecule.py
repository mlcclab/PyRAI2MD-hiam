######################################################
#
# PyRAI2MD 2 module for creating molecule objects
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

import sys
import numpy as np

from PyRAI2MD.Molecule.atom import Atom
from PyRAI2MD.Molecule.qmmm_helper import auto_boundary
from PyRAI2MD.Molecule.qmmm_helper import compute_hcap
from PyRAI2MD.Molecule.pbc_helper import compute_primitives
from PyRAI2MD.Molecule.pbc_helper import apply_pbc
from PyRAI2MD.Utils.coordinates import read_initcond
from PyRAI2MD.Utils.coordinates import read_coord
from PyRAI2MD.Utils.coordinates import read_charge
from PyRAI2MD.Utils.coordinates import read_tinker_key
from PyRAI2MD.Utils.coordinates import verify_xyz

class Molecule:
    """ Molecular property class

        Parameters:          Type:
            mol              str         name of the xyz file
                             or list     list of coordinates in xyz format
                             or ndarray  array of coordinates in xyz format

            keywords         dict    molecule keyword list

        Attribute:           Type:
            ci               int         ci dimension, a.k.a total number of state per spin multiplicity
            spin             int         total spin angular momentum per spin multiplicity
            mult             int         multiplicity per spin
            statemult        list        multiplicity per state
            coupling         list        list of interstate coupling pairs
            nac_coupling     list        list of non-adiabatic coupling pairs
            soc_coupling     list        list of spin-orbit coupling pairs
            nstate           int         number of electronic states
            nnac             int         number of non-adiabatic couplings
            nsoc             int         number of spin-orbit couplings
            active           list        index of active atoms
            inact            list        index of inactive atoms
            link             list        index of link atoms
            natom            int         number of atoms (active)
            ninac            int         number of inactive atoms
            nlink            int         number of link atoms (Hcaps)
            atoms            ndarray     atom name in all level region
            coord            ndarray     nuclear coordinates in all level region (active)
            atomic_number    ndarray     atomic number in all level region
            mass             ndarray     atomic mass in all level region
            velo             ndarray     velocity in all level region
            kinetic          float       kinetic energy in the present step
            energy           ndarray     potential energy in the present step
            grad             ndarray     gradient in the present step
            nac              ndarray     non-adiabatic coupling vectors in Hartree/Bohr (numerator)
            soc              ndarray     spin-orbit coupling in cm-1
            err_energy       float       error of energy in adaptive sampling
            err_grad         float       error of gradient in adaptive sampling
            err_nac          float       error of nac in adaptive sampling
            err_soc          float       error of soc in adaptive sampling
            qmmm_key         str         qmmm key file (Tinker)
            txyz             list        full tinker xyz file
            qm_atoms         ndarray     atom name in high level region
            qm_coord         ndarray     nuclear coordinates in high level region
            qm_energy        ndarray     potential energy in the present step in high level region
            qm_grad          ndarray     gradient in the present step in high level region
            qm_nac           ndarray     non-adiabatic coupling vectors in Hartree/Bohr (numerator) in high level region
            qm_soc           ndarray     spin-orbit coupling in cm-1 in high level region
            charges          ndarray     total charges
            qm1_charge       ndarray     charges in qm 1 region
            qm2_charge       ndarray     charges in qm 2 region (external)
            qmqm2_atoms      ndarray     atom name in high level region and middle level region
            qmqm2_coord      ndarray     nuclear coordinates in high level region and middel level region
            energy_qm2_1     float       qm2 energy for high level region
            energy_qm2_2     float       qm2 energy for high level region and middle level region
            energy_mm1       float       mm energy for high level region and middle level region
            energy_mm2       float       mm energy for high level region, middle level region and low level region
            Hcap_atoms       ndarray     atom name of capping H
            Hcap_coord       ndarray     nuclear coordinates of capping H
            Hcap_jacob       ndarray     Jacobian between caped and uncaped coordinates
            highlevel        ndarray     atoms in high level region, region 1
            midlevel         ndarray     atoms in middle level region, region 2
            lowlevel         ndarray     atoms in low level region, region 3
            boundary         ndarray     index of atoms at high and low level boundary
            embedding        ndarray     embed surrounding charge for high level
            read_charge      bool        read static charge from .charge file
            relax            ndarray     index of relaxed atoms
            freeze           ndarray     index of frozen atoms
            constrain        ndarray     index of constrained atoms
            center           ndarray     the center of the constraining potential
            record_center    bool        record the center for the following step
            ext_pot          float       external potential energy
            primitive        ndarray     primitive translation vectors in 1D 2D 3D
            lattice          ndarray     lattice constant
            status           int         molecular property calculation status

        Function:            Returns:
            reload           self        reload data from existing trajectory
            apply_pbc        self        setup periodic boundary condition for molecule
            apply_qmmm       self        apply qmmm for molecule
    """

    __slots__ = ['inact', 'active', 'link', 'ninac', 'nlink', 'qmmm_key', 'qmmm_xyz', 'txyz', 'embedding',
                 'read_charge', 'ci', 'nstate', 'spin', 'mult', 'statemult', 'coupling',
                 'nac_coupling', 'soc_coupling', 'nnac', 'nsoc', 'natom', 'atoms', 'coord', 'atomic_number', 'mass',
                 'velo', 'kinetic', 'energy', 'grad', 'nac', 'soc', 'err_energy', 'err_grad', 'err_nac', 'err_soc',
                 'qm_atoms', 'qm_coord', 'Hcap_atoms', 'Hcap_coord', 'Hcap_jacob', 'boundary', 'nhigh', 'nmid', 'nlow',
                 'highlevel', 'midlevel', 'lowlevel', 'relax', 'freeze', 'constrain', 'primitive', 'lattice', 'status',
                 'charges', 'qm1_charge', 'qm2_charge', 'qm_energy', 'qm_grad', 'qm_nac', 'qm_soc', 'qmqm2_index',
                 'qmqm2_atoms', 'qmqm2_coord', 'energy_qm2_1', 'energy_qm2_2', 'energy_mm1', 'energy_mm2',
                 'center', 'record_center', 'ext_pot']

    def __init__(self, mol, keywords=None):
        key_dict = keywords['molecule'].copy()
        title = keywords['control']['title']
        ## initialize variables for molecule
        self.atoms = np.zeros(0)
        self.coord = np.zeros(0)
        self.atomic_number = np.zeros(0)
        self.inact = []
        self.active = []
        self.link = []
        self.mass = np.zeros(0)
        self.velo = np.zeros(0)
        self.kinetic = np.zeros(0)
        self.energy = np.zeros(0)
        self.grad = np.zeros(0)
        self.nac = np.zeros(0)
        self.soc = np.zeros(0)
        self.err_energy = None
        self.err_grad = None
        self.err_nac = None
        self.err_soc = None
        self.Hcap_coord = np.zeros(0)
        self.Hcap_jacob = np.zeros(0)
        self.qm_atoms = np.zeros(0)
        self.qm_coord = np.zeros(0)
        self.qm_energy = np.zeros(0)
        self.qm_grad = np.zeros(0)
        self.qm_nac = np.zeros(0)
        self.qm_soc = np.zeros(0)
        self.charges = np.zeros(0)
        self.qm1_charge = np.zeros(0)
        self.qm2_charge = np.zeros(0)
        self.qmqm2_index = np.zeros(0)
        self.qmqm2_atoms = np.zeros(0)
        self.qmqm2_coord = np.zeros(0)
        self.energy_qm2_1 = 0
        self.energy_qm2_2 = 0
        self.energy_mm1 = 0
        self.energy_mm2 = 0
        self.midlevel = np.zeros(0)
        self.lowlevel = np.zeros(0)
        self.center = np.zeros(3)
        self.record_center = False
        self.ext_pot = 0
        self.txyz = []

        ## load variables from key_dict
        self.qmmm_key = key_dict['qmmm_key']
        self.qmmm_xyz = key_dict['qmmm_xyz']
        self.ci = key_dict['ci']
        self.spin = key_dict['spin']
        self.coupling = key_dict['coupling']
        self.highlevel = key_dict['highlevel']
        self.midlevel = key_dict['midlevel']
        self.boundary = key_dict['boundary']
        self.embedding = key_dict['embedding']
        self.read_charge = key_dict['read_charge']
        self.freeze = key_dict['freeze']
        self.constrain = key_dict['constrain']
        self.primitive = key_dict['primitive']
        self.lattice = key_dict['lattice']
        self.status = 0

        ## read coordinates from a file or a list
        xyz_type = verify_xyz(mol)

        if xyz_type == 'xyz':
            self.atoms, self.coord, self.velo = read_coord(mol)
        elif xyz_type == 'array':
            self.atoms, self.coord, self.velo = read_initcond(mol)
        elif xyz_type == 'dict':
            mol_info = read_tinker_key(mol, self.qmmm_key, dtype='dict')
            self.atoms = mol_info['atoms']
            self.coord = mol_info['coord']
            self.velo = mol_info['velo']
            self.inact = mol_info['inact']
            self.active = mol_info['active']
            self.link = mol_info['link']
            self.boundary = mol_info['boundary']
            self.highlevel = mol_info['highlevel']
            self.txyz = mol_info['txyz']
        elif xyz_type == 'tinker':
            mol_info = read_tinker_key('%s.xyz' % mol, self.qmmm_key, dtype='file')
            self.atoms = mol_info['atoms']
            self.coord = mol_info['coord']
            self.velo = mol_info['velo']
            self.inact = mol_info['inact']
            self.active = mol_info['active']
            self.link = mol_info['link']
            self.boundary = mol_info['boundary']
            self.highlevel = mol_info['highlevel']
            self.txyz = mol_info['txyz']
        else:
            sys.exit('\n  FileTypeError\n  PyRAI2MD: cannot recognize coordinate file %s' % mol)

        if xyz_type == 'xyz' and self.qmmm_xyz != 'Input':
            mol_info = read_tinker_key(self.qmmm_xyz, self.qmmm_key, dtype='file')
            self.inact = mol_info['inact']
            self.active = mol_info['active']
            self.link = mol_info['link']
            self.boundary = mol_info['boundary']
            self.highlevel = mol_info['highlevel']
            self.txyz = mol_info['txyz']

        ## get molecule information
        self.natom = len(self.atoms)
        self.atomic_number = np.array([Atom(x).name for x in self.atoms.reshape(-1)])
        self.mass = np.array([Atom(x).get_mass() * 1822.8852 for x in self.atoms.reshape(-1)]).reshape((-1, 1))
        self.relax = np.setdiff1d(np.arange(self.natom), self.freeze)

        ## partition atom in regions
        if len(self.highlevel) == 0:
            self.highlevel = np.arange(self.natom)
        else:
            if len(self.midlevel) == 0:
                self.midlevel = np.setdiff1d(np.arange(self.natom), self.highlevel)
            else:
                self.lowlevel = np.setdiff1d(np.arange(self.natom), np.concatenate((self.highlevel, self.midlevel)))

        self.qmqm2_index = np.concatenate((self.highlevel, self.midlevel))

        ## auto generate qmmm boundary if the request has no qmmm key
        if len(self.boundary) == 0 and len(self.lowlevel) > 0:
            self.link, self.boundary, self.primitive = auto_boundary(self.coord, self.highlevel, self.primitive)

        ## initialize charge and read charge from a file
        if self.embedding:
            ncharges = len(self.highlevel) + len(self.midlevel) + len(self.lowlevel)
            self.charges = np.zeros((ncharges, 4))
            self.charges[:self.natom, 1: 4] = self.coord
            if self.read_charge:
                self.qm2_charge = read_charge(title)
                self.charges[self.midlevel] = np.copy(self.qm2_charge)

        ## get additional molecule information
        self.ninac = len(self.inact)
        self.nlink = len(self.link)
        self.nhigh = len(self.highlevel)
        self.nmid = len(self.midlevel)
        self.nlow = len(self.lowlevel)
        self.Hcap_atoms = np.array([['H'] for _ in self.boundary])

        ## initialize spin state info
        self.nstate = int(np.sum(self.ci))
        self.mult = []
        self.statemult = []
        for n, s in enumerate(self.ci):
            mult = int(self.spin[n] * 2 + 1)
            self.mult.append(mult)
            for m in range(s):
                self.statemult.append(mult)

        self.nac_coupling = []
        self.soc_coupling = []
        for n, pair in enumerate(self.coupling):
            s1, s2 = pair
            s1 -= 1
            s2 -= 1
            if self.statemult[s1] != self.statemult[s2]:
                self.soc_coupling.append(sorted([s1, s2]))
            else:
                self.nac_coupling.append(sorted([s1, s2]))

        self.nnac = len(self.nac_coupling)
        self.nsoc = len(self.soc_coupling)

        ## initialize pbc
        if len(self.primitive) == 0 and len(self.lattice) == 6:
            self.primitive = compute_primitives(self.lattice)

        ## apply constraints
        self.velo[self.freeze] = np.array([0., 0., 0.])

    def apply_pbc(self):
        self.coord = apply_pbc(self.coord, self.primitive)

        return self

    def apply_qmmm(self):
        self.Hcap_coord, self.Hcap_jacob = compute_hcap(self.atoms, self.coord, self.boundary)
        if len(self.Hcap_coord) > 0:
            self.qm_atoms = np.concatenate((self.atoms[self.highlevel], self.Hcap_atoms))
            self.qm_coord = np.concatenate((self.coord[self.highlevel], self.Hcap_coord))
        else:
            self.qm_atoms = self.atoms[self.highlevel]
            self.qm_coord = self.coord[self.highlevel]

        # MM part does not support H cap yet
        self.qmqm2_atoms = self.atoms[self.qmqm2_index]
        self.qmqm2_coord = self.coord[self.qmqm2_index]

        if len(self.charges) > 0:
            self.qm1_charge = self.charges[self.highlevel]
            self.qm2_charge = self.charges[self.midlevel]
        else:
            self.qm1_charge = np.zeros(0)
            self.qm2_charge = np.zeros(0)

        return self
