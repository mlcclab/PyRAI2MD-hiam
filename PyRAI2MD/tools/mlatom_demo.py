######################################################
#
# PyRAI2MD 2 module for MLatom
#
# Author Jingbai Li
# Jan 10 2025
#
######################################################

import sys
import json
import copy
import numpy as np
import mlatom as ml
import multiprocessing

from PyRAI2MD.Molecule.atom import Atom


def assign_data(var):
    idx, nstate, coord, energy, grad = var
    coordinates = np.array(coord)[:, 1: 4].astype(float)
    species = np.array([Atom(x[0]).get_nuc() for x in coord])
    mol = ml.molecule.from_numpy(coordinates=coordinates, species=species)
    mol.electronic_states = [mol.copy(atomic_labels=[], molecular_labels=[]) for _ in range(nstate)]
    for n in range(nstate):
        mol.electronic_states[n].energy = energy[n]
        mol.electronic_states[n].energy_gradients = np.array(grad[n]) / 0.529177249

    return idx, mol

def MLatom():
    with open(sys.argv[1]) as indata:
        data = json.load(indata)
    print('==> Data loaded! ')

    xyz = data['xyz']  # coordinates of each molecule, the data shape is (N, 4) including the element symbol, str
    energy = data['energy']  # two energies of each molecule, the data shape is (N, 2), float
    grad = data['grad']
    nstate = data['nstate']
    ndata = len(energy)
    cpus = 16
    cpus = min([ndata, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    dataset = [[] for _ in range(ndata)]

    variables_wrapper = [(n, nstate, xyz[n], energy[n], grad[n]) for n in range(ndata)]
    for val in pool.imap_unordered(assign_data, variables_wrapper):
        n, mol = val
        dataset[n] = mol
    pool.close()

    print('==> Data prepared! ')
    database = ml.molecular_database()
    database.molecules = dataset

    print('==> Build Dataset! ')
    trainDB, valDB = database.split(fraction_of_points_in_splits=[0.9, 0.1])
    testDB = copy.deepcopy(valDB)
    print('==> Data set ready! ')

    msani_eg = ml.models.msani(model_file='msani.pt', nstates=nstate)
    print('==> Model is ready! ')

    msani_eg.train(molecular_database=trainDB,
                   validation_molecular_database=valDB,
                   property_to_learn='energy',
                   xyz_derivative_property_to_learn='energy_gradients',
                   hyperparameters={
                       'max_epochs': 100,
                       'learning_rate': 1e-4,
                       'batch_size': 71,
                       # "neurons": [[400, 400, 400, 400]],
                   },  # 100 epochs is not enough, only for test
                   )

    print('==> Trained! ')

    msani_eg.predict(molecular_database=testDB,
                     calculate_energy=True,
                     calculate_energy_gradients=True,  # 100 epochs is not enough, only for test
                     )

    print('==> predicted! ')

    ref_e = valDB.get_properties('energy')
    ref_g = valDB.get_xyz_vectorial_properties('energy_gradients')
    pred_e = testDB.get_properties('energy')
    pred_g = testDB.get_xyz_vectorial_properties('energy_gradients')

    print(ref_e)


MLatom()
