######################################################
#
# PyRAI2MD 2 module for manually updating the codes
#
# Author Jingbai Li
# Jun 13 2023
#
######################################################

import filecmp

# directly run this script inside the ./tool folder

try:
    import PyRAI2MD
    module_path = PyRAI2MD.__file__.split('__init__.py')[0]

except ModuleNotFoundError:
    module_path = ''
    exit('\n PyRAI2MD is not installed or not found\n')

modules = [
    '/pyrai2md.py',
    '/variables.py',
    '/methods.py',
    '/Molecule/atom.py',
    '/Molecule/molecule.py',
    '/Molecule/trajectory.py',
    '/Molecule/pbc_helper.py',
    '/Molecule/qmmm_helper.py',
    '/Molecule/constraint.py',
    '/Keywords/key_bagel.py',
    '/Keywords/key_control.py',
    '/Keywords/key_dimenet.py',
    '/Keywords/key_e2n2.py',
    '/Keywords/key_grid_search.py',
    '/Keywords/key_md.py',
    '/Keywords/key_mlp.py',
    '/Keywords/key_molcas.py',
    '/Keywords/key_molecule.py',
    '/Keywords/key_nn.py',
    '/Keywords/key_openqp.py',
    '/Keywords/key_orca.py',
    '/Keywords/key_read_file.py',
    '/Keywords/key_schnet.py',
    '/Keywords/key_templ.py',
    '/Keywords/key_xtb.py',
    '/Quantum_Chemistry/qc_molcas.py',
    '/Quantum_Chemistry/qc_bagel.py',
    '/Quantum_Chemistry/qc_molcas_tinker.py',
    '/Quantum_Chemistry/qc_openqp.py',
    '/Quantum_Chemistry/qc_orca.py',
    '/Quantum_Chemistry/qc_xtb.py',
    '/Quantum_Chemistry/qmqm2.py',
    '/Machine_Learning/Dimenet.py',
    '/Machine_Learning/model_demo.py',
    '/Machine_Learning/model_NN.py',
    '/Machine_Learning/model_pyNNsMD.py',
    '/Machine_Learning/model_GCNNP.py',
    '/Machine_Learning/model_DimeNet.py',
    '/Machine_Learning/model_templ.py',
    '/Machine_Learning/model_helper.py',
    '/Machine_Learning/hyper_nn.py',
    '/Machine_Learning/hyper_pynnsmd.py',
    '/Machine_Learning/hyper_gcnnp.py',
    '/Machine_Learning/hyper_dimenet.py',
    '/Machine_Learning/hyper_templ.py',
    '/Machine_Learning/training_data.py',
    '/Machine_Learning/permutation.py',
    '/Machine_Learning/adaptive_sampling.py',
    '/Machine_Learning/grid_search.py',
    '/Machine_Learning/search_nn.py',
    '/Machine_Learning/search_GCNNP.py',
    '/Machine_Learning/remote_train.py',
    '/Dynamics/aimd.py',
    '/Dynamics/mixaimd.py',
    '/Dynamics/single_point.py',
    '/Dynamics/hop_probability.py',
    '/Dynamics/reset_velocity.py',
    '/Dynamics/verlet.py',
    '/Dynamics/Ensembles/ensemble.py',
    '/Dynamics/Ensembles/microcanonical.py',
    '/Dynamics/Ensembles/thermostat.py',
    '/Dynamics/Propagators/surface_hopping.py',
    '/Dynamics/Propagators/setup_fssh.py',
    '/Dynamics/Propagators/fssh.pyx',
    '/Dynamics/Propagators/gsh.py',
    '/Dynamics/Propagators/tsh_helper.py',
    '/Utils/extension.py',
    '/Utils/coordinates.py',
    '/Utils/read_tools.py',
    '/Utils/geom_tools.py',
    '/Utils/bonds.py',
    '/Utils/sampling.py',
    '/Utils/timing.py',
    '/Utils/logo.py',
]

print(' Files need to be updated')
print('---------------------------')
for mod in modules:
    installed = '%s%s' % (module_path, mod)
    source = '../PyRAI2MD/%s' % mod
    if not filecmp.cmp(installed, source, shallow=False):
        print(' cp %s %s' % (source, installed))
