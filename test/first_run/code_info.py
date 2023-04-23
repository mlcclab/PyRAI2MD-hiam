######################################################
#
# PyRAI2MD Code Structure Review
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################

import os

try:
    import PyRAI2MD

    ROOT = os.path.dirname(PyRAI2MD.__file__)

except ModuleNotFoundError:
    ROOT = ''

register = {
    'pyrai2md': '/pyrai2md.py',
    'variables': '/variables.py',
    'methods': '/methods.py',
    'atom': '/Molecule/atom.py',
    'molecule': '/Molecule/molecule.py',
    'trajectory': '/Molecule/trajectory.py',
    'pbc_helper': '/Molecule/pbc_helper.py',
    'qmmm_helper': '/Molecule/qmmm_helper.py',
    'key_bagel': '/Keywords/key_bagel.py',
    'key_control': '/Keywords/key_control.py',
    'key_dimenet': '/Keywords/key_dimenet.py',
    'key_e2n2': '/Keywords/key_e2n2.py',
    'key_grid_search': '/Keywords/key_grid_search.py',
    'key_md': '/Keywords/key_md.py',
    'key_mlp': '/Keywords/key_mlp.py',
    'key_molcas': '/Keywords/key_molcas.py',
    'key_molecule': '/Keywords/key_molecule.py',
    'key_nn': '/Keywords/key_nn.py',
    'key_orca': '/Keywords/key_orca.py',
    'key_read_file': '/Keywords/key_read_file.py',
    'key_schnet': '/Keywords/key_schnet.py',
    'key_templ': '/Keywords/key_templ.py',
    'key_xtb': '/Keywords/key_xtb.py',
    'constraint': '/Molecule/constraint.py',
    'qc_molcas': '/Quantum_Chemistry/qc_molcas.py',
    'qc_bagel': '/Quantum_Chemistry/qc_bagel.py',
    'qc_molcas_tinker': '/Quantum_Chemistry/qc_molcas_tinker.py',
    'qc_orca': '/Quantum_Chemistry/qc_orca.py',
    'qc_xtb': '/Quantum_Chemistry/qc_xtb.py',
    'qmqm2': '/Quantum_Chemistry/qmqm2.py',
    'Dimenet': '/Machine_Learning/Dimenet.py',
    'model_demo': '/Machine_Learning/model_demo.py',
    'model_NN': '/Machine_Learning/model_NN.py',
    'model_pyNNsMD': '/Machine_Learning/model_pyNNsMD.py',
    'model_GCNNP': '/Machine_Learning/model_GCNNP.py',
    'model_DimeNet': '/Machine_Learning/model_DimeNet.py',
    'model_templ': '/Machine_Learning/model_templ.py',
    'model_helper': '/Machine_Learning/model_helper.py',
    'hyper_nn': '/Machine_Learning/hyper_nn.py',
    'hyper_pynnsmd': '/Machine_Learning/hyper_pynnsmd.py',
    'hyper_gcnnp': '/Machine_Learning/hyper_gcnnp.py',
    'hyper_dimenet': '/Machine_Learning/hyper_dimenet.py',
    'hyper_templ': '/Machine_Learning/hyper_templ.py',
    'training_data': '/Machine_Learning/training_data.py',
    'permutation': '/Machine_Learning/permutation.py',
    'adaptive_sampling': '/Machine_Learning/adaptive_sampling.py',
    'grid_search': '/Machine_Learning/grid_search.py',
    'search_nn': '/Machine_Learning/search_nn.py',
    'search_GCNNP': '/Machine_Learning/search_GCNNP.py',
    'remote_train': '/Machine_Learning/remote_train.py',
    'aimd': '/Dynamics/aimd.py',
    'mixaimd': '/Dynamics/mixaimd.py',
    'single_point': '/Dynamics/single_point.py',
    'hop_probability': '/Dynamics/hop_probability.py',
    'reset_velocity': '/Dynamics/reset_velocity.py',
    'verlet': '/Dynamics/verlet.py',
    'ensemble': '/Dynamics/Ensembles/ensemble.py',
    'microcanonical': '/Dynamics/Ensembles/microcanonical.py',
    'thermostat': '/Dynamics/Ensembles/thermostat.py',
    'surface_hopping': '/Dynamics/Propagators/surface_hopping.py',
    'setup_fssh': '/Dynamics/Propagators/setup_fssh.py',
    'fssh': '/Dynamics/Propagators/fssh.pyx',
    'gsh': '/Dynamics/Propagators/gsh.py',
    'tsh_helper': '/Dynamics/Propagators/tsh_helper.py',
    'extension': '/Utils/extension.py',
    'coordinates': '/Utils/coordinates.py',
    'read_tools': '/Utils/read_tools.py',
    'bonds': '/Utils/bonds.py',
    'sampling': '/Utils/sampling.py',
    'timing': '/Utils/timing.py',
    'logo': '/Utils/logo.py',
}

num_line = 0
num_file = 0
length_dict = {}
for name, location in register.items():
    mod = '%s/%s' % (ROOT, location)
    num_file += 1
    if os.path.exists(mod):
        with open(mod, 'r') as file:
            n = len(file.readlines())
        num_line += n
        length_dict[name] = n
    else:
        length_dict[name] = 0


def review(length, totline, totfile):
    status = """
 File/Folder Name                                  Contents                                      Length
--------------------------------------------------------------------------------------------------------
 PyRAI2MD                                          source codes folder
  |--pyrai2md.py                                   PyRAI2MD main function                      %8s
  |--method.py                                     PyRAI2MD method manager                     %8s
  |--variables.py                                  PyRAI2MD input reader                       %8s
  |
  |--Keywords                                      default input values folder
  |   |--key_control.py                            keywords for calculation control            %8s
  |   |--key_molecule.py                           keywords for molecule specification         %8s
  |   |--key_md.py                                 keywords for molecular dynamic settings     %8s
  |   |--key_nn.py                                 keywords for neural network settings        %8s
  |   |--key_grid_search.py                        keywords for grid search settings           %8s
  |   |--key_molcas.py                             keywords for molcas calculation             %8s
  |   |--key_bagel.py                              keywords for bagel calculation              %8s
  |   |--key_orca                                  keywords for orca calculation               %8s
  |   |--key_xtb.py                                keywords for xtb calculation                %8s
  |   |--key_mlp.py                                keywords for mlp settings                   %8s
  |   |--key_schnet.py                             keywords for schnet settings                %8s
  |   |--key_e2n2.py                               keywords for e2n2 settings                  %8s
  |   |--key_dimenet.py                            keywords for dimenet (NAC model) setting    %8s
  |   |--key_read_file.py                          keywords for reading training data          %8s
  |    `-key_templ.py                              keywords class template                     %8s
  |
  |--Molecule                                      atom, molecule, trajectory code folder
  |   |--atom.py                                   atomic properties class                     %8s
  |   |--molecule.py                               molecular properties class                  %8s
  |   |--trajectory.py                             trajectory properties class                 %8s
  |   |--constraint.py                             external potential functions                %8s
  |   |--pbc_helper.py                             periodic boundary condition functions       %8s
  |    `-qmmm_helper.py                            qmmm functions                              %8s
  |
  |--Quantum_Chemistry                             quantum chemicial program interface folder
  |   |--qc_molcas.py                              OpenMolcas interface                        %8s
  |   |--qc_bagel.py                               BAGEL interface                             %8s
  |   |--qc_molcas_tinker                          OpenMolcas/Tinker interface                 %8s
  |   |--qc_orca                                   ORCA interface                              %8s
  |   |--qc_xtb                                    GFN-xTB interface                           %8s
  |    `-qmqm2                                     Multiscale calculation interface            %8s
  |
  |--Machine_Learning                              machine learning library interface folder
  |   |--model_demo.py                             demo version neural network                 %8s
  |   |--model_NN.py                               native neural network interface             %8s
  |   |--model_pyNNsMD.py                          pyNNsMD interface                           %8s
  |   |--model_GCNNP.py                            GCNNP interface                             %8s
  |   |--model_DimeNet.py                          DimeNet NAC model interface                 %s
  |   |--model_templ.py                            NN interface template                       %s
  |   |--model_helper.py                           additional tools for neural network         %8s
  |   |--hyper_nn.py                               native neural network hyperparameter        %8s
  |   |--hyper_pynnsmd.py                          pyNNsMD hyperparameter                      %8s
  |   |--hyper_gcnnp.py                            GCNNP hyperparameter                        %8s
  |   |--hyper_dimenet.py                          DimeNet NAC model hyperparameter            %8s
  |   |--hyper_templ.py                            hyperparameter template                     %8s
  |   |--training_data.py                          training data manager                       %8s
  |   |--permutation.py                            data permutation functions                  %8s
  |   |--adaptive_sampling.py                      adaptive sampling class                     %8s
  |   |--grid_search.py                            grid search manager                         %8s
  |   |--search_nn.py                              grid search function for native nn          %8s
  |   |--search_GCNNP.py                           grid search function for e2n2               %8s  
  |   |--remote_train.py                           remote training function                    %8s
  |   |--Dimenet.py                                Dimenet NAC model                           %8s
  |   |--NNsMD                                     demo version neural network library  
  |    `-pyNNsMD                                   native neural network library                  
  |
  |--Dynamics                                      ab initio molecular dynamics code folder
  |   |--aimd.py                                   molecular dynamics class                    %8s
  |   |--mixaimd.py                                ML-QC hybrid molecular dynamics class       %8s
  |   |--single_point.py                           single point calculation                    %8s
  |   |--hop_probability.py                        surface hopping probability calculation     %8s
  |   |--reset_velocity.py                         velocity adjustment functions               %8s
  |   |--verlet.py                                 velocity verlet method                      %8s
  |   |--Ensembles                                 thermodynamics control code folder
  |   |   |--ensemble.py                           thermodynamics ensemble manager             %8s
  |   |   |--microcanonical.py                     microcanonical ensemble                     %8s
  |   |    `-thermostat.py                         canonical ensemble                          %8s
  |   |
  |    `-Propagators                               electronic propagation code folder
  |       |--surface_hopping.py                    surface hopping manager                     %8s
  |       |--setup_fssh.py                         setup file to compile the C-lib of fssh.pyx %8s
  |       |--fssh.pyx                              fewest switches surface hopping method      %8s
  |       |--gsh.py                                generalized surface hopping method          %8s
  |        `-tsh_helper.py                         trajectory surface hopping tools            %8s
  |
   `-Utils                                         utility folder
      |--extension.py                              additional tools for setup                  %8s
      |--coordinates.py                            coordinates writing functions               %8s
      |--read_tools.py                             index reader                                %8s
      |--bonds.py                                  bond length library                         %8s
      |--sampling.py                               initial condition sampling functions        %8s
      |--timing.py                                 timing functions                            %8s
       `-logo.py                                   logo and credits                            %8s
--------------------------------------------------------------------------------------------------------
Total %4s/%4s files                                                                          %8s
""" % (length['pyrai2md'],
       length['methods'],
       length['variables'],
       length['key_control'],
       length['key_molecule'],
       length['key_md'],
       length['key_nn'],
       length['key_grid_search'],
       length['key_molcas'],
       length['key_bagel'],
       length['key_orca'],
       length['key_xtb'],
       length['key_mlp'],
       length['key_schnet'],
       length['key_e2n2'],
       length['key_dimenet'],
       length['key_read_file'],
       length['key_templ'],
       length['atom'],
       length['molecule'],
       length['trajectory'],
       length['constraint'],
       length['pbc_helper'],
       length['qmmm_helper'],
       length['qc_molcas'],
       length['qc_bagel'],
       length['qc_molcas_tinker'],
       length['qc_orca'],
       length['qc_xtb'],
       length['qmqm2'],
       length['model_demo'],
       length['model_NN'],
       length['model_pyNNsMD'],
       length['model_GCNNP'],
       length['model_dimenet'],
       length['model_templ'],
       length['model_helper'],
       length['hyper_nn'],
       length['hyper_pynnsmd'],
       length['hyper_gcnnp'],
       length['hyper_dimenet'],
       length['hyper_templ'],
       length['training_data'],
       length['permutation'],
       length['adaptive_sampling'],
       length['grid_search'],
       length['search_nn'],
       length['search_GCNNP'],
       length['remote_train'],
       length['Dimenet'],
       length['aimd'],
       length['mixaimd'],
       length['single_point'],
       length['hop_probability'],
       length['reset_velocity'],
       length['verlet'],
       length['ensemble'],
       length['microcanonical'],
       length['thermostat'],
       length['surface_hopping'],
       length['setup_fssh'],
       length['fssh'],
       length['gsh'],
       length['tsh_helper'],
       length['extension'],
       length['coordinates'],
       length['read_tools'],
       length['bonds'],
       length['sampling'],
       length['timing'],
       length['logo'],
       totfile,
       len(length),
       totline)

    return status


def main():
    print(review(length_dict, num_line, num_file))


if __name__ == '__main__':
    main()
