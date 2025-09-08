######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 20 2023
#
######################################################

from PyRAI2MD.Keywords.key_control import KeyControl
from PyRAI2MD.Keywords.key_molecule import KeyMolecule
from PyRAI2MD.Keywords.key_molcas import KeyMolcas
from PyRAI2MD.Keywords.key_bagel import KeyBagel
from PyRAI2MD.Keywords.key_orca import KeyOrca
from PyRAI2MD.Keywords.key_openqp import KeyOpenQP
from PyRAI2MD.Keywords.key_xtb import KeyXtb
from PyRAI2MD.Keywords.key_md import KeyMD
from PyRAI2MD.Keywords.key_grid_search import KeySearch
from PyRAI2MD.Keywords.key_nn import KeyNN
from PyRAI2MD.Keywords.key_mlp import KeyMLP
from PyRAI2MD.Keywords.key_schnet import KeySchNet
from PyRAI2MD.Keywords.key_e2n2demo import KeyE2N2Demo
from PyRAI2MD.Keywords.key_e2n2 import KeyE2N2
from PyRAI2MD.Keywords.key_dimenet import KeyDimeNet
from PyRAI2MD.Keywords.key_read_file import KeyReadFile


def read_input(ld_input):
    ## This function read keywords from input
    ## This function is expected to be expanded in future as more methods added

    keywords_list = {
        'control': KeyControl(),
        'molecule': KeyMolecule(),
        'molcas': KeyMolcas(),
        'bagel': KeyBagel(),
        'orca': KeyOrca(),
        'openqp': KeyOpenQP(),
        'xtb': KeyXtb(),
        'md': KeyMD(),
        'nn': KeyNN(nn_type='nn'),
        'mlp': KeyNN(nn_type='mlp'),
        'schnet': KeyNN(nn_type='schnet'),
        'e2n2': KeyNN(nn_type='e2n2'),
        'dimenet': KeyNN(nn_type='dimenet'),
        'search': KeySearch(),
        'eg': KeyMLP(key_type='eg'),
        'nac': KeyMLP(key_type='nac'),
        'soc': KeyMLP(key_type='soc'),
        'eg2': KeyMLP(key_type='eg'),
        'nac2': KeyMLP(key_type='nac'),
        'soc2': KeyMLP(key_type='soc'),
        'sch_eg': KeySchNet(key_type='eg'),
        'sch_nac': KeySchNet(key_type='nac'),
        'sch_soc': KeySchNet(key_type='soc'),
        'e2n2_eg': KeyE2N2(key_type='eg'),
        'e2n2_nac': KeyE2N2(key_type='nac'),
        'e2n2_soc': KeyE2N2(key_type='soc'),
        'dime_nac': KeyDimeNet(key_type='nac'),
        'file': KeyReadFile(),
    }

    # initialize a list for all variables using their default values
    variables_all = {}
    for key, keywords in keywords_list.items():
        variables_all[key] = keywords.default()

    # update the default values
    if isinstance(ld_input, dict):
        ## update variables if input is a dict
        ## be caution that the input dict must have the same data structure
        ## this is only use to load pre-stored input in json
        variables_all = deep_update(variables_all, ld_input)
    else:
        ## read variable if input is a list of string
        ## skip reading if input is a json dict
        for line in ld_input:
            line = line.splitlines()
            if len(line) == 0:
                continue
            key = line[0].lower()
            variables_all[key] = keywords_list[key].update(line)

        # make nested dict
        variables_all['nn']['eg'] = variables_all['eg']
        variables_all['nn']['nac'] = variables_all['nac']
        variables_all['nn']['soc'] = variables_all['soc']
        variables_all['nn']['eg2'] = variables_all['eg2']
        variables_all['nn']['nac2'] = variables_all['nac2']
        variables_all['nn']['soc2'] = variables_all['soc2']
        variables_all['mlp']['eg'] = variables_all['eg']
        variables_all['mlp']['nac'] = variables_all['nac']
        variables_all['mlp']['soc'] = variables_all['soc']
        variables_all['mlp']['eg2'] = variables_all['eg2']
        variables_all['mlp']['nac2'] = variables_all['nac2']
        variables_all['mlp']['soc2'] = variables_all['soc2']
        variables_all['schnet']['sch_eg'] = variables_all['sch_eg']
        variables_all['schnet']['sch_nac'] = variables_all['sch_nac']
        variables_all['schnet']['sch_soc'] = variables_all['sch_soc']
        variables_all['e2n2']['e2n2_eg'] = variables_all['e2n2_eg']
        variables_all['e2n2']['e2n2_nac'] = variables_all['e2n2_nac']
        variables_all['e2n2']['e2n2_soc'] = variables_all['e2n2_soc']
        variables_all['dimenet']['dime_nac'] = variables_all['dime_nac']
        variables_all['nn']['ml_seed'] = variables_all['control']['gl_seed']
        variables_all['mlp']['ml_seed'] = variables_all['control']['gl_seed']
        variables_all['schnet']['ml_seed'] = variables_all['control']['gl_seed']
        variables_all['e2n2']['ml_seed'] = variables_all['control']['gl_seed']
        variables_all['md']['gl_seed'] = variables_all['control']['gl_seed']
        variables_all['molcas']['molcas_project'] = variables_all['control']['title']
        variables_all['molcas']['verbose'] = variables_all['md']['verbose']
        variables_all['bagel']['bagel_project'] = variables_all['control']['title']
        variables_all['bagel']['verbose'] = variables_all['md']['verbose']
        variables_all['orca']['orca_project'] = variables_all['control']['title']
        variables_all['orca']['verbose'] = variables_all['md']['verbose']
        variables_all['openqp']['openqp_project'] = variables_all['control']['title']
        variables_all['openqp']['verbose'] = variables_all['md']['verbose']
        variables_all['xtb']['xtb_project'] = variables_all['control']['title']
        variables_all['xtb']['verbose'] = variables_all['md']['verbose']
        variables_all['demo'] = variables_all['nn']
        variables_all['e2n2_demo'] = variables_all['e2n2']

    # prepare starting information
    method_info_dict = {
        'demo':
            KeyNN(nn_type='demo').info(variables_all['demo']) +
            KeyMLP().info(variables_all['eg'], variables_all['nac'], variables_all['soc'], 1, nn_type='demo') +
            KeyMLP().info(variables_all['eg2'], variables_all['nac2'], variables_all['soc2'], 2, nn_type='demo'),
        'nn':
            KeyNN(nn_type='nn').info(variables_all['nn']) +
            KeyMLP().info(variables_all['eg'], variables_all['nac'], variables_all['soc'], 1, nn_type='native') +
            KeyMLP().info(variables_all['eg2'], variables_all['nac2'], variables_all['soc2'], 2, nn_type='native'),
        'mlp':
            KeyNN(nn_type='mlp').info(variables_all['nn']) +
            KeyMLP().info(variables_all['eg'], variables_all['nac'], variables_all['soc'], 1, nn_type='pyNNsMD') +
            KeyMLP().info(variables_all['eg2'], variables_all['nac2'], variables_all['soc2'], 2, nn_type='pyNNsMD'),
        'schnet':
            KeyNN(nn_type='schnet').info(variables_all['nn']) +
            KeySchNet().info(variables_all['sch_eg'], variables_all['sch_nac'], variables_all['sch_soc']),
        'e2n2':
            KeyNN(nn_type='e2n2').info(variables_all['nn']) +
            KeyE2N2().info(variables_all['e2n2_eg'], variables_all['e2n2_nac'], variables_all['e2n2_soc']),
        'e2n2_demo':
            KeyNN(nn_type='e2n2').info(variables_all['nn']) +
            KeyE2N2Demo().info(variables_all['e2n2_eg'], variables_all['e2n2_nac'], variables_all['e2n2_soc']),
        'dimenet':
            KeyNN(nn_type='dimenet').info(variables_all['nn']) +
            KeyDimeNet().info(variables_all['dime_nac']),
        'molcas': KeyMolcas().info(variables_all['molcas']),
        'mlctkr': KeyMolcas().info(variables_all['molcas']),
        'bagel': KeyBagel().info(variables_all['bagel']),
        'orca': KeyOrca().info(variables_all['orca']),
        'openqp': KeyOpenQP().info(variables_all['openqp']),
        'xtb': KeyXtb().info(variables_all['xtb']),
    }

    qm = variables_all['control']['qm']
    abinit = variables_all['control']['abinit']
    qm_info = ''.join([method_info_dict[m] for m in qm])
    ab_info = ''.join([method_info_dict[m] for m in abinit])
    control_info = KeyControl().info(variables_all['control'])
    molecule_info = KeyMolecule().info(variables_all['molecule'])
    search_info = KeySearch().info(variables_all['search'])
    adaptive_info = KeyControl().info_adaptive(variables_all['control'])
    md_info = KeyMD().info(variables_all['md'])
    hybrid_info = KeyMD.info_hybrid(variables_all['md'])

    jobtype_info_dict = {
        'sp': control_info + molecule_info + qm_info,
        'md': control_info + molecule_info + md_info + qm_info,
        'hop': control_info + molecule_info + md_info,
        'hybrid': control_info + molecule_info + md_info + qm_info + ab_info + hybrid_info,
        'adaptive': control_info + molecule_info + adaptive_info + md_info + qm_info + ab_info,
        'train': control_info + molecule_info + qm_info,
        'prediction': control_info + molecule_info + qm_info,
        'predict': control_info + molecule_info + qm_info,
        'search': control_info + molecule_info + qm_info + search_info,
    }

    jobtype = variables_all['control']['jobtype']
    log_info = jobtype_info_dict[jobtype]

    return variables_all, log_info


def deep_update(a, b):
    ## recursively update a with b
    for key, val in b.items():
        if key in a.keys():
            if isinstance(val, dict) and isinstance(a[key], dict):
                a[key] = deep_update(a[key], val)
            else:
                a[key] = val
        else:
            a[key] = val

    return a
