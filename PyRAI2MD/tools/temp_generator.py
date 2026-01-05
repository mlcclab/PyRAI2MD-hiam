## ----------------------
## Quantum chemical calculation template generator
## ----------------------
##
## New version Dec 13 2022 Jingbai Li

bool_dict = {
        '1': 1,
        'y': 1,
        'yes': 1,
        '0': 0,
        'n': 0,
        'no': 0,
}

def write_molcas_gateway(basis, do_rela):
    if do_rela == 1:
        rela = 'AMFI\nangmom\n0 0 0'
    else:
        rela = ''

    gateway = '&GATEWAY\ncoord=$MOLCAS_PROJECT.xyz\nbasis\n%s\nGroup=c1\nRICD\n%s' % (basis, rela)

    return gateway

def write_molcas_seward(do_rela):
    if do_rela == 1:
        rela = 'rela\nR02O\nrelint\nexpert'
    else:
        rela = ''
    seward = '&SEWARD\ndoanalytic\n%s' % rela

    return seward

def write_molcas_rasscf(cas):
    mult = cas['mult']
    forb = cas['forb']
    nel = cas['nel']
    norb = cas['norb']
    astate = cas['astate']

    if mult == 'singlet':
        fileorb = 'fileorb=$MOLCAS_PROJECT.StrOrb'
        spin = 1
    else:
        fileorb = 'LumOrb'
        spin = 3
    rasscf = '&RASSCF\n%s\nSpin=%s\nNactel=%s 0 0\nInactive=%s\nRas2=%s\nITERATIONS=200,100\nCIRoot=%s %s 1' % (
        fileorb, spin, nel, forb, norb, astate, astate
    )

    return rasscf

def all_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(n - i - 1):
            pairs.append([i + 1, j + i + 2])

    return pairs

def neighbors(n):
    pairs = []
    for i in range(n):
        if i + 1 < n:
            pairs.append([i + 1, i + 2])

    return pairs

def inter_pairs(m, n):
    pairs = []
    for i in range(m):
        for j in range(n):
            pairs.append([i + 1, j + 1])

    return pairs

def write_molcas_alaska(cas, grad, nac):
    # grad 1 all 0 no
    # nac 2 all 1 neighbor 0 no
    nstate = cas['nstate']
    alaska = ''

    if grad > nstate:
        grad = nstate

    if grad == 1:
        grad_list = [x + 1 for x in range(nstate)]
    elif grad > 1:
        grad_list = [x + 1 for x in range(grad)]
    else:
        grad_list = []

    for i in grad_list:
        alaska += '&ALASKA\nROOT=%s\n' % i

    if nac == 2 and grad <= 1:
        nac_list = all_pairs(nstate)
    elif nac == 1 and grad <= 1:
        nac_list = neighbors(nstate)
    elif nac == 2 and grad > 1:
        nac_list = all_pairs(grad)
    elif nac == 1 and grad > 1:
        nac_list = neighbors(grad)
    else:
        nac_list = []

    for pair in nac_list:
        a, b = pair
        alaska += '&ALASKA\nNAC=%s %s\n' % (a, b)

    return alaska

def write_molcas_rassi(singlet, triplet):
    ns = singlet['nstate']
    nt = triplet['nstate']
    ss = ' '.join([str(x + 1) for x in range(ns)])
    st = ' '.join([str(x + 1) for x in range(nt)])
    rassi = '&RASSI\nNrofJobIph=2 %s %s;%s;%s\nSpinOrbit\nEJob\nSOCOupling=0\n' % (ns, nt, ss, st)

    return rassi

def setup_molcas_input(keyword):
    basis = keyword['basis']
    do_rela = keyword['do_rela']
    singlet = keyword['singlet']
    triplet = keyword['triplet']
    grad = keyword['grad']
    nac = keyword['nac']

    gateway = write_molcas_gateway(basis, do_rela)
    seward = write_molcas_seward(do_rela)
    ras_s = write_molcas_rasscf(singlet)
    ala_s = write_molcas_alaska(singlet, grad, nac)
    if triplet is None:
        ras_t = ''
        ala_t = ''
        rassi = ''
        cp1 = ''
        cp2 = ''
    else:
        ras_t = write_molcas_rasscf(triplet)
        ala_t = write_molcas_alaska(triplet, grad, nac)
        rassi = write_molcas_rassi(singlet, triplet)
        cp1 = '>>COPY  $WorkDir/$Project.JobIph  $WorkDir/JOB001\n'
        cp2 = '>>COPY  $WorkDir/$Project.JobIph  $WorkDir/JOB002\n'

    template = '%s\n\n%s\n\n%s\n\n%s\n\n%s\n\n%s\n\n%s\n\n%s\n\n%s\n' % (
        gateway, seward, ras_s, ala_s, cp1, ras_t, ala_t, cp2, rassi
    )

    return template

def read_molcas(mult):
    print('\n    Setting up sections for %s\n' % mult)
    forb = input("""
    -----------------------------------------------------------------
    how many frozen orbitals does the %s have?
    """ % mult)

    nel = input("""
    how many activate electrons does the %s have?
    """ % mult)

    norb = input("""
    how many activate orbitals does the %s have?
    """ % mult)

    astate = input("""
    how many %s states does the CASSCF calculation average?
    """ % mult)

    nstate = input("""
    how many %s state does the NAMD propagate (<= CASSCF states)?
    """ % mult)

    print('\n    -----------------------------------------------------------------\n')

    if nstate > astate:
        nstate = astate

    key_dict = {
        'mult': mult,
        'forb': int(forb),
        'nel': int(nel),
        'norb': int(norb),
        'astate': int(astate),
        'nstate': int(nstate),
    }

    return key_dict

def determine_coupling(singlet, triplet, nac):
    ns = singlet['nstate']
    if triplet is None:
        nt = 0
    else:
        nt = triplet['nstate']

    if nac == 2:
        info = """
    -----------------------------------------------------------------
    you have requested molcas to compute all nonadiabatic couplings
    the following information is needed for the PyRAI2MD &MOLECULE keyword <coupling> 
    """
        nac_s = all_pairs(ns)
        nac_t = all_pairs(nt)
    elif nac == 1:
        info = """
    -----------------------------------------------------------------
    you have requested molcas to compute neighboring nonadiabatic couplings
    the following information is needed for the PyRAI2MD &MOLECULE keyword <coupling> 
        """
        nac_s = neighbors(ns)
        nac_t = neighbors(nt)
    else:
        info = """
    -----------------------------------------------------------------
    you have not requested molcas to compute any nonadiabatic couplings
    if this not what you intended, something was wrong with the number of state, please rerun this script.
    
    if you intended to use kTDC method in PyRAI2MD, do not forget to add the state pairs for nonadiabatic coupling
    in &MOLECULE keyword <coupling>. The following only show the spin-orbit coupling pairs if there is any.
    """
        nac_s = []
        nac_t = []

    soc = inter_pairs(ns, nt)
    coupling = []
    for pair in nac_s:
        a, b = pair
        coupling.append('%s %s' % (a, b))

    for pair in nac_t:
        a, b = pair
        coupling.append('%s %s' % (a + ns, b + ns))

    for pair in soc:
        a, b = pair
        coupling.append('%s %s' % (a, b + ns))

    print('\n    -----------------------------------------------------------------\n')
    print('\n    Checking interstate coupling information\n')
    print(info)
    print('\n    %s\n' % ', '.join(coupling))
    print('\n    -----------------------------------------------------------------\n')

def molcas_input(title):
    print('\n    Generating Molcas input template\n')

    basis = input("""
    -----------------------------------------------------------------
    what is the basis set?
        e.g. ano-s-vdzp or ano-rcc-vdzp
    """)

    choice = input("""
    do you want to include relativistic effects? (yes/no)
        for SOC, this is needed
    """)

    do_rela = bool_dict[choice.lower()]

    choice = input("""
    do you need to include triplet? (yes/no)
        for SOC, this is needed
    """)

    print('\n    -----------------------------------------------------------------\n')

    do_trip = bool_dict[choice.lower()]

    singlet = read_molcas('singlet')
    if do_trip:
        triplet = read_molcas('triplet')
    else:
        triplet = None

    grad = input("""
    do you want to compute gradients for all states? (yes/no)
        for dynamics using activestate 1, choose no
        for dynamics using gsh, choose yes
        for adaptive sampling, choose yes
    """)

    grad = bool_dict[grad]

    nac = input("""
    do you want to compute nonadiabatic couplings? (yes/no)
        for dynamics using fssh with nac, choose yes
        for dynamics using gsh or fssh with ktdc, choose no
        for adaptive sampling, choose yes to train NN for NAC prediction. Otherwise, choose no.
    """)

    nac = bool_dict[nac]
    if nac == 1:
        full_nac = input("""
    do you want to compute all nonadiabatic couplings? (yes/no)
        for full nonadiabatic coupling matrix, choose yes
        for nonadiabatic couplings between neighboring states, choose no
    """)
        nac += bool_dict[full_nac]

    keyword = {
        'basis': basis,
        'do_rela': do_rela,
        'singlet': singlet,
        'triplet': triplet,
        'grad': grad,
        'nac': nac,
    }

    template = setup_molcas_input(keyword)
    determine_coupling(singlet, triplet, nac)

    with open('%s.molcas' % title, 'w') as out:
        out.write(template)

    print('\n    saving input template -> %s.molcas\n' % title)
    print('\n    COMPLETE\n')

def write_pmd_control(title, jobtype):
    print('\n    &CONTROL section\n')
    print('\n    -----------------------------------------------------------------\n')

    if jobtype == 'train':
        ml_ncpu = 4
        qc_ncpu = 1

    elif jobtype == 'search':
        ml_ncpu = input("""
    how many threads do you want to use for grid search? (e.g., 20)
    """)
        qc_ncpu = 1

    elif jobtype == 'adaptive':
        ml_ncpu = input("""
    how many threads do you want to use for ml-namd calculations? (e.g., 20)
    """)
        qc_ncpu = input("""
    how many threads do you want to use for qm calculations? (e.g., 75)
    """)

    else:
        ml_ncpu = 1
        qc_ncpu = 1

    qm = input("""
    what is the qm method? (enter one of the following options)
        name            compatible jobs                     descriptions
        nn          train search adaptive md    use trained neural network for qm calculations
        library        train search adaptive md    use trained excited-equivariant neural network for qm calculations
        molcas                   adaptive md    use molcas for qm calculations
        bagel                    adaptive md    use bagel for qm calculations
        orca                     adaptive md    use orca for qm calculations
        xtb                      adaptive md    use xtb for qm calculations
        nn xtb                            md    use trained excited-equivariant nn and xtb for qmqm2 calculation
        library xtb                          md    use neural network for qm calculations
        molcas xtb                        md    use molcas and xtb for qmqm2 calculation
        bagel xtb                         md    use bagel and xtb for qmqm2 calculation
        orca xtb                          md    use orca and xtb for qmqm2 calculation
    """)

    methods = qm.split()[0:]

    control = '&CONTROL\ntitle %s\njobtype %s\nqc_ncpu %s\nml_ncpu %s\nqm %s\n' % (
        title, jobtype, qc_ncpu, ml_ncpu, qm
    )

    if methods[0] in ['nn', 'library'] and jobtype == 'md':
        check_error = True
    elif jobtype == 'adaptive':
        check_error = True
    else:
        check_error = False

    if check_error:
        maxenergy = input("""
    you are using neural networks for qm calculation
    what is the max deviation for uncertain energy predictions? (in Hartree, e.g., 0.05)
    """)
        maxgrad = input("""
    what is the max deviation for uncertain gradient predictions? (in Hartree/Bohr, e.g., 0.25)
    """)
        maxsoc = input("""
    what is the max deviation for uncertain soc predictions? (in cm-1, e.g., 50)
    """)
        control += 'minenergy %s\nmaxenergy %s\nmingrad %s\nmaxgrad %s\nminsoc %s\nmaxsoc %s\n' % (
            maxenergy, maxenergy, maxgrad, maxgrad, maxsoc, maxsoc
        )

    if jobtype == 'adaptive':
        abinit = input("""
    you are doing adaptive sampling
    what is the qm method for training data calculation?
        e.g., molcas, bagel, or orca
    """)
        maxiter = input("""
    how many iterations do you want to do for adaptive sampling?
    """)
        control += 'abinit %s\nmaxiter %s\n' % (abinit, maxiter)

    control += '\n'

    return control, methods

def auto_coupling(ns, nt, nac, soc):

    if nac == 2:
        nac_s = all_pairs(ns)
        nac_t = all_pairs(nt)
    elif nac == 1:
        nac_s = neighbors(ns)
        nac_t = neighbors(nt)
    else:
        nac_s = []
        nac_t = []

    if soc == 1:
        soc = inter_pairs(ns, nt)
    else:
        soc = []

    coupling = []
    for pair in nac_s:
        a, b = pair
        coupling.append('%s %s' % (a, b))

    for pair in nac_t:
        a, b = pair
        coupling.append('%s %s' % (a + ns, b + ns))

    for pair in soc:
        a, b = pair
        coupling.append('%s %s' % (a, b + ns))

    if len(coupling) > 0:
        coupling = ', '.join(coupling)
    else:
        coupling = None

    return coupling

def getindex(index):
    ## This function read single, range, separate range index and convert them to a list
    index_list = []
    for i in index:
        if '-' in i:
            a, b = i.split('-')
            a, b = int(a), int(b)
            index_list += range(a, b + 1)
        else:
            index_list.append(int(i))

    index_list = sorted(list(set(index_list)))  # remove duplicates and sort from low to high
    return index_list

def write_pmd_molecule():
    print('\n    &MOLECULE section\n')

    ns = input("""
    -----------------------------------------------------------------
    how many singlet state do you want to include? 
    """)

    ns = int(ns)

    nt = input("""
    -----------------------------------------------------------------
    how many triplet state do you want to include? 
    """)

    nt = int(nt)

    if nt != 0:
        ci = '%s %s' % (ns, nt)
        spin = '0 1'
    else:
        ci = '%s' % ns
        spin = '0'

    molecule = '&MOLECULE\nci %s\nspin %s\n' % (ci, spin)

    nac = input("""
    do you want to define nonadiabatic couplings? (yes/no)
    """)

    nac = bool_dict[nac]

    if nac == 1:
        full_nac = input("""
    do you want to define all nonadiabatic couplings? (yes/no)
        for full nonadiabatic coupling matrix, choose yes
        for nonadiabatic couplings between neighboring states, choose no
    """)

        nac += bool_dict[full_nac]

    if ns > 0:
        soc = input("""
    do you want to define spin-orbit couplings? (yes/no)
    """)

        soc = bool_dict[soc]
    else:
        soc = 0

    coupling = auto_coupling(ns, nt, nac, soc)

    if coupling is not None:
        molecule += 'coupling %s\n' % coupling

    qmqm2 = input("""
    do you want to do qmqm2 calculation? (yes/no)
    """)

    qmqm2 = bool_dict[qmqm2]

    if qmqm2 == 1:
        highlevel = input("""
    enter the atom indices in the qm region. e.g 1 2 3-5
    """)
        embedding = input("""
    do you want to do electronic embedding in the qm region? (yes/no)
    """)
        embedding = bool_dict[embedding]

        molecule += 'highlevel %s\nembedding %s\n' % (highlevel, embedding)

    fix = input("""
    do you want to freeze any atoms? (yes/no)
    """)

    fix = bool_dict[fix]

    if fix == 1:
        freeze = input("""
    enter the frozen atom indices. e.g 1 2 3-5
    """)
        molecule += 'freeze %s\n' % freeze

    constrain = input("""
    do you want to add external constraining potential? (yes/no)
    """)

    constrain = bool_dict[constrain]

    if constrain == 1:
        shape = input("""
    define the shape of the potential. (ellipsoid/cuboid)
    """)

        cavity = input("""
    define the radius of the potential.(in Angstrom)
    """)

        molecule += 'shape %s\ncavity %s\n' % (shape, cavity)

        set_center = input("""
    do you want to modify the center of the potential? (yes/no)
    """)

        set_center = bool_dict[set_center]

        if set_center == 1:
            center = input("""
    enter the atom indices to define a new center. e.g 1 2 3-5
    """)
            molecule += 'center %s\n' % center

    track = input("""
    do you want to track the distance changes during NAMD? (yes/no)
    """)

    track = bool_dict[track]

    if track == 1:
        track_type = input("""
    chose the type of distance to track during NAMD, e.g.:
        frag - tracking distance between two fragments
        dist - tracking distances between several pairs of atoms
    """)

        track_index = input("""
    enter the atom indices to define fragments or atom pairs, e.g.:
        frag - 1 2 3 4, 5 6 7 8
        dist - 1 2, 3 4, 5 6, 7 8
    """)

        track_thrhd = input("""
    enter threshold to early stop trajectories if tracked distance(s) exceed the threshold (in Angstrom).
    """)

        molecule += 'track_type %s\ntrack_index %s\ntrack_thrhd %s\n' % (track_type, track_index, track_thrhd)

    molecule += '\n'

    return molecule

def write_pmd_ml(m):
    print('\n    &%s section\n' % m.upper())
    ml = '&%s\n' % m.upper()

    set_modeldir = input("""
    -----------------------------------------------------------------
    do you want to load or train neural networks in the current folder? (yes/no)
    """)

    set_modeldir = bool_dict[set_modeldir]

    if set_modeldir == 0:
        modeldir = input("""
    enter the path to the neural network model
    """)

        ml += 'modeldir %s\n' % modeldir

    train_data = input("""
    enter the path to the training data
    """)

    nsplits = input("""
    how many folds do you want to split the training data for train and validation set
    """)

    num_eg = input("""
    how many different hyperparameter sets for the energy+gradient model?
    """)

    num_nac = input("""
    how many different hyperparameter sets for the nonadiabatic coupling model?
    """)

    num_soc = input("""
    how many different hyperparameter sets for the spin-orbit coupling model?
    """)

    gpu = input("""
    do you have GPU support? (yes/no)
    """)

    gpu = bool_dict[gpu]

    ml += 'train_data %s\nnsplits %s\nnn_eg_type %s\nnn_nac_type %s\nnn_soc_type %s\ngpu %s\n\n' % (
        train_data, nsplits, num_eg, num_nac, num_soc, gpu
    )

    return ml, int(num_eg), int(num_nac), int(num_soc)

def write_pmd_nn_search():
    depth = input("""
    -----------------------------------------------------------------
    enter the values to search the number of hidden layers
    """)

    nn_size = input("""
    enter the values to search the number of neuron per hidden layer
    """)

    batch_size = input("""
    enter the values to search the number of batch size
    """)

    reg_l1 = input("""
    enter the values to search the number of the l1 factor        
    """)

    reg_l2 = input("""
    enter the values to search the number of the l2 factor
    """)

    dropout = input("""
    enter the values to search the number of the dropout ratio
    """)

    hyp = '&SEARCH\ndepth %s\nnn_size %s\nbatch_size %s\nreg_l1 %s\nreg_l2 %s\ndropout %s\n\n' % (
        depth, nn_size, batch_size, reg_l1, reg_l2, dropout
    )

    return hyp

def write_pmd_nn_hyp(m, n, search=False):
    if n == 0:
        print('\n    &%s section\n' % m.upper())
        hyp = '&%s\n' % m.upper()
    else:
        print('\n    &%s2 section\n' % m.upper())
        hyp = '&%s2\n' % m.upper()

    hyp += 'activ leaky_softplus\nactiv_alpha 0.3\n'

    print('\n    -----------------------------------------------------------------\n')

    if not search:
        depth = input("""
    how many hidden layer do you want to use?
    """)

        nn_size = input("""
    how many neuron per hidden layer do you want to use?
    """)

        batch_size = input("""
    how many batches do you want to use?
    """)

        hyp += 'depth %s\nnn_size %s\nbatch_size %s\n' % (depth, nn_size, batch_size)

        use_reg_l1 = input("""
    do you want to apply l1 regularization? (yes/no)
    """)

        use_reg_l1 = bool_dict[use_reg_l1]

        if use_reg_l1 == 1:
            reg_l1 = input("""
    enter the l1 factor        
    """)
        else:
            reg_l1 = 0

        use_reg_l2 = input("""
    do you want to apply l2 regularization? (yes/no)
    """)

        use_reg_l2 = bool_dict[use_reg_l2]

        if use_reg_l2 == 1:
            reg_l2 = input("""
    enter the l2 factor        
    """)
        else:
            reg_l2 = 0

        if use_reg_l1 == 1 and use_reg_l2 == 1:
            hyp += 'use_reg_activ l1_l2\nuse_reg_weights l1_l2\nuse_reg_bias l1_l2\nreg_l1 %s\nreg_l2 %s\n' % (
                reg_l1, reg_l2
            )

        elif use_reg_l1 == 1 and use_reg_l2 == 0:
            hyp += 'use_reg_activ l1\nuse_reg_weights l1\nuse_reg_bias l1\nreg_l1 %s\n' % reg_l1

        elif use_reg_l1 == 0 and use_reg_l2 == 1:
            hyp += 'use_reg_activ l2\nuse_reg_weights l2\nuse_reg_bias l2\nreg_l2 %s\n' % reg_l2

        use_dropout = input("""
    do you want to apply dropout? (yes/no)
    """)

        use_dropout = bool_dict[use_dropout]

        if use_dropout == 1:
            dropout = input("""
    enter the dropout ratio
    """)
            hyp += 'use_dropout 1\ndropout %s\n' % dropout

    if m == 'eg':
        loss_weights = input("""
    what is the ratio between the energy and gradient loss? (e.g., 1 1)
    """)
        hyp += 'loss_weights %s\n' % loss_weights

    if m == 'nac':
        phase_less = input("""
    do you want to use phase less loss for training nonadiabatic coupling model? (yes/no)
    """)

        phase_less = bool_dict[phase_less]

        hyp += 'phase_less_loss %s' % phase_less

    epo = input("""
    how many total epochs do you want to train?
    """)

    steps = input("""
    what are the learning rates in the learning rate reductions? (e.g., 1e-3 1e-5 1e-5)
    """)

    lr_step = input("""
    how many epochs in each learning rate reduction? (e.g., 100 100 100)
    """)

    hyp += 'epo %s\nepostep 10\nlearning_rate 1e-3\nlearning_rate_step %s\nepoch_step_reduction %s\n\n' % (
        epo, steps, lr_step
    )

    return hyp

def write_pmd_nn(jobtype):
    search = jobtype == 'search'
    nn, num_eg, num_nac, num_soc = write_pmd_ml(m='nn')

    eg = ''
    for n in range(min(num_eg, 2)):
        eg += write_pmd_nn_hyp(m='eg', n=n, search=search)

    nac = ''
    for n in range(min(num_nac, 2)):
        nac += write_pmd_nn_hyp(m='nac', n=n, search=search)

    soc = ''
    for n in range(min(num_soc, 2)):
        soc += write_pmd_nn_hyp(m='soc', n=n, search=search)

    pmd_nn = '%s%s%s%s' % (nn, eg, nac, soc)

    return pmd_nn


def write_pmd_e2n2_search():
    n_features = input("""
    -----------------------------------------------------------------
    enter the values to search the number of the node features
    """)

    n_blocks = input("""
    enter the values to search the number of the interaction blocks
    """)

    l_max = input("""
    enter the values to search the number of the rotation order
    """)

    n_rbf = input("""
    enter the values to search the number of the radial basis function     
    """)

    rbf_layers = input("""
    enter the values to search the number of the hidden layer in radial basis
    """)

    rbf_neurons = input("""
    enter the values to search the number of the neurons per hidden layers in radial basis
    """)

    hyp = '&SEARCH\ndepth %s\nnn_size %s\nbatch_size %s\nreg_l1 %s\nreg_l2 %s\ndropout %s\n\n' % (
        n_features, n_blocks, l_max, n_rbf, rbf_layers, rbf_neurons
    )

    return hyp

def write_pmd_e2n2_hyp(m, search=False):
    print('\n    &E2N2_%s section\n' % m.upper())
    hyp = '&E2N2_%s\n' % m.upper()

    print('\n    -----------------------------------------------------------------\n')

    if not search:
        n_features = input("""
    how many node features do you want to use?
    """)

        n_blocks = input("""
    how many interaction blocks do you want to use?
    """)

        l_max = input("""
    what is the largest rotation order?
    """)

        n_rbf = input("""
    how many radial basis functions do you want to use?
    """)

        rbf_layer = input("""
    how many hidden layers do you want to use for the radial basis?
    """)

        rbf_neurons = input("""
    how many neurons per hidden layers do you want to use for the radial basis?
    """)
        hyp += 'n_features %s\nn_blocks %s\nl_max %s\nn_rbf %s\nrbf_layers %s\nrbf_neurons %s\n' % (
            n_features, n_blocks, l_max, n_rbf, rbf_layer, rbf_neurons
        )

    maxradius = input("""
    what is the radius to find edges for each atom? (in Angstrom)
    """)

    n_edges = input("""
    how many edges do you want to include for each atomic center? (0 for all edges)
    """)

    batch_size = input("""
    what is the batch size for training? (e.g., 5)
    """)

    val_batch_size = input("""
    what is the batch size for validation? (e.g., 5)
    """)

    hyp += 'maxradius %s\nn_edges %s\nbatch_size %s\nval_batch_size %s\n' % (
        maxradius, n_edges, batch_size, val_batch_size
    )

    if m == 'eg':
        loss_weights = input("""
    what is the ratio between the energy and gradient loss? (e.g., 1 1)
    """)
        hyp += 'loss_weights %s\n' % loss_weights

    epo = input("""
    how many total epochs do you want to train?
    """)

    steps = input("""
    what are the learning rates in the learning rate reductions? (e.g., 1e-3 1e-e 1e-5)
    """)

    lr_step = input("""
    how many epochs in each learning rate reduction? (e.g., 100 100 100)
    """)

    hyp += 'epo %s\nepostep 10\nlearning_rate 1e-3\nlearning_rate_step %s\nepoch_step_reduction %s\n\n' % (
        epo, steps, lr_step
    )

    return hyp

def write_pmd_e2n2(jobtype):
    search = jobtype == 'search'
    nn, num_eg, num_nac, num_soc = write_pmd_ml(m='library')
    eg = ''
    if num_eg > 0:
        eg += write_pmd_e2n2_hyp(m='eg', search=search)

    nac = ''
    if num_nac > 0:
        nac += write_pmd_e2n2_hyp(m='nac', search=search)

    soc = ''
    if num_soc > 0:
        soc += write_pmd_e2n2_hyp(m='soc', search=search)

    pmd_e2n2 = '%s%s%s%s' % (nn, eg, nac, soc)

    return pmd_e2n2

def write_pmd_molcas(_):
    print('\n    &MOLCAS section\n')
    pmd_molcas = '&MOLCAS\n'

    molcas = input("""
    -----------------------------------------------------------------
    enter the path to the molcas executable
    """)

    pmd_molcas += 'molcas %s\n' % molcas

    set_calcdir = input("""
    do you want to specify a folder to run molcas calculation? (yes/no)
        for adaptive sampling, choose yes 
    """)

    set_calcdir = bool_dict[set_calcdir]

    if set_calcdir == 1:
        calcdir = input("""
    enter the path to molcas calculation folder
    """)
        pmd_molcas += 'molcas_calcdir %s\n' % calcdir
    else:
        print('\n    molcas calculation will run in the current folder\n')

    set_workdir = input("""
    do you want to run molcas on local disk? (yes/no) 
    """)

    set_workdir = bool_dict[set_workdir]

    if set_workdir == 1:
        pmd_molcas += 'molcas_workdir AUTO\n'
    else:
        print('\n    molcas calculation will run in the current folder\n')

    hpc = input("""
    do you want to submit molcas calculations to job schedular? (yes/no)
        for md choose no
        for adaptive sampling choose yes 
    """)

    hpc = bool_dict[hpc]

    if hpc == 1:
        pmd_molcas += 'use_hpc 1\n'

    pmd_molcas += '\n'

    return pmd_molcas

def write_pmd_bagel(_):
    print('\n    &BAGEL section\n')
    pmd_bagel = '&BAGEL\n'

    bagel = input("""
    -----------------------------------------------------------------
    enter the path to the bagel executable
    """)

    pmd_bagel += 'bagel %s\n' % bagel

    set_calcdir = input("""
    do you want to specify a folder to run bagel calculation? (yes/no)
        for adaptive sampling, choose yes 
    """)

    set_calcdir = bool_dict[set_calcdir]

    if set_calcdir == 1:
        calcdir = input("""
    enter the path to bagel calculation folder
    """)
        pmd_bagel += 'bagel_workdir %s\n' % calcdir
    else:
        print('\n    bagel calculation will run in the current folder\n')

    oneapi = input("""
    do you have intel OneAPI for mkl and mpi?
    """)

    oneapi = bool_dict[oneapi]

    if oneapi == 1:
        mkl = input("""
    enter the path to the OneAPI setvar.sh file.
    """)
        pmd_bagel += 'mkl %s\n' % mkl
    else:
        mpi = input("""
    enter the path to the MPI library. (openmpi, intel mpi, or mavpich)
    """)
        mkl = input("""
    enter the path to the MKL library.
    """)
        arch = input("""
    enter your CPU architecture for MKL. (e.g., intel64)
    """)

        pmd_bagel += 'mpi %s\nmkl %s\narch %s\n' % (mpi, mkl, arch)

    blas = input("""
    enter the path to the BLAS library.
    """)

    lapack = input("""
    enter the path to the LAPACK library.
    """)

    boost = input("""
    enter the path to the BOOST library.
    """)

    pmd_bagel += 'blas %s\nlapack %s\nboost %s\n' % (blas, lapack, boost)

    hpc = input("""
    do you want to submit bagel calculations to job schedular? (yes/no)
        for md choose no
        for adaptive sampling choose yes 
    """)

    hpc = bool_dict[hpc]

    if hpc == 1:
        pmd_bagel += 'use_hpc 1\n'

    pmd_bagel += '\n'

    return pmd_bagel

def write_pmd_orca(_):
    print('\n    &ORCA section\n')
    pmd_orca = '&ORCA\n'

    orca = input("""
    -----------------------------------------------------------------
    enter the path to the orca executable
    """)

    pmd_orca += 'orca %s\n' % orca

    set_calcdir = input("""
    do you want to specify a folder to run orca calculation? (yes/no)
        for adaptive sampling, choose yes 
    """)

    set_calcdir = bool_dict[set_calcdir]

    if set_calcdir == 1:
        calcdir = input("""
    enter the path to orca calculation folder
    """)
        pmd_orca += 'orca_workdir %s\n' % calcdir
    else:
        print('\n    orca calculation will run in the current folder\n')

    mpi = input("""
    enter the path to the openmpi library
    """)

    pmd_orca += 'mpi %s\n' % mpi

    hpc = input("""
    do you want to submit orca calculations to job schedular? (yes/no)
        for md choose no
        for adaptive sampling choose yes 
    """)

    hpc = bool_dict[hpc]

    if hpc == 1:
        pmd_orca += 'use_hpc 1\n'

    pmd_orca += '\n'

    return pmd_orca

def write_pmd_xtb(_):
    print('\n    &XTB section\n')
    pmd_xtb = '&XTB\n'

    xtb = input("""
    -----------------------------------------------------------------
    enter the path to the xtb executable
    """)

    pmd_xtb += 'xtb %s\n' % xtb

    set_calcdir = input("""
    do you want to specify a folder to run xtb calculation? (yes/no)
        for adaptive sampling, choose yes 
    """)

    set_calcdir = bool_dict[set_calcdir]

    if set_calcdir == 1:
        calcdir = input("""
    enter the path to xtb calculation folder
    """)
        pmd_xtb += 'xtb_workdir %s\n' % calcdir
    else:
        print('\n    xtb calculation will run in the current folder\n')

    xtb_nproc = input("""
    how many threads do you want to run xtb?
    """)

    gfnver = input("""
    which version of GFN do you want to use?
        -1  force field
        0   GFN0
        1   GFN1
        2   GFN2
    """)

    mem = input("""
    specify xtb memory stacksize (1000atoms for 1000MB)
    """)

    pmd_xtb += 'xtb_nproc %s\ngfnver %s\nmem %s\n\n' % (xtb_nproc, gfnver, mem)

    return pmd_xtb

def write_pmd_search(m):
    print('\n    &SEARCH section\n')

    if m == 'nn':
        search = write_pmd_nn_search()
    elif m == 'library':
        search = write_pmd_e2n2_search()
    else:
        search = ''

    return search

def write_pmd_method(methods, jobtype):
    method_dict = {
        'nn': write_pmd_nn,
        'library': write_pmd_e2n2,
        'molcas': write_pmd_molcas,
        'bagel': write_pmd_bagel,
        'orca': write_pmd_orca,
        'xtb': write_pmd_xtb,
    }

    method = ''
    for m in methods:
        method += method_dict[m](jobtype)

    return method

def write_pmd_md():
    print('\n    &MD section\n')

    step = input("""
    -----------------------------------------------------------------
    how many steps do you want to simulate? 
    """)

    size = input("""
    how long is the step size? (in atomic unit, e.g, 20.67 for 0.5 fs) 
    """)

    root = input("""
    which state is the initial state?
    """)

    sfhp = input("""
    which surface hopping algorithm do you want to use?
        fssh for fewest switches surface hopping
        gsh for Zhu-Nakamura surface hopping
        no for no surface hopping
    """)

    md = '&MD\nstep %s\nsize %s\nroot %s\n' % (step, size, root)

    if sfhp == 'fssh':
        nactype = input("""
    what type of nonadiabatic coupling do you want to use for fssh?
        nac     compute nonadiabatic coupling from wave function
        ktdc    compute nonadiabatic coupling from energies
    """)
        md += 'sfhp %s\nnactype %s\n' % (sfhp, nactype)
        if nactype == 'nac':
            md += 'phasecheck 1\n'

    elif sfhp == 'gsh':
        gap = input("""
    what is the energy gap threshold for gsh internal conversion? (in eV, e.g., 0.5)
    """)
        dosoc = input("""
    do you want to include intersystem crossing? (yes/no)
    """)
        dosoc = bool_dict[dosoc]

        md += 'sfhp %s\ngap %s\ndosoc %s\n' % (sfhp, gap, dosoc)

        if dosoc == 1:
            gapsoc = input("""
    what is the energy gap threshold for gsh intersystem crossing? (in eV, e.g., 0.5)
    """)
            md += 'gapsoc %s\n' % gapsoc
    else:
        md += 'sfhp no\n'

    grad = input("""
    do you want to only compute gradient for the current state? 
        for fssh dynamics, choose yes to save time
        for gsh dynamics, choose no
    """)
    grad = bool_dict[grad]

    md += 'activestate %s\n' % grad

    thermo = input("""
    do you want to use thermostat? (yes/no)
    """)

    thermo = bool_dict[thermo]

    if thermo == 1:
        temp = input("""
    what is the target temperature? (in K)
    """)
        md += 'thermo nvt\ntemp %s\n' % temp
    else:
        md += 'thermo off\n'

    direct = input("""
    how many steps do you want to save in log directly?
    """)

    buffer = input("""
    how many steps do you want to skip after direct saving?
    """)

    md += 'direct %s\nbuffer %s\n\n' % (direct, buffer)

    return md

def setup_pmd_md(title):
    print('\n    Setting for molecular dynamics\n')
    control, methods = write_pmd_control(title, 'md')
    molecule = write_pmd_molecule()
    method = write_pmd_method(methods, 'md')
    md = write_pmd_md()

    template = '%s%s%s%s' % (control, molecule, method, md)

    return template

def setup_pmd_train(title):
    print('\n    Setting for training neural networks\n')
    control, methods = write_pmd_control(title, 'train')
    method = write_pmd_method(methods, 'train')

    template = '%s%s' % (control, method)

    return template

def setup_pmd_search(title):
    print('\n    Setting for neural network grid search\n')
    control, methods = write_pmd_control(title, 'search')
    search = write_pmd_search(methods[0])
    method = write_pmd_method(methods, 'search')

    template = '%s%s%s' % (control, search, method)

    return template

def setup_pmd_adaptive(title):
    print('\n    Setting for adaptive sampling\n')
    control, methods = write_pmd_control(title, 'adaptive')
    molecule = write_pmd_molecule()
    method = write_pmd_method(methods, 'adaptive')
    md = write_pmd_md()

    template = '%s%s%s%s' % (control, molecule, method, md)

    return template

def pmd_input(title):
    print('\n    Generating PyRAI2MD input\n')

    jobtype = input("""
    -----------------------------------------------------------------
    what is the job type? (enter one of the following options)
        md          nonadiabatic molecular dynamics
        train       training neural networks
        search      neural network hyperparameter grid search 
        adaptive    adaptive sampling training neural networks
    """)

    job_dict = {
        'md': setup_pmd_md,
        'train': setup_pmd_train,
        'search': setup_pmd_search,
        'adaptive': setup_pmd_adaptive,
    }

    template = job_dict[jobtype](title)

    with open('input', 'w') as out:
        out.write(template)

    print('\n    saving PyRAI2MD input -> input\n')
    print('\n    COMPLETE\n')

def bagel_input(title):
    print('\n    Generating BAGEL input template\n')

    basis = input("""
    -----------------------------------------------------------------
    what is the basis set?
        e.g. cc-pvdz or svp
    """)

    df_basis = input("""
    -----------------------------------------------------------------
    what is the df basis set?
        e.g. cc-pvdz-jkfit or svp-jkfit
    """)

    forb = input("""
    -----------------------------------------------------------------
    how many frozen orbitals?
    """)

    charge = input("""
    how many charges?
    """)

    norb = input("""
    how many activate orbitals?
    """)

    astate = input("""
    how many states for CASSCF state-averaging?
    """)

    nstate = input("""
    how many states for NAMD simulation(<=CASSCF states) or gradient data calculation?
    """)

    caspt2 = input("""
    do you want to do XMS-CASPT2? (yes/no)
    """)
    
    nac = input("""
    do you want to compute nonadiabatic couplings? (yes/no)
        for dynamics using fssh with nac, choose yes
        for dynamics using gsh or fssh with ktdc, choose no
        for adaptive sampling, choose yes to train NN for NAC prediction. Otherwise, choose no.
    """)

    nac = bool_dict[nac.lower()]
    if nac == 1:
        full_nac = input("""
    do you want to compute all nonadiabatic couplings? (yes/no)
        for full nonadiabatic coupling matrix, choose yes
        for nonadiabatic couplings between neighboring states, choose no
    """)
        nac += bool_dict[full_nac.lower()]

    caspt2 = bool_dict[caspt2.lower()]

    keyword = {
        'title': title,
        'basis': basis,
        'df_basis': df_basis,
        'forb': forb,
        'charge': charge,
        'norb': norb,
        'astate': astate,
        'nstate': int(nstate),
        'nac': nac,
        'caspt2': caspt2
    }

    template = setup_bagel_input(keyword)

    with open('%s.bagel' % title, 'w') as out:
        out.write(template)

    print('\n    saving input template -> %s.bagel\n' % title)
    print('\n    COMPLETE\n')

def setup_bagel_input(keyword):
    title = keyword['title']
    basis = keyword['basis']
    df_basis = keyword['df_basis']

    grad = setup_bagel_grad(keyword)
    method = setup_bagel_method(keyword)

    template = """{ "bagel" : [
{
  "title" : "molecule",
  "basis" : "%s",
  "df_basis" : "%s",
"geometry" : [
]
},
{
"title" : "load_ref",
"file" : "%s",
"continue_geom" : false
},
{
  "title"   : "forces",
  "grads" : [
%s
],
  "export" : true,
  "method" : [ {
%s
} ]
},
{
 "title" : "print",
 "file" : "%s.orbital.molden",
 "orbitals" : true
},
{
"title" : "save_ref",
"file" : "%s"
}
]}  
""" % (basis, df_basis, title, grad, method, title, title)
    
    return template

def setup_bagel_grad(keyword):
    nstate = keyword['nstate']
    nac = keyword['nac']
    grad = []
    for n in range(nstate):
        line = '{ "title" : "force", "target" : %s }' % n
        grad.append(line)

    if nac == 2:
        nac_list = all_pairs(nstate)
    elif nac == 1:
        nac_list = neighbors(nstate)
    else:
        nac_list = []

    for pair in nac_list:
        a, b = pair
        line = '{ "title" : "nacme", "target" : %s, "target2" : %s, "nacmtype" : "noweight" }' % (a, b)
        grad.append(line)

    grad = ',\n'.join(grad)

    return grad

def setup_bagel_method(keyword):
    forb = keyword['forb']
    charge = keyword['charge']
    norb = keyword['norb']
    astate = keyword['astate']
    caspt2 = keyword['caspt2']

    if caspt2 == 0:
        method = """  "title" : "casscf",
  "natocc"  : true,
  "nstate"  : %s,
  "nact"    : %s,
  "nclosed" : %s,
  "charge"  : %s,
  "maxiter" : 200""" % (astate, norb, forb, charge)
    else:
        method = """ "title": "caspt2",
 "smith":{
   "method" : "caspt2",
   "ms"     : true,
   "xms"    : true,
   "sssr"   : true,
   "shift"  : 0.5,
   "thresh" : 1.0e-8,
   "maxiter": 800
 },
 "natocc"  : true,
 "nstate"  : %s,
 "nact"    : %s,
 "nclosed" : %s,
 "charge"  : %s,
 "maxiter" : 200""" % (astate, norb, forb, charge)

    return method

def orca_input(title):
    print('\n    Generating ORCA input template\n')

    default = input("""
    -----------------------------------------------------------------
    Do you want to generate the default orca template? (yes/no)
        For ground-state calculation, choose no
        For excited-state calculation, it uses wB97X-d3 cc-pvdz def2/J rijcosx
    """)
    default = bool_dict[default.lower()]

    if default == 1:
        basis = ''
        functional = ''
        roots = ''
        irootlist = ''
        engrad = ''
        mult = ''
        pal = ''
    else:
        basis = input("""
    -----------------------------------------------------------------
    what is the basis set?
        e.g. cc-pvdz or cc-pvdz def2/J rijcosx
    """)

        functional = input("""
    what is the density functional
        e.g. wB97X-d3, cam-b3lyp
    """)

        roots = input("""
    How many roots do you want to compute in TDDFT calculation? (for nroots)
        for excited-state calculation, enter a number greater than 1
        for ground-state calculation, enter 1
    """)

        iroot = input("""
    How many states do you want to compute in TDDFT gradient calculation? (for irootlist)
        for excited-state calculation, enter a number smaller than above one
        for ground-state calculation, enter any number
    """)

        irootlist = ','.join([str(x) for x in list(range(int(iroot)))])

        engrad = input("""
    Do you call ORCA for NAMD simulation? (yes/no)
        for NAMD with ORCA, choose no
        for training data calculation, choose yes
    """)

        engrad = bool_dict[engrad.lower()]
        if engrad == 1:
            engrad = 'engrad'
        else:
            engrad = ''

        mult = input("""
    What is the charge and multiplicity? (two values)
    e.g. 0 1
    """)

        pal = input("""
    How many CPUs do you want to use? (for nprocs)
    """)

    print('\n    -----------------------------------------------------------------\n')

    keyword = {
        'default': default,
        'basis': basis,
        'dft': functional,
        'roots': int(roots),
        'irootlist': irootlist,
        'engrad': engrad,
        'mult': mult,
        'pal': pal,
    }

    template = setup_orca_input(keyword)

    with open('%s.orca' % title, 'w') as out:
        out.write(template)

    print('\n    saving input template -> %s.orca\n' % title)
    print('\n    COMPLETE\n')

def setup_orca_input(keyword):
    default = keyword['default']
    basis = keyword['basis']
    dft = keyword['dft']
    roots = keyword['roots']
    irootlist = keyword['irootlist']
    engrad = keyword['engrad']
    mult = keyword['mult']
    pal = keyword['pal']

    if roots > 1:
        template = """!%s  %s tightscf printbasis %s
%%pal nprocs %s end
%%tddft nroots %s
       triplets false
       tda true
       maxdim 5
       irootlist %s
       maxiter 500
end
%%scf print[p_mos] 1 end
* xyz %s
""" % (dft, basis, engrad, pal, roots, irootlist, mult)
    else:
        template = """!%s  %s tightscf printbasis %s
%%pal nprocs %s end
%%scf print[p_mos] 1 end
* xyz %s
""" % (dft, basis, engrad, pal, mult)

    if default == 1:
        template = """!wb97x-d3 cc-pvdz def2/J rijcosx tightscf printbasis engrad
%pal nprocs 6 end
%tddft nroots 5
       triplets false
       tda true
       maxdim 5
       irootlist 0,1,2
       maxiter 500
end
%scf print[p_mos] 1 end
* xyz 0 1
"""
    return template

def main():
    print('\n\n    PyRAI2MD QC calculation input template generator\n')

    prog_dict = {
        '0': 'pyrai2md',
        '1': 'molcas',
        '2': 'bagel',
        '3': 'orca',
        'pyrai2md': 'pyrai2md',
        'molcas': 'molcas',
        'bagel': 'bagel',
        'orca': 'orca',
    }

    prog_func = {
        '0': pmd_input,
        '1': molcas_input,
        '2': bagel_input,
        '3': orca_input,
        'pyrai2md': pmd_input,
        'molcas': molcas_input,
        'bagel': bagel_input,
        'orca': orca_input,
    }

    title = input("""
    -----------------------------------------------------------------
    what is the title of the calculation?
    """)

    prog = input("""
    what is quantum chemical program? (enter the number or name below)
        0.pyrai2md
        1.molcas
        2.bagel
        3.orca
    """)

    print('\n    selected: %s\n' % prog_dict[prog])
    print('\n    template name: %s.%s\n' % (title, prog_dict[prog]))
    print('\n    -----------------------------------------------------------------\n')
    prog_func[prog](title)


if __name__ == '__main__':
    main()
