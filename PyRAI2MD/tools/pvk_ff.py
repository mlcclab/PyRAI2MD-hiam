# m-MYP parameters for PbI from Journal of Physics and Chemistry of Solids 180 (2023) 111383,
# https://doi.org/10.1016/j.jpcs.2023.111383
# IFF parameters for NIO from J. Chem. Theory Comput. 2023, 19, 8293−8322
# https://doi.org/10.1021/acs.jctc.3c00750
# others are from on GAFF

pair_fixed = """
    pair_coeff @atom:pvk_Pb @atom:pvk_Pb   buck/coul/long 371321.8 0.099492 25480.4
    pair_coeff @atom:pvk_Pb @atom:pvk_I    buck/coul/long 158835   0.339893 11784.8
    pair_coeff @atom:pvk_I @atom:pvk_I     buck/coul/long 43145.4  0.464671 10774.0
    pair_coeff @atom:iff_O @atom:iff_O     lj/charmm/coul/long 0.40 2.95778
    pair_coeff @atom:iff_Ni @atom:iff_Ni   lj/charmm/coul/long 0.35 1.66598
"""

pair_1 = """
    pair_coeff @atom:pvk_Pb @atom:pvk_Pb lj/charmm/coul/long 0.05570 3.598300
    pair_coeff @atom:pvk_I @atom:pvk_I lj/charmm/coul/long 0.07000 5.400000
"""

pair_2 = """
    pair_coeff @atom:iff_O @atom:iff_O lj/charmm/coul/long 0.40 2.95778
    pair_coeff @atom:iff_Ni @atom:iff_Ni lj/charmm/coul/long 0.35 1.66598
    pair_coeff @atom:hc @atom:hc lj/charmm/coul/long 0.0208 2.600176998764394
    pair_coeff @atom:ha @atom:ha lj/charmm/coul/long 0.0161 2.62547852235958
    pair_coeff @atom:hn @atom:hn lj/charmm/coul/long 0.0100 1.1064962079303013
    pair_coeff @atom:ho @atom:ho lj/charmm/coul/long 0.0047 0.5379246460131368
    pair_coeff @atom:hs @atom:hs lj/charmm/coul/long 0.0124 1.0890345930547507
    pair_coeff @atom:hp @atom:hp lj/charmm/coul/long 0.0144 1.0746020338208773
    pair_coeff @atom:o @atom:o lj/charmm/coul/long 0.1463 3.048120874245357
    pair_coeff @atom:os @atom:os lj/charmm/coul/long 0.0726 3.156097798883966
    pair_coeff @atom:op @atom:op lj/charmm/coul/long 0.0726 3.156097798883966
    pair_coeff @atom:oq @atom:oq lj/charmm/coul/long 0.0726 3.156097798883966
    pair_coeff @atom:oh @atom:oh lj/charmm/coul/long 0.0930 3.242871334030835
    pair_coeff @atom:c3 @atom:c3 lj/charmm/coul/long 0.1078 3.397709531243626
    pair_coeff @atom:c2 @atom:c2 lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:c1 @atom:c1 lj/charmm/coul/long 0.1596 3.478959494338025
    pair_coeff @atom:n @atom:n lj/charmm/coul/long 0.1636 3.1808647832482673
    pair_coeff @atom:s @atom:s lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:p2 @atom:p2 lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:f @atom:f lj/charmm/coul/long 0.0832 3.0342228542423677
    pair_coeff @atom:cl @atom:cl lj/charmm/coul/long 0.2638 3.465952373053176
    pair_coeff @atom:br @atom:br lj/charmm/coul/long 0.3932 3.6125943020590756
    pair_coeff @atom:i @atom:i lj/charmm/coul/long 0.4955 3.841198913133887
    pair_coeff @atom:n1 @atom:n1 lj/charmm/coul/long 0.1098 3.2735182499348627
    pair_coeff @atom:n2 @atom:n2 lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:n3 @atom:n3 lj/charmm/coul/long 0.0858 3.36510263815969
    pair_coeff @atom:na @atom:na lj/charmm/coul/long 0.2042 3.2058099473561965
    pair_coeff @atom:nh @atom:nh lj/charmm/coul/long 0.2150 3.189951950173299
    pair_coeff @atom:n+ @atom:n+ lj/charmm/coul/long 0.7828 2.8558649308706716
    pair_coeff @atom:n9 @atom:n9 lj/charmm/coul/long 0.0095 4.04468018035714
    pair_coeff @atom:h1 @atom:h1 lj/charmm/coul/long 0.0208 2.4219972551363265
    pair_coeff @atom:h2 @atom:h2 lj/charmm/coul/long 0.0208 2.243817511508259
    pair_coeff @atom:h3 @atom:h3 lj/charmm/coul/long 0.0208 2.0656377678801907
    pair_coeff @atom:hx @atom:hx lj/charmm/coul/long 0.0208 1.8874580242521226
    pair_coeff @atom:h4 @atom:h4 lj/charmm/coul/long 0.0161 2.536388650545546
    pair_coeff @atom:h5 @atom:h5 lj/charmm/coul/long 0.0161 2.4472987787315117
    pair_coeff @atom:cx @atom:cx lj/charmm/coul/long 0.1078 3.397709531243626
    pair_coeff @atom:cy @atom:cy lj/charmm/coul/long 0.1078 3.397709531243626
    pair_coeff @atom:c @atom:c lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cs @atom:cs lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:ca @atom:ca lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cc @atom:cc lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cd @atom:cd lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:ce @atom:ce lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cf @atom:cf lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cp @atom:cp lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cq @atom:cq lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cz @atom:cz lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cg @atom:cg lj/charmm/coul/long 0.1596 3.478959494338025
    pair_coeff @atom:ch @atom:ch lj/charmm/coul/long 0.1596 3.478959494338025
    pair_coeff @atom:cu @atom:cu lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:cv @atom:cv lj/charmm/coul/long 0.0988 3.3152123099438304
    pair_coeff @atom:nb @atom:nb lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:nc @atom:nc lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:nd @atom:nd lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:ne @atom:ne lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:nf @atom:nf lj/charmm/coul/long 0.0941 3.3841678707278926
    pair_coeff @atom:no @atom:no lj/charmm/coul/long 0.0858 3.36510263815969
    pair_coeff @atom:n7 @atom:n7 lj/charmm/coul/long 0.0522 3.5076464330621437
    pair_coeff @atom:n8 @atom:n8 lj/charmm/coul/long 0.0323 3.650190227964598
    pair_coeff @atom:n4 @atom:n4 lj/charmm/coul/long 3.8748 2.499505443614536
    pair_coeff @atom:nx @atom:nx lj/charmm/coul/long 2.5453 2.58859531542857
    pair_coeff @atom:ny @atom:ny lj/charmm/coul/long 1.6959 2.6776851872426035
    pair_coeff @atom:nz @atom:nz lj/charmm/coul/long 1.1450 2.766775059056638
    pair_coeff @atom:ns @atom:ns lj/charmm/coul/long 0.1174 3.269954655062301
    pair_coeff @atom:nt @atom:nt lj/charmm/coul/long 0.0851 3.3590445268763354
    pair_coeff @atom:nu @atom:nu lj/charmm/coul/long 0.1545 3.2790418219873327
    pair_coeff @atom:nv @atom:nv lj/charmm/coul/long 0.1120 3.368131693801367
    pair_coeff @atom:ni @atom:ni lj/charmm/coul/long 0.1636 3.1808647832482673
    pair_coeff @atom:nj @atom:nj lj/charmm/coul/long 0.1636 3.1808647832482673
    pair_coeff @atom:nk @atom:nk lj/charmm/coul/long 2.5453 2.58859531542857
    pair_coeff @atom:nl @atom:nl lj/charmm/coul/long 2.5453 2.58859531542857
    pair_coeff @atom:nm @atom:nm lj/charmm/coul/long 0.2150 3.189951950173299
    pair_coeff @atom:nn @atom:nn lj/charmm/coul/long 0.2150 3.189951950173299
    pair_coeff @atom:np @atom:np lj/charmm/coul/long 0.0858 3.36510263815969
    pair_coeff @atom:nq @atom:nq lj/charmm/coul/long 0.0858 3.36510263815969
    pair_coeff @atom:n5 @atom:n5 lj/charmm/coul/long 0.0522 3.5076464330621437
    pair_coeff @atom:n6 @atom:n6 lj/charmm/coul/long 0.0522 3.5076464330621437
    pair_coeff @atom:s2 @atom:s2 lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:s4 @atom:s4 lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:s6 @atom:s6 lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:sx @atom:sx lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:sy @atom:sy lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:sh @atom:sh lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:ss @atom:ss lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:sp @atom:sp lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:sq @atom:sq lj/charmm/coul/long 0.2824 3.532413417426445
    pair_coeff @atom:p3 @atom:p3 lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:p4 @atom:p4 lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:p5 @atom:p5 lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:pb @atom:pb lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:px @atom:px lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:py @atom:py lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:pc @atom:pc lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:pd @atom:pd lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:pe @atom:pe lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:pf @atom:pf lj/charmm/coul/long 0.2295 3.6940224448971026
    pair_coeff @atom:Cu @atom:Cu lj/charmm/coul/long 0.1729 3.9377723341802997   # Esitmated by Junmei
    pair_coeff @atom:hw @atom:hw lj/charmm/coul/long 0.0    1.0 # (default parameters)
    pair_coeff @atom:ow @atom:ow lj/charmm/coul/long 0.1521 3.1507 # (default parameters)
"""

output = ''

output += pair_fixed

for line in pair_1.splitlines():
    if len(line) == 0:
        continue
    atom_type_1 = line.split()[1].split(':')[-1]
    k_1 = float(line.split()[4])
    c_1 = float(line.split()[5])
    for lin in pair_2.splitlines():
        if len(lin) == 0:
            continue
        atom_type_2 = lin.split()[1].split(':')[-1]
        k_2 = float(lin.split()[4])
        c_2 = float(lin.split()[5])
        k = (k_1 * k_2) ** 0.5
        c = (c_1 + c_2) / 2
        output += '    pair_coeff @atom:%s @atom:%s lj/charmm/coul/long %8.4f %24.16f\n' % (
            atom_type_1, atom_type_2, k, c
        )

print(output)
