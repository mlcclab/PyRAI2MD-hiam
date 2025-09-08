######################################################
#
# PyRAI2MD 2 module for printing logo
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

def print_logo(version):

    credit = """
  -------------------------------------------------------------------
                              /\\
   |\\    /|                  /++\\
   ||\\  /||                 /++++\\
   || \\/ || ||             /++++++\\
   ||    || ||            /PyRAI2MD\\
   ||    || ||           /++++++++++\\                    __
            ||          /++++++++++++\\    |\\ |  /\\  |\\/| | \\
            ||__ __    *==============*   | \\| /--\\ |  | |_/

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics



                      Author @Jingbai Li
                      
                      
    2022 – present   Hoffmann Institute of Advanced Materials
                     Shenzhen Polytechnic, Shenzhen, China    
                                
    2019 – 2022      Department of Chemistry and Chemical Biology
                     Northeastern University, Boston, USA

                          version:   %s

  With contributions from (in alphabetic order):
    Jingbai Li     - Fewest switches surface hopping
                     Zhu-Nakamura surface hopping
                     Velocity Verlet
                     OpenMolcas interface
                     OpenMolcas/Tinker interface
                     BAGEL interface
                     ORCA interface
                     OpenQP interface
                     GFN-xTB interface
                     Adaptive sampling
                     Grid search
                     Multilayer ONIOM (QM:QM' and QM:QM':MM)
                     Periodic boundary condition
                     Wall potential
                     QC/ML hybrid NAMD
                     Excited-State Neural Network Potentials (ESNNP)

    Patrick Reiser - Fully connected neural networks (pyNNsMD)
                     SchNet (pyNNsMD)

  Special acknowledgement to:
    Steven A. Lopez            - Project co-founder
    Pascal Friederich          - Project co-founder

""" % version

    return credit
