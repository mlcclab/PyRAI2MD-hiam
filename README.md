# Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics
<pre>

                              /\
   |\    /|                  /++\
   ||\  /||                 /++++\
   || \/ || ||             /++++++\
   ||    || ||            /PyRAI2MD\
   ||    || ||           /++++++++++\                    __
            ||          /++++++++++++\    |\ |  /\  |\/| | \
            ||__ __    *==============*   | \| /--\ |  | |_/

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics



                      Author @Jingbai Li


    2022 – present   Hoffmann Institute of Advanced Materials
                     Shenzhen Polytechnic, Shenzhen, China    
                                
    2019 – 2022      Department of Chemistry and Chemical Biology
                     Northeastern University, Boston, USA

                          version:   2.4.0
                          

  With contribution from (in alphabetic order):
    Jingbai Li     - Fewest switches surface hopping
                     Zhu-Nakamura surface hopping
                     Velocity Verlet
                     OpenMolcas interface
                     OpenMolcas/Tinker interface
                     BAGEL interface
                     ORCA interface
                     GFN-xTB interface
                     Adaptive sampling
                     Grid search
                     Two-layer ONIOM
                     Periodic boundary condition (coming soon)
                     Wall potential
                     QC/ML hybrid NAMD
                     Excited-state Equivariant Neural Network

    Patrick Reiser - Fully connected neural networks (pyNNsMD)
                     SchNet (pyNNsMD)

  Special acknowledgement to:
    Steven A. Lopez            - Project co-founder
    Pascal Friederich          - Project co-founder

</pre>
## Features
 - Machine learning nonadiabatic molecular dynamics (ML-NAMD).
 - Neural network training and grid search.
 - Active learning with ML-NAMD trajectories.
 - Support BAGEL, Molcas, ORCA, GFN-xTB for QM, and Molcas/Tinker for QM/MM calculations.
 - Generalized FSSH and ZNSH with nonadiabatic coupling and spin-orbit coupling
 - Add curvature-driven time-dependent coupling for FSSH
 - Support QMQM2 scheme multiscale dynamics
 
## Prerequisite
 - **Python >=3.7** PyRAI2MD is written and tested in Python 3.7.4. Older version of Python is not tested and might not be working properly.
 - **TensorFlow >=2.2** TensorFlow/Keras API is required to load the trained NN models and predict energy and force.
 - **Cython** PyRAI2MD uses Cython library for efficient surface hopping calculation.
 - **Matplotlib/Numpy** Scientific graphing and numerical library for plotting training statistic and array manipulation.
 - **scikit-learn** Machine learning library

## Content
<pre>
 File/Folder Name                                  Description                                      
---------------------------------------------------------------------------------------------------
 PyRAI2MD                                          source codes folder
  |--pyrai2md.py                                   PyRAI2MD main function                           
  |--method.py                                     PyRAI2MD method manager                          
  |--variables.py                                  PyRAI2MD input reader                            
  |
  |--Keywords                                      default input values folder
  |   |--key_control.py                            keywords for calculation control                 
  |   |--key_molecule.py                           keywords for molecule specification              
  |   |--key_md.py                                 keywords for molecular dynamic settings          
  |   |--key_nn.py                                 keywords for neural network settings             
  |   |--key_grid_search.py                        keywords for grid search settings                
  |   |--key_molcas.py                             keywords for molcas calculation                  
  |   |--key_bagel.py                              keywords for bagel calculation                   
  |   |--key_orca                                  keywords for orca calculation                    
  |   |--key_xtb.py                                keywords for xtb calculation                      
  |   |--key_mlp.py                                keywords for mlp settings                        
  |   |--key_schnet.py                             keywords for schnet settings                     
  |   |--key_e2n2.py                               keywords for e2n2 settings                       
  |   |--key_dimenet.py                            keywords for dimenet (NAC model) setting         
  |   |--key_read_file.py                          keywords for reading training data                
  |    `-key_templ.py                              keywords class template                           
  |
  |--Molecule                                      atom, molecule, trajectory code folder
  |   |--atom.py                                   atomic properties class                          
  |   |--molecule.py                               molecular properties class                       
  |   |--trajectory.py                             trajectory properties class                      
  |   |--constraint.py                             external potential functions                     
  |   |--pbc_helper.py                             periodic boundary condition functions             
  |    `-qmmm_helper.py                            qmmm functions                                    
  |
  |--Quantum_Chemistry                             quantum chemicial program interface folder
  |   |--qc_molcas.py                              OpenMolcas interface                             
  |   |--qc_bagel.py                               BAGEL interface                                  
  |   |--qc_molcas_tinker                          OpenMolcas/Tinker interface                      
  |   |--qc_orca                                   ORCA interface                                   
  |   |--qc_xtb                                    GFN-xTB interface                                
  |    `-qmqm2                                     Multiscale calculation interface                 
  |
  |--Machine_Learning                              machine learning library interface folder
  |   |--model_demo.py                             demo version neural network                      
  |   |--model_NN.py                               native neural network interface                  
  |   |--model_pyNNsMD.py                          pyNNsMD interface                               
  |   |--model_GCNNP.py                            GCNNP interface                                  
  |   |--model_DimeNet.py                          DimeNet NAC model interface                 
  |   |--model_templ.py                            NN interface template                       
  |   |--model_helper.py                           additional tools for neural network              
  |   |--hyper_nn.py                               native neural network hyperparameter             
  |   |--hyper_pynnsmd.py                          pyNNsMD hyperparameter                           
  |   |--hyper_gcnnp.py                            GCNNP hyperparameter                             
  |   |--hyper_dimenet.py                          DimeNet NAC model hyperparameter                 
  |   |--hyper_templ.py                            hyperparameter template                           
  |   |--training_data.py                          training data manager                            
  |   |--permutation.py                            data permutation functions                       
  |   |--adaptive_sampling.py                      adaptive sampling class                         
  |   |--grid_search.py                            grid search manager                              
  |   |--search_nn.py                              grid search function for native nn               
  |   |--search_GCNNP.py                           grid search function for e2n2                    
  |   |--remote_train.py                           remote training function                         
  |   |--Dimenet.py                                Dimenet NAC model                                
  |   |--NNsMD                                     demo version neural network library  
  |    `-pyNNsMD                                   native neural network library                  
  |
  |--Dynamics                                      ab initio molecular dynamics code folder
  |   |--aimd.py                                   molecular dynamics class                         
  |   |--mixaimd.py                                ML-QC hybrid molecular dynamics class            
  |   |--single_point.py                           single point calculation                         
  |   |--hop_probability.py                        surface hopping probability calculation          
  |   |--reset_velocity.py                         velocity adjustment functions                    
  |   |--verlet.py                                 velocity verlet method                           
  |   |--Ensembles                                 thermodynamics control code folder
  |   |   |--ensemble.py                           thermodynamics ensemble manager                   
  |   |   |--microcanonical.py                     microcanonical ensemble                           
  |   |    `-thermostat.py                         canonical ensemble                                
  |   |
  |    `-Propagators                               electronic propagation code folder
  |       |--surface_hopping.py                    surface hopping manager                           
  |       |--setup_fssh.py                         setup file to compile the C-lib of fssh.pyx       
  |       |--fssh.pyx                              fewest switches surface hopping method           
  |       |--gsh.py                                generalized surface hopping method               
  |        `-tsh_helper.py                         trajectory surface hopping tools                 
  |
   `-Utils                                         utility folder
      |--extension.py                              additional tools for setup                        
      |--coordinates.py                            coordinates writing functions                    
      |--read_tools.py                             index reader                                     
      |--geom_tools.py                             compute geometrical derivatives
      |--bonds.py                                  bond length library                               
      |--sampling.py                               initial condition sampling functions            
      |--timing.py                                 timing functions                                  
       `-logo.py                                   logo and credits
</pre>

## Installation
Download the repository

    git clone https://github.com/mlcclab/PyRAI2MD-hiam.git

Install the codes, this will create a command **pyrai2md** to run calculations

    cd ./PyRAI2MD-hiam
    pip install .

Compile fssh library using **pyrai2md** command

    pyrai2md update

## Test PyRAI2MD
Go to the test folder

    cd ./test

Choose test jobs by editing test_case.py, set test_job = 1 to turn on the test or 0 to turn off the test

    nano test_case.py

Modify the environment variables in the template 

    nano run_test.sh

Run test, this might take a while

    bash run_test.sh
    
## Run PyRAI2MD

    pyrai2md input
    
# User manual
We are currently working on the tutorials for users manual.

# Cite us
- Jingbai Li, Patrick Reiser, Benjamin R. Boswell, André Eberhard, Noah Z. Burns, Pascal Friederich, and Steven A. Lopez, "Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations", Chem. Sci. 2021, 12, 5302-5314. DOI:10.1039/D0SC05610C
- Jingbai Li, Rachel Stein, Daniel Adrion, Steven A. Lopez, "Machine-learning photodynamics simulations uncover the role of substituent effects on the photochemical formation of cubanes", J. Am. Chem. Soc. 2021, 143, 48, 20166–20175. DOI:10.1021/jacs.1c07725
- Jingbai Li, Steven A. Lopez, “Excited-state distortions promote the reactivities and regioselectivities of photochemical 4π-electrocyclizations of fluorobenzenes”, Chem. A Eur J. 2022, 28, e202200651. DOI:10.1002/chem.202200651
- Jingbai Li, Steven A. Lopez, “A Look Inside the Black Box of Machine Learning Photodynamics Simulations”, Acc. Chem. Res., 2022, 55, 1972–1984. DOI:10.1021/acs.accounts.2c00288
- L. Wang, C. Salguero, S.A. Lopez, J. Li, "Machine learning photodynamics uncover blocked non-radiative mechanisms in aggregation-induced emission", Chem, 2024, 10, 2295–2310. DOI: 10.1016/j.chempr.2024.04.017
