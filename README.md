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

                          version:   2.2 alpha

  With contributions from (in alphabetic order):
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
                   - SchNet (pyNNsMD)

  Special acknowledgement to:
    Steven A. Lopez            - Project co-founder
    Pascal Friederich          - Project co-founder

</pre>
## Features
 - Machine learning nonadibatic molecular dyanmics (ML-NAMD).
 - Neural network training and grid search.
 - Active learning with ML-NAMD trajectories.
 - Support BAGEL, Molcas for QM, and Molcas/Tinker for QM/MM calculations.
 - Support ORCA and GFN2-xTB.
 - Generalized FSSH and ZNSH with nonadibatic coupling and spin-orbit coupling
 - Add curvature-driven time-depedent coupling for FSSH
 
## Prerequisite
 - **Python >=3.7** PyRAI2MD is written and tested in Python 3.7-3.9. Older version of Python is not tested and might not work properly.
 - **TensorFlow >=2.3** TensorFlow/Keras API is required to load the trained NN models and predict energy and force.
 - **Cython>=0.29.0** PyRAI2MD uses Cython library for efficient surface hopping calculation.
 - **Matplotlib>=3.5.0/Numpy>=1.20.0** Scientifc graphing and numerical library for plotting training statistic and array manipulation.

## Additional library
 - **pyNNsMD>=2.0.0** pyNNsMD provides the SchNet model

## Content
<pre>
 File/Folder Name                                  Description                                      
---------------------------------------------------------------------------------------------------
  PyRAI2MD                                         source codes folder
  |--pyrai2md.py                                   PyRAI2MD main function                         
  |--variables.py                                  PyRAI2MD input reader                           
  |--method.py                                     PyRAI2MD method manager                           
  |--Molecule                                      atom, molecule, trajectory code folder
  |   |--atom.py                                   atomic properties class                        
  |   |--molecule.py                               molecular properties class                    
  |   |--trajectory.py                             trajectory properties class                     
  |   |--pbc_helper.py                             periodic boundary condition functions             
  |    `-qmmm_helper.py                            qmmm functions                                    
  |
  |--Quantum_Chemistry                             quantum chemicial program interface folder
  |   |--qc_molcas.py                              OpenMolcas interface                           
  |   |--qc_bagel.py                               BAGEL interface                                  
  |   |--qc_molcas_tinker                          OpenMolcas/Tinker interface                      
  |   |--qc_orca                                   ORCA interface                                  
  |    `-qc_xtb                                    GFN-xTB interface                                
  |
  |--Machine_Learning                              machine learning library interface folder
  |   |--model_NN.py                               native neural network interface                  
  |   |--model_pyNNsMD.py                          pyNNsMD interface                               
  |   |--model_GCNNP.py                            GCNNP interface                                  
  |   |--model_helper.py                           additional tools for neural network              
  |   |--hyper_nn.py                               native neural network hyperparameter            
  |   |--hyper_pynnsmd.py                          pyNNsMD hyperparameter                       
  |   |--hyper_gcnnp.py                            GCNNP hyperparameter                            
  |   |--training_data.py                          training data manager                         
  |   |--permutation.py                            data permutation functions                   
  |   |--adaptive_sampling.py                      adaptive sampling class                      
  |   |--grid_search.py                            grid search class                               
  |   |--remote_train.py                           distribute remote training                      
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
  |	      |--setup_fssh.py                         setup file to compile the C-lib of fssh.pyx       
  |       |--fssh.pyx                              fewest switches surface hopping method           
  |       |--gsh.py                                generalized surface hopping method               
  |        `-tsh_helper.py                         trajectory surface hopping tools                 
  |
   `-Utils                                         utility folder
      |--extension.py                              additional tools for setup                        
      |--coordinates.py                            coordinates writing functions                    
      |--read_tools.py                             index reader                                     
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
We are currently working on the user manual.

# Cite us
- Jingbai Li, Patrick Reiser, Benjamin R. Boswell, André Eberhard, Noah Z. Burns, Pascal Friederich, and Steven A. Lopez, "Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations", Chem. Sci. 2021, 12, 5302-5314. DOI:10.1039/D0SC05610C
- Jingbai Li, Rachel Stein, Daniel Adrion, Steven A. Lopez, "Machine-learning photodynamics simulations uncover the role of substituent effects on the photochemical formation of cubanes", J. Am. Chem. Soc. 2021, 143, 48, 20166–20175. DOI:10.1021/jacs.1c07725
- Jingbai Li, Steven A. Lopez, “Excited-state distortions promote the reactivities and regioselectivities of photochemical 4π-electrocyclizations of fluorobenzenes”, Chem. A Eur J. 2022, 28, e202200651. DOI:10.1002/chem.202200651
- Jingbai Li, Steven A. Lopez, “A Look Inside the Black Box of Machine Learning Photodynamics Simulations”, Acc. Chem. Res., 2022, 55, 1972–1984. DOI:10.1021/acs.accounts.2c00288


