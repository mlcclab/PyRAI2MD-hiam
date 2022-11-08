
                   Trajectory Analysis Script Manual

=========================================================================

   !!! 	       	       	       	       	       	       	  !!!
   !!!                 Important Notes       	       	  !!!
   !!!                                                    !!!
   !!! The calculation folders should have the same names !!!
   !!! as the output files.                               !!! 
   !!! 	       	       	       	       	       	       	  !!!
   !!! The traj_analysis.py can read trajectories anywhere!!!
   !!! as long as a list of path is provided.             !!!
   !!! It saves trajectories in JSON for later analysis   !!!
   !!!                                                    !!!
   !!!                                                    !!!
   !!!                                                    !!!
   
1. Quick command line usage
-------------------------------------------------------------------------
    You can simply run this script on command line as below:

        python3 traj_analysis.py input_file

2. Options, keywords, and values
-------------------------------------------------------------------------
    Here are all options and keywords in this script. 
    Currently, all command line options are deprecated.
    Use the keywords in an input file

  Keyword                 Value
-------------------------------------------------------------------------
    title           Title of calculations or name of JSON file.
                    Required to run this script.

    cpus            Number of CPU.
                    Default is 1.
                    The calculations of geometrical parameters can be parallelized.

    mode            Analysis mode.
                    Default is 0. Available options are:
                    1 or diag or diagnosis       Diagnosis trajectory and check completeness.
                    2 or cons or conservation    Check energy conservation and remove none-conserved ones.
                    3 or read or extract         Read trajectory and save into a JSON file.
                    4 or pop or population       Compute average state population and hop energy gaps.
                    5 or cat or classify         Classify trajectory based on snapshot structures.
                    6 or plot or compute         Compute geometrical parameters for plot trajectory
                    7 or out or fetch            Fetch coordinates and more data for selected trajectory

    read_index      Index to read trajectory or a file of path to read trajectory
                    Default	is 1.
                    Index can be a single integer, a range of integers, or mixed (e.g. 1 2-4 5).
                    When using a file, each line contains a path to a trajectory

    save_traj       Save trajectory data to a json file.
                    Default is 1.
                    When trajectory data is too large, saving the JSON file could be slow
                    and re-reading them could be faster than loading the JSON file.
                    The traj_analysis.py will attempt to re-read if the JSON file is not available.

    minstep         Minimum step of trajectory to determine completion.
                    Default is 0. Any length of trajectory is complete.
                    If greater than 0, the shorter trajectory will not be included in complete list.
                    
    maxstep         Maximum step of trajectory to read.
                    Default is 0. All steps will be read.
                    If greater than 0, the longer part of trajectory will not be read.

    maxdrift        Maximum energy drift to determine none-conserved trajectory
                    Default is 0.5 a.u.

    classify        Classify structures of final or hop snapshot
                    Default is all. Available options are:
                    final   classify final snapshot
                    hop     classify hop snapshot
                    all     classify final and hop snapshot

    classify_state  Target state to classify snapshot structure
                    Default is 0, the lowest state.

    param           Parameters or a list file that has parameters for classification or plot data calculations
                    Available options are:
                    B X X           bond
                    A X X X         angel in [0, π]
                    D X X X X       dihedral angle in [-π, π]
                    D2 X X X X      dihedral angle in [0, π]
                    D3 X X X X      dihedral angle in [0, 2π]
                    DD X X X X X X  dihedral angle in [-π, π]
                    DD2 X X X X X X dihedral angle in [0, π]
                    DD3 X X X X X X dihedral angle in [0, 2π]
                    O X X X X       out-of-plane angle in [0, π]
                    P X X X X X X   plane-plane angle in [0, π]
                    RMSD            root-mean-square deviation.

    threshold       Threshold for each parameter
                    Default is None.
                    The number of thresholds must match with the number of parameters

    ref_geom        The xyz file of a reference structure.
                    Required for computing RMSD

    save_data       Save full state population, kinetic energy, potential energy, and total energy
                    Default is 0. Available options are:
                    1. save full state population
                    2. save full state population, kinetic energy, potential energy, and total energy

    prog            Program used to compute trajectory.
                    Support format are:
                    molcas (PyRAI2MD uses the same format)
                    newtonx
                    sharc
                    fromage

3. More info about RMSD
-------------------------------------------------------------------------
    RMSD has more optional inputs that use a format as RMSD input1=value1 input2=value2 ...etc.
    Space is NOT allowed on the both sides of the equal sign (=). 
    
    Input   Value                          Meaning   
    no=     element name (hydrogen)  Elements are not included in the structures.
                                     Default is empty.
                                     Multiple inputs are allowed. no=hydrogen no=carbon skip H and C.

    on=     element name (hydrogen)  Elements are only included in the structure.
                                     Default is empty.
                                     Multiple inputs are allowed. on=hydrogen on=carbon only count H and C.

    align=  hung or no               Align reference structure with trajectory structure using Hungarian Algorithm before calculating RMSD.
                                     Default is no.

    pick=   index with ','           Choose specific atoms for RMSD calculation.
                                     Default is empty. All atoms are included
                                     1,2,3,4,5 will only include the atom 1–5. 
                                     Note, the index must be connected with ',' without space

    coord=  cart or sym              Choose coordinates system for RMSD calculation.
                                     Default is cart.
                                     cart use Cartesian coordinates.
                                     sym  convert Cartesian coordinates to symmetry functions, Behler, J, Int. J. Quantum Chem., 2-15, 115 1032-1050.

    cut=    1 or 2                   Formula of cutoff function.
                                     Default is 1.
                                     1 cosine function.
                                     2 tanh^3 function.

    ver=    1, 2, 3, or 4            Formula of symmetry function.
                                     Default is 1.
                                     1 G1 function. Sum of cutoff functions.
       	       	       	       	     1 G2 function. Sum of Gaussian weighted cutoff functions.
       	       	       	       	     1 G3 function. Sum of three-pairs-angle and Gaussian weighted cutoff functions.
       	       	       	       	     1 G4 function. Sum of two-paris-angle and Gaussian weighted cutoff functions.

    rc=     number                   Cutoff radii. Default is 6 Angstrom.
          
    rs=     number                   Gaussian center. Default is 0.

    eta=    number                   Gaussian exponent factor. Default is 1.2.

    zeta=   number                   angular function exponent. Default is 1.

    lambda= number                   angular function factor. Default is 1. Only takes 1 and -1. 


4. General tips
-----------------------------------------------------------------------------

    Step 1. check the completeness of trajectory
    -------------------------------------------------------------------------
    The input file looks like below

    title file_name
    mode diag
    read_index list.txt

    The list.txt is a file containing the path to all trajectories, looks like below:

    /path/to/traj_1
    /path/to/traj_1
    /path/to/traj_1
    ...
    /path/to/traj_n

    It will compare the number of steps in all logfile and save the path to the
    completed one in a file, called 'complete'. We can use 'complete' to read
    completed trajectories instead of reading all of them from list.txt.
    -------------------------------------------------------------------------

    Step 2. check energy conservation of trajectory.
    -------------------------------------------------------------------------
    The input file looks like below:

    title file_name
    mode cons
    read_index complete
    maxdrift 0.05
    maxstep 200

    After checking the completeness of trajectory, we can determine an appropriate
    maxstep to process the trajectory data.

    It will read completed trajectory and compute energy drift. Any trajectory
    has total energy drift exceeding the maxdrift, will not be selected. It saves
    the path to the selected trajectories in a file, called 'conserved'. We can use
    'conserved' to read the trajectories for later analysis.

    The energy drift of each step are saved in energy_drift.txt.
    -------------------------------------------------------------------------

    Step 3. extract trajectory data
    -------------------------------------------------------------------------
    The input file looks like below:

    title file_name
    mode read
    read_index conserved
    save_traj 1
    maxstep 200

    This step is optional, because the following steps can automatically read trajectories
    if the trajectory data generated in this step is not found.

    Keep the maxstep to read the conserved trajectory. It will save the trajectory data
    in a portable $file_name.json. Note that when reading thousands of trajectories, the
    JSON file could be large, so saving and loading it might be slower than re-reading them
    from logfile. In this case, set save_traj to 0 to save time.

    In any case, it will save all hop and final snapshots in geom-$file_name.json and individual
    xyz files according to the final state of the trajectory, e.g., Final.Sn.xyz Hop.Sn.xyz

    The following steps can be done independently.
    -------------------------------------------------------------------------

    Step 4. compute state population
    -------------------------------------------------------------------------
    The input file looks like below:

    title file_name
    mode pop
    read_index conserved
    save_traj 1
    maxstep 200
    save_data 0

    When $file_name.json is available, read_index, save_traj, maxstep have no effect. Otherwise,
    they will be used to automatically read trajectory data before compute state population.

    In default, save_data 0 only compute the average state population and write them in average-$file_name.dat.
    The option 1 will write state population for all trajectories in state-pop-$file_name.json.
    The option 2 will write kinetic energy, potential energy, and total energy of all trajectories in
    energy-profile-$file_name.json.

    Sometimes, trajectory could fail with exploded structures, which should be removed from analysis.
    To do so, use keyword 'select' as following:

    select 1-5 7-10

    This will skip the 6th trajectory. Alternatively, you can provide a file containing this information:

    select file.txt

    The file.txt can have multiple line, each contain can have multiple index or group of index, e.g.:

    1-5 7-10
    11 12 13 16-20

    -------------------------------------------------------------------------
    Step 5. classify trajectory
    The input file looks like below:

    title file_name
    mode cat
    read_index conserved
    save_traj 1
    maxstep 200
    classify final
    classify_state 1
    param B 1 2 B 3 4
    threshold 2.5 2.5

    It will find geom-$file_name.json generated in Step 3. It will automatically re-read
    trajectory if geom-$file_name.json is not available.

    The keyword 'classify' decides the snapshot to classify trajectory. The keyword
    'classify_state' decides final state of the trajectories to be classified starting from 0.
    The keyword 'param' defines the geometrical parameters used for classification.
    The keyword threshold provide the values to classify trajectory. It labels trajectory with
    a series of 1 or 0 if the geometrical parameters are greater or smaller than the thresholds.
    The computed geometrical parameters are saved in param-%file_name.Sn.fin for final snapshots
    and param-%file_name.Sn.hop for hop snapshots.

    The keyword 'select' is also available to remove undesired trajectory in classification.

    -------------------------------------------------------------------------
    Step 6. compute geometrical parameters for trajectory plot
    The input file looks like below:

    title file_name
    mode plot
    read_index conserved
    save_traj 1
    maxstep 200
    param B 1 2 B 3 4
    select file.txt

    When $file_name.json is available, read_index, save_traj, maxstep have no effect. Otherwise,
    they will be used to automatically read trajectory data before compute plot data.

    The keyword 'param' defines the geometrical parameters used for plot data calculation
    The keyword 'select' decides the trajectories to compute plot data. If not provided, all trajectories
    will be included.

    It will save the data in plot-$file_name.json

    -------------------------------------------------------------------------
    Step 7. fetch a selected trajectory for visualization
    The input file looks like below:

    title file_name
    mode out
    read_index conserved
    save_traj 1
    maxstep 200
    select 1

    When $file_name.json is available, read_index, save_traj, maxstep have no effect. Otherwise,
    they will be used to automatically read trajectory data before fetching trajectory data.

    The keyword 'select' decides the trajectories to fetch. If not provided, it will stop.

    It will save the selected trajectory in select-$file_name-$index.xyz and the corresponding data
    in select-$file_name-$index.dat