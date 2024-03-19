
                   Trajectory Generator Manual

========================================================================

   !!! 	       	       	       	       	       	       	  !!!
   !!!                 Important Notes       	       	  !!!
   !!! 	       	       	       	       	       	       	  !!!
   !!! Normal Mode Format:                                !!!
   !!! Molcas   -   mass-weighted unnormalized            !!!
   !!! Gaussian - unmass-weighted   normalized            !!!
   !!! ORCA     - unmass-weighted   normalized            !!!
   !!! BAGEL    - unmass-weighted   normalized            !!!
   !!! This script generates  unmass-wighted velocities   !!!
   !!! Coordinates are in Angstrom                        !!!
   !!! Velocities are in Bohr/au                          !!!
   !!!                                                    !!!
   !!!                 Useful Conversion                  !!!
   !!!                                                    !!!
   !!! length[Bohr] 1 au   = 0.529177249        Angstrom  !!!
   !!! velocity     1 au   = 2.18769126364e+6   m/s       !!!
   !!! time         1 au   = 2.41888432658e−17  s         !!!
   !!! temperature  1 au   = 315775.024804      K         !!!
   !!! mass         1 au   = 9.1093837015e−31   kg        !!!
   !!! wavenumber   1 cm-1 = 4.55633518e-6      au        !!!
   !!! molar mass   1 g/mol= 1822.8885          au or amu !!!

1. Prepare frequency calculation and file
------------------------------------------------------------------------
    A frequency file is required. 

    Molcas   -  file_name.freq.molden (need &MCKINLEY)

    Gaussian -  file_name.freq.g16 (Gaussian .log file)
             -  file_name.freq.fchk (Gaussian .fchk file, need Freq=SaveNormalModes in input)

    BAGEL    -  file_name.freq.bagel (BAGEL .log file, need "title" : hessian)

    ORCA     - file_name.freq.orca (ORCA .hess file)

    NewtonX  - file_name.init.newtonx (NewtonX sampled initial conditions, final_output)
    
    XYZ      - file_name.init.xyz (traj_generator.py sampled initial conditions, .init)

    Support programs:

    Molcas   - file_name.StrOrb (Initial Orbitals)
             - file_name.inp (Molcas molecular dynamics input, need velocities=1)

    NewtonX/ -  bagelinput.basis.inp (BAGEL basis info)
    BAGEL    -  bagelinput.part1.inp (BAGEL headline)
             -  bagelinput.part2.inp (BAGEL gradient info)
             -  bagelinput.part3.inp (BAGEL method info)
             -  control.dyn (Newton molecular dynamics input)
             -  jiri.inp and sh.inp (optional)

    pyrai2mdnn     - pyrai2md dynamics with nn
    pyrai2mdmolcas - pyrai2md dynamics with molcas
    pyrai2mdbagel  - pyrai2md dynamics with bagel
    pyrai2mdorca   - pyrai2md dynamics with orca

    ORCA     -  file_name.inp

    OQP      -  file_name.inp
             -  file_name.json (OQP orbital data, optional)

2. Quick command line usage (Not recommended)
------------------------------------------------------------------------
    You can simply run this script on command line as below:

        python traj_generator.py input_file

    You can find more options with explanations as below:

        python traj_generator.py for help

3. Input file  
------------------------------------------------------------------------
    The name of input file is arbitrary. It follows a tow-columns format
    (keywords on the left and values on the right) in free order:

      input      file_name.freq.molden
      seed	 1
      temp	 0.00001
      method     wigner
      partition  fullnode
      time       2-23:59:59
      memory   	 3000
      nodes    	 1 
      cores    	 28
      jobs     	 28
      index    	 1
      prog       molcas
      molcas     /share/apps/molcas-ext

4. Keywords and values
------------------------------------------------------------------------
    Here are all keywords in this script.

           keyword                 Value    
    ------------------------------------------------------
            input       This is the input file (file_name.freq.molden).
                        The Molcas input template (file_name.inp) must be in current directory!!!
                        The orbital file (file_name.StrOrb) must be in current directory!!!

            iseed       Random number seed. 
                        Default is -1, means a random seed. 
                        You can choose any integer from 0 to + infinite to make sampling reproducible.

            temp        Sampling temperature in K.
                        Default is 298.15 K.
                        This temperature only control the sampling but NOT the trajectory calculation.
                        Make sure the temperature matches with the actual calculation.

            method      Sampling method.
                        Default is boltzmann.
                        Available sampling methods are Boltzmann and Wigner distribution. 

            partition   Slurm partition.
       	       	       	Default	is normal.

            time        Slurm time limit.
                        Default is 1-00:00:00.
                        The time must follow slurm format.

            memory      Slurm memory limit per node in MB.
       	       	       	Default is 100.
                        This will limit the Molcas memory usage, the actual limit for slurm is
                        1.5 times the number jobs per node to ensure the Molcas only take 75% of allocated memory.

            nodes       Slurm nodes number.
       	       	       	Default is 1.
                        For each node, individual runscript will be generated.

            cores       Slurm cores number.
                        Default is 1.
                        This is the maximum number of cores per node.                        

            jobs        Slurm jobs number.
                        Default is 1.
                        This number cannot exceed the cores number but can be smaller if you want to run less jobs per node.

            index       Job initial index.
                        Default is 1.
                        The trajectory calculations are named as file_name-index. Change the initial index
                        can set up new calculations following the existed ones.

            ncpus       Number of CPUs for generating trajectory in parallel
                        Default is 1.

            repeat      Reuse the first sampled initial condition for all requested calculation
                        Default is 0. Use all sampled conditions

            notraj      Only perform initial condition sampling
                        Default is 0. Generate trajectories after sampling

            molcas      Path to Molcas.
                        Default is /share/apps/molcas-ext
                        This path is used to set up molcas root in slurm submission script


5. Additional keywords
------------------------------------------------------------------------
    These keywords are only used for NewtonX/BAGEL FSSH calculation.

    newton              Path to NewtonX.
                        Default is /share/apps/NX-2.4-B06

    bagel               Path to BAGEL.
       	       	       	Default	is /share/apps/bagel

    lib_blas            Path to BLAS library
                        Default is /share/apps/blas-3.10.0

    lib_lapack          Path to LAPACK library
                        Default is /share/apps/lapack-3.10.1

    lib_scalapack       Path to SCALAPACK library
                        Default is /share/apps/scalapack-2.2.0

    lib_boost           Path to Boost library
                        Default is /share/apps/boost_1_80_0

    mkl                 Path to Intel MKL (Not used anymore)
                        Default is /share/apps/intel/oneapi

    mpi                 Path to Intel MPI (Not used anymore)
                        Default is None


6. Insert shell commands in the generated submission scripts
------------------------------------------------------------------------
    Users may have their customized slurm setting or environments, shell, conda etc.
    To provide the largest flexibility, this script can read user-defined shell commands
    in the following way with a pair of delimiter '>shell'. For example,

    ... other keywords
    >shell
    #SBATCH constraint=["some slurm constraints"]
    module load some_module
    export PATH=/some/path:$PATH
    conda activate some_environments
    >shell
    ... other keywords

    In this example, we can specify some slurm setting, load modules, export environment
    variables, activate conda environments, and more shell command you would like to use.
    Note, the syntax must use 2 '>shell' delimiters, less or more will stop the script.
    If no '>shell' delimiter is used, it will be automatically skipped.

7. Useful tips
------------------------------------------------------------------------
    User need to change the original extension of frequency file so Gen-FSSH can identify the format
    Molcas         - .freq.molden === .freq.molden
    Gaussian       - .log         --> .freq.g16 
    BAGEL          - .log         --> .freq.bagel
    ORCA           - .hess        --> .freq.orca

    traj_generator.py can read initial conditions from previous sampling
    NewtonX        - final_output --> .init.newtonx
    traj_generator - .init        --> .init.xyz

    Molcas FSSH can only run with 1 CPU. So cores and jobs should be set to the same.

    BAGEL can run with MPI. The number of CPUs for parallelization will be determined by
    cores/jobs. The cores should be integer times of jobs. The traj_generator.py doesn't check it.


