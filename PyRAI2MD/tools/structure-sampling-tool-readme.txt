
              Structure Sampling Tool Manual

=========================================================================

   !!! 	       	       	       	       	       	       	  !!!
   !!!                 Important Notes       	       	  !!!
   !!!                                                    !!!
   !!! The structure_sampling_tool.py read normal modes   !!!
   !!! in the following format:                           !!!
   !!! Molcas   -   mass-weighted unnormalized            !!!
   !!! Gaussian - unmass-weighted   normalized            !!!
   !!! ORCA     - unmass-weighted   normalized            !!!
   !!! BAGEL    - unmass-weighted   normalized            !!!
   !!!                                                    !!!


1. The aim of this tool
------------------------------------------------------------------------------------------------------
   This tool provides a quick generation of structures for training data
   calculations. There are two main techniques to generate structures,
   Wigner sampling and geometrical interpolations. The Wigner sampling use
   PyRAI2MD's sampling code and the geometrical interpolation requires the
   geodesic_interpolate package (https://github.com/virtualzx-nad/geodesic-interpolate).

   Interpolation can be done in two ways.
   1. Use the optimized structure of the reactant, meci, and product to do
   two or three points interpolation. This approach first give you a single
   pathway. To broaden the interpolated pathway, the atomic displacement of
   Wigner sampling for reactant can be added to the interpolated structures.
   This approach is fast because only one pathway is interpolated and Wigner
   sampling is usually fast too. However, some problems happen when the
   vibration modes are notably different in the reactant and another point on
   the interpolated path, for instance, the product. The Wigner sampled
   displacements of reactant unphysically distort the product structure,
   especially for the C-H bonds.

   2. Use the Wigner sampled structures of reactant, meci, and product to
   do two or three points interpolation. This approach avoid the unphysical
   structures of product as they are all sampled according to their vibrational
   modes. In this case, several pathways can be interpolated by selecting
   a series of reactant, meci, product structures. Thus, it is more time-consuming
   than the first approach.

   In our experience, approach 1 works well for small molecules, where the
   reactant and product structures are not different very much in space. For
   large molecule, such as a chromophore with a long conjugated chain,
   approach 2 produces better structures for training data calculation.

   This script can perform standalone Wigner sampling, geometrical interpolation,
   and mixed Wigner sampling and interpolations using the two approach explained
   above.

2. Quick command line usage
------------------------------------------------------------------------------------------------------
    You can simply run this script on command line as below:

        python3 structure_sampling_tool.py sampling

    You can find more options with explanations as below:

        python traj_generator.py for help

3. Input file
------------------------------------------------------------------------------------------------------
    The name of input file is arbitrary. It follows a tow-columns format
    (keywords on the left and values on the right) in free order:

      cpus          1 number of cpu for parallel generation
      wigner        reac.freq.molden meci.freq.molden prod.freq.molden
      seed          1
      temp          298.15
      nw            10
      scale         1
      refxyz        reac.xyz
      interp        reac.xyz meci.xyz prod.xyz
      ni            10
      skip_wigner   0
      skip_first    0
      skip_last     0

4. Keywords and values
------------------------------------------------------------------------------------------------------
    Here are all keywords in this script.

           keyword                 Value
------------------------------------------------------------------------------------------------------
            cpus        Number of CPUs for interpolating path between Wigner sampled structure
                        This keyword is only used for interpolation approach 2.
                        Default is 1.

            wigner      Specify the input files for Wigner sampling.
                        The default is None, which will skip Wigner sampling.
                        You can give up to three filenames. The number of files will decide the
                        job type. See example in Section 5.
                        The filename follow the same style as the traj_generator.py input.
                        For reading vibrational modes of different program, it reads
                        .freq.molden, .freq.g16, freq.orca, and freq.bagel file.
                        For loading previous Wigner sampling results, it reads
                        .init.xyz file

            iseed       Random number seed.
                        Default is 1.
                        You can choose any number from 0 to + infinite to make sampling reproducible.

            temp        Wigner sampling temperature in K.
                        Default is 298.15 K.

            nw          Set a target number of Wigner sampling structures.
                        Default is 10.

            scale       Scaling factor of Wigner sampling structure.
                        Default is 1.
                        This factor linear scale the atomic displacements and velocities.

            refxyz      Specify a xyz file as the reference structure for Wigner sampling.
                        Default is None.
                        The reference structures will be used to compute the atomic displacement.
                        The scaled Wigner sampling structures will not be generated if this file
                        is not given.

            interp      Specify the input files for interpolation.
                        The default is None, which will skip interpolation using these structures.
                        You can give up to three filenames. The number of files will decide the
                        job type. See example in Section 5.
                        This keyword is only used for interpolation approach 1.

            ni          Set a target number of interpolation structures.
                        Default is 10.
                        In two points interpolation, this value is the final number of interpolated
                        structures. In three points interpolation, the final number of interpolated
                        structures is ni * 2.

            skip_wigner Number of skipped structures in Wigner sampling.
                        Default is 0
                        The Wigner sampling is always sequential. This keyword allows you to
                        use Wigner sampling structures later in the structure list.
                        This value do not change the target number of Wigner sampling structures,
                        set by nw.

            skip_first  Number of skipped structures from the beginning of the interpolation.
                        Default is 0
                        This value does not change the target number of interpolated structures.

            skip_last   Number of skipped structures from the end of the interpolation.
                        Default is 0
                        This value does not change the target number of interpolated structures.

5. Job types
------------------------------------------------------------------------------------------------------

    Decide the job you want to run

    This tool can automatically detect the job type according to the number of input
    files provides in keywords wigner and interp. The following table summarize the
    meaning of the jobs with different number of input files.

------------------------------------------------------------------------------------------------------
          Number     Number of
 Case   of Wigner  Interpolation        Job type
          files        files
------------------------------------------------------------------------------------------------------
   1        0           <2         No job will be done

   2        0          >=2         Two or three points interpolation

   3        1            0         One point Wigner sampling

   4        1            1         One point Wigner sampling and read interpolated structures
                                   and add Wigner sampled atomic displacement to interpolated structures

   5        1          >=2         One point Wigner sampling and two or three points interpolation
                                   and add Wigner sampled atomic displacement to interpolated structures

   6      >=2          >=0         Two or three point Wigner sampling
                                   and interpolation between the Wigner sampled structures
------------------------------------------------------------------------------------------------------

    Here are detailed explanations
    Case 1:
        0 Wigner input file skips the Wigner sampling.
        <2 interpolation input files does not have enough structure to do interpolation.
        Thus, the job terminates immediately.
        In this case, nothing will happen.

    Case 2:
        0 Wigner input file skip the Wigner sampling.
        >=2 interpolation input files have enough structure for interpolation.
        2 interpolation input files will compute 1 path.
        3 interpolation input files will compute 2 path.
        In this case, the job is equivalent to a standalone geometrical interpolation.

    Case 3:
        1 Wigner input file can do Wigner sampling.
        <2 interpolation input files does not have enough structure to do interpolation.
        In this case, the job is equivalent to a standalone Wigner sampling.

    Case 4:
        1 Wigner input file can do Wigner sampling.
        1 interpolation input file will attempt to read all structures from the given file.
        It allows you to read the previously interpolated structures saved in one file.
        The Wigner sampling structures will be mixing into the interpolated structures.
        In this case, the job is equivalent to interpolation approach 1.

    Case 5:
        1 Wigner input file can do Wigner sampling.
        >=2 interpolation input files have enough structure for interpolation.
        2 interpolation input files will compute 1 path.
        3 interpolation input files will compute 2 path.
        The Wigner sampling structures will be mixing into the interpolated structures.
        In this case, the job is equivalent to interpolation approach 1.

    Case 6:
        >=2 Wigner input files will do Wigner sampling for each provided structure.
        2 Wigner input files will sample 2 list of structures.
        3 Wigner input files will sample 3 list of structures.
        The interpolation input files are no longer used in this case. Instead, the
        interpolation will be done using the Wigner sampled structures.
        2 Wigner input files will compute 1 path.
        3 Wigner input files will compute 2 path.
        In this case, the job is equivalent to interpolation approach 2.

6. Examples
------------------------------------------------------------------------------------------------------
    6.1 Standalone Wigner sampling
------------------------------------------------------------------------------------------------------
    The sampling file is like:

        cpus 10
        wigner file.freq.molden
        seed 1
        temp 298.15
        nw 200

    file.freq.molden is a frequency file generated by OpenMolcas calculations.
    nw 200 will generate 200 Wigner sampling structure. It generates the .init
    file, which can be reused in future jobs. To reuse .init file, just rename
    it as .init.xyz and change the filename in keyword wigner accordingly, e.g.,

        wigner file.init.xyz

    If you wish to reduce the Wigner sampling magnitude by 0.7, you need to add
    the following to the sampling file:

        refxyz file.xyz
        scale 0.7

    file.xyz must be the optimized structures that was used in the frequency calculation.
    Otherwise, the atomic displacements in the Wigner sampling structures would be wrong.
------------------------------------------------------------------------------------------------------
    6.2 Standalone geometrical interpolation
------------------------------------------------------------------------------------------------------
    The sampling file is like:

        interp reac.xyz meci.xyz prod.xyz
        ni 20
        skip_first 1
        skip_last 1

    reac.xyz, meci.xyz and prod.xyz are the optimized structure of the reactant, meci and
    product.
    ni 20 will interpolate 20 structures between them (2 pathways), which give 40 structures in total.
    skip_first 1 and skip_last 1 will not include the first and last structure in the
    interpolated structure list. To understand these keywords correctly, the actual number
    of structure in the interpolation is the target number of structure adding with the
    skipped numbers of structures (42). Thus, there is no need to manually remove any
    structures in the output file, saved as interp.xyz

    If only two files are provided, a single pathway between them will be computed.
    And then the number of interpolated structures is 20.
------------------------------------------------------------------------------------------------------
    6.3 Interpolation approach 1
------------------------------------------------------------------------------------------------------
    The sampling file is like:

        wigner file.freq.molden
        seed 1
        temp 298.15
        nw 20
        ni 10
        refxyz file.xyz
        interp reac.xyz meci.xyz prod.xyz

    It first performs Wigner sampling by reading molcas frequency file and then interpolate structures
    between the given optimized structures.
    The atomic displacements are computed using the reference structure and then applied to the
    interpolated structures.
    The output structures are saved in two formats:
    wigner-interp-merged.xyz and wigner-interp-merged.init.xyz

    Note, if the reference structures (file.xyz) is same as the first interpolation input file (reac.xyz),
    you can ignore keyword refxyz. The script will automatically use the first interpolated structure
    as reference to compute the atomic displacement. In either case, the script will compute the RMSD
    between the first Wigner sampling structure, the reference structures and the first interpolated
    structures. A large RMSD (> 1A) indicates the structure inconsistency to help you debug.

    The following sampling file show how to reuse the previous file:

        wigner file.init.xyz
        nw 20
        ni 10
        interp interp.xyz

    The Wigner sampling structure and interpolated structures are directly read from the file.init.xyz
    and interp.xyz. The interpolation will start immediately.
------------------------------------------------------------------------------------------------------
    6.4.Interpolation approach 2
------------------------------------------------------------------------------------------------------
    The sampling file is like:

        cpus 10
        wigner reac.freq.molden meci.freq.molden prod.freq.molden
        seed 1
        temp 298.15
        nw 20
        ni 10

    It first performs Wigner sampling for three structures by reading their frequency files. It stores
    three list of Wigner sampling structures. Following the order, each structure in the lists are used
    for interpolation. Since three structures are available, the number of the interpolated pathways is 2
    and the number of the interpolated structures is 10 * 2 = 20. Thus, 20 Wigner sampling structures
    generate 20 * 20 = 400 structures, which are saved in two formats:
    interpolated-wigner.xyz and interpolated-wigner.init.xyz

