#!/bin/sh
##### ------ Setup BAGEL  ------
export BAGEL=/share/apps/bagel
export MPI=''
export BLAS=/share/apps/blas-3.10.0
export LAPACK=/share/apps/lapack-3.10.1/lib
export BOOST=/share/apps/boost_1_80_0/
export MKL=/share/apps/intel/oneapi/setvars.sh
export ARCH=''

##### ------ Setup MOLCAS ------
export MOLCAS=/share/apps/molcas
export TINKER=/share/apps/molcas/tinker-6.3.3/bin

##### ------ Setup ORCA ------
export ORCA=/share/apps/orca_5_0_3_linux_x86-64_openmpi411

##### ------ Setup OpenQP ------
export OPENQP=/share/apps/openqp
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=SEQUENTIAL
export LD_LIBRARY_PATH=/share/apps/gcc-11.3.0/lib64:/share/apps/intel/oneapi/mkl/2022.1.0/lib/intel64/:/home/guojd/nbo6/bin:

##### ------ Setup GFN-xTB ------
export XTB=/share/apps/xtb-6.5.1

python3 test_case.py
