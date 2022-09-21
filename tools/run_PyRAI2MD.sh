#!/bin/sh
## script for PyRAI2MD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:59:59
#SBATCH --job-name=tod-8mf-1
#SBATCH --partition=large,long,short,lopez
#SBATCH --mem=11000mb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export INPUT=input
export WORKDIR=$PWD

cd $WORKDIR
pyrai2md $INPUT

