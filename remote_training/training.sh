#!/bin/bash -l
# Batch script to run a serial job on Legion with the upgraded
# software stack under SGE.

# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=2

# 3. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=6:00:0

# 4. Request 4 gigabyte of RAM (must be an integer)
#$ -l mem=32G

# 6. Set the name of the job.
#$ -N new_cnn_training

# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /home/ucesxc0/Scratch/output/second_training_images/new_cnn

# 8. load the cuda module (in case you are running a CUDA program
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-10.0
source activate tf-2.0
./CNN_network.py
source deactivate
