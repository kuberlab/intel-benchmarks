#!/bin/sh
cd ~
source ../env.sh
export MKL_NUM_THREADS=4
export I_MPI_FABRICS=tcp
mpirun -np 16 -machinefile $PBS_NODEFILE python3 intel-benchmarks/mnist/tf_mnist_estimator.py --data-dir ~/data --iterations 12512 --batch-size 64
