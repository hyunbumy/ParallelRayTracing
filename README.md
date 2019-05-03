# ParallelRayTracing
EE 451 Project for Ray Tracing

## MPI
for running mpi code, use srun --mpi=pmi2 -n8 ./rtMPI

## CUDA
  1. Login to HPC
  2. Enable CUDA and NVCC:
    `source /usr/usc/cuda/default/setup.sh`
  3. Compile CUDA code:
    `cd src/cuda && make`
  4. Run the compiled CUDA executable on a GPU V100 enabled node:
    `srun -n1 --gres=gpu:v100:1 ./build/main`
