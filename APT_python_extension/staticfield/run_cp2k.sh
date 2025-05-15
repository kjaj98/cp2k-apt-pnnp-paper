#!/bin/bash



# Request 4 nodes using 128 cores per node for 128 MPI tasks per node.



#SBATCH --job-name=cp2kdev

#SBATCH --nodes=8

#SBATCH --tasks-per-node=1

#SBATCH --cpus-per-task=128

#SBATCH --time=00:20:00



# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=e686

#SBATCH --partition=standard

#SBATCH --qos=short


# Setup the batch environment
module -q load PrgEnv-gnu
module -q load cray-fftw
module -q load cray-python
module -q load mkl
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
module use /work/e686/e686/kjoll/software/privatemodules
module load libffi/3.4.4
module load openssl/3.1.3
module load python/3.11.3
export PYTHONPATH=/work/e686/e686/shared/apt-devel/apt:${PYTHONPATH}


OMP_NUM_THREADS=128 srun --overlap --oversubscribe python3 /work/e686/e686/shared/apt-devel/apt/scripts/server.py --host '/tmp/qiskit_apt.server.socket' > outputserver.txt 2>outputserver.err &
sleep 160
OMP_NUM_THREADS=128 srun --overlap --oversubscribe --hint=nomultithread --distribution=block:block --nodes=8 --ntasks=1024 --ntasks-per-node=128 --cpus-per-task=1 /work/e686/e686/shared/apt-devel/cp2k-2023.1/exe/ARCHER2/cp2k.psmp -i start.inp -o hybrid.out
