echo "bash_profile: Loading modules..."

module load clang/6.0.0
module load gcc/8.2.0
#module load gcc

module load make/4.2.1
module load cmake/3.12.1

module load nvidia/cuda/10.0
# module load intel/mpi/2019.1
module load openmpi/mpi_thread_multiple/cuda/2.1.5

module load python/3.6.4
module load git/2.19.2

echo "bash_profile: Seting additional env variables..."

export TERM=xterm-256color
export CC=gcc
export CXX=g++
