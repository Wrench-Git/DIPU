PLATFORM=/nvme/share/share/platform
ENV_NAME=pt2.0_diopi
export PATH=`python ${PLATFORM}/env/clear_path.py PATH`
export LD_LIBRARY_PATH=`python ${PLATFORM}/env/clear_path.py LD_LIBRARY_PATH`
GCC_ROOT=${PLATFORM}/dep/gcc-7.5
CONDA_ROOT=${PLATFORM}/env/miniconda3.8
export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export CUDA_PATH=${PLATFORM}/dep/cuda11.0-cudnn8.0
export MPI_ROOT=${PLATFORM}/dep/openmpi-4.0.5-cuda11.0
export NCCL_ROOT=${PLATFORM}/dep/nccl-2.9.8-cuda11.0
export GTEST_ROOT=${PLATFORM}/dep/googletest-gcc5.4


export LD_LIBRARY_PATH=${CONDA_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PLATFORM}/dep/binutils-2.27/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib/:$LD_LIBRARY_PATH
export PIP_CONFIG_FILE=${CONDA_ROOT}/envs/${ENV_NAME}/.pip/pip.conf
# alias gbd='/nvme/share/share/xt/xplatform/bins/gdb'

export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib/
export DIPU_ROOT=$(pwd)/torch_dipu
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}
export PYTORCH_DIR=/nvme/share/share/platform/env/miniconda3.8/envs/pt2.0_diopi/lib/python3.8/site-packages
export LIBRARY_PATH=$DIPU_ROOT:${DIOPI_ROOT}:${LIBRARY_PATH}; LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PYTHONPATH=${PYTORCH_DIR}:${PYTHONPATH}
export PATH=/nvme/share/share/platform/dep/patchelf-0.12/bin:${CONDA_ROOT}/envs/dipu_poc/bin:${CONDA_ROOT}/bin:${PATH}
export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6
export PYTHON_INCLUDE_DIR=/nvme/share/share/platform/env/miniconda3.8/envs/pt2.0_diopi/include/python3.8
export PYTORCH_DIR_110=/nvme/share/share/platform/env/miniconda3.8/envs/pt2.0_diopi/pytorch1.10
export PYTORCH_TEST_DIR=/nvme/share/share/platform/env/miniconda3.8/envs/pt2.0_diopi/pytorch2.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export NCCL_INCLUDE_DIRS=${NCCL_ROOT}/include
export VENDOR_INCLUDE_DIRS=${CUDA_PATH}/include

export CUDA_LAUNCH_BLOCKING=1
#export DIPU_FORCE_FALLBACK_OPS_LIST=_index_put_impl_,index.Tensor_out
export CPLUS_INCLUDE_PATH=${CONDA_ROOT}/envs/${ENV_NAME}/include/python3.8:$CPLUS_INCLUDE_PATH
export PYTHONPATH=${PWD}:$PYTHONPATH

source activate $ENV_NAME
