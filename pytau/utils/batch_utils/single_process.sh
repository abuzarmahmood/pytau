# Set up environment for running a single process on a single core
# =======================
export OMP_NUM_THREADS=1

# Set theano compiledir for this run
# ==================================

# Get local directory
DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))

# Get job id
JOBBASE=$(basename $1)
# Take off extension
JOBID=${JOBBASE%.*}

export THEANO_FLAGS="base_compiledir=${DIR}/parallel_temp/temp_compile_dir/${JOBID}/.theano"

# Run fit
# =======
python fit_from_file.py $1
