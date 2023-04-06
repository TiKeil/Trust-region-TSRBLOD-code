#!/bin/bash
 
#SBATCH --nodes=1                  # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=4
#SBATCH --mem=80G
#SBATCH --exclusive
#SBATCH --partition=express          # on which partition to submit the job
#SBATCH --time=2:00:00              # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=test_minimal      # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_keil02@uni-muenster.de # your mail address


# set an output file
#SBATCH --output /scratch/tmp/t_keil02/tr_tsrblod/final/minimal_express.dat

# run the application
module load intel
module load Python
module load SuiteSparse

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMPI_MCA_mpi_warn_on_fork=0

echo "Launching job:"
srun /home/t/t_keil02/tr_tsrblod_github/venv/bin/python -u /home/t/t_keil02/tr_tsrblod_github/scripts/minimal_test.py

if [ $? -eq 0 ]
then
    echo "Job ${SLURM_JOB_ID} completed successfully!"
else
    echo "FAILURE: Job ${SLURM_JOB_ID}"
fi
