#!/bin/bash
 
#SBATCH --nodes=40                  # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=10
#SBATCH --mem=80G
#SBATCH --exclusive
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=4:00:00              # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=test_exp_1       # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_keil02@uni-muenster.de # your mail address


# set an output file
#SBATCH --output /scratch/tmp/t_keil02/lrblod/FINAL/exp1.dat

# run the application
module load intel
module load Python
module load SuiteSparse

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMPI_MCA_mpi_warn_on_fork=0

echo "Launching job:"
srun /home/t/t_keil02/lrb-for-pde-opt-code/venv_new/bin/python -u /home/t/t_keil02/lrb-for-pde-opt-code/notebooks/FINAL_exp_1.py

if [ $? -eq 0 ]
then
    echo "Job ${SLURM_JOB_ID} completed successfully!"
else
    echo "FAILURE: Job ${SLURM_JOB_ID}"
fi
