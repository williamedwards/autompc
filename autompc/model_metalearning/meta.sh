#!/bin/bash
#SBATCH --time=04:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=multi-serial_job      # Name of batch job
#SBATCH --partition=secondary            # Partition (queue)
#SBATCH --output=multi-serial.o%j        # Name of batch job output file

# Run the serial executable
# ./a.out < input > output
# module load anaconda/3

module load openmpi
mpiexec -np 16 /home/baoyul2/.conda/envs/autompc/bin/python /home/baoyul2/autompc/autompc/model_metalearning/get_portfolio_configurations.py