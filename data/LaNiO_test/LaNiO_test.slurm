#!/bin/bash
#SBATCH --job-name=LaNiO_RR_test
#SBATCH --time=01:00:00
#SBATCH --output=test_output
#SBATCH --partition=parallel
#SBATCH --account=iqm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=wbunsti1@jhu.edu

module load GCC/12.3.0 Python/3.11.3-GCCcore-12.3.0
source ~/virtual/py_netlogo/bin/activate
python /data/tmcquee2/wbunsti1/AutoRiet/LaNiO_test/LaNiO_test.py > test.log