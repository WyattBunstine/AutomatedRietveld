import subprocess
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def write_slurm(loc):
    with open(loc + "test.slurm","w+") as f:
        f.write("""#!/bin/bash
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
python /data/tmcquee2/wbunsti1/AutoRiet/test/LaNiO_test.py > test.log""")

def send_to_rf(files, local_loc, remote_loc):
    for file in files:
        subprocess.run(["scp", local_loc + file,
                        "wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/AutoRiet/" + remote_loc + file])


def download_remote(loc: str):
    """
    This simply downloads a materials documents from remote.
    :param loc: The material location for which to download
    :return: if download was successful
    """
    if not os.path.exists(loc):
        os.mkdir(loc)

    #["scp", "-r", "wbunsti1@login.rockfish.jhu.edu:/home/wbunsti1/elk/elk-8.8.26/BSPOperDir/" + loc + "*", loc])

    subprocess.run(
        ["scp", "-r", "wbunsti1@login.rockfish.jhu.edu:/data/tmcquee2/wbunsti1/AutoRiet/" + loc + "*", "data/" + loc])





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    local_dir = "/Users/wyatt/PycharmProjects/AutomatedReitveld/data/LaNiO_test/"
    #write_slurm(local_dir)
    download_remote("LaNiO_test/")

    #send_to_rf(os.listdir(local_dir), local_dir, "LaNiO_test/")
