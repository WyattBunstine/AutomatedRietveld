import subprocess
import os
import json
import shutil

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def write_slurm(loc, filename, jobname= "Python Job"):
    with open(loc + filename[:-3] + ".slurm","w+") as f:
        f.write("""#!/bin/bash
#SBATCH --job-name="""+jobname+"""
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
python /data/tmcquee2/wbunsti1/AutoRiet/""" + loc + filename + " > fit.log""")

def send_to_rf(local_loc, remote_loc):
    for file in os.listdir(local_loc):
        if ".cif" in file or ".slurm" in file or ".prm" in file or ".py" in file:
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


def main():
    '''
    This is the main function that creates a project script to run locally or on rockfish
    :return:
    '''
    dir = "data/SrTiO/"
    remote = False
    config = {"detectors": [dict(
        data_file="./ML_Bjork_20250202_1_WLB_0_(EXT_FS_N19G017)_(EXT_BTC_50000729)-SrTiO_n=0_1070C_15min.raw",
        detector_file="./LabXRD.prm",
        min_two_theta=5.0,
        max_two_theta=60.0)],
        "phases": [dict(phase_file="./SrTiO3.cif",
                        phase_label="SrTiO3"),
                   dict(phase_file="./Sr2TiO4.cif",
                        phase_label="Sr2TiO4"),
                   dict(phase_file="./Sr3Ti2O7.cif",
                        phase_label="Sr3Ti2O7"),
                   dict(phase_file="./Sr4Ti3O10.cif",
                        phase_label="Sr4Ti3O10"),
                   dict(phase_file="./SrO.cif",
                        phase_label="SrO")],
        "generations": 100,
        "evaluations": 100}

    with open(dir + 'config.json', 'w') as f:
        json.dump(config, f)

    shutil.copyfile("AutoRietveldScript.py", dir + "AutoRietveldScript.py")

    if remote:
        write_slurm(dir, "LaNiO_test_2.py", jobname="LaNiO_test_2")
        subprocess.run(
            ["ssh", "wbunsti1@login.rockfish.jhu.edu", "cd", "/data/tmcquee2/wbunsti1/AutoRiet;", "mkdir",
             dir + ";", "exit"])

if __name__ == '__main__':
    main()




    config = {"detectors": [dict(
        data_file="./ML_Bodie_20241106_6_RYH_(EXT_BTC_50001777)_(EXT_Strem_L02402203)_(EXT_MPB_U1123352219)-RYH1065F_LaNiO3_3gKBr_100Chrup_6hrdw_1100C_100Chrdn_ grey_fluxrem_15min.raw",
        detector_file="./LabXRD.prm",
        min_two_theta=5.0,
        max_two_theta=60.0)],
        "phases": [dict(phase_file="./LaNiO3.cif",
                        phase_label="LaNiO3"),
                   dict(phase_file="./La2NiO4.cif",
                        phase_label="La2NiO4"),
                   dict(phase_file="./La3Ni2O7.cif",
                        phase_label="La3Ni2O7"),
                   dict(phase_file="./La4Ni3O10.cif",
                        phase_label="La4Ni3O10"),
                   dict(phase_file="./NiO.cif",
                        phase_label="NiO")],
    "generations": 100,
    "evaluations": 100}