import os
import subprocess
from tunamelt import REPO_PATH, log

# NOTE: each ablation test run will take ~9 hours
# Using 20 (best recall/most sensitive)
# Using 88 (best precision/most frames removed)
# Using 75 (best mixture of FR/TDR (for tracklet))

log.warning("Running this will take a very long time!\n\nPress enter to proceed.")
_ = input()
os.chdir(f"{REPO_PATH}/scripts/experiments")

# no intensity
cmd = (
    "python multirun.py -d test "
    + "-p "
    + "scripts/experiments/params/20.json "
    + "scripts/experiments/params/88.json "
    + "scripts/experiments/params/75.json "
    + "-f "
    + "mean_filter "
    + "turbine_filter "
    + "denoise_filter "
)
subprocess.call(cmd, shell=True)

# no denoise
cmd = (
    "python multirun.py -d test "
    + "-p "
    + "scripts/experiments/params/20.json "
    + "scripts/experiments/params/88.json "
    + "scripts/experiments/params/75.json "
    + "-f "
    + "mean_filter "
    + "turbine_filter "
    + "intensity_filter "
)
subprocess.call(cmd, shell=True)

# no turbine
cmd = (
    "python multirun.py -d test "
    + "-p "
    + "scripts/experiments/params/20.json "
    + "scripts/experiments/params/88.json "
    + "scripts/experiments/params/75.json "
    + "-f "
    + "mean_filter "
    + "denoise_filter "
    + "intensity_filter "
)
subprocess.call(cmd, shell=True)

# no mean
cmd = (
    "python multirun.py -d test "
    + "-p "
    + "scripts/experiments/params/20.json "
    + "scripts/experiments/params/88.json "
    + "scripts/experiments/params/75.json "
    + "-f "
    + "turbine_filter "
    + "denoise_filter "
    + "intensity_filter "
)
subprocess.call(cmd, shell=True)
