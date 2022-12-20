import subprocess

# NOTE: each ablation test run will take ~9 hours
# Using 20 (best recall/most sensitive)
# Using 88 (best precision/most frames removed)
# Using 75 (best mixture of FR/TDR (for tracklet))

# no intensity
cmd = (
    "python multirun.py -d test "
    + "-p "
    + "experiments/20.json "
    + "experiments/88.json "
    + "experiments/75.json "
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
    + "experiments/20.json "
    + "experiments/88.json "
    + "experiments/75.json "
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
    + "experiments/20.json "
    + "experiments/88.json "
    + "experiments/75.json "
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
    + "experiments/20.json "
    + "experiments/88.json "
    + "experiments/75.json "
    + "-f "
    + "turbine_filter "
    + "denoise_filter "
    + "intensity_filter "
)
subprocess.call(cmd, shell=True)
