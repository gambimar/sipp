# MRIgait benchmark repository. Here, all data and code is listed that is needed to run the MRIgait benchmark. 

## Getting started

1. Download this repository
2. Create a python environment with the following packages:
    - `numpy`
    - `pandas`
    - `opensim`
3. Run `python scripts/mrigait_benchmark/run_mrigait_benchmark.py --model1='path/to/your/model1.osim' --model2='path/to/your/model2.osim' --trajectory='path/to/a/folder/with/trajectories' to run the benchmark, the output will be printed to the console. 

4. Using the function `mrigait_benchmark.main`, you can provide a folder containing IK data for the benchmark evaluation. In the script `run_mrigait_benchmark`, you can define a loading path for own models to compare your personalization against the existing baselines.
