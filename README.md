# MRIGait_EmpkinS
Repository for C01 - Phase I - WP4: MRI Validation of MSK model personalization. Attention:
**We cannot provide all data, such as the raw MRI data, smartphone pictures and initial avatars, due to privacy concerns. Therefore, running the "mesh_pipeline.py" and "reshape_mri.py" scripts will not work out of the box. However, we provide the SMPL fits and segmentation masks in the "results" folder in the data repository. The remaining scripts can be run with these results to reproduce the results in the paper.**

## Getting started

1. Clone this repository
2. Install requirements
2. Get our fork of SMPL-Fitting from [here](github), follow the instructions to install it.
3. Get SKEL from [here](github) and follow the instructions to install it - the dependencies are already included in our environment file.
4. Get HIT from [here](github) and follow the instructions to install it. You will need to create a new environment, because the dependencies are incompatible with the the main environment file.

### Get our data
1. Download the data from [here](todo) and extract it in the root of this repository.
2. The data should contain two folders:
```
data/...
results/...
```
We do not provide raw MRI data, but the SMPL fits together with the segmentation masks are available in the "results" folder.

3. Obtain addBiomechanics from [here](https://app.addbiomechanics.org/data/8cf6c9e4-5b47-4c5e-9b8f-90ee4aa39575/MRI2025/), download them and structure them as follows:
```
results/P01/
    gait/
        addB-ignore_phys/ <-- S01 folder on addBiomechanics
            IK/
            ID/
            Models/
        addB-with_phys/
            S01_HIT/ <-- from addBiomechanics
            S01_SGEN/ <-- from addBiomechanics
            generic/ <-- S01_with_phys folder on addBiomechanics
    ...
```

### Reproduce experiments
1. Get all results from the [results repository](todo) or run the scripts below to reproduce the results:
```
python scripts/bodyscan/mesh_pipeline.py
python scripts/bodyscan/hit_predictions.py
python scripts/bodyscan/all_bisp_estimations.py

python scripts/smpl_generic/create_generic_models.py
python scripts/smpl_generic/generic_models_scale_and_id.py

python scripts/gait/create_all_osim_models.py
python scripts/gait/run_id_all.py

python -m src.mri.reshape_mri --scan_result_path results/P01/bodyhull/smpl_fit/scan/ --mri_hull_path results/P01/MRI/final_masks --scan_name scaneca --participant P01 --hull_name_mri all --output_path results/P01/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P02/bodyhull/smpl_fit/scan/ --mri_hull_path results/P02/MRI/final_masks --scan_name scaneca --participant P02 --hull_name_mri all --output_path results/P02/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P03/bodyhull/smpl_fit/scan/ --mri_hull_path results/P03/MRI/final_masks --scan_name scaneca --participant P03 --hull_name_mri all --output_path results/P03/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P04/bodyhull/smpl_fit/scan/ --mri_hull_path results/P04/MRI/final_masks --scan_name scaneca --participant P04 --hull_name_mri all --output_path results/P04/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P05/bodyhull/smpl_fit/scan/ --mri_hull_path results/P05/MRI/final_masks --scan_name scaneca --participant P05 --hull_name_mri all --output_path results/P05/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P06/bodyhull/smpl_fit/scan/ --mri_hull_path results/P06/MRI/final_masks --scan_name scaneca --participant P06 --hull_name_mri all --output_path results/P06/MRI/bodyhull/
python scripts/mri/mri_id.py
````
2. Reproduce the results for the paper:
Run `python scripts/mrigait_benchmark/run_mrigait_benchmark.py` to reproduce the MRIgait benchmark results. Then, run all cells in the Jupyter notebook `notebooks/paper_plots.ipynb` to reproduce the plots/other tables in the paper. 

