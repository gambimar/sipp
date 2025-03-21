## Function to reshape the MRI data to the tight hull
### Steps ###
# 1. Fit SMPL to the MRI data. Freeze the betas and scale from scaneca's loose SMPL fit.
# 2. Tight fit the SMPL model to the MRI hull.
# 3. Repose the tight scaneca fit to the MRI fit.
# 4. Find deformation fields from tight MRI to tight scaneca fit.
# 5. Save the deformed part meshes.
# 6. Align SKEL to the loose MRI fit.

import argparse
import os


os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import sys
import pandas as pd
import torch

# Ignore segments, MRI has a good head, therefore we can add it to the SMPL fit
ignore_segments = ["leftArm","leftHandIndex1","leftForeArm","leftHand","rightArm","rightHandIndex1","rightForeArm","rightHand"]

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Reshape the MRI data to the tight scaneca hull')

    # Input path to the body scan data
    parser.add_argument('--scan_result_path', type=str,
                        default='results/P01/bodyhull/smpl_fit/scan/')
    parser.add_argument('--mri_hull_path', type=str,
                        default='results/P01/MRI/final_masks/all.stl')
    parser.add_argument('--scan_name', type=str, default='scaneca')
    parser.add_argument('--participant', type=str, default='P01')

    # Output path to save the reshaped MRI data
    parser.add_argument('--output_path', type=str,
                        default='results/P01/MRI/bodyhull/')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    participant_info = pd.read_csv("data/participant_info.csv")

    ### Step 1: Fit SMPL to the MRI data. Freeze the betas and scale from scaneca's loose SMPL fit.
    # Load scan loose fit result
    scan_result = np.load(os.path.join(args.scan_result_path, 'loose', f'{args.scan_name}_ignored_segments.npz'))
    betas, scale = scan_result['shape'], scan_result['scale']

    # Fitting procedure
    if True:
        sys.path.append("SMPL-Fitting")
        import fit_vertices
        import fit_body_model
        from utils import (load_config, process_body_model_fit_verts,
                           process_dataset_name, save_configs, set_seed,
                           initialize_fit_verts_loss_weights, load_loss_weights_config,
                           initialize_fit_bm_loss_weights)
        from src.bodyscan import mesh_utils

        print('Fitting SMPL to the MRI data, loose fit')
        # This is a bit copy-pasta from the mesh pipeline, could be refactored @todo
        cfg = load_config("SMPL-Fitting/configs/config.yaml")
        cfg_tight_fit = cfg['fit_vertices_optimization']
        cfg['fit_body_model_optimization']['use_landmarks'] = []
        cfg['fit_body_model_optimization']['ignore_segments'] = ignore_segments
        cfg['fit_body_model_optimization']['volume_target'] = False
        sex = participant_info[participant_info["participant"] == args.participant].sex.values[0]
        #
        landmarks_path = "results/P01/bodyhull/transformed/bodyscan_landmarks.json" # This is useless, but we need to provide it
        scan_info = {"participant_id": args.participant,
                    "scan_name": "MRI",
                    "fit_type": "loose"}
        cfg = mesh_utils.process_cfg_body_model(cfg, scan_info, args.mri_hull_path, landmarks_path)
        cfg['body_model_gender'] = "FEMALE" if sex == "F" else "MALE"  #
        init_coords = torch.tensor([-0.2, 1.5, 0.15])
        cfg['init_params'] = {'trans': init_coords,
                              'shape': torch.tensor(betas),
                              'scale': torch.tensor(scale)}
        ## TODO: Fix the betas and scale
        cfg['refine_params'] = ['pose','trans'] # Only optimize pose and translation
        cfg["save_path"] = args.output_path+"/loose"
        os.makedirs(cfg["save_path"], exist_ok=True)
        fit_body_model.fit_body_model_onto_scan(cfg)

        res_path = f"{args.output_path}/loose/{args.mri_hull_path.split('/')[-1].split('.')[0]}.npz"
        smpl_result = np.load(res_path, allow_pickle=True)
        mesh_utils.smpl_result_to_obj(smpl_result, res_path.replace(".npz", ".obj"), has_faces=False)

if __name__ == "__main__":
    main()
