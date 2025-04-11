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
from src.bodyscan.mesh_utils import deform_mesh
import numpy as np
import sys
import pandas as pd
import torch

# Ignore segments, MRI has a good head, therefore we can add it to the SMPL fit. Some participants have missing toes
ignore_segments = {
    "default": ["leftArm","leftHandIndex1","leftForeArm","leftHand","rightArm","rightHandIndex1","rightForeArm","rightHand"],
    "P02": ["leftArm","leftHandIndex1","leftForeArm","leftHand","rightArm","rightHandIndex1","rightForeArm","rightHand", "leftToeBase", "rightToeBase"],
    "P06": ["leftArm","leftHandIndex1","leftForeArm","leftHand","rightArm","rightHandIndex1","rightForeArm","rightHand", "leftToeBase", "rightToeBase"],
}

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Reshape the MRI data to the tight scaneca hull')

    # Input path to the body scan data
    parser.add_argument('--scan_result_path', type=str,
                        default='results/P01/bodyhull/smpl_fit/scan/')
    parser.add_argument('--mri_hull_path', type=str,
                        default='results/P01/MRI/final_masks')
    parser.add_argument('--scan_name', type=str, default='scaneca')
    parser.add_argument('--participant', type=str, default='P01')
    parser.add_argument('--hull_name_mri', type=str, default='all')

    # Output path to save the reshaped MRI data
    parser.add_argument('--output_path', type=str,
                        default='results/P01/MRI/bodyhull/')
    parser.add_argument('--use_tight_fit', default=True)
    parser.add_argument('--deform_no_ignored_segments', default=True,
                        help='Deform the mesh without taking ignored segments into account')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    participant_info = pd.read_csv("data/participant_info.csv")

    ### Step 1: Fit SMPL to the MRI data. Freeze the betas and scale from scaneca's loose SMPL fit.
    # Load scan loose fit result
    scan_result = np.load(os.path.join(args.scan_result_path, 'loose', f'{args.scan_name}_ignored_segments.npz'))
    betas, scale = scan_result['shape'], scan_result['scale']

    # Fitting procedure, known scaneca shape --> pose to MRI
    if True:
        sys.path.append("SMPL-Fitting")
        import fit_vertices
        import fit_body_model
        from utils import (load_config, process_body_model_fit_verts,
                           process_dataset_name, save_configs, set_seed,
                           initialize_fit_verts_loss_weights, load_loss_weights_config,
                           initialize_fit_bm_loss_weights)
        from src.bodyscan import mesh_utils

        for ssm in ['smpl', 'skel']: # 
            print('Fitting SMPL to the MRI data, loose fit')
            # This is a bit copy-pasta from the mesh pipeline, could be refactored @todo
            cfg = load_config("SMPL-Fitting/configs/config.yaml")
            cfg_tight_fit = cfg['fit_vertices_optimization']
            cfg['fit_body_model_optimization']['use_landmarks'] = []
            ign_ = ignore_segments[args.participant] if args.participant in ignore_segments else ignore_segments["default"]
            cfg['fit_body_model_optimization']['ignore_segments'] = ign_
            cfg['fit_body_model_optimization']['volume_target'] = False
            sex = participant_info[participant_info["participant"] == args.participant].sex.values[0]
            #
            landmarks_path = "results/P01/bodyhull/transformed/bodyscan_landmarks.json" # This is useless, but we need to provide it
            scan_info = {"participant_id": args.participant,
                        "scan_name": "MRI",
                        "fit_type": "loose"}
            scan_info["fit_type"] = "loose" if ssm == "smpl" else "loose_skel"
            cfg["body_model"] = ssm
            if ssm == "skel":
                cfg['reference_smpl'] = f"{args.output_path}/loose/MRI_ignored_segments.obj"#
                cfg['loss_weights'] = initialize_fit_bm_loss_weights(
                                                load_loss_weights_config(which_strategy="fit_bm_loss_weight_strategy",
                                                which_option="skel",
                                                path="SMPL-Fitting/configs/loss_weight_configs.yaml"))
            cfg = mesh_utils.process_cfg_body_model(cfg, scan_info, args.mri_hull_path+"/all.stl", landmarks_path)
            cfg['body_model_gender'] = "FEMALE" if sex == "F" else "MALE"  #
            init_coords = torch.tensor([-0.2, 1.5, 0.15])
            cfg['init_params'] = {'trans': init_coords,
                                  'shape': torch.tensor(betas),
                                  'scale': torch.tensor(scale)}
            ## TODO: Fix the betas and scale
            cfg['refine_params'] = ['pose','trans','scale'] # Only optimize pose and translation, maybe scale is off

            cfg["save_path"] = args.output_path+"/loose" if ssm == "smpl" else args.output_path+"/loose_skel"
            os.makedirs(cfg["save_path"], exist_ok=True)

            #fit_body_model.fit_body_model_onto_scan(cfg)
            if ssm == "smpl":
                res_path = f"{args.output_path}/loose/{args.hull_name_mri}.npz"
            else:
                res_path = f"{args.output_path}/loose_skel/{args.hull_name_mri}.npz"
            smpl_result = np.load(res_path, allow_pickle=True)
            mesh_utils.smpl_result_to_obj(smpl_result, res_path.replace(".npz", ".obj"), has_faces=False)

        if True: # Tight vertex fit
            scan_info["fit_type"] = "tight" 
            cfg_loss_weights = load_loss_weights_config(
                "fit_verts_loss_weight_strategy",
                "default_option",
                path="SMPL-Fitting/configs/loss_weight_configs.yaml")
            for key in cfg_loss_weights.keys():
                cfg_loss_weights[key]['landmark'] = 0
                cfg_loss_weights[key]['smooth'] = 10
                cfg_loss_weights[key]['normal'] = 10
                cfg_loss_weights[key]['neighbordistance'] = 1e3 # We need the "skin" to be warped to the new shape, so that features beneath that do not get distorted -> That problem does not apply to the standing shapes, where variations are smaller in general
            cfg_loss_weights = initialize_fit_verts_loss_weights(cfg_loss_weights)
            cfg.update(cfg_tight_fit)
            cfg["loss_weights"]=cfg_loss_weights
            cfg['start_from_previous_results'] = cfg['save_path']
            # Go aggressive on the loss weights
    
            cfg['start_from_body_model'] = None
            cfg['save_path'] =  args.output_path+"/tight"
            os.makedirs(cfg["save_path"], exist_ok=True)
            cfg = process_body_model_fit_verts(cfg)
            #cfg['ignore_segments'] #+= ["head"] # Should not remove head
            cfg['use_losses'] = ["neighbordistance","smooth","normal", "data"]
            cfg['random_init_A'] = False
            set_seed(cfg["seed"])
            save_configs(cfg)
            # Fit the model to vertices
            fit_vertices.fit_vertices_onto_scan(cfg)
            
            res_path = f"{args.output_path}/tight/{args.hull_name_mri}.npz"

            smpl_result = np.load(res_path, allow_pickle=True)
            mesh_utils.smpl_result_to_obj(smpl_result, res_path.replace(".npz", ".obj"), has_faces=False)

        
        # RBF interpolate the meshes
        res_path = f"{args.output_path}/tight/{args.hull_name_mri}.obj"
        input_mesh = mesh_utils.load_trimesh_from_file(res_path)
        res_path = f"{args.output_path}/loose/{args.hull_name_mri}.obj"
        target_mesh = mesh_utils.load_trimesh_from_file(res_path)
        input_points = input_mesh.vertices
        target_points = target_mesh.vertices

        if args.use_tight_fit: # Apply the loose-tight fit transformation to the loose fit that is rigged to match the MRI
            start_mesh = mesh_utils.load_trimesh_from_file(f"{args.scan_result_path}/loose/{args.scan_name}_ignored_segments.obj")
            tight_mesh = mesh_utils.load_trimesh_from_file(f"{args.scan_result_path}/tight/{args.scan_name}_ignored_segments.obj")
            target_points = mesh_utils.project_mesh_to_deformed(start_mesh, tight_mesh.vertices, target_mesh)
            
        if args.deform_no_ignored_segments:
            # Remove the _ignored segments from the target and input meshes 
            import json
            ignore_verts = []
            with open("SMPL-Fitting/smpl_vert_segmentation.json", 'r') as f:
                vert_segmentation = json.load(f)
            for key in ign_:
                ignore_verts.extend(vert_segmentation[key])
            mask = torch.ones(6890, dtype=torch.bool)
            mask[ignore_verts] = False
            input_points = input_points[mask]
            target_points = target_points[mask]


        print("Interpolating points..., this may take a while")
        for mesh_name in ["all", "cortical_bone", "trabecular_bone", "lung", "fat", "muscle", "other"]:
            print('Deforming mesh:', mesh_name)
            input_path = f"{args.mri_hull_path}/{mesh_name}.stl"
            os.makedirs(f"{args.mri_hull_path}/deformed", exist_ok=True)
            target_path = f"{args.mri_hull_path}/deformed/{mesh_name}.obj"


            mesh = mesh_utils.load_trimesh_from_file(input_path)
            initial_volume = mesh.volume
            scan_vertices = mesh.vertices
            scan_vertices[:, [1,2]] = scan_vertices[:, [2,1]]  # Swap y and z axis - stl has a different orientation
            scan_vertices[:, 2] *= -1  # Invert z axis

            #mesh = mesh_utils.load_trimesh_from_file(f"{args.output_path}/loose/{args.hull_name_mri}.obj")
            mesh.vertices = deform_mesh(input_points, target_points, scan_vertices, kernel='thin_plate_spline', neighbors=None, smoothing=100)
            mesh.export(target_path)
            print(f"Deformed mesh saved to {target_path}")
            print(f"Volume of mesh {mesh_name} [L]: Before: {initial_volume*1000:.2f}, After: {mesh.volume*1000:.2f}")
             






if __name__ == "__main__":
    main()
