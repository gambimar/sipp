import json

from src.bodyscan import mesh_utils
import pandas as pd
import numpy as np

body_scan_name = "final_fit_invtrans_bodyscan.obj"
face_scan_name = "final_fit_invtrans_facescan.obj"
skin_layer_name = "skin_layer.obj"
musc_layer_name = "musc_layer.obj"
skel_layer_name = "skel_layer.obj"

if __name__ == "__main__":
    # Add the parent-parent directory to the path
    import os
    import sys
    curr_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_file_path)))
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    print(os.getcwd())

    # Load external files
    landmarks = pd.read_csv(f"data/resources/landmarks.csv")
    participant_info = pd.read_csv("data/participant_info.csv")
    ignore_segments = json.load(open("data/resources/ignore_segments.json"))
    vertex_segmentation = json.load(open("SMPL-Fitting/smpl_vert_segmentation.json"))

    # Main loop
    for participant in range(1, 7):
        pid = f"P{str(participant).zfill(2)}"
        data_path = f"data/{pid}/bodyhull"
        result_path = f"results/{pid}/bodyhull"

        skin_layer = mesh_utils.load_trimesh_from_file(f"{data_path}/{skin_layer_name}")

        if False:
            for scan_name in [body_scan_name, face_scan_name]:
                ### Scan alignment
                # Load all meshes for the participant
                scan = mesh_utils.load_trimesh_from_file(f"{data_path}/{scan_name}")
    
                # Transform the scanned meshes to the correct scale - taken from the skin layer
                scan = mesh_utils.register_scanned_meshes(skin_layer, scan)
    
                # Save the transformed meshes
                mesh_utils.save_trimesh_to_obj(scan, f"{result_path}/transformed/{scan_name[-12:]}")

                # Load landmarks for this model
                landmark_positions = {row["name"]: list(scan.vertices[row["id_avatar_scanning"]]) for _, row in landmarks.iterrows()}
                # Dump the landmarks to json
                mesh_utils.save_landmarks_to_json(landmark_positions,
                                                  f"{result_path}/transformed/{scan_name[-12:-4]}_landmarks.json")

            ### 3-step SMPL fitting
            for inside_humans_layer in [skel_layer_name, musc_layer_name, skin_layer_name]:
                # Load the inside humans layer
                inside_humans_mesh = mesh_utils.load_trimesh_from_file(f"{data_path}/{inside_humans_layer}")
                # Find landmarks on the inside humans layer
                landmark_positions = {row["name"]: list(inside_humans_mesh.vertices[row["id_inside_humans"]]) for _, row in landmarks.iterrows()}
                # Dump the landmarks to json
                mesh_utils.save_landmarks_to_json(landmark_positions,
                                                  f"{result_path}/transformed/{inside_humans_layer[:-4]}_landmarks.json")
        if True: # SMPL fitting
            sys.path.append("SMPL-Fitting")
            import fit_body_model
            from utils import load_config
            for mesh_name in [skel_layer_name, musc_layer_name, skin_layer_name, body_scan_name, face_scan_name]:
                if mesh_name.startswith("final_fit_invtrans"):
                    landmarks_path = f"{result_path}/transformed/{mesh_name[-12:-4]}_landmarks.json"
                    mesh_path = f"{result_path}/transformed/{mesh_name[-12:]}"
                else:
                    landmarks_path = f"{result_path}/transformed/{mesh_name[:-4]}_landmarks.json"
                    mesh_path = f"{data_path}/{mesh_name}"
                # SMPL Fitting
                print('1. Fit; fitting the full model')
                cfg = load_config("SMPL-Fitting/configs/config.yaml")

                ## 1. Fit - Full model
                cfg['fit_body_model_optimization']['use_landmarks'] = landmarks.name.to_list()
                cfg['fit_body_model_optimization']['ignore_segments'] = []
                cfg['fit_body_model_optimization']['volume_target'] = False
                sex = participant_info[participant_info["participant"] == pid].sex.values[0]
                scan_info = {"participant_id": pid,
                             "scan_name": mesh_name[-12:-8] if mesh_name.startswith("final_fit_invtrans") else mesh_name[:4],
                             "fit_type": "initial"}
                cfg = mesh_utils.process_cfg_body_model(cfg, scan_info, mesh_path, landmarks_path)
                cfg['body_model_gender'] = "FEMALE" if sex == "F" else "MALE" #
                fit_body_model.fit_body_model_onto_scan(cfg)
                res_path = f"results/{pid}/bodyhull/smpl_fit/{scan_info['scan_name']}/{scan_info['fit_type']}/{mesh_name[:-4]}.npz"
                smpl_result = np.load(res_path)
                mesh_utils.smpl_result_to_obj(smpl_result, res_path.replace(".npz", ".obj"), has_faces=False)

                ## 2. Fit - Define ROI's
                print('2. Fit; ignoring segments that usually have artifacts')
                cfg['ignore_segments'] = ignore_segments
                # Create a version of the mesh where the ignored segments are removed
                curr_scan = mesh_utils.load_trimesh_from_file(mesh_path)
                new_res_path = mesh_utils.create_mesh_without_ignored_segments(curr_scan, smpl_result,
                                                                ignore_segments, vertex_segmentation, res_path)
                break
        break
        
