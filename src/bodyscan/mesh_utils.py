import trimesh
import numpy as np
import src.bodyscan.mesh_utils as mesh_utils
import sys
sys.path.append("SMPL-Fitting")
sys.path.append("SKEL")

import os
import torch
from skel.skel_model import SKEL

def load_trimesh_from_file(file_path):
    with open(file_path, 'r'):
        return trimesh.load(file_path)
    
def register_scanned_meshes(target, source, head_vertices_path = "data/resources/avatar_head_faces.txt"):
    assert isinstance(target, trimesh.Trimesh)
    assert isinstance(source, trimesh.Trimesh)
    assert target.faces.shape[0] == source.faces.shape[0]
    assert target.faces.shape[0] == 41792, "The number of faces in the AvatarScanning mesh should be 41792"

    # Load the head vertices, as they are assumed to be constant
    head_faces = np.loadtxt(head_vertices_path, dtype=int)
    head_vertices_source = source.vertices[(source.faces[head_faces].flatten())]
    head_vertices_target = target.vertices[(target.faces[head_faces].flatten())]
    # Find a rigid scale, translation and rotation to align the head vertices
    matrix, transformed, cost = trimesh.registration.procrustes(head_vertices_source, head_vertices_target, 
                                                                reflection=False, translation=True, scale=True,
                                                                return_cost=True)
    print('Registered head vertices with cost:', cost)
    print('Mean mesh error:', np.mean(np.linalg.norm(head_vertices_target - transformed, axis=1)))
    source.vertices = trimesh.transform_points(source.vertices, matrix)
    return source

def save_trimesh_to_obj(mesh, file_path):
    with open(file_path, 'w') as f:
        mesh.export(file_obj=f, file_type='obj')
    # Delete all rows that start with 'mtllib' and 'usemtl' in the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines:
            if not line.startswith('mtllib') and not line.startswith('usemtl'):
                f.write(line)
                

def save_landmarks_to_json(landmarks, file_path):
    import json
    with open(file_path, 'w') as f:
        json.dump(landmarks, f)

def create_results_dir(scan_info):
    folder = scan_info['participant_id']+'/bodyhull/smpl_fit/'+scan_info['scan_name']+'/'+scan_info['fit_type']
    #folder = f"{scan_info["participant_id"]}/bodyhull/{scan_info["scan_name"]}/{scan_info["fit_type"]}"
    path = os.path.join("results", folder)
    os.makedirs(path, exist_ok=True)
    return path

def process_cfg_body_model(cfg, scan_info, mesh_path, landmarks_path):
    """
        Process the configuration file for the SMPL fitting. In fit_body_model.py, this gets parsed from the command line arguments.
    """
    from utils import (load_loss_weights_config, initialize_fit_bm_loss_weights, process_body_model_path,
                       process_default_dtype,
                       process_landmarks, process_visualize_steps, save_configs) # This should be in another file because the import break subprocessing
    cfg_optimization = cfg["fit_body_model_optimization"]
    cfg_datasets = cfg["datasets"]
    cfg_paths = cfg["paths"]
    cfg_general = cfg["general"]
    cfg_web_visualization = cfg["web_visualization"]
    cfg_loss_weights = load_loss_weights_config(
            which_strategy="fit_bm_loss_weight_strategy",
            which_option=cfg_optimization["loss_weight_option"],
            path="SMPL-Fitting/configs/loss_weight_configs.yaml")
    cfg_loss_weights = initialize_fit_bm_loss_weights(cfg_loss_weights)
    cfg = {}
    cfg.update(cfg_optimization)
    cfg.update(cfg_datasets)
    cfg.update(cfg_paths)
    cfg.update(cfg_general)
    cfg.update(cfg_web_visualization)    
    cfg["loss_weights"] = cfg_loss_weights
    cfg["continue_run"] = cfg["continue_run"] if "continue_run" in cfg.keys() else None
    cfg["save_path"] = create_results_dir(scan_info)
    cfg = process_default_dtype(cfg)
    cfg = process_visualize_steps(cfg)
    cfg = process_landmarks(cfg)
    cfg = process_body_model_path(cfg)    
    cfg['scale_scan'] = 1.0
    cfg['scale_landmarks'] = 1.0
    cfg['scan_path'] = mesh_path # Path to the scan, supply after processing the cfg
    cfg['landmark_path'] = landmarks_path
    cfg['body_models_path'] = cfg['body_models_path']
    save_configs(cfg)
    return cfg

def smpl_result_to_obj(smpl_result, file_path, has_faces = True):
    if not has_faces:
        faces = np.load("data/resources/smpl_faces.npy")
        mesh = trimesh.Trimesh(vertices=smpl_result['vertices'], faces=faces)
    else:
        mesh = trimesh.Trimesh(vertices=smpl_result['vertices'], faces=smpl_result['faces'])
    save_trimesh_to_obj(mesh, file_path)

def create_mesh_without_ignored_segments(scan, smpl_fit, ignore_segments, vertex_segmentation, res_path):
    """
        Create a version of the mesh where the ignored segments are removed
    """
    # Get the vertices that are part of the ignored segments
    ignore_vert_idx = np.hstack([vertex_segmentation[k] for k in 
                                 vertex_segmentation if k in ignore_segments])

    # Remove the vertices that are part of the ignored segments    
    # 1. Get the NN of each scan vertex
    dists = torch.cdist(torch.from_numpy(scan.vertices).to(torch.float32), 
                        torch.from_numpy(smpl_fit['vertices']).to(torch.float32),
                        p=2)
    nn = np.argmin(dists, axis=1)
    # Check if the entries in NN are in the ignore_vert_idx
    mask = np.isin(nn, ignore_vert_idx)
    vertices_to_keep = scan.vertices[~mask]
    # Only keep the faces that have all vertices in the vertices_to_keep
    vertices_to_keep_set = np.where(~mask)[0] # Convert to a set for O(1) lookups
    faces_array = np.array(scan.faces)  # Convert faces to NumPy array if it's not already
    # Create a mask for faces where all vertices are in `vertices_to_keep_set`
    mask = np.all(np.isin(faces_array, vertices_to_keep_set), axis=1)
    # Apply mask to filter faces
    faces_to_keep = faces_array[mask]   
    scan.faces = faces_to_keep
    scan.remove_unreferenced_vertices()
    res_path_ignored_segments = res_path.replace(".npz", "_ignored_segments.obj")
    print(f"Saving mesh without ignored segments to {res_path_ignored_segments}")
    save_trimesh_to_obj(scan, res_path_ignored_segments)
    return res_path_ignored_segments

def get_approximate_landmarks_scaneca(height): 
    landmarks = {}
    landmarks['Lt. Calcaneous Post.'] = [0.3258, 0.31, 0.03] # Fixed scaneca foot position
    landmarks['Rt. Calcaneous Post.'] = [0.5622, 0.31, 0.03]
    wrist_height = 0.5 * height
    landmarks['Lt. Ulnar Styloid'] = [0.1139, 0.44, wrist_height] # The wrist x-z position is fixed, height is adjustable 
    landmarks['Rt. Ulnar Styloid'] = [0.8020, 0.44, wrist_height] 
    suprasternale_height = 0.8 * height
    sellion_height = 0.93 * height
    landmarks['Suprasternale'] = [0.45, 0.51, suprasternale_height]
    landmarks['Sellion'] = [0.45, 0.58, sellion_height]   
    for k in landmarks:
        landmarks[k] = list(rotate_scaneca_coordinate_system(np.array(landmarks[k])))
    return landmarks

def rotate_scaneca_coordinate_system(vertices):
    if len(vertices.shape) == 2:
        vertices_new = vertices[:,[0,2,1]]
        vertices_new[:,0] *= -1
    else:
        vertices_new = vertices[[0,2,1]]
        vertices_new[0] *= -1
    return vertices_new

def save_np_res_as_pkl(np_res_path):
    import pickle
    np_res = np.load(np_res_path, allow_pickle=True)
    pkl_res_path = np_res_path.replace(".npz", ".pkl")
    res_as_dict = {}
    for k in np_res.keys():
        if k in ['body_model','name']:
            continue
        if k == "shape":
            res_as_dict["betas"] = np_res[k].T.squeeze()
        elif k == "pose":
            res_as_dict["pose"] = np_res[k].T.squeeze()
        else:
            res_as_dict[k] = np_res[k]
    with open(pkl_res_path, 'wb') as f:
        pickle.dump(res_as_dict, f)

def get_skel_output(npz_file, gender):
    data = np.load(npz_file, allow_pickle=True)
    pose = torch.tensor(data['pose'])
    betas = torch.tensor(data['shape'])
    trans = torch.tensor(data['trans']).unsqueeze(0)
    # Vis skel fit when these are coming in
    model = SKEL(gender)
    joint_idx_fixed_beta = [0, 5, 10, 13, 18, 23] # Pytorch 1.x monkey patch
    model.register_buffer('joint_idx_fixed_beta', torch.LongTensor(joint_idx_fixed_beta)) # Pytorch 1.x monkey patch
    output = model(poses=pose,
                   betas=betas,
                   trans=torch.zeros(1,3).to("cuda" if torch.cuda.is_available() else "cpu"))
    skin_verts = (output['skin_verts'] * data['scale']) + trans
    joints = (output['joints'] * data['scale']) + trans
    skel_verts = (output['skel_verts'] * data['scale']) + trans
    return {
        'skin_verts': skin_verts.squeeze(0).detach().cpu().numpy(),
        'joints': joints.squeeze(0).detach().cpu().numpy(),
        'skel_verts': skel_verts.squeeze(0).detach().cpu().numpy(),
        'skin_faces': model.skin_f,
        'skel_faces': model.skel_f,
        'pose': pose,
        'betas': betas,
        'trans': trans,
        'scale': data['scale'],
    }

def adjust_hit_scaling(hit_path, smpl_fit):
    # HIT does not make use of David Boja's SMPL Fitting "scale" parameter, so we need to adjust the scaling
    for output in ['AT', 'BT', 'LT']:
        mesh = load_trimesh_from_file(os.path.join(hit_path, f"{output}_mesh.obj"))
        # Tranlate mesh to the origin and then apply the scaling
        mesh.vertices -= smpl_fit['trans']
        mesh.vertices *= smpl_fit['scale']
        mesh.vertices += smpl_fit['trans']
        save_trimesh_to_obj(mesh, os.path.join(hit_path, f"{output}_mesh_adjusted.obj"))