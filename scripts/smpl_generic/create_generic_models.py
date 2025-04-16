# Create an instance of each a female, male and neutral SMPL model

import os
import sys
curr_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_file_path)))
sys.path.append(parent_dir)
os.chdir(parent_dir)
import numpy as np
import torch
from src.bodyscan import mesh_utils
from src.opensim import opensim_utils
from SKEL.skel.skel_model import SKEL
import trimesh
import subprocess

sys.path.append("SMPL-Fitting")
from body_models import *


for model_name in ['generic_male', 'generic_female']: # There's no neutral HIT
    """
        Create a generic model for the SMPL model, for a given sex
        1. Create a SKEL instance, where the pose is modified to have shoulderwide stance
        2. Fit SMPL to the SKEL instance
        3. Run HIT inference on the SMPL instance
        4. Scale OpenSim models to the landmarks of the SKEL/SMPL instance
    """

    ### --- SKEL model ---
    gender = model_name.split("_")[-1]

    # Create a SKEL model for the generic model, 
    joint_idx_fixed_beta = [0, 5, 10, 13, 18, 23] # Pytorch 1.x monkey patch
    model = SKEL(gender)
    model.register_buffer('joint_idx_fixed_beta', torch.LongTensor(joint_idx_fixed_beta)) # Pytorch 1.x monkey patch
    betas = torch.zeros(1, 10)
    pose = torch.zeros(1, 46)
    pose[0,4] = -np.pi/24 # 7.5 degrees abduction, --> eliminate self-intersections between the thighs
    pose[0,11] = -np.pi/24 # Same for the other leg
    output = model(poses=pose,
                   betas=betas, 
                   trans=torch.zeros(1,3).to("cuda" if torch.cuda.is_available() else "cpu"))
    
    os.makedirs(os.path.join("results", model_name, "bodyhull", "smpl_fit", "","loose_skel"), exist_ok=True)
    savepath = os.path.join("results", model_name, "bodyhull", "smpl_fit", "", "loose_skel", "generic.npz")

    np.savez(savepath,
             body_model = 'SKEL_generic',
             vertices=output['skin_verts'].detach().cpu().data.numpy(),
             pose=pose.detach().cpu().numpy(),
             shape=betas.detach().cpu().numpy(), 
             scale = torch.ones(1).float().detach().cpu().numpy(),
             trans = torch.zeros(3).float().detach().cpu().numpy(),
    )

    # Save mesh as .obj
    savepath_obj = os.path.join("results", model_name, "bodyhull", "smpl_fit", "", "loose_skel", "generic.obj")
    mesh = trimesh.Trimesh(vertices=output['skin_verts'].detach().cpu().data.numpy().squeeze(), faces=model.skin_f)
    mesh.export(savepath_obj)

    ### --- SMPL model ---
    from utils import load_config, initialize_fit_bm_loss_weights, load_loss_weights_config
    import fit_body_model
    cfg = load_config("SMPL-Fitting/configs/config.yaml")
    cfg['fit_body_model_optimization']['use_landmarks'] = []
    cfg['fit_body_model_optimization']['ignore_segments'] = []
    cfg['fit_body_model_optimization']['volume_target'] = False
    sex = gender.upper()
    landmarks_path = "results/P01/bodyhull/transformed/bodyscan_landmarks.json" # This is useless, but we need to provide it
    scan_info = {"participant_id": model_name,
                "scan_name": "",
                "fit_type": "loose"}

    cfg = mesh_utils.process_cfg_body_model(cfg, scan_info, savepath_obj, landmarks_path)
    cfg['body_model_gender'] = gender.upper()

    # Load reference smpl model - the skel model in this case
    cfg['reference_smpl'] = os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose_skel", "generic.obj")
    cfg['loss_weights'] = initialize_fit_bm_loss_weights(
                                load_loss_weights_config(which_strategy="fit_bm_loss_weight_strategy",
                                which_option="scaneca",
                                path="SMPL-Fitting/configs/loss_weight_configs.yaml"))

    ## Fix the betas and scale while fitting
    # cfg['refine_params'] = ['pose','trans','scale']
    cfg['save_path'] = os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose", "")
    os.makedirs(os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose"), exist_ok=True)
    cfg['body_model'] = 'smpl'
    #fit_body_model.fit_body_model_onto_scan(cfg)

    # save as .obj
    savepath_obj = os.path.join("results", model_name, "bodyhull", "smpl_fit","loose", "generic.obj")
    smpl_result = np.load(cfg['save_path'] + "generic.npz", allow_pickle=True)
    # Copy the generic.npz file as a generic.pkl file
    mesh_utils.save_np_res_as_pkl(f"{cfg['save_path']}/generic.npz")
    mesh_utils.smpl_result_to_obj(smpl_result, savepath_obj, has_faces=False)

    ### --- HIT inference ---
    # We can't run HIT in this python version, so we run it in the bash script
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hit_out_folder = os.path.join("results", model_name,"bodyhull" ,"hit_prediction", "generic")
    savepath_pkl = os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose", "generic.pkl")
    # Run /scripts/smpl_generic/hit_inference.sh and supply the command as an argument
    #subprocess.run(f"scripts/smpl_generic/hit_inference.sh {gender} {savepath_pkl} {device} {hit_out_folder}", shell=True)

    ### --- BSIP estimation ---
    # We need to copy the generic.npz file to generic_ignored_segments.npz -- SIPP convention
    path_npz = os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose_skel")
    path_new = os.path.join("results", model_name, "bodyhull", "smpl_fit","generic", "loose_skel")
    subprocess.run(f"cp {path_npz}/generic.npz {path_new}/generic_ignored_segments.npz", shell=True)
    subprocess.run(f"cp {path_npz}/generic.obj {path_new}/generic_ignored_segments.obj", shell=True)
    path_npz = os.path.join("results", model_name, "bodyhull", "smpl_fit","", "loose")
    path_new = os.path.join("results", model_name, "bodyhull", "smpl_fit","generic", "loose")
    subprocess.run(f"cp {path_npz}/generic.npz {path_new}/generic_ignored_segments.npz", shell=True)
    subprocess.run(f"cp {path_npz}/generic.obj {path_new}/generic_ignored_segments.obj", shell=True)


    model_c = f"--model generic"
    scan_c = f"--pipeline HIT"
    input_path = f"--input_path results/{model_name}/bodyhull/"
    os.makedirs(f"results/{model_name}/bsip/", exist_ok=True)
    output_path = f"--output_path results/{model_name}/bsip/"
    mass_c = f"--mass {71.1 if gender=='female' else 85.3}"
    gender_c = f"--gender={gender}"
    hull_c = f"--hull=loose"
    extra_commans = f"--debug=True --use_skel=True"
    skel_c = "_w_skel"
    print(os.getcwd())
    command = f"python src/bodyscan/estimate_bsip.py {model_c} {scan_c} {input_path} {output_path} {mass_c} {gender_c} {hull_c} {extra_commans}"
    subprocess.run(command, shell=True)


    ### --- OpenSim model scaling ---
    from scripts.smpl_generic import vicon_smpl_mapping
    import landmarks
    # Create a landmark dict for SMPL / SKEL
    marker_dict = {}
    for marker in vicon_smpl_mapping.marker_to_smpl:
        if vicon_smpl_mapping.marker_to_smpl[marker] in landmarks.SMPL_INDEX_LANDMARKS:
            marker_dict[marker] = landmarks.SMPL_INDEX_LANDMARKS[vicon_smpl_mapping.marker_to_smpl[marker]]
        else: 
            print(f"Marker {vicon_smpl_mapping.marker_to_smpl[marker]} not in SMPL landmarks")

    print("Writing T-Pose landmarks to file")
    os.makedirs(os.path.join("results", model_name, "marker"), exist_ok=True)
    with open(f"results/{model_name}/marker/static.trc", "w") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t1\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigDataNumFrames\n")
        f.write(f"100\t100\t10\t{len(marker_dict)}\tmm\t100\t0\t10\n") # 10 frames, 100Hz - write the static pose
        f.write("Frame#\tTime\t")
        for marker in marker_dict:
            f.write(f"{marker}\t\t\t")
        f.write("\n")
        f.write("\t\t")
        for idx, marker in enumerate(marker_dict):
            f.write(f"X{idx+1}\tY{idx+1}\tZ{idx+1}\t")
        f.write("\n")
        for i in range(10):
            f.write(f"{i+1}\t{i/100}\t")
            for marker in marker_dict:
                x = smpl_result['vertices'][marker_dict[marker]]*1000
                f.write(f"{x[0]}\t{x[1]}\t{x[2]}\t")
            f.write("\n")

    continue
    # Scale the generic opensim model to the SMPL model
    import opensim
    osim_model = opensim.Model("data/resources/unscaled_generic.osim")
    osim_model.initSystem()
    scaleTool = opensim.ScaleTool("scripts/smpl_generic/setup_scale.xml")
    time_array = opensim.ArrayDouble()
    time_array.set(0, 0.0)
    time_array.set(1, 0.1)
    scaleTool.setSubjectMass(80.0)
    scaleTool.getModelScaler().setApply(True)
    scaleTool.getModelScaler().setMarkerFileName(f"results/{model_name}/marker/static.trc")
    scaleTool.getModelScaler().setTimeRange(time_array)
    scaleTool.getModelScaler().setOutputModelFileName(f"results/{model_name}/opensim/smpl_{model_name}_scaled_no_bsip.osim")
    scaleTool.getModelScaler().setOutputScaleFileName(f"results/{model_name}/opensim/smpl_{model_name}.xml")
    scaleTool.getModelScaler().processModel(osim_model, '', 80.0)


    ### --- Insert BSIP into OpenSim model ---
    opensim_model = f"results/{model_name}/opensim/smpl_{model_name}_scaled_no_bsip.osim"
    opensim_model = opensim_utils.get_model(opensim_model)
    hull_bsip = pd.read_pickle(f"results/{model_name}/bsip/generic_ignored_segments.pkl")
    new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim_bsip(hull_bsip, opensim_model, arms_from_hull=True)
    opensim_utils.save_model(opensim_model, f"results/{model_name}/opensim/smpl_{model_name}.osim")