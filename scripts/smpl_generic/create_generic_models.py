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

sys.path.append("SMPL-Fitting")
from body_models import *


for model in ['generic_male', 'generic_female']: # There's no neutral HIT
    cfg = {}
    cfg["model"] = "smpl"
    cfg["body_model_gender"] = model.split("_")[-1]
    cfg["body_models_path"] = os.path.join("SMPL-Fitting","data", "body_models","smpl")

    bm = SMPLBodyModel(cfg)

    # Save the model
    os.makedirs(os.path.join("results", model, "bodyhull", "smpl_fit","generic", "loose"), exist_ok=True)
    savepath = os.path.join("results", model, "bodyhull", "smpl_fit", "generic","loose", "generic_ignored_segments.npz")

    pose = torch.zeros(1, 72).float()# + torch.randn(1,72)
    beta = torch.zeros(1, 10).float()#+ torch.randn(1,10)
    trans = torch.zeros(3).float()
    scale = torch.ones(1).float()
    with torch.no_grad():
        body_model_verts = bm.deform_verts(pose, beta, trans, scale)
        fitted_body_model_verts = body_model_verts.detach().cpu().data.numpy()
        fitted_pose = pose.detach().cpu().numpy()
        fitted_shape = beta.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
    np.savez(savepath, 
                 body_model = bm.body_model_name,
                 vertices=fitted_body_model_verts, 
                 pose=fitted_pose, 
                 shape=fitted_shape, 
                 trans=trans, 
                 scale=scale, 
                 name="generic",
                 scan_index=0)
    smpl_result = np.load(savepath, allow_pickle=True)
    mesh_utils.smpl_result_to_obj(smpl_result, savepath.replace(".npz", ".obj"), has_faces=False)
    mesh_utils.save_np_res_as_pkl(savepath)


    # Create a SKEL model for the generic model
    ### Actually, we need to fit this model 
    cfg["model"] = "skel"
    bm = SKELBodyModel(cfg)

    # Save the model
    os.makedirs(os.path.join("results", model, "bodyhull", "smpl_fit", "generic","loose_skel"), exist_ok=True)
    savepath = os.path.join("results", model, "bodyhull", "smpl_fit","generic", "loose_skel", "generic_ignored_segments.npz")

    pose = torch.zeros(1, 46).float()
    beta = torch.zeros(1, 10).float()
    trans = torch.zeros(3).float()
    scale = torch.ones(1).float()
    with torch.no_grad():
        body_model_verts = bm.deform_verts(pose, beta, trans, scale)
        fitted_body_model_verts = body_model_verts.detach().cpu().data.numpy()
        fitted_pose = pose.detach().cpu().numpy()
        fitted_shape = beta.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
    np.savez(savepath, 
                 body_model = bm.body_model_name,
                 vertices=fitted_body_model_verts, 
                 pose=fitted_pose, 
                 shape=fitted_shape, 
                 trans=trans, 
                 scale=scale, 
                 name="generic",
                 scan_index=0)
    smpl_result = np.load(savepath, allow_pickle=True)
    mesh_utils.smpl_result_to_obj(smpl_result, savepath.replace(".npz", ".obj"), has_faces=False)
    mesh_utils.save_np_res_as_pkl(savepath)

# Scale opensim models
smpl_bodyheights = {
    'male': 1.79,
    'female': 1.66,
}
generic_height = 1.70
generic_model_path = "data/resources/unscaled_generic.osim"
# for P06 only: generic_model_path = "data/resources/unscaled_generic_P06_no_CLAV.osim"

import opensim
# scale the model

def scale_opensim_model(model_path, bodyheight, output_model_file):
    # Load the model
    os.makedirs(os.path.join(*output_model_file.split('/')[:-1]), exist_ok=True)
    # Scale all semgents manually
    scale_factor = bodyheight / generic_height
    model = opensim.Model(model_path)
    jointSet = model.getJointSet()
    for i in range(jointSet.getSize()):
        joint = jointSet.get(i)
        # Scale location in parent frame
        translation = joint.get_frames(0).get_translation()
        new_t = opensim.Vec3(
            translation.get(0) * scale_factor,
            translation.get(1) * scale_factor,
            translation.get(2) * scale_factor
        )
        joint.get_frames(0).set_translation(new_t)
        
    
    model.printToXML(output_model_file)


input('run HIT inferences before this part, oterwise cancel; I ran marching cubes cause the male mesh wouldnt be broken')

scale_opensim_model(generic_model_path, smpl_bodyheights["male"], "results/generic_male/opensim/smpl_male_generic.osim")
scale_opensim_model(generic_model_path, smpl_bodyheights["female"], "results/generic_female/opensim/smpl_female_generic.osim")


from src.opensim import opensim_utils
import subprocess
import pandas as pd
name = "generic_female"
model_c = f"--model generic"
scan_c = f"--pipeline HIT"
input_path = f"--input_path results/generic_female/bodyhull"
output_path = f"--output_path results/generic_female/{name}"
mass_c = f"--mass 72"
gender_c = f"--gender=female"
hull_c = f"--hull=loose"
extra_commans = f"--debug=True --use_skel=True"
skel_c = "_w_skel"
# 
command = f"python src/bodyscan/estimate_bsip.py {model_c} {scan_c} {input_path} {output_path} {mass_c} {gender_c} {hull_c} {extra_commans}"
subprocess.run(command, shell=True)

opensim_model = f"data/resources/unscaled_generic_P06_no_CLAV.osim"
opensim_model = opensim_utils.get_model(opensim_model)
hull_bsip = pd.read_pickle(f"results/generic_female/generic_female/HIT_generic_loose_w_skel.pkl")
new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight = 0, arms_from_hull=False) # Arms from hull would be ideal
opensim_utils.save_opensim_model(osim_model, f"results/generic_female/smpl_female_P06_no_CLAV.osim")

opensim_model = f"data/resources/unscaled_generic.osim"
opensim_model = opensim_utils.get_model(opensim_model)
hull_bsip = pd.read_pickle(f"results/generic_female/generic_female/HIT_generic_loose_w_skel.pkl")
new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight=0, arms_from_hull=False) # Arms from hull would be ideal, please check
opensim_utils.save_opensim_model(osim_model, f"results/generic_female/smpl_female.osim")



name = "generic_male"
model_c = f"--model generic"
scan_c = f"--pipeline HIT"
input_path = f"--input_path results/generic_male/bodyhull"
output_path = f"--output_path results/generic_male/{name}"
mass_c = f"--mass 80"
gender_c = f"--gender=male"
hull_c = f"--hull=loose"
extra_commans = f"--debug=True --use_skel=True"
skel_c = "_w_skel"
# 
command = f"python src/bodyscan/estimate_bsip.py {model_c} {scan_c} {input_path} {output_path} {mass_c} {gender_c} {hull_c} {extra_commans}"
subprocess.run(command, shell=True)

opensim_model = f"data/resources/unscaled_generic_P06_no_CLAV.osim"
opensim_model = opensim_utils.get_model(opensim_model)
hull_bsip = pd.read_pickle(f"results/generic_male/generic_male/HIT_generic_loose_w_skel.pkl")
new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight = 0, arms_from_hull=False) # Arms from hull would be ideal
opensim_utils.save_opensim_model(osim_model, f"results/generic_male/smpl_male_P06_no_CLAV.osim")

opensim_model = f"data/resources/unscaled_generic.osim"
opensim_model = opensim_utils.get_model(opensim_model)
hull_bsip = pd.read_pickle(f"results/generic_female/generic_male/HIT_generic_loose_w_skel.pkl")
new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight=0, arms_from_hull=False) # Arms from hull would be ideal, please check
opensim_utils.save_opensim_model(osim_model, f"results/generic_male/smpl_male.osim")