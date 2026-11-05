
# Scale the generic opensim model to the SMPL model
import opensim
import pandas as pd
from src.opensim import opensim_utils

osim_model = opensim.Model("sipp_runmad/gait3d_pelvis213.osim")
generic_heights = pd.read_csv("data/generic_heights.csv")
print(generic_heights)
h_rajogop = generic_heights[generic_heights['model'] == 'rajogopal'].height.values[0]
h_smpl = generic_heights[generic_heights['model'] == "smpl_male"].height.values[0]
generic_osim_model = f"sipp_runmad/gait3d_pelvis213.osim"
#load the generic model
generic_model = opensim_utils.get_model(generic_osim_model)
bodys_gen = generic_model.getBodySet()
joints_gen = generic_model.getJointSet()

### --- Insert BSIP into OpenSim model ---
hull_bsip = pd.read_pickle(f"results/generic_male/bsip/HIT_generic_loose_w_skel.pkl")
mass_hit = hull_bsip['mass'].sum()
import numpy as np
mass_generic = np.sum([bodys_gen.get(i).getMass() for i in range(bodys_gen.getSize())])
# scale mass to match the mass of the generic model
scale_factor = mass_generic / mass_hit

for body in ["toes","calcn","tibia","femur","humerus","hand","ulna"]:
    l, r = body+"_l", body+"_r"
    il = list(hull_bsip.segment).index(l)
    ir = list(hull_bsip.segment).index(r)
    # average to retain symmetry

    avg_mass = (hull_bsip[hull_bsip.segment == l].mass.values[0] + hull_bsip[hull_bsip.segment == r].mass.values[0]) / 2
    hull_bsip.loc[il, "mass"] = avg_mass
    hull_bsip.loc[ir, "mass"] = avg_mass
    com_r = np.array(list(hull_bsip[hull_bsip.segment == r].mass_center))
    com_l = np.array(list(hull_bsip[hull_bsip.segment == l].mass_center))
    if body in ["toes","calcn","tibia","femur"]:
        com_mirror_axis = 2
    else:
        com_r = com_r[:, [1,0,2]] 
        com_l = com_l[:, [1,0,2]]
        com_l[:,1] = -com_l[:,1]
        com_mirror_axis = 2
    com_r[0,com_mirror_axis] = -com_r[0,com_mirror_axis]
    avg_com = (com_l + com_r) / 2
    hull_bsip.at[il, "mass_center"] = avg_com[0].copy()
    avg_com[0,com_mirror_axis] = -avg_com[0,com_mirror_axis]
    hull_bsip.at[ir, "mass_center"] = avg_com[0].copy()

    inertia_r = np.array(list(hull_bsip[hull_bsip.segment == r].inertia))
    inertia_l = np.array(list(hull_bsip[hull_bsip.segment == l].inertia))
    avg_inertia = (inertia_l + inertia_r) / 2
    if body not in ["toes","calcn","tibia","femur"]:
        avg_inertia[0,[0,1,2],[0,1,2]] = avg_inertia[0,[1,0,2],[1,0,2]]
    hull_bsip.at[il, "inertia"] = avg_inertia[0] * np.eye(3) # zero out off-diagonal terms
    hull_bsip.at[ir, "inertia"] = avg_inertia[0] * np.eye(3)



for body in ["pelvis", "lumbar_body"]:
    # set y-components to zero
    com_mask = np.array([1,1,0])
    inertia_mask = np.eye(3)

    hull_bsip.loc[hull_bsip.segment == body, "mass_center"] = hull_bsip.loc[hull_bsip.segment == body, "mass_center"].apply(lambda x: np.array(x) * com_mask)
    hull_bsip.loc[hull_bsip.segment == body, "inertia"] = hull_bsip.loc[hull_bsip.segment == body, "inertia"].apply(lambda x: np.array(x) * inertia_mask)

hull_bsip['mass'] *= scale_factor
hull_bsip['inertia'] *= scale_factor
# hull_bsip_to_opensim renames the 'segment' column to 'name' in place, so snapshot
# the pre-rename frame here for the second call below.
hull_bsip_pre_rename = hull_bsip.copy()
new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, generic_model, arms_from_hull=True)
opensim_utils.save_opensim_model(generic_model, f"sipp_runmad/sipp_generic_runmad.osim")

# Create sipp_runmad sneaker model
foot_mass = hull_bsip[(hull_bsip.name.isin(["calcn_r","talus_r","toes_r"]))].mass.sum()
foot_mass_shod = foot_mass + .45
scale_factor_shod = foot_mass_shod / foot_mass
hull_bsip_shod = hull_bsip_pre_rename.copy()
hull_bsip_shod.loc[hull_bsip_shod.segment.isin(["calcn_r","talus_r","toes_r"]), "mass"] *= scale_factor_shod
hull_bsip_shod.loc[hull_bsip_shod.segment.isin(["calcn_l","talus_l","toes_l"]), "mass"] *= scale_factor_shod
hull_bsip_shod.loc[hull_bsip_shod.segment.isin(["calcn_r","talus_r","toes_r"]), "inertia"] *= scale_factor_shod
hull_bsip_shod.loc[hull_bsip_shod.segment.isin(["calcn_l","talus_l","toes_l"]), "inertia"] *= scale_factor_shod
print(hull_bsip_shod, hull_bsip)
generic_model = opensim_utils.get_model(generic_osim_model)
new_hull_bsip_shod, osim_model_shod = opensim_utils.hull_bsip_to_opensim(hull_bsip_shod, generic_model, arms_from_hull=True)
opensim_utils.save_opensim_model(osim_model_shod, f"sipp_runmad/sipp_sneaker_runmad.osim")