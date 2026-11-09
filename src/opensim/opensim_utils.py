"""
This file contains utility functions for OpenSim models. As OpenSim is incompatible with numpy, we need to convert between the two.
"""
import opensim as osim
import numpy as np
import pandas as pd
import os

def get_model(model_file):
    mod = osim.Model(model_file)
    state = mod.initSystem()
    return mod

def get_joint_translation(model, body_name):
    parent_joint_name = get_parent_joint(model, body_name)
    for joint in range(model.getJointSet().getSize()):
        jour = model.getJointSet().get(joint)
        if jour.getName() == parent_joint_name:    
            translation = jour.get_frames(0).get_translation()
    return np.array([translation.get(0), translation.get(1), translation.get(2)])

def to_numpy3(vec3): # For com vectors
    return np.array([vec3.get(0), vec3.get(1), vec3.get(2)])

def to_numpy3_3(mat3): # for inertia matrices
    return np.array([[mat3.get(0), mat3.get(3), mat3.get(4)], 
                     [mat3.get(3), mat3.get(1), mat3.get(5)], 
                     [mat3.get(4), mat3.get(5), mat3.get(2)]])

def get_bsip_dataframe(model):
    body_set = model.getBodySet()
    nSeg = body_set.getSize()
    df = pd.DataFrame(columns=['name', 'mass', 'mass_center', 'inertia'])
    for i in range(nSeg):
        body = body_set.get(i)
        df.loc[i] = [body.getName(), body.getMass(), to_numpy3(body.getMassCenter()), to_numpy3_3(body.get_inertia())]
    return df

def get_joint_dataframe(model):
    joint_set = model.getJointSet()
    nJoints = joint_set.getSize()
    df = pd.DataFrame(columns=['name', 'parent', 'child', 'location', 'orientation'])
    for i in range(nJoints):
        joint = joint_set.get(i)
        df.loc[i] = [joint.getName(), joint.getParentFrame().getName(), joint.getChildFrame().getName(), to_numpy3(joint.get_location_in_parent()), to_numpy3(joint.get_orientation_in_parent())]
    return df

def get_parent_joint(model, body_name):
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        if joint.getChildFrame().getName() == body_name+'_offset':
            return joint.getName()
    return None

def composite_bsip(masses, coms, inertias):
    """
        Combine rigid bodies into one. All inputs are expressed in the SAME frame (orientation and origin).
        masses: (n,), coms: (n, 3) body centers of mass, inertias: (n, 3, 3) body inertias about their own center of mass.
        Returns total mass, composite center of mass and the composite inertia about the composite center of mass,
        using the parallel axis theorem I = sum_i [ I_i + m_i * ((d_i . d_i) E - d_i d_i^T) ], with d_i = com_i - com_composite.
    """
    masses = np.asarray(masses, dtype=float)
    coms = np.asarray(coms, dtype=float).reshape(-1, 3)
    inertias = np.asarray(inertias, dtype=float).reshape(-1, 3, 3)
    mass = masses.sum()
    com = (masses[:, None] * coms).sum(axis=0) / mass
    inertia = np.zeros((3, 3))
    for m_i, c_i, I_i in zip(masses, coms, inertias):
        d = c_i - com
        inertia += I_i + m_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def _body_bsip_in_frame(model, state, body_name, reference_name):
    """ Mass, center of mass and inertia of a body, expressed in the frame of the reference body (default pose of the model). """
    body = model.getBodySet().get(body_name)
    T = body.findTransformBetween(state, model.getBodySet().get(reference_name))  # body frame -> reference frame
    R = T.R().asMat33()
    R = np.array([[R.get(i, j) for j in range(3)] for i in range(3)])
    p = to_numpy3(T.p())
    return body.getMass(), R @ to_numpy3(body.getMassCenter()) + p, R @ to_numpy3_3(body.get_inertia()) @ R.T


def get_bsip_dataframe_with_feet_and_HAT(model, foot = ['calcn','talus','toes'], HAT = ['pelvis','torso']):
    """
        BSIP dataframe of all bodies plus the combined bodies foot_l, foot_r (expressed in the frame of the first foot body, calcn)
        and HAT (expressed in the frame of the first HAT body, pelvis). The relative position and orientation of the bodies
        is taken from the model in its default pose; the combined inertia is about the combined center of mass.
    """
    bodies_df = get_bsip_dataframe(model)
    state = model.initSystem()

    def combine(names, reference):
        parts = [_body_bsip_in_frame(model, state, n, reference) for n in names]
        return composite_bsip([p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts])

    for side in ['_l', '_r']:
        names = [f + side for f in foot]
        bodies_df.loc[bodies_df.shape[0]] = ['foot' + side, *combine(names, names[0])]
    bodies_df.loc[bodies_df.shape[0]] = ['HAT', *combine(HAT, HAT[0])]
    return bodies_df

def set_opensim_bsip_parameters(model, bodies_df):
    ''' 
        Set the mass, mass center and inertia parameters of the bodies in the OpenSim model from a pandas dataframe.
    '''
    # Filter out the HAT and foot segments if they appear in the dataframe
    bodies_df = bodies_df[~bodies_df['name'].isin(['HAT', 'foot_l', 'foot_r'])]
    for idx, body in bodies_df.iterrows():
        seg = model.getBodySet().get(body['name'])
        seg.setMass(float(body['mass']))
        seg.setMassCenter(osim.Vec3(body['mass_center'][0], body['mass_center'][1], body['mass_center'][2]))
        seg.setInertia(osim.Inertia(body['inertia'][0,0], body['inertia'][1,1], body['inertia'][2,2], body['inertia'][0,1], body['inertia'][0,2], body['inertia'][1,2]))
    return model

def save_opensim_model(model, file_name):
    model.printToXML(file_name)


def concat_addbiomechanics_data(ik_data_path):
    # Find all mot files in the folder
    mot_files = sorted([f for f in os.listdir(ik_data_path) if f.endswith('.mot')])
    # Check unique trials
    trials = set([f.split('_')[0] for f in mot_files])
    for t in trials:
        # Skip trials without _, as they are not produced by addbiomechanics
        if t.find('.mot') > 0:
            continue
        # Find all files for the trial
        trial_files = [f for f in mot_files if f.startswith(t) and '_segment' in f]
        # Sort the trial files by the segment number as int
        trial_files = sorted(trial_files, key=lambda x: int(x.split('_')[-2].split('_')[0]))

        # Load all trials
        df = pd.concat([pd.read_csv(os.path.join(ik_data_path, f), sep='\t', skiprows=10) for f in trial_files])
        nRows = df.shape[0]
        # Create a new file, with the same name as the set
        with open(os.path.join(ik_data_path, f"{t}.mot"), 'w', newline='\n') as f:
            with open(os.path.join(ik_data_path, trial_files[0]), 'r') as f0:
                # Write the first two lines from the original file
                for i in range(2):
                    f.write(f0.readline())
                f0.readline() # Skip the third line
                f.write(f"nRows={nRows}\n")
                for i in range(7):
                    f.write(f0.readline())
            df.to_csv(f, sep='\t', index=False)



def filter_data(data_path, lowpass_f, lowpass_order, mode='ik', return_df=False, clean=False):
    from scipy.signal import butter, sosfiltfilt
    if mode == 'ik':
        skiprows = 10
    elif mode == 'grf':
        skiprows = 5
    else:
        raise ValueError('Filter mode should be either marker or grf')
    

    data = pd.read_csv(data_path, sep='\t', skiprows=skiprows)
    if clean:
        # Clamp the center of pressure values to min/max values of 1.5m (bigger than the treadmill dimensions)
        # and the torques to -100Nm to 100N*m
        # This is done to avoid unrealistic values in the data, which would then get filtered and cause more issues
        for col in data.columns:
            if col[-2] == 'p':
                data[col] = data[col].clip(-1.5, 1.5)
            if col.endswith('_torque_y'):
                # Clamp the torque values to -100Nm to 100N*m
                data[col] = data[col].clip(-100, 100)

    # Lowpass filter the data
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        y = sosfiltfilt(sos, data)
        return y

    for col in data.columns:
        if col in ['time', 'Time', 'Frame#']:
            continue # Do not filter time, looking at you opensim
        data[col] = butter_lowpass_filter(data[col], lowpass_f, 100, lowpass_order)
        data[col] = data[col].apply(lambda x: round(x, 8))

    # Testwise: Set the mean of the force plate data to 0 - it is constantly biased somehow...
    if mode == 'grf':
        for col in data.columns:
            if col in ['ground_force_vx', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vz']:
                print('Mean of', col, 'is', data[col].mean())

    with open(data_path.replace('.mot', '_filtered.mot'), 'w', newline='\n') as f:
        with open(data_path, 'r') as f0:
            for i in range(skiprows+1):
                f.write(f0.readline())
        data_arr = data.to_numpy()
        # Somehow, thats the weird thing: the data is not written correctly if we use pd.to_csv
        # But literally 10 lines above this it works fine. I have no idea why
        np.savetxt(f, data_arr, delimiter='\t', newline='\n', fmt='%.8f')
        f.flush()

    if return_df:
        return data_path.replace('.mot', '_filtered.mot'), data
    return data_path.replace('.mot', '_filtered.mot')

# --- SKEL/SMPL body frame -> OpenSim body frame -------------------------------------------------------------
# estimate_bsip returns CoM and inertia in the SKEL/SMPL body frames (X to the left, Y up, Z forward).
# OpenSim body frames are X forward, Y up, Z to the right. The change of basis (x, y, z) -> (z, y, -x) maps one to the other
# for pelvis, torso, legs and feet. SKEL's arm frames are in the T-pose (arm along +-X), OpenSim's arms hang down (arm along -Y),
# so the arms need an additional rotation about the (forward) X axis: right arm (x, y, z) -> (z, x, y), left arm (z, -x, -y).
# All matrices are proper rotations (det = +1), which keeps the inertia tensor valid: I' = R I R^T.
_R_SKEL_TO_OPENSIM = np.array([[0., 0., 1.],
                               [0., 1., 0.],
                               [-1., 0., 0.]])
_R_SKEL_TO_OPENSIM_ARM = {
    '_r': np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
    '_l': np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]),
}
_ARM_BODIES = ('humerus', 'ulna', 'radius', 'hand')


def skel_to_opensim_rotation(body_name):
    """ Rotation matrix R with v_opensim = R @ v_skel for the given body name. """
    for arm in _ARM_BODIES:
        for side, R in _R_SKEL_TO_OPENSIM_ARM.items():
            if body_name == arm + side:
                return R
    return _R_SKEL_TO_OPENSIM


def skel_to_opensim_bsip(bsip_df):
    """
        Convert mass centers and inertias of a BSIP dataframe (column 'name' or 'segment') from the SKEL/SMPL body
        frames to the OpenSim body frames. Returns a copy.
    """
    df = bsip_df.copy()
    name_col = 'name' if 'name' in df.columns else 'segment'
    com, inertia = [], []
    for _, row in df.iterrows():
        R = skel_to_opensim_rotation(row[name_col])
        com.append(R @ np.asarray(row['mass_center'], dtype=float).reshape(3))
        inertia.append(R @ np.asarray(row['inertia'], dtype=float).reshape(3, 3) @ R.T)
    df['mass_center'] = com
    df['inertia'] = inertia
    return df


def hull_bsip_to_opensim(hull_bsip, model,
                         arms_from_hull=False,
                         extra_shoe_weight=0,
                         keep_bodies=False):

    hull_bsip = hull_bsip.copy() # Do not modify the caller's dataframe

    # Check if lumbar_body is in the hull_bsip.names and replace it with "torso"
    if 'lumbar_body' in hull_bsip['segment'].values:
        hull_bsip.loc[hull_bsip['segment'] == 'lumbar_body', 'segment'] = 'torso'

    hull_bsip.rename(columns={'segment': 'name'}, inplace=True)

    # The hull BSIPs are in the SKEL/SMPL body frames, OpenSim uses different body frames
    hull_bsip = skel_to_opensim_bsip(hull_bsip)

    # Define arms segments
    if not arms_from_hull:
        arm_segments = ['humerus_r', 'humerus_l', 'radius_r', 'radius_l', 'ulna_r', 'ulna_l', 'hand_r', 'hand_l']
    else: 
        arm_segments = []
    if type(keep_bodies) == list: # We simply keep the bodies that are desired (usually the toes in MRI, they are out of volume)
        arm_segments += keep_bodies

    model_bsip = get_bsip_dataframe(model)
    bodymass = hull_bsip['mass'].sum()

    if not arms_from_hull or keep_bodies:
        # Use the original BSIP of the model for the arms
        # Remove the arms from the hull_bsip
        hull_bsip = hull_bsip[~hull_bsip['name'].isin(arm_segments)]
        # Add the arms from the model
        hull_bsip = pd.concat([hull_bsip, model_bsip[model_bsip['name'].isin(arm_segments)]])

        # check the total mass and adjust mass and inertia parameters
        bodymass_new = hull_bsip['mass'].sum()
        hull_bsip['mass'] = hull_bsip['mass'] / bodymass_new * bodymass
        hull_bsip['inertia'] = hull_bsip['inertia'].apply(lambda x: x / bodymass_new * bodymass)
    # else: arms come from the hull and were already rotated into the OpenSim frame by skel_to_opensim_bsip

    if extra_shoe_weight > 0:
        # Add the weight of the shoes to the feet
        feet = ['calcn_r', 'calcn_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l']

        starting_foot_mass = hull_bsip[hull_bsip['name'].isin(feet)]['mass'].sum()
        scale_factor = (extra_shoe_weight + starting_foot_mass) / starting_foot_mass

        hull_bsip.loc[hull_bsip['name'].isin(feet), 'mass'] *= scale_factor
        hull_bsip.loc[hull_bsip['name'].isin(feet), 'inertia'] = hull_bsip.loc[hull_bsip['name'].isin(feet), 'inertia'].apply(lambda x: x * scale_factor)

    # Set all undefined bisp values to (almost) 0 - patella (not segmented), radius (same body as ulna), talus (same body as calcn)
    missing_bsip = model_bsip[~model_bsip['name'].isin(hull_bsip['name'])]
    missing_bsip.loc[:,'mass'] = 1e-5
    missing_bsip.loc[:,'mass_center'] = missing_bsip['mass_center'].apply(lambda x: np.array([1e-5, 1e-5, 1e-5]))
    missing_bsip.loc[:,'inertia'] = missing_bsip['inertia'].apply(lambda x: np.eye(3) * 1e-5)
    hull_bsip = pd.concat([hull_bsip, missing_bsip])

    return hull_bsip, set_opensim_bsip_parameters(model, hull_bsip)





