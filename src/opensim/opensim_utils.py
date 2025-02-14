"""
This file contains utility functions for OpenSim models. As OpenSim is incompatible with numpy, we need to convert between the two.
"""
import opensim as osim
import numpy as np
import pandas as pd

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

def get_bsip_dataframe_with_feet_and_HAT(model, foot = ['calcn','talus','toes'], HAT = ['pelvis','torso']):
    bodies_df = get_bsip_dataframe(model)
    #joint_df = get_joint_dataframe(model)
    for side in ['_l', '_r']:
        # Add translations 
        foot_mass = 0
        for f in foot:
            foot_mass += bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0]
        foot_mass_center = np.zeros(3)
        trans = np.zeros(3)
        for f in foot:
            if foot != 'calcn': # calcn is the parent segment for the foot
                trans += get_joint_translation(model, f+side)
            foot_mass_center += bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0]*(bodies_df.loc[bodies_df['name'] == f+side, 'mass_center'].values[0]+trans)
        foot_mass_center /= foot_mass
        foot_inertia = np.zeros((3,3))
        trans = np.zeros(3)
        for f in foot:
            if foot != 'calcn':
                trans += get_joint_translation(model, f+side)
            foot_inertia += bodies_df.loc[bodies_df['name'] == f+side, 'inertia'].values[0] + \
                    bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0] * \
                    np.outer(foot_mass_center+trans, foot_mass_center+trans)
        bodies_df.loc[bodies_df.shape[0]] = ['foot'+side, foot_mass, foot_mass_center, foot_inertia]
    HAT_mass = 0
    for h in HAT:
        HAT_mass += bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0]
    HAT_mass_center = np.zeros(3)
    trans = np.zeros(3)
    for h in HAT: # pelvis is the parent segment for the torso
        if h != 'pelvis':
            trans += get_joint_translation(model, h)
        HAT_mass_center += bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0]*(bodies_df.loc[bodies_df['name'] == h, 'mass_center'].values[0]+trans)
    HAT_mass_center /= HAT_mass
    HAT_inertia = np.zeros((3,3))
    trans = np.zeros(3)
    for h in HAT:
        if h != 'pelvis':
            trans += get_joint_translation(model, h)
        HAT_inertia += bodies_df.loc[bodies_df['name'] == h, 'inertia'].values[0] + \
                bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0] * \
                np.outer(HAT_mass_center+trans, HAT_mass_center+trans)
    bodies_df.loc[bodies_df.shape[0]] = ['HAT', HAT_mass, HAT_mass_center, HAT_inertia]
    return bodies_df

def set_opensim_bsip_parameters(model, bodies_df):
    ''' 
        Set the mass, mass center and inertia parameters of the bodies in the OpenSim model from a pandas dataframe.
    '''
    # Filter out the HAT and foot segments if they appear in the dataframe
    bodies_df = bodies_df[~bodies_df['name'].isin(['HAT', 'foot_l', 'foot_r'])]
    for idx, body in bodies_df.iterrows():
        seg = model.getBodySet().get(body['name'])
        seg.setMass(body['mass'])
        seg.setMassCenter(osim.Vec3(body['mass_center'][0], body['mass_center'][1], body['mass_center'][2]))
        seg.setInertia(osim.Inertia(body['inertia'][0,0], body['inertia'][1,1], body['inertia'][2,2], body['inertia'][0,1], body['inertia'][0,2], body['inertia'][1,2]))
    return model

def save_opensim_model(model, file_name):
    model.printToXML(file_name)