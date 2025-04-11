""" 
    Runs the MRIGait benchmark for two given opensim models on a given set of trajectories.
"""

import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import opensim
import pandas as pd
import numpy as np
import argparse
from src.mrigait_benchmark.exportSignalsFromIK import get_signal
from src.opensim import opensim_utils


def main(gt_model, model2, trajectory_files, verbose = False):
    """
    Main function to run the MRIGait benchmark.
    
    Args:
        model1 (str): Path to the first opensim model.
        model2 (str): Path to the second opensim model.
        trajectory_files (list of str): List of paths to the trajectory files.
        
    Returns:
        pd.DataFrame: Result of the MRIGait benchmark.
    """
    results = pd.DataFrame(columns=['mass', 'mass_center', 'inertia','kinetic_energy', 'rel_mass', 'rel_mass_center', 'rel_inertia', 'rel_kinetic_energy'])

    kin1, kin2 = [], []
    for trajectory_file in trajectory_files:
        # Get the signals for the models, according to raitors script. This will be easier in 
        kin1.append(get_signal(gt_model, trajectory_file).kinetic_energy.to_numpy())
        kin2.append(get_signal(model2, trajectory_file).kinetic_energy.to_numpy())

    if len(kin1) > 0: # Just for debugging
        kin1 = np.concatenate(kin1)
        kin2 = np.concatenate(kin2)

        kin_diff = (kin1 - kin2).mean()
        rel_kin_diff = ((kin1 - kin2) / kin1).mean()

    df_gt = opensim_utils.get_bsip_dataframe_with_feet_and_HAT(opensim_utils.get_model(gt_model))
    df2 = opensim_utils.get_bsip_dataframe_with_feet_and_HAT(opensim_utils.get_model(model2))
    # remove: HAT, talus_r, calcn_r, toes_r, toes_l, talus_l, calcn_l

    df_gt = df_gt[~df_gt['name'].isin(['HAT', 'talus_r', 'calcn_r', 'toes_r', 'toes_l', 'talus_l', 'calcn_l'])]
    df2 = df2[~df2['name'].isin(['HAT', 'talus_r', 'calcn_r', 'toes_r', 'toes_l', 'talus_l', 'calcn_l'])]

    mass_e = (df_gt['mass'] - df2['mass']).mean()
    rel_mass_e = ((df_gt['mass'] - df2['mass']) / df_gt['mass']).mean()

    mass_center_e, rel_mass_center_e = [], []
    for segment in df_gt['name']:
        mc_gt = df_gt[df_gt['name'] == segment]['mass_center'].values[0]
        mc_2 = df2[df2['name'] == segment]['mass_center'].values[0]
        mass_center_e.append(np.linalg.norm(mc_gt - mc_2))
        rel_mass_center_e.append(np.linalg.norm(mc_gt - mc_2) / np.linalg.norm(mc_gt))

    mass_center_e = np.linalg.norm((df_gt['mass_center'] - df2['mass_center']),axis=0).mean()
    rel_mass_center_e = (np.linalg.norm((df_gt['mass_center'] - df2['mass_center']),axis=0) / np.linalg.norm(df_gt['mass_center'],axis=0)).mean()

    inertia_e, rel_inertia_e = [], []
    for segment in df_gt['name']:
        inertia_gt = np.diag(df_gt[df_gt['name'] == segment]['inertia'].values[0])
        inertia_2 = np.diag(df2[df2['name'] == segment]['inertia'].values[0])
        inertia_e.append((inertia_gt - inertia_2).mean())
        rel_inertia_e.append(((inertia_gt - inertia_2) / inertia_gt).mean())

    inertia_e = np.linalg.norm((df_gt['inertia'] - df2['inertia']),axis=0).mean()
    rel_inertia_e = (np.linalg.norm((df_gt['inertia'] - df2['inertia']),axis=0) / np.linalg.norm(df_gt['inertia'],axis=0)).mean()

    if verbose:
        print(f"mass difference: {mass_e}")
        print(f"relative mass difference: {rel_mass_e}")

        print(f"mass_center difference: {mass_center_e}")
        print(f"relative mass_center difference: {rel_mass_center_e}")

        print(f"inertia difference: {inertia_e}")
        print(f"relative inertia difference: {rel_inertia_e}")

        print(f"kinetic energy difference: {kin_diff}")
        print(f"relative kinetic energy difference: {rel_kin_diff}")

    results.loc[len(results)] = [np.abs(mass_e), np.abs(mass_center_e), np.abs(inertia_e), np.abs(kin_diff),
                                 np.abs(rel_mass_e), np.abs(rel_mass_center_e), np.abs(rel_inertia_e), np.abs(rel_kin_diff)]
    return results




def call_cmd():
    """
    Testing function to run the MRIGait benchmark. Only supplies a single trajectory file.
    """
    parser = argparse.ArgumentParser(description='Run the MRIGait benchmark.')
    parser.add_argument('--model1', type=str, required=True, help='Path to the first opensim model.')
    parser.add_argument('--model2', type=str, required=True, help='Path to the second opensim model.')
    parser.add_argument('--trajectories', type=str, required=True, help='Path to the trajectories file.')
    args = parser.parse_args()
    main(args.model1, args.model2, [args.trajectories], verbose=True)


if __name__ == '__main__':
    call_cmd()
    # I think opensim kinda likes global paths