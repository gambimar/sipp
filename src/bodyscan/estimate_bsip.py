## Functions to estimate the bsips from the body scan data
# Number of final models to be used for the bsip estimation:
# 1. HIT models
#   1.1 Original Pose (Body)
#   1.2 Reposed to T (Face)
#   1.3 Reposed to wide stance T (Skin)
#   1.4 Scaneca scan - Scaneca Pose (Scan)
# 2. SMPL shape-only model
#   2.1 Original Pose (Body)
#   2.2 Reposed to T (Face)
#   2.3 Reposed to wide stance T (Skin)
#   2.4 Scaneca scan - Scaneca Pose (Scan)
# 3. InsideHumans Model
#   3.1 Reposed to wide stance T (Skin)
# 4. Meta models (not in this file - we could weighted average multiple models)
### These models x2 - on the loose and on the tight hull

import argparse
import os
import pandas as pd

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Estimate the BSIPs from body scan data')
    # Loose or tight hull
    parser.add_argument('--hull', type=str, default='loose', choices=['loose', 'tight'],
                        help='Whether to use the loose or tight hull for BSIP estimation')
    # Pipeline - HIT, SMPL, InsideHumans
    parser.add_argument('--pipeline', type=str, default='HIT', choices=['HIT', 'SMPL', 'InsideHumans'],
                        help='Which pipeline to use for BSIP estimation')
    # Input path to the body scan data
    parser.add_argument('--input_path', type=str, default='results/P01/bodyhull',
                        help='Path to the body scan data')
    # Which model to use for the BSIP estimation
    parser.add_argument('--model', type=str, default='body', choices=['body', 'face', 'skin', 'scan'],
                        help='Which model to use for the BSIP estimation')
    # Output path to save the estimated BSIPs as a pandas dataframe
    parser.add_argument('--output_path', type=str, default='results/P01/bodyhull/bsip',
                        help='Path to save the estimated BSIPs')
    # Exact body mass
    parser.add_argument('--mass', type=float, default=70.0,
                        help='Exact body mass of the participant')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    """
        What happens in this function
        1. Load the body models
            1.1 SMPL && SKEL fit
            1.2 if tight: also load tight
            1.3 if HIT/Inside Humans - load additional data into dictionary
        2. Joint regression
            2.1 Use skel model
            2.2 if tight, sanity check whether the joints are close to non-tight
        3. If tight && HIT
            3.1 Reshaping needs to be done
        4. BSIP estimation
            4.1 Estimate cut planes
            4.2 Estimate BSIPs
    """
    loaded_data = {}
    scan_names = {
        'body': 'bodyscan',
        'face': 'facescan',
        'scan': 'scaneca',
        'skin': 'skin_layer'
    }
    smpl_fit = np.load(os.path.join(args.input_path, 'smpl_fit', args.model, 'loose', f'{scan_names[args.model]}_ignored_segments.npz'), allow_pickle=True)
    skel_fit = np.load(os.path.join(args.input_path, 'skel_fit', args.model, 'loose', f'{scan_names[args.model]}_ignored_segments.npz'), allow_pickle=True)



"""
    Some in between project management ketchup
    Gait:
        - process all data in addbiomechanics
        - do ID for all participants and models (scaled, addB, our models)
        (- process new initial guess in addbiomechanics) (if results are okish)
        (- do ID for that data again -- addB is not exactly solving the same thing as opensim)
        (- Compare with OpenSim RRA, NMSM builder)
    MRI:
        - correct labels, export meshes
        - body model fit loose and tight, reshapeing
        - NMS builder for a more exact skeleton?
        - ID with loose + tight, addB and NMS models
    Bodyhull:
        - Get bsip (this file)
"""



if __name__ == "__main__":
    main()