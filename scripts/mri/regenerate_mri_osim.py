"""
Rebuild results/<P>/MRI/osim_models/<out_name> from the already estimated MRI BSIPs
(results/<P>/bodyhull/bsip/MRI_skin_loose.pkl), without redoing the SMPL fit, mesh deformation or BSIP estimation.
This is the last step of src/mri/reshape_mri.py. The pickles stay in the SKEL/SMPL frame, hull_bsip_to_opensim converts them.

Usage: python scripts/mri/regenerate_mri_osim.py [--participants P01 P02 ...] [--out_name mri.osim]
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.opensim.opensim_utils as opensim_utils

parser = argparse.ArgumentParser()
parser.add_argument('--participants', nargs='+', default=['P01', 'P02', 'P03', 'P04', 'P05', 'P06'])
parser.add_argument('--out_name', default='mri.osim')
args = parser.parse_args()

for p in args.participants:
    model_file = f"results/{p}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim"
    pkl = f"results/{p}/bodyhull/bsip/MRI_skin_loose.pkl"
    out = f"results/{p}/MRI/osim_models/{args.out_name}"
    hull_bsip = pd.read_pickle(pkl)
    model = opensim_utils.get_model(model_file)
    # extra_shoe_weight=0 as in the current reshape_mri.py
    _, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, model, extra_shoe_weight=0, arms_from_hull=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    opensim_utils.save_opensim_model(osim_model, out)
    print(f"{p}: wrote {out}")
