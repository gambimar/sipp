"""
Check that the BSIPs written into OpenSim models by hull_bsip_to_opensim are in the OpenSim body frames
(X forward, Y up, Z right), using the unscaled Rajagopal-based generic model as the reference for axis conventions.

Checks per model and body (values are read back from the OpenSim model):
  1. CoM: the dominant axis and its sign equal the template's (pelvis: -X, legs/arms: -Y, calcn/toes: +X, torso: +Y)
  2. Inertia: the axis of the smallest moment equals the template's (long bones: Y, calcn: X)
  3. Left/right: X and Y of the CoM have the same sign, Z the opposite sign (mirror images), unless |Z| is tiny
  4. Inertia is a valid tensor: positive definite and the principal moments satisfy the triangle inequality

Usage (needs opensim, pandas):
  python scripts/bodyscan/verify_bsip_frames.py [--utils path/to/opensim_utils.py] [--root .] [-v]
"""
import argparse, glob, importlib.util, os, sys
import numpy as np
import pandas as pd

TEMPLATE = 'data/resources/unscaled_generic_P06_no_C7.osim'
AXES = 'XYZ'
# bodies that are checked and the reason why their axes are clear cut
COM_BODIES = ['pelvis', 'femur', 'tibia', 'calcn', 'toes', 'torso', 'humerus', 'ulna', 'hand']
INERTIA_BODIES = ['femur', 'tibia', 'calcn', 'torso', 'humerus', 'ulna']  # unambiguous smallest (calcn) / largest-pair axes
MIN_AXIS_BODIES = ['femur', 'tibia', 'calcn', 'humerus', 'ulna']


def load_utils(path):
    spec = importlib.util.spec_from_file_location('opensim_utils_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def valid_inertia(I):
    w = np.linalg.eigvalsh((I + I.T) / 2)
    return bool(w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--utils', default='src/opensim/opensim_utils.py')
    ap.add_argument('--root', default='.')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    os.chdir(args.root)
    u = load_utils(args.utils)

    ref = u.get_bsip_dataframe(u.get_model(TEMPLATE)).set_index('name')

    jobs = []  # (label, pickle, model_file, arms_from_hull)
    for p in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']:
        m = f'results/{p}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim'
        for pkl in sorted(glob.glob(f'results/{p}/bodyhull/bsipw_shoes/*_loose*.pkl') + glob.glob(f'results/{p}/bodyhull/bsip/MRI_skin_loose.pkl')):
            if os.path.exists(m):
                jobs.append((f'{p}:{os.path.basename(pkl)}', pkl, m, False))
    for g in ['generic_male', 'generic_female']:
        pkl, m = f'results/{g}/bsip/HIT_generic_loose_w_skel.pkl', f'results/{g}/opensim/scaled_generic_model.osim'
        if os.path.exists(pkl) and os.path.exists(m):
            jobs.append((f'{g}:HIT_generic', pkl, m, True))

    total_fail = 0
    for label, pkl, mfile, arms in jobs:
        hull = pd.read_pickle(pkl)
        hull_names = set(hull['segment'].replace({'lumbar_body': 'torso'}))  # before the call, old versions rename in place
        model = u.get_model(mfile)
        try:
            _, model = u.hull_bsip_to_opensim(hull, model, arms_from_hull=arms)
        except Exception as e:
            print(f'{label}: ERROR {type(e).__name__}: {e}')
            total_fail += 1
            continue
        df = u.get_bsip_dataframe(model).set_index('name')
        fails = []
        for name, row in df.iterrows():
            base = name[:-2] if name.endswith(('_r', '_l')) else name
            if name not in hull_names or row['mass'] < 1e-3:
                continue  # only check bodies that really come from the hull
            com, I, tcom, tI = row['mass_center'], row['inertia'], ref.loc[name, 'mass_center'], ref.loc[name, 'inertia']
            if base in COM_BODIES and np.linalg.norm(tcom) > 0.02 and np.linalg.norm(com) > 0.01:
                ax = int(np.argmax(np.abs(tcom)))
                if int(np.argmax(np.abs(com))) != ax or np.sign(com[ax]) != np.sign(tcom[ax]):
                    fails.append(f'{name}: CoM {np.round(com, 3)} vs template {np.round(tcom, 3)} (dominant axis/sign)')
            if base in MIN_AXIS_BODIES and np.argmin(np.diag(I)) != np.argmin(np.diag(tI)):
                fails.append(f'{name}: smallest moment about {AXES[np.argmin(np.diag(I))]}, template {AXES[np.argmin(np.diag(tI))]}')
            if not valid_inertia(I):
                fails.append(f'{name}: invalid inertia tensor, diag {np.diag(I)}')
        for base in ['femur', 'tibia', 'calcn', 'toes', 'humerus', 'ulna', 'hand']:
            r, l = f'{base}_r', f'{base}_l'
            if r in hull_names and l in hull_names and df.loc[r, 'mass'] > 1e-3 and df.loc[l, 'mass'] > 1e-3:
                cr, cl = df.loc[r, 'mass_center'], df.loc[l, 'mass_center']
                if base in ('hand',):
                    continue  # tiny and noisy
                for ax in (0, 1):
                    if abs(cr[ax]) > 0.005 and abs(cl[ax]) > 0.005 and np.sign(cr[ax]) != np.sign(cl[ax]):
                        fails.append(f'{base}: {AXES[ax]} of CoM has different sign on the two sides ({cr[ax]:+.3f}, {cl[ax]:+.3f})')
                if abs(cr[2]) > 0.005 and abs(cl[2]) > 0.005 and np.sign(cr[2]) == np.sign(cl[2]):
                    fails.append(f'{base}: Z of CoM has the same sign on both sides ({cr[2]:+.3f}, {cl[2]:+.3f}), should be mirrored')
        status = 'OK' if not fails else f'{len(fails)} problem(s)'
        print(f'{label:<45} {status}')
        if args.verbose or fails:
            for f in fails:
                print('     -', f)
        total_fail += len(fails)
    print(f'\nTotal problems: {total_fail} in {len(jobs)} models')
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
