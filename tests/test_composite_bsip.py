"""
Tests for the combined bodies (foot_l/foot_r/HAT) in src/opensim/opensim_utils.py.
Run: python tests/test_composite_bsip.py   (needs opensim; model tests are skipped if the result files are missing)
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.opensim import opensim_utils as u


def cloud_bsip(points, masses):
    """ Mass, center of mass and inertia about the center of mass of a set of point masses, straight from the definition. """
    M = masses.sum()
    c = (masses[:, None] * points).sum(0) / M
    r = points - c
    I = sum(m * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)) for m, ri in zip(masses, r))
    return M, c, I


def test_composite_matches_point_masses():
    rng = np.random.default_rng(0)
    clouds = [(rng.normal(size=(40, 3)) * s + o, rng.uniform(0.1, 1.0, 40)) for s, o in [(0.1, [0, 0, 0]), (0.05, [0.3, -0.2, 0.1]), (0.2, [-0.1, 0.4, 0.0])]]
    parts = [cloud_bsip(p, m) for p, m in clouds]
    M, c, I = u.composite_bsip([p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts])
    M_ref, c_ref, I_ref = cloud_bsip(np.vstack([p for p, _ in clouds]), np.concatenate([m for _, m in clouds]))
    assert np.isclose(M, M_ref) and np.allclose(c, c_ref) and np.allclose(I, I_ref), "composite differs from the point mass definition"


def test_offset_along_axis_does_not_add_inertia_about_that_axis():
    # a point mass 1 m along X from a massless-ish partner: no inertia about X (this is what the old outer(r, r) term got wrong)
    M, c, I = u.composite_bsip([1.0, 1.0], [[-0.5, 0, 0], [0.5, 0, 0]], [np.zeros((3, 3))] * 2)
    assert np.allclose(np.diag(I), [0.0, 0.5, 0.5]), np.diag(I)


def valid(I):
    w = np.linalg.eigvalsh(I)
    return w.min() > 0 and w[0] + w[1] >= w[2] - 1e-12


def test_models():
    done = 0
    for p in ['P02', 'P05']:
        for path in [f'results/{p}/MRI/osim_models/mri3.osim', f'results/{p}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim']:
            if not os.path.exists(path):
                continue
            model = u.get_model(path)
            df = u.get_bsip_dataframe_with_feet_and_HAT(model).set_index('name')
            for side in ['_l', '_r']:
                names = [f + side for f in ['calcn', 'talus', 'toes']]
                assert np.isclose(df.loc['foot' + side, 'mass'], sum(df.loc[n, 'mass'] for n in names))
                # independent calculation: calcn is the reference frame, toes sit at the mtp joint location, talus origin at the negative subtalar location
                off = {names[0]: np.zeros(3), names[2]: u.get_joint_translation(model, names[2]), names[1]: -u.get_joint_translation(model, names[0])}
                m, c, I = u.composite_bsip([df.loc[n, 'mass'] for n in names], [df.loc[n, 'mass_center'] + off[n] for n in names], [df.loc[n, 'inertia'] for n in names])
                assert np.allclose(df.loc['foot' + side, 'mass_center'], c, atol=1e-6), (path, side, df.loc['foot' + side, 'mass_center'], c)
                assert np.allclose(df.loc['foot' + side, 'inertia'], I, atol=1e-7)
                if df.loc['foot' + side, 'mass'] > 0.5:
                    assert valid(df.loc['foot' + side, 'inertia']), (path, side)
            # HAT: pelvis frame, torso offset is the back joint location
            m, c, I = u.composite_bsip([df.loc['pelvis', 'mass'], df.loc['torso', 'mass']],
                                       [df.loc['pelvis', 'mass_center'], df.loc['torso', 'mass_center'] + u.get_joint_translation(model, 'torso')],
                                       [df.loc['pelvis', 'inertia'], df.loc['torso', 'inertia']])
            assert np.allclose(df.loc['HAT', 'mass_center'], c, atol=1e-6) and np.allclose(df.loc['HAT', 'inertia'], I, atol=1e-6)
            assert valid(df.loc['HAT', 'inertia'])
            done += 1
    print(f"model tests: {done} models checked")


if __name__ == '__main__':
    test_composite_matches_point_masses()
    test_offset_along_axis_does_not_add_inertia_about_that_axis()
    test_models()
    print("all tests passed")
