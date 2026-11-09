# Errata: coordinate frame of body segment inertial parameters (BSIP)

## Summary
Centers of mass and inertia tensors estimated from the body hulls were stored in the SKEL/SMPL body frame (X left, Y up, Z forward) but written into OpenSim bodies, which use X forward, Y up, Z right. The values were therefore rotated by 90° about the vertical axis (arm segments, which lie along X in the T-pose, were affected differently). Masses were not affected.

Two further problems were found and fixed while checking this:
* The merged `foot_l`/`foot_r`/`HAT` bodies computed by `get_bsip_dataframe_with_feet_and_HAT` were incorrectly accumulating joint offsets, which led to incorrect CoM and inertia values. 
* The residual-force table in the paper was mixed up with two low-pass cutoffs: the *scaled* and *SIPP (HIT)* rows were computed with a 12 Hz cutoff, all other rows with 15 Hz. We now report all residuals at 15 Hz.

## Affected models and files
* All models created with `hull_bsip_to_opensim` (MRI ground truth, SIPP, SIPP (IH), SIPP (SMPL), SIPP-generic / S-gen), including the published
  `sipp_generic_male.osim`, `sipp_generic_female.osim` and `sipp_generic.zip`. **Please re-download the corrected versions.**
* Models derived from them, e.g. the AddBiomechanics runs started from SIPP / S-gen models (addB+SIPP, addB+S-gen). These have not been re-run yet. 
* Not affected: scaled generic models, plain AddBiomechanics models, segment masses.

## Effect on the reported results

All main findings of the paper are unchanged. Numbers change as follows (mean over participants; paper → corrected).

**MRIgait benchmark**

We reomved the merged foot rows from the benchmark as the previous values were incorrect. Because of that the relative mass errors are now much lower. In general, SIPP-based models are still much better than scaled models in relative CoM and inertia errors, with only the mass error being slightly higher in SIPP-based models. 

| Model | E_rel mass | E_rel CoM | E_rel inertia | E_rel kin. energy | E mass [kg] | E CoM [m] | E inertia [kg m²] | E kin. energy [J] |
|---|---|---|---|---|---|---|---|---|
| scaled | 0.282 → **0.261** | 0.604 → **0.331** | 0.376 → **0.663** | 0.264 → **0.239** | 0.082 → **0.081** | 0.087 → **0.050** | 0.209 → **0.043** | 2.079 → **1.876** |
| addB | 0.221 → **0.201** | 0.611 → **0.326** | 0.326 → **0.545** | 0.196 → **0.172** | 0.033 → **0.034** | 0.087 → **0.049** | 0.220 → **0.056** | 1.533 → **1.329** |
| SIPP | 0.102 → **0.115** | 0.171 → **0.128** | 0.163 → **0.205** | 0.110 → **0.135** | 0.057 → **0.055** | 0.032 → **0.018** | 0.173 → **0.009** | 1.041 → **1.318** |
| S-gen | 0.056 → **0.069** | 0.184 → **0.181** | 0.116 → **0.148** | 0.099 → **0.132** | 0.056 → **0.055** | 0.034 → **0.026** | 0.186 → **0.012** | 0.988 → **1.247** |
| SIPP (IH) | 0.096 → **0.109** | 0.170 → **0.133** | 0.121 → **0.140** | 0.098 → **0.124** | 0.056 → **0.054** | 0.032 → **0.019** | 0.173 → **0.009** | 0.946 → **1.233** |
| SIPP (SMPL) | 0.099 → **0.111** | 0.172 → **0.131** | 0.147 → **0.173** | 0.107 → **0.133** | 0.057 → **0.056** | 0.032 → **0.019** | 0.173 → **0.007** | 1.029 → **1.307** |
| addB+SIPP* | 0.171 → **0.171** | 0.266 → **0.563** | 0.233 → **0.185** | 0.185 → **0.189** | 0.054 → **0.055** | 0.044 → **0.073** | 0.188 → **0.031** | 1.628 → **1.764** |
| addB+S-gen* | 0.057 → **0.063** | 0.196 → **0.651** | 0.037 → **0.190** | 0.093 → **0.108** | 0.092 → **0.093** | 0.037 → **0.082** | 0.179 → **0.012** | 0.908 → **1.109** |

**Pelvis residual forces / moments** (31 trials, all models at 15 Hz, F in %BW, M in %BW·height)

Our bug only affected our contributed BSIP values, not the scaled models. Now that the BSIP values are correct, the SIPP-models show even lower residuals than reported in the paper. The residual forces for the scaled model were only affected by the mix-up in the low-pass cutoff. Thus, the corrected values are slightly higher than the paper values. 

| Model | F_norm | M_norm |
|---|---|---|
| MRI | 7.70 → 7.20 | 3.13 → 2.72 |
| SIPP (HIT) | 6.56* → 6.88 | 2.55* → 2.49 |
| S-gen | 7.77 → 7.22 | 3.00 → 2.74 |
| SIPP (IH) | 7.34 → 6.94 | 2.61 → 2.44 |
| SIPP (SMPL) | 7.29 → 6.89 | 2.68 → 2.50 |
| scaled | 7.71* → 8.43 | 2.94* → 3.11 |

\* These paper values were computed at 12 Hz. 

**Relative residual reduction compared with the scaled model (pelvis residual norms, mean |x| over 31 trials)**

The following table shows the relative reduction of the pelvis residuals compared with the scaled model, as reported in the paper and after correction. This table shows that the SIPP-based models better by an even larger margin than reported in the paper. The biggest improvement is with the MRI model and the SIPP-generic models, which now show a 15 % and 14 % reduction in F_norm, respectively, compared with the scaled model.


| Model | F_norm vs scaled, paper | F_norm vs scaled, **corrected** | M_norm vs scaled, paper | M_norm vs scaled, **corrected** |
|---|---|---|---|---|
| MRI | -0 % | **-15 %** | +6 % | **-13 %** |
| SIPP (HIT) | -15 % | **-18 %** | -13 % | **-20 %** |
| S-gen | +1 % | **-14 %** | +2 % | **-12 %** |
| SIPP (IH) | -5 % | **-18 %** | -11 % | **-22 %** |
| SIPP (SMPL) | -5 % | **-18 %** | -9 % | **-20 %** |
| SIPP (HIT), both at 12 Hz | −15 % | **−19 %** | −13 % | **−20 %** |

## Known remaining limitations
* AddBiomechanics-based models (addB+SIPP, addB+S-gen) still have to be re-run.

## Code changes
* `src/opensim/opensim_utils.py`: `hull_bsip_to_opensim` converts CoM (`c' = R c`) and inertia (`I' = R I Rᵀ`) to the OpenSim frame (`skel_to_opensim_bsip`);
  `get_bsip_dataframe_with_feet_and_HAT` rewritten with `composite_bsip`. Tests: `tests/test_composite_bsip.py`.
* `scripts/bodyscan/verify_bsip_frames.py` checks the frame of the BSIP in any generated model.
* `src/mrigait_benchmark/mrigait_benchmark.py` excludes the merged foot rows.
