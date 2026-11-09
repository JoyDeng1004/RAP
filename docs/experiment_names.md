# Experiment names

This file is the single source of truth for mapping experiment names. Reports, tables and slides use the new names. Physical names (W&B run id = `experiment_name`, ckpt dir, Hydra dir, `$EVAL/ckpts` alias) are never renamed.

## Rule

`D{n}-K{k}[-strict|-scene][-sources][-usage][-budget][_s{seed}]`

- **D{n}**: number of independent data sources that contribute training data. A dataset that only supplies camera geometry (e.g. NAVSIM rig used for rendering, with no NAVSIM logs) is not counted.
- **K{k}**: percentage of target-rig (NAVSIM) real images in training (0 / 1 / 2 / 5 / 10 / 100; K-shot splits `navsim_kshot_p{k:03d}_seed*`).
- **K=0 only**:
  - `-strict`: no target data at all, not even target-scene raster.
  - `-scene`: target-scene raster is given, target real images are not.
- **sources** (fixed-budget grid only): the external datasets, `nusc` / `womd`, in that order. Older names leave them implicit (nuScenes).
- **usage** (how the external dataset is used):

| suffix | meaning |
|---|---|
| `-nat` | nuScenes native rig, real + raster pairs |
| `-tgt` | external data rendered into the target (NAVSIM) rig, raster only (nuScenes or WOMD; WOMD has no other form) |
| `-mix` | `-nat` + `-tgt` |
| `-ft` | pretrained on external data, then fine-tuned on target data |
| `-aug` | official RAP data augmentation |

- **budget** (fixed-budget grid only): total external samples per epoch, e.g. `-23k`, split equally over the listed sources. Cells with the same budget have the same steps/epoch and target share, so they differ only in which sources fill the budget.

No `+` or `=` in names (both break `qsub -v` / Hydra). New runs use the new name directly as `EXP`, with seed as `_s0`.

## Mapping

| new name | old name | physical experiment_name | status | training data |
|---|---|---|---|---|
| D1-K0-strict-nat | E1 | `E1_nusc_nativerig_s0` | done | nuScenes → nuScenes rig, real + raster (23,099) |
| D1-K10 | E2a | `E2a_navsim_p010_s0` | done | NAVSIM p010 → NAVSIM rig, real + raster (8,555) |
| D2-K10-ft | E2b | `E2b_e1init_p010_s0` | done | E2a data, init from E1 `last.ckpt` |
| D2-K10-mix | E3 | `E3_mix_p010_s0` | done | NAVSIM p010 (8,555) + nuScenes native pair (23,099) + nuScenes → NAVSIM-rig raster (23,099) |
| D2-K10-tgt | E4 | `E4_navsimrig_raster_p010_s0_r2` (eval alias `E4_navsimrig_raster_p010_s0`) | done | NAVSIM p010 (8,555) + nuScenes → NAVSIM-rig raster (23,099) |
| D1-K100-aug | official RAP | — (official weights) | reference | NAVSIM navtrain full + official augmentation |
| D1-K100 | E2a_p100 | `E2a_navsim_p100_s0` | training (jobs 9060304 / 9060312) | NAVSIM p100 → NAVSIM rig, real + raster (85,016), no augmentation |
| D2-K0-scene-mix | K0-b | (new name) | planned | NAVSIM p010 raster only (8,555, `use_real=false`) + nuScenes native pair + nuScenes → NAVSIM-rig raster |
| D1-K0-strict-mix | K0-a | (new name) | planned | nuScenes native pair + nuScenes → NAVSIM-rig raster |
| D1-K1 | K1%-DS1 | (new name) | planned | NAVSIM p001 → NAVSIM rig, real + raster (~730, est.) |
| D2-K1-mix | K1%-DS2 | (new name) | planned | NAVSIM p001 + nuScenes native pair + nuScenes → NAVSIM-rig raster |
| D3-K1-mix | — | (new name) | done (v1 86.09, v2 79.20) | D2-K1-mix + WOMD training_20s → NAVSIM-rig raster, scenario-level subset `rap_womd_navsimrig_23k` (≈23,099, raster only, rig 0, no score). Uses everything available; not part of the fixed-budget grid |
| D2-K1-womd-tgt-23k | — | `D2-K1-womd_s0`, `D2-K1-womd_s1` (`RUN=D2-K1-womd`) | s0 done (v1 84.62, v2 78.71); s1 done (e7, v1 82.95, v2 76.71) | NAVSIM p001 (9,152) + WOMD → NAVSIM-rig raster `rap_womd_navsimrig_23k` (23,103) |
| D2-K1-nusc-tgt-23k | — | (new name) | s0 done (v1 83.20, v2 77.63); s1 done (e8, v1 81.82, v2 76.78) | NAVSIM p001 (9,152) + nuScenes → NAVSIM-rig raster `rap_nusc_navsimrig` (23,099) |
| D1-K1-stepmatch-23k | — | (new name) | s0 done (jobs 9220413 + 9266604 resume; e9, v1 80.86, v2 74.66) | NAVSIM p001 only, repeated to ≥ 32,255 rows per epoch (same steps/epoch as the 23k cells); control for "more data" vs "more steps" |
| D3-K1-nusc-womd-tgt-23k | — | (new name) | s0 done (v1 83.79, v2 78.01); s1 done (e6, v1 82.35, v2 76.44) | NAVSIM p001 (9,152) + nuScenes raster `rap_nusc_navsimrig_11k5` (≈11.5k) + WOMD raster `rap_womd_navsimrig_11k5` (≈11.5k) |

## Reading the grid

- DS=2 shot curve: `D2-K*-mix` (K0-scene, K1, K10).
- K=10% column: `D1-K10`, `D2-K10-tgt`, `D2-K10-mix`, `D2-K10-ft`.
- Fixed-budget grid at K1 (external = target-rig raster, 23k per epoch): `D1-K1` (no external) → `D2-K1-nusc-tgt-23k` vs `D2-K1-womd-tgt-23k` (which source) → `D3-K1-nusc-womd-tgt-23k` (more sources at the same volume); `D1-K1-stepmatch-23k` is the no-external-data control at the same steps. Compare `-mix` runs with these only as a practical recipe, since they add real images, more rows and more steps.
- `-strict` points are D1 by definition (NAVSIM contributes only geometry) and are not on the D2 curve.

## R2R alignment ablation (`D1-R*`)

Separate from the data-scaling grid above. Question (11/2 1v1): in NAVSIM-only RAP, does the R2R alignment module matter for PDMS? Job script `scripts/jobs/train_r2r.qsub`.

`D1-R{r}-{align|noalign|raster}[_s{seed}]`

- **R{r}**: real ratio as defined in RAP §4.2. Real images are kept for the logs of the K-shot split `navsim_kshot_p{r:03d}_seed0`; the other logs of the 978-log pool give raster only, so the total is always the full pool (85,016 samples). This differs from **K**, where only K% of the logs are used at all.
- **align**: official weights (spatial MSE `distill_feature_weight=0.002`, GRL `domain_loss_weight=0.001`; RAP paper Table 8 says 0.1 for the global term). **noalign**: both weights 0, losses still computed and logged. **raster**: R0, no real image, so there is nothing to switch.
- Everything else matches D1-K100: from scratch, no official augmentation, 32×2×4, 20 epochs, val = the same 214 logs. The split is always seed0; `_s{seed}` only changes the training seed.

| name | physical experiment_name | status | training data |
|---|---|---|---|
| D1-R100-align | `E2a_navsim_p100_s0` (= D1-K100) | done (v1 92.30) | NAVSIM p100, real + raster (85,016) |
| D1-R100-noalign | `D1-R100-noalign_s0` | planned | same as D1-R100-align, alignment off |
| D1-R1-align | `D1-R1-align_s{0,1,2}` | s0 queued (job 9266260, AR 9404) | NAVSIM p001_seed0 real + raster (832) + the other 968 logs raster only (`cache/rap_ego_rest_p001_seed0`, ≈84,184) |
| D1-R1-noalign | `D1-R1-noalign_s{0,1,2}` | s0 queued (jobs 9266262 → 9266263, normal queue) | same as D1-R1-align, alignment off |
| D1-R0-raster | `D1-R0-raster_s0` | planned (optional) | NAVSIM p100 raster only (85,016, `train_use_real=false`) |
