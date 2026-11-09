## Sources

| model | kind | csv |
|---|---|---|
| stepmatch | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1-stepmatch-23k_e9_scorer/9266630/2026.11.05.12.51.05.csv |
| stepmatch | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1-stepmatch-23k_e9_medoid/9266631/2026.11.05.12.51.05.csv |
| stepmatch | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D1-K1-stepmatch-23k_e9_scorer/9266632/2026.11.05.12.54.57.csv |
| stepmatch | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D1-K1-stepmatch-23k_e9_scorer/9266633/2026.11.05.12.35.34.csv |
| stepmatch | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D1-K1-stepmatch-23k_e9/9266634/2026.11.05.14.14.01.csv |
| D1K1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_scorer/9095481/2026.10.22.10.30.13.csv |
| D1K1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_medoid/9095482/2026.10.22.10.30.13.csv |
| D1K1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D1-K1_e16_scorer/9095484/2026.10.22.10.33.36.csv |
| D1K1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D1-K1_e16_scorer/9095486/2026.10.22.10.15.53.csv |
| D1K1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D1-K1_e16/9095487/2026.10.22.11.51.57.csv |
| D2K1womd | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_e16_scorer/9220138/2026.11.01.12.28.13.csv |
| D2K1womd | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_e16_medoid/9220139/2026.11.01.12.28.13.csv |
| D2K1womd | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-womd_e16_scorer/9220140/2026.11.01.12.31.33.csv |
| D2K1womd | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-womd_e16_scorer/9220141/2026.11.01.12.09.35.csv |
| D2K1womd | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-womd_e16/9220142/2026.11.01.13.43.17.csv |
| D2K1nusctgt | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-nusc-tgt-23k_e14_scorer/9230413/2026.11.02.15.38.24.csv |
| D2K1nusctgt | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-nusc-tgt-23k_e14_medoid/9230414/2026.11.02.15.40.04.csv |
| D2K1nusctgt | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-nusc-tgt-23k_e14_scorer/9230415/2026.11.02.15.49.39.csv |
| D2K1nusctgt | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-nusc-tgt-23k_e14_scorer/9230416/2026.11.02.15.24.27.csv |
| D2K1nusctgt | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-nusc-tgt-23k_e14/9230417/2026.11.02.17.01.30.csv |

## 1. Headline (x100)

| model | v1 scorer | v1 medoid | v1 Δ(s−m) | v2 navtest scorer | v2 navtest medoid | v2 Δ(s−m) | navhard scorer combined (s1 / s2) | navhard medoid |
|---|---|---|---|---|---|---|---|---|
| stepmatch | 80.86 | 75.81 | 5.05 | 74.66 | — | — | 24.63 (54.85 / 42.92) | — |
| D1K1 | 80.36 | 69.97 | 10.38 | 73.70 | — | — | 28.05 (58.15 / 46.90) | — |
| D2K1womd | 84.62 | 74.16 | 10.46 | 78.71 | — | — | 27.19 (59.44 / 44.35) | — |
| D2K1nusctgt | 83.20 | 75.70 | 7.50 | 77.63 | — | — | 26.96 (60.44 / 44.60) | — |

## 2. v1 navtest sub-metrics, scorer (x100)

| model | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| stepmatch | 95.56 | 93.45 | 75.72 | 88.43 | 99.50 | 95.59 | 80.86 |
| D1K1 | 96.52 | 94.76 | 72.80 | 89.26 | 97.11 | 95.30 | 80.36 |
| D2K1womd | 96.22 | 95.39 | 80.03 | 90.66 | 99.93 | 94.90 | 84.62 |
| D2K1nusctgt | 95.91 | 94.59 | 79.23 | 88.87 | 99.90 | 95.10 | 83.20 |

## 3. v2 navtest sub-metrics, scorer (x100)

| model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| stepmatch | 95.56 | 93.45 | 98.50 | 99.00 | 85.18 | 93.64 | 93.78 | 97.16 | 53.75 | 74.66 |
| D1K1 | 96.51 | 94.76 | 98.70 | 99.24 | 80.04 | 94.42 | 92.47 | 87.97 | 49.09 | 73.70 |
| D2K1womd | 96.22 | 95.39 | 98.07 | 99.08 | 87.59 | 95.38 | 93.70 | 98.15 | 68.31 | 78.71 |
| D2K1nusctgt | 95.91 | 94.59 | 98.08 | 98.88 | 87.65 | 94.76 | 93.72 | 97.18 | 67.16 | 77.63 |

## 4. navhard two-stage sub-metrics, scorer (x100)

| model | stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stepmatch | s1 | 94.56 | 78.44 | 96.78 | 99.33 | 79.05 | 93.78 | 90.44 | 97.11 | 37.33 | 54.85 |
| stepmatch | s2 | 82.73 | 74.94 | 86.30 | 98.37 | 78.12 | 79.34 | 51.02 | 97.09 | 46.62 | 42.92 |
| D1K1 | s1 | 94.56 | 84.44 | 96.11 | 99.56 | 74.19 | 93.56 | 88.22 | 86.00 | 33.78 | 58.15 |
| D1K1 | s2 | 85.77 | 77.73 | 89.21 | 98.21 | 73.16 | 83.72 | 51.33 | 91.95 | 53.33 | 46.90 |
| D2K1womd | s1 | 93.67 | 81.78 | 95.33 | 99.33 | 81.74 | 92.00 | 92.67 | 97.56 | 56.44 | 59.44 |
| D2K1womd | s2 | 83.15 | 78.47 | 84.68 | 98.23 | 83.12 | 79.95 | 47.72 | 96.50 | 56.73 | 44.35 |
| D2K1nusctgt | s1 | 92.89 | 82.89 | 95.22 | 99.33 | 81.36 | 92.00 | 91.78 | 97.78 | 53.33 | 60.44 |
| D2K1nusctgt | s2 | 84.29 | 77.16 | 83.79 | 98.15 | 81.12 | 82.38 | 49.88 | 96.26 | 54.63 | 44.60 |

## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))

| model | oracle64 | mean64 | scorer | medoid | capture | regret | pick=0 | pick=oracle | pick pctile | best of scorer top-1/4/8/16 | spearman(logit,score) | safe cand. | minADE64 / ADE scorer / ADE oracle (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stepmatch | 95.56 | 70.35 | 80.86 | 75.81 | 0.42 | 14.70 | 10.47 | 29.98 | 56.76 | 80.86 / 85.48 / 87.67 / 89.87 | 0.271 | 79.09 | 0.402 / 1.740 / 2.254 |
| D1K1 | 93.99 | 63.83 | 80.36 | 69.97 | 0.55 | 13.63 | 8.32 | 26.49 | 60.60 | 80.36 / 85.63 / 87.74 / 89.70 | 0.378 | 76.04 | 0.852 / 2.104 / 1.991 |
| D2K1womd | 98.09 | 68.13 | 84.62 | 74.16 | 0.55 | 13.47 | 8.12 | 30.32 | 64.17 | 84.62 / 89.36 / 91.40 / 93.61 | 0.412 | 76.68 | 0.282 / 1.406 / 2.112 |
| D2K1nusctgt | 97.24 | 71.31 | 83.20 | 75.70 | 0.46 | 14.04 | 9.08 | 30.83 | 62.59 | 83.20 / 87.68 / 89.77 / 92.02 | 0.366 | 80.05 | 0.282 / 1.429 / 1.686 |

consistency: stepmatch: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.029, n=12146; D1K1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146; D2K1womd: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.010, n=12146; D2K1nusctgt: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.014, n=12146

## 6. v1 gap to D1K1: scorer part = Δcapture × (oracle64−mean64) of either model; candidate part = rest (x100)

| model | PDMS gap | scorer part | candidate part | oracle64 gap | mean64 gap |
|---|---|---|---|---|---|
| stepmatch | -0.50 | 3.31 … 3.96 | -4.46 … -3.81 | -1.56 | -6.52 |
| D2K1womd | -4.26 | -0.07 … -0.07 | -4.19 … -4.19 | -4.10 | -4.30 |
| D2K1nusctgt | -2.84 | 2.32 … 2.70 | -5.54 … -5.16 | -3.25 | -7.48 |

token→log map: 12146 tokens, 136 logs (from /gs/bs/tga-RLA/qdeng/RAP/metric_cache_v1_navtest)

## 7. paired stepmatch − D1K1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | stepmatch | D1K1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 80.86 | 80.36 | 0.50 | [-0.47, 1.61] | 50.2 | 25.1 | 24.8 |
| v1_medoid.score | 12146 | 75.81 | 69.97 | 5.84 | [4.56, 7.21] | 48.5 | 28.8 | 22.7 |
| v2_scorer.score | 12146 | 74.66 | 73.70 | 0.96 | [0.05, 1.97] | 49.2 | 23.0 | 27.9 |
| oracle64.oracle64 | 12146 | 95.56 | 93.99 | 1.56 | [1.05, 2.07] | 46.9 | 44.6 | 8.4 |
| oracle64.mean64 | 12146 | 70.35 | 63.83 | 6.52 | [5.66, 7.45] | 68.4 | 0.6 | 31.0 |
| oracle64.score_scorer | 12146 | 80.86 | 80.36 | 0.50 | [-0.47, 1.61] | 50.2 | 25.1 | 24.8 |
| oracle64.minADE64 | 12146 | 0.402 | 0.852 | -0.450 | [-0.562, -0.353] | 42.1 | 0.0 | 57.9 |

## 7. paired D2K1womd − stepmatch (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D2K1womd | stepmatch | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 84.62 | 80.86 | 3.76 | [2.91, 4.60] | 47.8 | 28.8 | 23.4 |
| v1_medoid.score | 12146 | 74.16 | 75.81 | -1.65 | [-3.16, -0.24] | 36.8 | 30.9 | 32.3 |
| v2_scorer.score | 12146 | 78.71 | 74.66 | 4.04 | [3.22, 4.90] | 48.5 | 26.7 | 24.8 |
| oracle64.oracle64 | 12146 | 98.09 | 95.56 | 2.54 | [2.05, 3.06] | 24.9 | 61.1 | 14.0 |
| oracle64.mean64 | 12146 | 68.13 | 70.35 | -2.22 | [-3.28, -1.25] | 39.0 | 0.7 | 60.3 |
| oracle64.score_scorer | 12146 | 84.62 | 80.86 | 3.76 | [2.91, 4.60] | 47.8 | 28.8 | 23.4 |
| oracle64.minADE64 | 12146 | 0.282 | 0.402 | -0.120 | [-0.139, -0.103] | 30.2 | 0.0 | 69.8 |

## 7. paired D2K1nusctgt − stepmatch (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D2K1nusctgt | stepmatch | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 83.20 | 80.86 | 2.34 | [1.48, 3.21] | 48.9 | 28.8 | 22.3 |
| v1_medoid.score | 12146 | 75.70 | 75.81 | -0.12 | [-1.39, 1.10] | 36.0 | 31.6 | 32.3 |
| v2_scorer.score | 12146 | 77.63 | 74.66 | 2.97 | [2.11, 3.84] | 49.6 | 26.8 | 23.6 |
| oracle64.oracle64 | 12146 | 97.24 | 95.56 | 1.68 | [1.24, 2.20] | 20.5 | 58.5 | 21.0 |
| oracle64.mean64 | 12146 | 71.31 | 70.35 | 0.96 | [0.10, 1.80] | 51.7 | 0.7 | 47.6 |
| oracle64.score_scorer | 12146 | 83.20 | 80.86 | 2.34 | [1.48, 3.21] | 48.9 | 28.8 | 22.3 |
| oracle64.minADE64 | 12146 | 0.282 | 0.402 | -0.120 | [-0.137, -0.106] | 27.7 | 0.0 | 72.3 |
