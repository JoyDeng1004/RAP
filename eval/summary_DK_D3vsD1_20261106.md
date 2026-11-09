## Sources

| model | kind | csv |
|---|---|---|
| D3s0 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-nusc-womd-tgt-23k_e16_scorer/9242225/2026.11.03.12.51.46.csv |
| D1K1s0 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_scorer/9095481/2026.10.22.10.30.13.csv |
| D3s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-nusc-womd-tgt-23k_s1_e6_scorer/9258284/2026.11.04.19.10.48.csv |
| D1K1s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_s1_e10_scorer/9103914/2026.10.23.01.10.15.csv |

## 1. Headline (x100)

| model | v1 scorer | v1 medoid | v1 Δ(s−m) | v2 navtest scorer | v2 navtest medoid | v2 Δ(s−m) | navhard scorer combined (s1 / s2) | navhard medoid |
|---|---|---|---|---|---|---|---|---|
| D3s0 | 83.79 | — | — | — | — | — | — | — |
| D1K1s0 | 80.36 | — | — | — | — | — | — | — |
| D3s1 | 82.35 | — | — | — | — | — | — | — |
| D1K1s1 | 78.12 | — | — | — | — | — | — | — |

## 2. v1 navtest sub-metrics, scorer (x100)

| model | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| D3s0 | 96.52 | 94.12 | 80.09 | 89.86 | 99.86 | 95.22 | 83.79 |
| D1K1s0 | 96.52 | 94.76 | 72.80 | 89.26 | 97.11 | 95.30 | 80.36 |
| D3s1 | 96.80 | 96.29 | 74.48 | 89.46 | 99.81 | 96.62 | 82.35 |
| D1K1s1 | 96.20 | 96.28 | 66.15 | 88.94 | 97.13 | 96.25 | 78.12 |

## 3. v2 navtest sub-metrics, scorer (x100)

| model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|

## 4. navhard two-stage sub-metrics, scorer (x100)

| model | stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|

## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))

| model | oracle64 | mean64 | scorer | medoid | capture | regret | pick=0 | pick=oracle | pick pctile | best of scorer top-1/4/8/16 | spearman(logit,score) | safe cand. | minADE64 / ADE scorer / ADE oracle (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

consistency: 

token→log map: 12146 tokens, 136 logs (from /gs/bs/tga-RLA/qdeng/RAP/metric_cache_v1_navtest)

## 7. paired D3s0 − D1K1s0 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3s0 | D1K1s0 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 83.79 | 80.36 | 3.43 | [2.22, 4.62] | 57.2 | 25.5 | 17.3 |

## 7. paired D3s1 − D1K1s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3s1 | D1K1s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 82.35 | 78.12 | 4.23 | [3.35, 5.06] | 61.6 | 18.9 | 19.5 |
