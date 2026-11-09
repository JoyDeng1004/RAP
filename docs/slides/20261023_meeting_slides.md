# 2026-10-23 Meeting Slides（逐页文字稿）

参会：佐藤先生、関川さん、池畑先生。每页两块：**Slide（英文，直接放到页面上）** / **中文说明（给你自己看：要讲清什么、数字来源、可能被问到什么）**。

主线：外部数据集对 target rig 的帮助主要来自"生成更好的候选"；剩下两个瓶颈是没有 target 真图时的"选择"，以及 rig 真正不同时（相机高度）的"解码器几何"。三个 claim ↔ 三个假说 ↔ 三个方向一一对应（①D↔覆盖↔A，②K↔选择↔B，③+α↔几何↔C）。

图：`assets/mtg1023_fig1…fig7_*.png`（`scripts/analysis/plot_mtg1023_figs.py` 生成）。P2、P15 的示意图和 P11 的对照表在 PPT 里直接做。

数字来源：D-K 评测 = `eval/summary_DK_20261021.md`、`summary_DK_K1pair_20261022.md`、`summary_DK_K100_20261022.md`、`summary_DK_K1seeds_20261022.md`（navtest，×100，按 136 个 log 做的配对 bootstrap 95% CI）；T2 = `eval/t2_results/t2_summary_full.md`、`t2_swap_full.md`（官方 v1 权重，raster，navtest 1,501 token）；论文数字 = 各论文 PDF 原文。标「Inferred」的都是推断。

---

# Part 0　Opening

## Slide 0｜Title

**Slide**
> **Using Multiple Datasets for a Target Camera Rig:**
> **Where the Gain Comes From, and What Still Breaks**
> Joy Deng · 2026-10-23

**中文说明**
开场页，不停留。

---

## Slide 1｜Since 10/16

**Slide**
> **Since 10/16: feedback → where it is answered**
>
> | Feedback (10/16) | This week | Slide |
> |---|---|---|
> | Goal = scaling with the number of datasets D; few/full-shot included; zero-shot as the extreme case | Axis ② now has D2 at K0 / K1 / K10 | 3, 5–8 |
> | Single target rig: two axes; Multi target rig: D axis | Axis ① has D1 only; Multi not started | 3, 19 |
> | One framework, Single/Multi differ by +α; Single also needs a twist beyond RAP | Generate–select framework, directions A / B / C | 15–18 |
> | Define scope B; RAP zero-shot = conventional method | K0-strict / K0-scene defined; RAP zero-shot = 43.8 | 4 |
> | Rig token / rig positional encoding | T2 tells where to put it (decoder) and what to encode (height, focal) | 18 |
> | "Shifting the image ≠ moving the camera" | T2 moves the camera in 3D and re-renders | 10–12 |
> | NAVSIM and nuScenes rigs are too similar | D2 cannot test rig effects; prior work predicts naive mixing fails once rigs differ | 10, 16 |
> | Survey prior work on D-scaling and zero/few-shot | What is solved vs. open | 14 |
> | (Optional) fewer cameras at test time | Existing data: drop F0 −48.9, drop L0/R0 −0.4, drop B0 0.0 | — |

**中文说明**
- 池畑先生没参加 10/16，这页顺带讲清"目的已扩大为 P(D,K) scaling"。
- 最后一行的数字来自 T1（真图，官方 v1 权重，1,501 token）：去掉 B0 逐 token 完全不变，因为轨迹 query 只在候选航点处采样，后视永远采不到。只口头带过，不展开。
- Simulator + RL/IL → journal，口头一句即可。

---

## Slide 2｜Goal, three claims, and how we measure

**Slide**
> **Goal: improve planning on a known target rig by training on several datasets — P(D, K)**
> D = number of independent data sources · K = % of target-rig (NAVSIM) real images · K = 0 is the extreme case
>
> **What we need to show**
> | | Claim | Status today |
> |---|---|---|
> | ① D | Adding an independent source helps the target rig — beyond data volume | Partial: D1 → D2 only, K ≤ 10, volume not controlled |
> | ② K | External data substitutes for target real images, down to K = 0 | Partial: D2 at K0 / K1 / K10 |
> | ③ +α | When rigs differ, a rig-aware module is needed | Motivation only (T2) |
>
> **One lens for every result** — RAP generates 64 candidates, a scorer picks one
> - **oracle@64** = best of 64 → *coverage*
> - **mean64** = average of 64 → *candidate quality*
> - **capture** = (scorer − mean64) / (oracle64 − mean64) → *selection*
>
> *[diagram: images → generator (64 candidates) → scorer → trajectory, with the three quantities marked]*

**中文说明**
- 佐藤先生要求的"先写目的、要实证什么"就是这一页。后面每一部分都回到这三个 claim。
- 这三个量贯穿全场：Part 1 用它解释外部数据为什么有用，Part 2 用它解释 rig 变化坏在哪里。

---

## Slide 3｜Where we are on the two axes

**Slide**
> **Where we are: D × K grid (v1 PDMS)**
> *[fig5: mtg1023_fig5_grid_status.png]*
> - Axis ② (D = 2, K 0 → full): K0 / K1 / K10 done, **K100 missing**
> - Axis ① (target full, D 1 → 4): **only D1 done**
> - D3 / D4 blocked by data: WOD-E2E has no 3D GT (no raster, no PDM labels); no Argoverse adapter yet

**中文说明**
- 先给地图，今天主要讲 D2 这一行（轴②）。
- 时间紧时这页可以删，fig5 挪到 Slide 19。

---

# Part 1　Axis ②: how much target data can external data replace?

## Slide 4｜Setup

**Slide**
> **Setup: names, budget, and what K0 means**
> - **D** counts independent data sources; a dataset that only supplies camera geometry is not counted
> - **K0-strict**: nothing from the target · **K0-scene**: target-scene raster, no target real images
> - NVS / 3DGS methods (CoIn3D, VR-Drive) also use **0 target real images** → K0 is the shared protocol for comparing with them
> - K0 points use **-mix**: nuScenes native rig (real + raster) + nuScenes rendered to the NAVSIM rig (raster)
> - **Budget**: 20 epochs; the target source gives ≥ 8,555 samples per epoch (repeated if smaller)
>   e.g. D2-K1-mix = 9,152 (1% × 11) + 23,099 + 23,099 = 55,350 per epoch
> - **Conventional method**: RAP zero-shot (D1-K0-strict-nat) = 43.8
> - **100% point**: D1-K100 (no augmentation, 85,016 samples). Official weights add only +1.50 [0.90, 2.14] (v1) / +0.50 [0.02, 0.98] (v2)
> - All checkpoints are selected on 214 NAVSIM val logs with real images; K counts training logs only

**中文说明**
- 回答関川さん的"Scope B 要定义清楚"：K0 是和 3DGS/NVS 类比较的共同协议，因为它们在 target 真图上同样是 0。
- 为什么 100% 点不用官方权重：两者差距很小，而且官方配方不同（多了增强、batch 不同）。
- 可能被问："ckpt 用真图 val 选，K0 还算 0 吗？" → Slide 8 回答。

---

## Slide 5｜External data helps at every K

**Slide**
> **Result 1 — External data helps at every K**
> *[fig1: mtg1023_fig1_shot_curve.png]*
> - At every K, **D2 > D1**
> - **D2 @ K1 (85.73) > D1 @ K10 (82.83)** — on navtest
> - D2-tgt (88.66) ≈ D2-mix (89.40): within seed noise, consistent with near-identical rigs
> - *Inferred:* D2 @ K1 ≈ D1 @ ~20% · D2 @ K10 ≈ D1 @ ~49% (log interpolation, 3 D1 points)
> - **navhard does not follow**: D2 ≈ D1 at K1 (30.07 vs 29.91), D2 < D1 at K10 (32.11 vs 36.12) — but navhard seed s.d. is ±3.18, so these gaps are within noise

**中文说明**
- 数字：D1 = 79.08 ± 1.15（K1，3 seed）/ 82.83 / 92.30；D2-mix = 82.30（K0 e6）/ 77.66（K0 last）/ 85.73 / 89.40；官方 93.80。v2 同趋势。
- 加性差距在 K1 和 K10 差不多（对 D1-K1 三个 seed 均值是 +6.65，K10 上是 +6.56），因为 D1 在 K1–K10 这段很平。按"等效 target 数据量"看，target 越少，外部数据越值钱（推断）。
- 第三个小图是 navhard（two-stage EPDMS），和前两个小图同样式、同图例：
  - D1 = 29.91 ± 3.18（K1，3 seed：28.05 / 33.58 / 28.09）/ 36.12 / 37.38；
  - D2-mix = 28.32（K0 e6）/ 20.72（K0 last）/ 30.07 / 32.11；D2-tgt 34.18；官方 36.16。
- navhard 上 D2 并没有高于 D1，所以"D2@K1 > D1@K10"只在 navtest 上成立。但单个 seed 的 navhard 就能差 5.5 分（28.05 对 33.58），K10 上的 4.01 差距和 seed 波动同一量级，不能说 D2 更差。
- 能确定的只有两点：K0 last（20.72）明显最低，和 Slide 8 的选择端漂移一致；D1-K100（37.38）最高。

---

## Slide 6｜Which differences are real?

**Slide**
> **Which differences are larger than seed noise?**
> *[fig4: mtg1023_fig4_paired_forest.png]*
> - Robust: external data at fixed K (**+5.4 to +7.6**), K10 → K100 (**+9.47**)
> - Not distinguishable: D2-K0 (e6) vs D1-K10 (−0.54 [−1.61, 0.48])
> - Bootstrap CIs cover test sampling only; seed s.d. is ±1.15 (v1) and ±3.18 (navhard) → **navhard single-run comparisons are uninformative**

**中文说明**
- 先划定能说的结论，避免被追问"是不是噪声"。
- 推断：两个单 seed run 之差的噪声约 √2 × 1.15 ≈ 1.6。
- navhard 的曲线已放进 Slide 5 的第三个小图：新点都低于 D1-K10 的 36.12，D1-K100 最高（37.38）；单 seed 的差距不可信，只看趋势，不做配对比较。

---

## Slide 7｜The gain is in the generator

**Slide**
> **Result 2 — The gain is in the generator, not the selector**
> *[fig2: mtg1023_fig2_pool_vs_scorer.png]*
> - Coverage: D1 oracle ≈ 94 until K100 (93.99 / 94.30 / 98.55); **D2 oracle 98.2–98.5 at every K, incl. K0**
> - D2-K1-mix − D1-K10: oracle **+3.94** [3.40, 4.51], mean64 **+3.05** [2.17, 3.94], final +2.90
> - Selection: capture at K1 is **0.55 for both D1 and D2**; it rises with target data (D1: 0.55 → 0.58 → 0.70)
> - *Inferred:* at K100 coverage is already saturated → on axis ①, more datasets can only help through mean64 / capture — and naive mixing can even hurt (HEAT, CSA) → **D2-K100 is the key next point**

**中文说明**
- 这页决定方法改哪一块：外部数据修的是生成端，选择端靠 target 真图。
- K10 上 capture 也有提升（0.58 → 0.67），所以"只在生成端"严格只对 K1 成立；K10 是生成端占大头。
- HEAT：朴素混训使 LAW 在 NAVSIM 上从 84.6 掉到 72.3。CSA：检测上朴素合并 6 个数据集几乎不涨（Slide 14）。

---

## Slide 8｜Zero-shot extreme: the selector drifts

**Slide**
> **Result 3 — At K = 0 the selector drifts while the candidates stay good**
> *[fig3: mtg1023_fig3_val_curves.png]*
>
> | D2-K0-scene-mix | e6 (val-selected) | last (strict K0) |
> |---|---|---|
> | v1 PDMS | 82.30 | 77.66 |
> | oracle@64 / mean64 | 98.17 / 61.11 | 98.54 / 64.04 |
> | capture | 0.57 | **0.39** |
> | NC / DAC / TTC | 96.68 / 94.82 / 88.97 | **92.97 / 91.74 / 81.28** |
> | EP | 76.47 | 77.91 |
>
> - last − e6: **−4.64** [−5.63, −3.56], although mean64 is +2.93 → riskier, less safe picks
> - The e6 checkpoint is chosen with target real images → **strict K0 needs a selection rule that uses no target real images** (→ Direction B)
> - *Inferred:* the scorer is supervised only on raster + nuScenes, and its ranking drifts away from real-image NAVSIM ranking

**中文说明**
- 训练 val：val/score 在 e6 达到峰值 0.820，之后降到 e19 的 0.757；val/best_score 从 e3 起保持 0.970–0.985，和 D2-K1/K10-mix 同一水平。
- 这就是佐藤先生说的"zero-shot 作为极端情况"的失败模式。

---

## Slide 9｜The representation-side SOTA: ViRA

**Slide**
> **Is data the right lever? The representation-side SOTA: ViRA (arXiv 2610.09695)**
> - **Method**: align planner encoder features to a frozen VFM at training time (SDA: KL on spatial attention; SSA: contrastive); inference unchanged; planner-agnostic
> - **SOTA**: ViRA-Diffusion **92.3 EPDMS** on NAVSIM v2 navtest, ≥ 1.9 above recent methods in their comparison
>   — a DiffusionDrive planner; on RAP the best is **86.3** (DINOv3-L 83.8 → + VGGT alignment), ≈ official RAP 86.67
> - **On RAP**: +10.9 for RAP\* (ResNet-34, no augmentation); **+2.5** with a DINOv3 backbone — the case comparable to ours
> - Across three planners: +7.9 (navtest) / +8.6 (navhard); zero-shot closed loop on HUGSIM: RAP\* HD-Score +3.0
> - **Not tested**: low K · multiple datasets · rig change → exactly our axes
>
> **Two experiments that position us against it**
> 1. ViRA @ K10 / K1, alone and on top of D2 → substitute or complementary?
> 2. ViRA-trained RAP under T2 → *Inferred:* encoder-side alignment will not fix the height loss (Slide 12)

**中文说明**
- 关键价值：① NAVSIM v2 的天花板参照；② 和我们"加数据"路线竞争的"改表示"路线，评审一定会问"用 ViRA 不就够了"；③ 它没做的三件事正好是我们的三条轴；④ 它是编码器侧方法，正好用来检验 Slide 12 的预测。
- 注意事项（被问再说）：
  - RAP\* 是不同 backbone、没有原始增强的复现。
  - 我们的 D1-K100 v2 是 86.16，已高于它的 RAP\* 基线 83.8，所以它在我们设定上的增益可能小于 +2.5（推断）。
  - 它的 navhard 用 SimScale 两阶段协议，和我们的 navhard 数字不能直接比。
  - HUGSIM 的相机 rig 是否和 NAVSIM 一致，ViRA 原文没说，要查 HUGSIM 原文。
  - SDA 对齐对象应是 PV 特征、教师投影是否 stop-grad、SSA 的聚合方式，原文都没写，复现时要自己定。
- 这页是 Part 1 → Part 2 的桥：表示能不能替代数据，以及换 rig 时表示能不能救。

---

# Part 2　What breaks when the rig really changes

## Slide 10｜Why a true camera-move test

**Slide**
> **Why we need a true camera-move test**
> - **Our D2 cannot test rig effects**: NAVSIM vs nuScenes (4 cameras) — height 1.43–1.49 m vs ≈ 1.5 m, focal 618 vs ≈ 608 px (after resize); differ only in rear FOV (63.7° vs ≈ 89°), distortion, calibration quality
> - **Image transforms ≠ moving the camera**: our T1 shifted images; MosaicVT (ICLR'26 submission) simulated mounting height by cropping — reviewers: rig dependence "not quantitatively demonstrated" (4/4/6/6)
> - VR-Drive renders new viewpoints by per-scene reconstruction on nuScenes (> 8 h per sequence)
>
> **T2 design** (criteria registered before running)
> - Official raster renderer; move the camera in 3D, update the calibration, re-render
> - Official RAP v1 weights · navtest 1,501 tokens
> - Factors: height −0.7 … +1.0 m · longitudinal ±1.0 m · lateral ±0.5 m · focal ×0.8 / ×1.25
> - Raster is a fair proxy: c1 raster − real = +1.4; drop F0: raster −48.8 vs real −48.9

**中文说明**
- 正面回答関川さん的两条批评：两个数据集 rig 太近（所以 D2 测不到）；平移图像不等于移动相机（所以 T2 真的移动相机）。
- 局限要主动说：官方 raster 渲染时相机抬高了 0.8 m（raster F0 约 2.29 m，真图约 1.49 m），所以 raster 的结论不能直接外推到真图。焦距效应 raster 略高估（−3.9 / −4.9，真图 T1 是 −2.1 / −3.5）。

---

## Slide 11｜Only camera height (and focal) hurt RAP

**Slide**
> **Result 4 — Only camera height (and focal) hurt RAP**
> *[fig6: mtg1023_fig6_t2_factors.png]*
> - Height: dose-dependent on both sides — −0.3 / −0.5 / −0.7 m: **−5.6 / −11.1 / −15.3**; +0.3 / +0.5 / +1.0 m: **−5.8 / −10.4 / −10.7**
> - Translation is harmless: longitudinal ±1.0 m −0.5 / −0.4; lateral ±0.5 m −0.6 / +0.1 · focal ×0.8 / ×1.25: −4.9 / −3.9
> - Loss is in **generation**: height −0.7 / +1.0 m → mean64 −15.1 / −10.5, oracle only −0.6 / −1.3, selection-only −3.3 / −3.4
> - Stale calibration is worse in every condition → RAP **does use** the calibration; the problem is how
>
> **Which factor matters depends on the architecture**
> | Work | Model / metric | Height | Longitudinal | Focal |
> |---|---|---|---|---|
> | CSA (CoRL'23) | DETR3D / BEVDet, detection | ego height matters; BEVDet fails from "wrong heights" | t_x changes object size → E-aware | most crucial: focal sync alone > +24 mAP |
> | VR-Drive (NeurIPS'25) | DiffusionDrive, open-loop L2 (m) | 0.57 → 1.21 (−0.7 m) / 1.46 (+1.0 m) | 0.57 → 1.57 (+1.0 m) | — |
> | Hashimoto (CVPRW'26) | World4Drive, L2 (m) | mostly recovered (5.01 → 0.90) | barely recovered (7.10 → 6.61) | — |
> | **Ours (T2)** | **RAP, PDMS** | **−5.6 to −15.3** | **≈ 0** | **−3.9 / −4.9** |

**中文说明**
- 这页决定 +α 要处理什么：高度和焦距，不处理平移。rig token 只编码高度和焦距，回应関川さん和佐藤先生的建议。
- "selection-only" = 候选固定为原 rig 的候选、只换图重新选（fix_all）。
- 标定过期时正确标定的收益：焦距 +11.9，高度 +1.0 m 时 +8.5。
- 对照表注意：
  - VR-Drive 没说明测试时有没有更新标定；我们标定过期时纵向 +1.0 m 是 −3.7。
  - VR-Drive 的开环 L2 上，只用 ego 状态的 AD-MLP 在所有设定下都是 0.29，胜过所有相机模型，说明开环 L2 基本测不到视觉；我们用 PDMS。
  - PCCR（ECCV'26）的 RigCD 用 ‖Δt‖，分不出高度和纵向：同为 1 m 时我们是 −10.7 对 −0.4。
  - CSA 的 "+24 mAP" 是 nuScenes→AV2、KITTI 的迁移，只做焦距对齐。
- 机制（H_par 只有 1/3 支持，H_scale 强形式被否）不上主页，放 backup。

---

## Slide 12｜Not the encoder

**Slide**
> **Result 5 — The height loss is not in the encoder features**
> *[fig7: mtg1023_fig7_swap_rescue.png]*
> - Intervention: replace each new-rig feature cell that shows a surface with the identity-rig feature of **the same surface**; decoder and calibration unchanged
> - Height: rescues only **1%** [−6, 8] (+1.0 m) and **2%** [−3, 7] (−0.7 m) → **ENC-DEAD** (pre-registered: both CIs end ≤ 25%)
> - Focal (positive control): **24%** [9, 39] / **23%** [6, 40] → the test can detect an encoder effect
> - → The loss lies in how the decoder **uses** features geometrically (reference points / sampling), not in what the encoder sees
> - *Inferred:* encoder- or input-side fixes (ViRA, CSA's focal sync) will not fix height. **Not ruled out**: VR-Drive-style consistency at the same 3D **anchor** (→ Slide 18)

**中文说明**
- 这页给方向 C 放在解码器提供依据，也给 ViRA 定位（编码器侧）。
- 局限：只替换了约 11–12% 的 F0 格（画出来的表面），背景格保持新 rig 特征；未 warp 的对照在 +1.0 m 上救回 40% [21, 58]、在 −0.7 m 上只有 6%，几何上不自洽，原因未解释（backup）。
- 机制拆解到此为止，接下来转向干预实验。

---

# Part 3　From findings to directions

## Slide 13｜Three findings → three hypotheses

**Slide**
> **Three findings → three hypotheses → three directions**
>
> | Hypothesis | Key evidence | Strength | Would be falsified by | Direction |
> |---|---|---|---|---|
> | **Coverage**: external data improves the generator | oracle ≈ 94 → ≥ 98.2; capture unchanged at K1 | navtest, 1 seed, volume not controlled | a volume-matched D1 control giving the same gain | A |
> | **Selection**: selection needs target real images; without them the scorer drifts | capture 0.55 → 0.58 → 0.70; K0 val/score 0.820 → 0.757 | one run; cause inferred | drift not reproduced across seeds | B |
> | **Geometry**: among rig factors only height (and focal) break RAP, and the break is in the decoder | T2 dose curve; swap 1–2% vs focal 23–24% | pre-registered; raster, official weights | encoder alignment (ViRA) or anchor consistency (VR-Drive) recovering most of the height loss | C |

**中文说明**
- 替代草稿里的"假说 1/2/3"。三个假说都能用同一套 oracle / mean64 / capture 测，并分别对应佐藤先生的三条轴（①、②的 K0 端、Multi）。
- 每个假说都写了"什么结果会推翻它"，表明我们知道证据有多强。

---

## Slide 14｜Prior work: solved vs open

**Slide**
> **Prior work: what is solved, what is open**
>
> | Work | Task | Done | Not done |
> |---|---|---|---|
> | **CSA** (CoRL'23) | camera 3D detection, 6 datasets | adding datasets one by one (1 → 6): naive merging barely helps; aligning focal, camera translation and ego frame makes joint training beat single-dataset models | no planning, no K axis; ego alignment by grid search with a reference model |
> | **HEAT** (RA-L) | camera planning, NAVSIM closed loop | nuScenes + NAVSIM + WOD-E2E; naive mixing drops LAW 84.6 → 72.3 | no D / K curves; rig only by padding blank views |
> | **VR-Drive** (NeurIPS'25) | planning, nuScenes open loop + CARLA | first study of viewpoint change in end-to-end driving; online feed-forward 3DGS views + consistency distillation | single dataset, fixed training rig, no K axis, no PDM scorer |
> | CoIn3D (CVPR'26) · Klinghoffer (ICCV'23) · 123D | perception | random-camera 3DGS; NVS to a known target rig; multi-dataset at fixed budget | no planning |
> | UniPlanner · MOSAIC · UDGA | planning / data selection / detection | abstracts only | to read |
>
> **Solved**: D-curve + sensor alignment for detection · viewpoint robustness for planning · mixed-dataset camera planning
> **Open**: P(D, K) for camera planning · generator vs selector decomposition · rig handling in multi-dataset planning

**中文说明**
- 回答 10/16 的文献调查 action。
- "首次"只能说到这一层：检测上的 D 曲线（CSA）和规划上的视角鲁棒性（VR-Drive 自称首个）都已经有人做了。
- UniPlanner 摘要自称"首个多数据集规划框架"，精读后可能还要再收窄。
- CoIn3D：随机相机高度 t_z ~ U(1.5, 2.2) m，需要 LiDAR + 4D 标注 + 源 rig 真图。Klinghoffer'23 的设定（渲染到已知 target rig）和我们的 -tgt/K0 相同，只是在感知侧。

---

## Slide 15｜One framework

**Slide**
> **One framework, one +α per bottleneck**
> *[diagram: data interface (real on native rig / raster on target or random rigs, per-camera rig descriptor) → generator (64 candidates) → selector]*
> - **A — data** (coverage): which sources, how much, how they enter
> - **B — selection at low K**: scorer consistency + checkpoint selection without target real images
> - **C — geometry**: rig-conditioned decoder (height, focal) + height-randomized raster
> - **Single target rig = A + B** (B is the twist beyond RAP) · **Multi target rig = A + B + C** (+α = C) · Single may also need C once D ≥ 3 brings different rigs
>
> **Baselines**: RAP zero-shot · target-only RAP · naive mixing (D2-mix / HEAT) · ViRA (stackable) · CSA-style sensor alignment · VR-Drive-style novel views + distillation (raster version) · CoIn3D · UniDrive · official RAP (reference only)

**中文说明**
- 回应"一个框架、Single/Multi 只差 +α、Single 也要一捻り"。
- 讨论点：10/16 也讨论过另一种分工（Multi 是提案法、Single 只是 fine-tune），请老师选。

---

## Slide 16｜Direction A — data

**Slide**
> **Direction A — Does D scale beyond volume?**
> - **To show**: claim ① — the gain comes from a new source, not just more samples
> - **Solved**: CSA — a detection D-curve; without alignment it does not scale · HEAT · MOSAIC (domain scaling laws within one dataset)
> - **Explanation + prediction** *(Inferred)*: our D2 scales without alignment because NAVSIM ≈ nuScenes rig; a D3 with different focal / height will likely fail under naive mixing → that is where +α becomes necessary
>
> **Plan**
> 1. Volume-matched control: D1-K1 + 46,198 raster samples from other NAVSIM logs vs D2-K1-mix
> 2. D2-K100 (second point on axis ①; tests the saturation prediction)
> 3. D3 audit: focal and camera height of candidate datasets; focal sync (CSA's K-sync) as the first alignment baseline
> 4. More seeds for D2-K1
> - **Possible novelty**: first P(D, K) with a generator/selector decomposition for camera planning — an empirical contribution

**中文说明**
- 计划 1 的 46,198 = nuScenes 两个来源每 epoch 的样本数（23,099 × 2）。这些 NAVSIM log 已有 metric cache，不用新建。它把"数据量"和"数据源"分开，是 claim ① 最关键的对照。
- CSA 用过的数据集：AV2、KITTI、KITTI-360、Lyft、nuScenes、Waymo。

---

## Slide 17｜Direction B — selection at low K

**Slide**
> **Direction B — Make the selector work without target images**
> - **To show**: claim ② at K = 0 — train and select a good selector with no target real images
> - **Solved**: RAP's own raster↔real alignment (feature MSE) and domain GRL act on encoder features; **nothing constrains the scorer's ranking**
> - **Gap**: no checkpoint-selection protocol for K0 planning that avoids target real images
>
> **Plan**
> 1. Cheap check: does val/score on NAVSIM **raster** val also peak near e6? → a legal K0 selection rule
> 2. Seeds for D2-K0: is the drift reproducible?
> 3. Method: ranking consistency of the scorer on paired raster / real (nuScenes native pairs) + PDM labels on target-rig raster
> - **Possible novelty**: moderate; its main value is making the K0 end point valid

**中文说明**
- 计划 1 要先确认 D2-K0 的 ckpt 是否每个 epoch 都存了（目前确定存了 e6 和 last）。
- 代码事实：R2R = raster↔real 的 FPN 特征 MSE（×0.002）；GRL 只作用在 F0（×0.001）。

---

## Slide 18｜Direction C — geometry

**Slide**
> **Direction C — Condition the decoder on camera height and focal**
> - **To show**: claim ③ — under a rig change (mainly height), a rig-aware decoder recovers the loss, better than augmentation, canonicalization or encoder alignment alone
> - **Solved**
>   - VR-Drive: biggest gain from feeding novel views (training ranges: height [−0.7, 1.0] m, pitch [−10°, 5°], longitudinal [−0.2, 1.0] m), plus consistency distillation at high-confidence anchors
>   - CSA: canonicalization (focal sync, ego-frame alignment) · CoIn3D · UniDrive (virtual camera)
>   - MosaicVT / VAPE: camera-centric positional encoding (make the representation invariant)
>   - Atomizer-IO / Perceiver IO: observations + metadata, cross-attention in physical coordinates → long-term Multi-rig architecture
> - **Gap**: no method targets the planner decoder's geometric sampling under camera-height change
> - **Sketch**: rig token = per-camera height and focal → modulates SCA offset / weight generation (multiplicative vs additive as ablation) + height-randomized raster
>
> **Two Go / Kill gates — both aimed at VR-Drive**
> 1. **Anchor-swap** (forward only): identity-rig features at the same 3D anchors = upper bound of VR-Drive's distillation in RAP
> 2. **Data only**: D1-K10 + height-randomized raster, no conditioning = raster version of VR-Drive's main component
> → If either removes the height loss, the rig token is not needed

**中文说明**
- 两个门都不过，rig token 才有存在理由。风险要主动说：VR-Drive 已显示"随机视角作为输入"单独就是最大的增益来源，所以门 2 通过的可能性不低（推断）。
- VR-Drive 用"不与测试设定重叠的小范围"（高度 [−0.3, 0.5] m）训练也稳健。
- 准备给池畑先生的一句话：Atomizer-IO（arXiv 2609.40320）把每个观测表示成带元数据的 atom，在物理坐标里做局部 cross-attention，实验是遥感栅格和 LiDAR 点云，没有透视相机。RAP 的轨迹 query 已经按 3D 参考点投影取特征（结构同类），差别在于相机像素只给射线、不给 3D 位置，而 T2 的高度问题正出在这一步。所以它是 Multi rig 的长期参考。
- RAP 的 SCA 是 deformable，没有 dense q·k，所以 Perceiver IO 式的 dense cross-attention 没有直接的插入点。

---

# Part 4　Plan and discussion

## Slide 19｜Plan

**Slide**
> **Plan for the next two weeks**
>
> | # | Experiment | Direction | Cost |
> |---|---|---|---|
> | 1 | K0 selection rule on raster val | B | forward only |
> | 2 | Anchor-swap | C | forward only |
> | 3 | Volume-matched D1 control | A | ≈ 27 h (est.) |
> | 4 | Height-randomized raster control | C | renderer change + 1 training run |
> | 5 | ViRA @ K10 (and + D2) | Slide 9 | check code release first |
> | 6 | D2-K100 | A | well over 53 h (est.) |
> | 7 | Seeds (D2-K1, D2-K0) | A / B | ≈ 27 h each (est.) |
> | 8 | D3 audit + focal sync | A / C | survey |
>
> Deferred: SCA bev_mask fix · D1-K0-strict-mix · navhard deep dive

**中文说明**
- 前两项只跑前向，本周就能出结果，而且分别决定方向 B 和 C 的去留。
- 估计依据：D2-K1-mix 实测约 27 h；D1-K100 实测约 53 h，D2-K100 的每 epoch 样本约多 50%。
- SCA bev_mask 先不修，为了和 D2-K10-mix（E3）保持可比。

---

## Slide 20｜Limitations

**Slide**
> **Limitations**
> - Single seed except D1-K1 (seed s.d.: v1 ±1.15, navhard ±3.18 — wider than bootstrap CIs)
> - All checkpoints selected with 214 NAVSIM val logs of real images
> - -mix runs carry the SCA sample-0 bev_mask defect
> - NAVSIM and nuScenes rigs are nearly identical → D2 says nothing about rig differences
> - No D2-K100 yet
> - T2 is raster + official weights; the raster camera sits 0.8 m above the nominal calibration
> - "D2 @ K1 > D1 @ K10" holds on navtest only

**中文说明**
- 主动列出，比被问出来好。

---

## Slide 21｜Decisions needed

**Slide**
> **Decisions needed today**
> 1. **Contribution split**: A as an empirical contribution, B / C as the method — acceptable?
> 2. **Third dataset**: by T2 and CSA, pick one whose focal / camera height differ from NAVSIM — Waymo or Argoverse?
> 3. **Compute priority**: D2-K100 · volume-matched control · height-randomized control · ViRA reproduction
> 4. **3DGS-type baselines** (CoIn3D / VR-Drive): now, or after Direction C shows a gain?
> 5. (Ikehata-sensei) Is an Atomizer-IO-style "observation + metadata" interface worth pursuing for the Multi-rig model?

**中文说明**
- 第 2 题的背景：WOD-E2E 没有 3D GT，不能 raster、没有 PDM 标签；仓库里还没有 Argoverse 的 adapter。
- 第 5 题既表明看过池畑先生推荐的论文，也请他给方向。

---

# Backup slides（只放图表，不需要讲稿）

- B1 完整结果表（v1 / v2 / navhard s1·s2 / oracle64 / mean64 / capture）— `eval/summary_DK_*.md`
- B2 v1 / v2 子指标表 — 同上第 2、3 节
- B3 navhard s1 / s2 分阶段数字 + seed 波动说明（综合分的曲线已在 Slide 5）
- B4 gap-to-official 分解（scorer part / candidate part）— 同上第 6 节
- B5 D-K runs：配置、作业号、选用 ckpt、每 epoch 样本数
- B6 T2 全部预设、标定过期对照、视野遮罩 — `t2_summary_full.md` 第 1–2 节
- B7 T2 机制检验：H_par（剂量支持、远近与平面场景不支持）、H_scale（S1 否；S2 只有位移方向对，幅度 10–40%）、表面对应点余弦（约 2/3 来自几何）
- B8 feature swap 细节（swap_f0 / swap_all / swap_all_bg、替换覆盖率）— `t2_swap_full.md`
- B9 ViRA：3 处未写明的实现细节、navhard 协议差异、HUGSIM
- B10 VR-Drive：Table 1 全表、组件消融（新视角作为输入是最大增益）、训练范围的 Superset / Subset 结果
- B11 CSA：三种对齐方法、Ego-sync 的网格搜索、局限（没有研究相机旋转）
- B12 MosaicVT / VAPE · Atomizer-IO / Perceiver IO / EPIO
- B13 NAVSIM vs nuScenes 标定对比
- B14 T1 结果（图像平移的问题，以及去掉相机的数字）

## 可能被问到的问题

| 问题 | 回答在哪 |
|---|---|
| D2 的增益是不是只是数据变多了？ | 目前排除不了 → Slide 16 计划 1 |
| e6 用真图选 ckpt，不算 zero-shot | 两个 ckpt 都报了 → Slide 8、Slide 17 计划 1 |
| T2 是 raster，不是真图 | 代理验证 + 0.8 m 渲染偏移 → Slide 10 |
| VR-Drive 已经做了，区别在哪？ | 多数据集、K 轴、PDMS、因子拆分；两个门直接测它的组件 → Slide 18 |
| CSA 已经有 D 曲线 | 那是检测；它还预测 D3 不对齐会失败 → Slide 16 |
| 用 ViRA 不就够了？ | 两个定位实验 → Slide 9 |
