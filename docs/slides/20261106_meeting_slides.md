# 2026-11-06 Meeting Slides（逐页文字稿，v2：按 10/23 实际讲过的内容接续）

参会：佐藤先生、関川さん、池畑先生。每页两块：**Slide（英文，上页面）** / **中文说明（自己看）**。**承接句**是每页开头口头说的那句话，用来把这一页挂回 10/23 讲过的内容。

## 设计原则

**接续的锚点只用 10/23 实际放映过的东西**（`MTG_20261023.pptx` 共 19 页）：
- p2 Recap（10/16 手写笔记）
- p3 Goal + claims ①② + oracle/mean64/capture 这个视角
- p5 D×K 格子图
- p6–p9 Result 1–3
- p10–p14 T2 / Result 4–5
- p15–16 ViRA
- p17 Discussion

10/23 的草稿里有、但没有放映的内容（"Since 10/16"表、三假说表、Plan、Decisions、claim ③）**不能当作"上次说过"来引用**。

**顺序**：今天按 10/23 的 Result 1 → 5 逐个"回访"。每一节的第一页都用同一个模板：上次说了什么 → 还留着什么问题 → 这次做了什么 → 结果。

**佐藤先生和池畑先生没参加 11/2**：凡是 11/2 或 Slack 上出现的东西（warp 想法、alignment 的疑问、K0 诊断细节），第一次出现时都要用一句话交代出处。

**数字来源**：
- navtest：`eval/summary_DK_*.md`（×100，按 136 个 log 做配对 bootstrap 95% CI）。
- K0：`eval/k0_results/`。
- T2：`eval/t2_results/`。
- D1-R1：训练日志 `qsub_outputs/rap-r2r.o9266260`（训练中，非 navtest）。

**时间**：讲约 30 分钟，讨论约 25 分钟。

---

# Part 0　Recap：上次停在哪里

## Slide 1｜Last time (10/23) in one page

**Slide**
> **Goal**: improve planning on a known target rig by training on several datasets — P(D, K)
> D = independent data sources · K = % of target-rig (NAVSIM) real images · K = 0 is the extreme case
> Lens: RAP generates 64 candidates, a scorer picks one (oracle@64 / mean64 / selection)
>
> | 10/23 result | What I said |
> |---|---|
> | Result 1 | External data helps at every K; D2 @ K1 (85.73) > D1 @ K10 (82.83); navhard does not follow |
> | Result 2 | The gain is in the generator, not the selector (capture 0.55 for D1 and D2 at K1) |
> | Result 3 | At K = 0 the selector drifts while the candidates stay good |
> | Result 4 | Under a rig change, only camera height (and focal) hurt RAP |
> | Result 5 | Height breaks where the decoder looks, not what the encoder sees |
>
> Claim ① (D) and ② (K) were **"Partial — volume not controlled"**

**承接句**："上次我报告了 5 个结果。今天的内容就是逐个回访这 5 个结果：哪些被加固了，哪些被修正了，哪些还停在原地。"

**中文说明**
- 文字和数字直接取自 10/23 的 p3、p6–p14，不加新内容，让导师确认"这就是上次的状态"。
- "volume not controlled"是 10/23 p3 上你自己写的限制，今天第一部分正好把它补上，这是最自然的接续点。

---

## Slide 2｜What came out of the discussion — and where each item is answered today

**Slide**
> | From the 10/23 discussion | Today |
> |---|---|
> | NAVSIM and nuScenes are too similar — try another pair (Sato-sensei) | S4–S5: added WOMD |
> | Count epochs on the target, fix the mini-batch ratio, try ~3 ratios (Sato-sensei) | S4: step control first; 3 ratios not yet |
> | Dataset / rig ID token → camera parameters → Plücker (Sekikawa-san) | S10: not yet; where it fits |
> | T2 lateral: GT and collisions did not move (Sato-sensei) | S9: checked |
> | How does T2 connect to the first half? What is the improvement point? (Ikehata-sensei) | S6 (improvement point) · S11 (connection) |
> | Appearance, not only the rig, can break transfer (Ikehata-sensei) | S8 |
> | Raster is not required; main contribution = absorbing rig geometry, alignment = sub-contribution — not settled (all) | S3 premise · S11 |
> | Red line above the official RAP; navhard not the focus (Sato-sensei) | S11 · Backup |

**承接句**："讨论里大家给了这些意见，我把每一条都对应到了今天的某一页。"

**中文说明**
- 10/23 的 deck 里没有这张表，它是讨论（VTT）的整理。用"Today"一列告诉大家每一条在哪一页回答，后面讲到时只要说"这是 S2 的第几行"。
- 池畑先生 10/23 问过"T2 和前半部分怎么连"（00:58:41–01:03:21），当时你承认没有连接。今天 S11 正面回答，一定要讲到。

---

## Slide 3｜Today: the same grid, what changed, and one premise

**Slide**
> *[10/23 p5 的 D×K 格子图原样复用。新做的格子用颜色标出：K1 列加上 nuSc-23k / WOMD-23k / nuSc+WOMD-23k / step-matched；D1 行加上 "D1-R1 (running)"]*
>
> **This week went into the K1 column** — where the controls were missing — rather than into new rows.
>
> **One premise to state first** (from my 11/2 meeting with Sekikawa-san): RAP's raster↔real alignment is assumed to be needed, and all our discussions so far assumed it. I doubted it for PDMS, because RAP's evidence is MinADE; my earlier on/off run (8/26) was too small to decide. **Working premise today: needed.** A clean on/off test is running (S7).

**承接句**："这还是上次那张格子图。这一周我没有往下加新的行，而是把 K1 这一列补扎实，因为上次 claim ①② 的限制'没有控制数据量'就出在这里。"

**中文说明**
- 用同一张图，导师不用重新理解框架。
- alignment 立场按関川さん 11/2 的要求在开头讲（01:50:13–01:53:32）。佐藤先生和池畑先生第一次听到，所以只讲立场和"正在测"，细节放 S7。

---

# Part 1　回访 Result 1：外部数据的增益是真的吗？

## Slide 4｜Result 1 revisited — design

**Slide**
> **Last time**: "External data helps at every K" — but volume was not controlled, and NAVSIM ≈ nuScenes (S2, rows 1–2)
> **Open questions** → my hypotheses: H1 the gain is not just more training steps · H2 at the same budget, more sources help · H3 a source closer to NAVSIM helps more
>
> K1 (1 % NAVSIM real, 832 samples); all external data rendered to the NAVSIM rig; same 2,520 steps; target = 28.4 % of rows; seeds 0 and 1
> | Cell | External raster | Tests |
> |---|---|---|
> | D1-K1 | — | baseline |
> | D1-K1-stepmatch | — (target repeated to the same steps) | H1 |
> | D2-K1-nusc-tgt-23k / D2-K1-womd-tgt-23k | 23k nuScenes / 23k WOMD | H3 |
> | D3-K1-nusc-womd-tgt-23k | 11.5k + 11.5k | H2 |
>
> Why not the 3 ratios first: changing the ratio also changes the number of steps, so I fixed the steps first. Concat + shuffle at 28.4 % is, in expectation, one cell of the fixed-ratio design.

**承接句**："上次 Result 1 的限制是数据量没控制，佐藤先生也指出 NAVSIM 和 nuScenes 太像。所以这周我先把这两个问题变成三个可以检验的假说。"

**中文说明**
- 10/23 的 D2-K1-mix（85.73）与 D1-K1 的步数相差约 6 倍，这就是"没控制数据量"的具体含义。主动讲，这是自我修正。
- WOMD 没有相机图像，只能用 raster。所以它检验的是"场景不同"，不是"rig 不同"。这一点放在 S11 再说。

---

## Slide 5｜Result 1 revisited — results

**Slide**
> v1 PDMS (navtest), vs D1-K1 of the same seed (s0 / s1)
> | External (23k) | Δ PDMS | Δ oracle | Δ regret |
> |---|---|---|---|
> | nuScenes | +2.84 / +3.70 | +3.25 / +4.22 | +0.41 / +0.53 (worse) |
> | WOMD | +4.26 / +4.83 | +4.10 / +4.65 | −0.16 / −0.17 |
> | nuScenes + WOMD | +3.43 / +4.23 | +3.68 / +3.74 | +0.25 / −0.49 |
>
> - **H1 ✔** vs the step-matched control: still +2.34 [1.48, 3.21] (nuScenes), +3.76 [2.91, 4.60] (WOMD); steps alone: +0.50 [−0.47, 1.61]
> - **H3 ✗** WOMD beats nuScenes in both seeds (−1.42 [−2.06, −0.76]; −1.13 [−1.83, −0.38]), although nuScenes shares cities with NAVSIM
> - **H2 ✗** two sources ≈ their average (83.07 vs 83.15, two-seed means)
> → **Result 1 holds with the controls**; and it is not "similar data, of course it helps"

**承接句**："结论是：Result 1 在控制了步数之后仍然成立，而且不是因为数据像。"

**中文说明**
- H3 直接回答佐藤先生 10/23 的担心（01:16:50 "似てるから良くなって当たり前"）。
- H1 的口径（主动讲）：
  - v1 上外部增益保留约 82–88%，但 stepmatch 的 scorer 过拟合（832 个样本重复 39 倍，regret 变差 1.07），把这个比例抬高了。
  - 按 oracle 算，步数本身贡献 +1.57；外部数据在此之上 +1.68 / +2.54。
  - 所以要说"外部数据在步数对齐后仍显著"，不说"85% 与步数无关"。stepmatch 只有 1 个 seed。
- 表里第三列 regret 是 S6 引入的新指标，这里先放着，下一页解释。

---

# Part 2　回访 Result 2：增益在 generator

## Slide 6｜Result 2 revisited — still the generator; the gap to K100 is selection

**Slide**
> **Last time**: "The gain is in the generator, not the selector" (capture 0.55 for D1 and D2)
> **Metric change**: capture → **regret = oracle − PDMS of the pick** (capture's denominator moves with mean64; its split flipped between seeds). Same idea, more stable.
>
> | navtest v1 | Real | oracle | regret | PDMS |
> |---|---|---|---|---|
> | D1-K1 (s0) | 1 % | 94.0 | 13.6 | 80.4 |
> | D1-K10 | 10 % | 94.3 | 11.5 | 82.8 |
> | K1 + 23k external raster (6 runs) | 1 % | 97.1–98.1 | 13.5–15.8 | 81.8–84.6 |
> | D1-K100 | 100 % | 98.6 | 6.3 | 92.3 |
>
> - Result 2 holds: external raster lifts the best candidate to near K100; selection stays at the K1 level
> - **Of the 8–10 points left to K100, 7–9.5 are selection** → this is the improvement point
> - Why external data cannot help selection directly: the scorer learns from PDM labels, which need nuPlan maps → nuScenes / WOMD rows have none

**承接句**："上次 Result 2 说增益在 generator。这次控制了变量、换了更稳的指标，结论不变；而且现在能回答池畑先生上次的问题——改进点在哪：在挑选。"

**中文说明**
- 指标改名必须讲清，否则导师会以为换了结论。说法是"同一个问题，换了一个两 seed 一致的量"（Backup B2）。
- 最后一条是関川さん 11/2 说"大家可能都不知道"的细节（00:30:31–00:33:28）。

---

# Part 3　回访 Result 3：K0 时 selector 漂移

## Slide 7｜Result 3 revisited — where the K0 selector breaks

**Slide**
> **Last time**: "At K = 0 the selector drifts while the candidates stay good." Possible explanation: the scorer is supervised only on raster, and its ranking drifts away from real images
> **Expected** (and found): without any NAVSIM real image, nothing trains real ↔ raster correspondence. **New: where it breaks**
> - raster − real input: +3.7 (epoch 6) → +10.9 (last); almost all of it is picks that score 0 (13.3 % vs 3.5 %)
> - **The selector, not the generator**: selecting with raster features inside the real candidate pool rescues 93 % / 87 % of failures (collision / off-road)
> - **Front-camera object patches**: 6 % of the patches rescue 87 % / 83 % (random patches of the same area: 49 % / 56 %); for collisions, the lead vehicle alone rescues 78 %
> - Vehicles are still readable from real features (AUC 0.994 at K0 and K1), but their encoding no longer matches raster (0.88 vs 0.94)
> - With 1 % real images the gap disappears (Δ ≈ 0 for every K ≥ 1)
> → Same rig, same calibration → an **appearance** failure, on vehicles

**承接句**："上次 Result 3 我给的解释是'scorer 只在 raster 上监督，所以漂移'。这次我去看了它具体坏在哪里。"

**中文说明**
- "预期就是不一致"：関川さん 11/2 说过在他预料之中（01:12:22）。新的是"坏在挑选、坏在 F0 相机的车上"。
- 最后一行接住池畑先生 10/23 的外观差担心（01:17:57–01:22:01）。
- 限制：这个 run 的 scorer 从未在带标签的真图上训练过；e6 是用真图 val 选的（10/23 p9 已经说过）。
- 物体加权对齐会改变问题设定，当前不做（11/2 已与関川さん确认）。被问再说。

---

## Slide 8｜Result 3 extended — does selection need real images, or only labels?

**Slide**
> **Question from S6 + S7**: at K1 the gap is gone, yet selection still lags K100. External raster lacks two things — real images and PDM labels. Which one does the selector need?
> **Test** = RAP's own real-ratio protocol (§4.2): 85,016 NAVSIM samples; 1 % keep real + raster, 99 % raster only **with PDM labels**; same scenes and steps as D1-K100 → **D1-R1**
> **Reading, fixed before the navtest result**: regret near D1-K100 (≈ 6) → labels suffice · near the K1 cells (≈ 14) → selection needs real images
>
> **Early signal** (training-time val on NAVSIM real images; epoch 13 of 20; alignment on; 1 seed)
> | | val PDMS of the pick | val best of 64 | val regret |
> |---|---|---|---|
> | D1-K1 | 0.795 | 0.924 | 0.129 |
> | K1 + WOMD 23k | 0.837 | 0.979 | 0.142 |
> | **D1-R1** (running) | 0.842 | 0.976 | 0.134 |
> | D1-K100 | 0.915 | 0.979 | 0.064 |
> → So far: labeled raster lifts generation to the K100 level; selection stays at the K1 level. navtest ~11/7.
>
> **The same runs test the premise from S3**: D1-R1 with alignment on vs off (off is queued). Rule fixed on 11/3: |Δ| ≤ 1.2 → not needed for PDMS at 1 %; Δ ≤ −2.4 → needed

**承接句**："上次 Result 3 的解释说 scorer 只有 raster 监督。那么反过来：给 raster 加上 PDM 标签够不够？这个问题正好可以用 RAP 论文自己的实验协议来测。"

**中文说明**
- 今天最重要的一页。它是 Result 2（改进点在挑选）和 Result 3（K0 漂移）两条线的交汇，不是新方向。
- 必讲的限制：val 不等于 navtest；训练未完（R1 从 e3 起就平在 0.82–0.84，K100 是一路升到 e15）；只有 alignment 开的版本；单 seed。**开会前重新抽一次数字。**
- 和 RAP 的关系：RAP §4.2 用 MinADE 证明"raster 能替代真图"。如果 MinADE 是对候选取最小，它衡量的是生成端；我们在 PDMS 上补的是挑选端。**会前核实 RAP 图 5 的定义。**

---

# Part 4　回访 Result 4–5：rig 真变了的时候

## Slide 9｜Result 4–5 revisited — the lateral check

**Slide**
> **Last time**: only camera height (and focal) hurt RAP; the loss is in the decoder, not the encoder. Question raised: in the lateral shift, GT and collisions did not move
> **Checked**: the shift changes only the camera's position relative to the ego; ego pose, GT and metric cache are unchanged, and both rigs are scored on the same metric cache = **"same car, camera remounted"** (Sekikawa-san's first reading on 10/23)
> → Result 4 stands under this definition: lateral ±0.5 m −0.6 / +0.1 · longitudinal ±1.0 m −0.5 / −0.4 · height ±0.3 m −5.6 / −5.8 · focal −4.9 / −3.9
> Wording tightened for Result 5: "not in the encoder" is measured (swap rescues 1–2 %, focal control 23–24 %); "in the decoder's geometry" is inferred — the mechanism is not yet identified

**承接句**："上次 T2 的部分留下一个问题：横向平移时 GT 没动。我查了实现。"

**中文说明**
- 只讲定义，不需要重做 GT（10/29 已查代码：只改了 `sensor2lidar_translation`）。
- 主动收紧 Result 5 的措辞：H_par 只确认了 1/3，H_scale 未确认。这比被问出来要好。

---

## Slide 10｜The +α candidates, and what blocks them

**Slide**
> At the end of 10/23 the +α was open. Candidates since then:
> - (a) **warp alignment across rigs** — Sekikawa-san's idea (Slack, 10/28; Sato-sensei's memo the same day): warp rig-A features to rig B and align, so rig B needs no real images
> - (b) **ID token → camera-parameter encoding → Plücker** — Sekikawa-san, 10/23
> - (c) **height/focal-conditioned sampling + height-randomized raster** — from Result 5
>
> **Blocker — data**: all our external data is rendered to the NAVSIM rig (WOMD has no images; nuScenes ≈ NAVSIM rig) → no rig difference to absorb yet
> - Virtual rig (focal via an exact homography on NAVSIM): fast; tests (a)(b), not height
> - Height needs a real source with a different mounting (Waymo / Argoverse with images, as suggested on 10/16 and 10/23); the target stays NAVSIM, so PDMS still applies
>
> Cheapest gates: G1 height-randomized raster only (1 run + T2 eval) · G2 alignment off ≈ on in D1-R1 → drop (a) · G3 dataset ID on the D3-23k cell (1 run)

**承接句**："上次讨论结束时 +α 还是开放的。这两周出现了三个候选，但它们都卡在同一件事上：我们手上还没有 rig 真正不同的数据。"

**中文说明**
- 池畑先生可能不知道 warp 想法，这里一句话交代出处。
- (b) 是関川さん 10/23 的明确建议。"未着手"的理由要讲清：当前所有来源都在同一个 rig 上，ID 只能区分来源；G3 是在现有数据上最便宜的一步。
- G2 的逻辑：D1-R1 只能排除 (a)，不能确认 (a)。它测的是同一 rig、1% 真图下的 alignment，warp 针对的是跨 rig、0 真图。

---

# Part 5　把两半连起来

## Slide 11｜How the two halves connect — and what that means for the papers

**Slide**
> Ikehata-sensei asked on 10/23 how T2 relates to the first half. My answer today:
> | | Results 1–3 (data) | Results 4–5 (rig) |
> |---|---|---|
> | Question | What can raster from other datasets replace for a new rig? | How to train one planner when rigs really differ? |
> | Status | Nearly complete: controls, 2 seeds, K0 diagnosis, D1-R1 running | Motivation only; no test bed, no method |
> | Message | Raster replaces **generation**, not **selection** | — |
>
> - They are two questions. The link (if D1-R1 confirms S8): selection needs real images → abundant real images come from other cars, with other rigs → the rig question
> - **Proposal**: two papers — paper 1 = the left column, paper 2 = the right column (the multi-rig method)
> - Still needed for paper 1: D1-R1 navtest + alignment off; R10 (how much real the selector needs); step control seed 1
> - Limits of paper 1: every external source shares the target rig · navhard not improved · D1-K10 / D1-K100 / step control single-seed · red line vs official: D1-K100 92.30 vs 93.80, D2-K100 not run

**承接句**："上次池畑先生问 T2 和前半部分是什么关系，我当时答不上来。现在我的回答是：它们是两个问题，可以分成两篇，中间的连接点就是 S8 的那个实验。"

**中文说明**
- 这一页把 10/23 遗留的"前后半没连上"变成了今天的结构性回答，所以分两篇不会显得突然。
- 定位风险：佐藤先生 9/30 说 Proposal 1（target-rig raster）novelty 弱，10/16 要求 Single 也要有一捻り，10/23 说 paper 1 是多 rig 方法。要讲清论文 1 的贡献是"替代的边界"这个发现，而不是配方。
- 10/23 讨论里"主贡献 = rig 几何，alignment = 副"没有定论（01:58:20 vs 02:04:25）。按今天的分法：rig 几何对应论文 2，alignment 由 D1-R1 决定在论文 1 里占多大分量。

---

## Slide 12｜Decisions I'd like today

**Slide**
> 1. **Paper 1 as an analysis paper** (generation vs selection) — acceptable as the first paper?
>    My view: yes — nearly complete, and it qualifies RAP's own real-ratio claim. If a method is required, the closest candidate is on the selector side; it is not designed yet (weeks, not days)
> 2. **Venue and deadline** → which "still needed" runs must go in
> 3. **Compute, next ~10 days** — my proposal: D1-R1 off → R10 → step control seed 1 → G1 → G3 (ID) → 3 ratios
> 4. **Test bed for paper 2**: virtual focal rig first, or get a real different-rig data set now?

**承接句**："最后，有四件事想请大家判断。每一条我都先说自己的想法。"

**中文说明**
- 10/23 最后是一页空白的 Discussion，导师不知道你想要什么决定。今天反过来：先给推荐，再请判断。
- 3 档比例放在最后：它调的是配方，不改变"生成 vs 挑选"的结论。如果导师认为必须做，就往前挪。

---

# Backup

- **B1** T2 lateral 定义细节：`sensor2lidar_translation`；幅度横向 ±0.5 m、纵向 ±1.0 m、高度 +1.0 / −0.7 m，另有 ±0.3 / ±0.5 m 剂量点；用 `run_t2_swap_v1.py` 在同一 metric_cache 上打分。
- **B2** capture → regret：s0 和 s1 对 womd 的 scorer 份额分别给出约 0 和 +3.2–4.0；regret 在两个 seed 上一致。
- **B3** 完整结果表：v1 / v2 / navhard（s1/s2）/ oracle / mean64 / regret，s0 和 s1 分列。
- **B4** navhard 与子指标：
  - navhard 两 seed 均值：nusc 28.3、womd 28.9、D3 29.9，≈ D1-K1 的 29.9，低于 D1-K10 的 36.1。
  - 原因是"开得快 ↔ stage 2 违规"的取舍。
  - EC 的 seed 间标准差 8.1。
- **B5** stepmatch 细节：e9；spearman 0.271；oracle +1.57；regret 变差 1.07。
- **B6** K0 个案图（token 04e3f71d 等）。
- **B7** D1-R1 的 val 曲线，与 D1-K100 对照。
- **B8** alignment 现有证据：
  - RAP §4.2：MinADE；1% 时 w/o 1.125 vs w/ 1.05。
  - 8/26：2,498 样本、400 步，75.82 vs 75.80，CI 约 ±3.5 → 不能下结论。
- **B9** 10/23 p9 留下的"strict K0 需要不用真图的选 ckpt 规则"：未做。
