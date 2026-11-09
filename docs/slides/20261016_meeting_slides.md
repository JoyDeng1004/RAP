# 2026-10-16 Meeting Slides（逐页文字稿）

每页三块：**Slide（英文，直接放到页面上）** / **中文说明（给你自己看：这页要讲清什么、数字来源、可能被问到什么）** / **日本語発言稿（照着讲）**。

整体顺序：开场 → Part 1 回顾 → Part 2 问题设定（scope → 先行研究 → 实验矩阵 → 两条 story 的前提）→ Part 3 T1 → Part 4 ViRA → Part 5 讨论。
每个 Part 最后一页都有一句"桥"，把下一部分要回答的问题抛出来。

数字来源：E1–E4 = 9/29–10/2 的 NAVSIM 评测；T1 = `eval/t1_results/t1_summary_official.md` 和 `t1_probe_official.md`（官方 v1 权重，navtest 1,501 token 子集，log 级 bootstrap 95% CI）。

---

## Slide 0｜Title

**Slide**
> **Camera-Rig Transfer for RAP: Recap, Scope, and a First Diagnosis (T1)**
> Joy Deng · 2026-10-16

**中文说明**
开场页，不用停留。

**日本語発言稿**
本日はよろしくお願いします。前回のミーティングからの進捗をご報告します。E1〜E4 の振り返り、問題設定とスコープ、昨日回した T1 実験の結果、それから ViRA の位置づけの順にお話しして、最後に 3 点ご相談させてください。

---

## Slide 1｜Today

**Slide**
> **Agenda**
> 1. Recap: E1–E4
> 2. Problem, prior work, and our scope
> 3. T1: does a rig change hurt RAP — and where?
> 4. Where ViRA fits
> 5. Discussion
>
> **Three things I'd like to decide today**
> ① **Zero-shot definition** — may we use target-domain logs and maps (but no target-rig images)?
> ② **Test bed** — which target rig makes the rig effect measurable?
> ③ **Story** — which premise do we build on: Story 1 or Story 2?

**中文说明**
先把要决定的三件事亮出来，后面每一部分都是在为这三件事准备证据。佐藤先生听的时候会知道"这段为什么要讲"。

**日本語発言稿**
先に、今日ご判断いただきたいことを 3 つ挙げておきます。1 つ目は zero-shot の定義で、target 側のログや地図を使ってよいかどうかです。2 つ目は、rig の違いをきちんと測れる target rig をどう用意するか。3 つ目は、ストーリー 1 と 2 のどちらの前提に乗るかです。このあとの話は、全部この 3 つにつながるように並べています。

---

# Part 1　Recap

## Slide 2｜What each run trains on

**Slide**
> **Recap: E1–E4 (all evaluated on NAVSIM)**
>
> | Run | Training data | NAVSIM real images |
> |---|---|---|
> | E1 | nuScenes, own rig (real + raster) | 0 |
> | E2a | NAVSIM 10% (98 logs, real + raster) | 10% |
> | E2b | E1 weights → fine-tune on NAVSIM 10% | 10% |
> | E3 | NAVSIM 10% + nuScenes (own rig) + nuScenes scenes rendered in the NAVSIM rig | 10% |
> | **E4 (= P1)** | NAVSIM 10% + nuScenes scenes rendered in the NAVSIM rig (raster only) | 10% |

**中文说明**
- E4 是 9/30 定下的 Proposal 1 的第一个干净实例：外部数据只以 target rig 的 raster 形式进来。
- E3 只是"朴素的多 rig 混合"，不是提案法（9/30 佐藤先生定调）。被问到时这样回答。

**日本語発言稿**
まず前回の実験の振り返りです。E1 は nuScenes だけで学習していて、NAVSIM の実画像は 1 枚も見ていません。E2a は NAVSIM の 10% だけ、E2b は E1 の重みから同じ 10% でファインチューニングしたものです。E3 と E4 は、NAVSIM 10% に外部データを足したものです。E4 は、nuScenes のシーンを NAVSIM の rig でラスタにして足しただけなので、提案 1 のいちばんきれいな形になっています。

---

## Slide 3｜Results and what they tell us

**Slide**
> **Results**
>
> | Run | v1 PDMS | v2 EPDMS | navhard (stage 1 / stage 2) |
> |---|---|---|---|
> | Official RAP (100% NAVSIM) | 93.8 | 86.7 | 36.2 (74.1 / 48.7) |
> | E2a | 82.8 | 76.9 | 36.1 (67.2 / 53.7) |
> | E2b | 87.3 | 82.2 | 33.0 (66.3 / 49.4) |
> | E3 | 89.4 | 83.3 | 32.1 (67.0 / 47.2) |
> | E4 | 88.7 | 82.7 | 34.2 (69.4 / 49.3) |
> | E1 (zero-shot) | 61.0* | 59.1* | 14.3* |
>
> \*medoid selection; E1's scorer is untrained (nuScenes has no nuPlan map)
>
> - **navtest:** external data helps — **+4.5 to +6.6** over E2a (v1, 95% CI excludes 0)
> - **navhard:** no consistent gain. The drop is only in stage 2, where even official RAP (48.7) is below E2a (53.7) → not yet evidence of negative transfer (no paired test yet)
> - Remaining gap to official: mostly the candidate pool, not the scorer
> - Caveats: E2a ran ~680 optimizer steps vs. 2,480 for E4; one seed per run
>
> → **These runs change the dataset, not the rig** (NAVSIM rig ≈ nuScenes rig). Whether the rig itself matters is still unknown.

**中文说明**
- **改掉了"navhard 反而降"的说法**：stage 1 上 E4 是 p010 模型里最高的（69.4 vs E2a 67.2）；下降只在 stage 2，而全量数据的官方 RAP 在 stage 2 也低于 E2a。stage 2 的场景权重又取决于 stage 1 选出的终点，有混杂。所以只能说"没有稳定收益"。
- 配对 CI：E2b−E2a +4.47 [3.77, 5.15]，E3−E2a +6.56 [5.73, 7.33]，E4−E2a +5.83 [5.04, 6.56]。
- 可能被问："外部数据的收益是不是只是步数多？" → 老实说有这个混杂（E2a 约 680 步，E4 2,480 步），下一步补一个步数对齐的 E2a。
- 最后一句是全场的第一座桥：外部数据有用，但 rig 有没有影响，这组实验回答不了。

**日本語発言稿**
結果です。navtest では、外部データを足すと E2a より 4.5〜6.6 点上がっていて、信頼区間も 0 を含みません。一方、navhard では安定した改善は見られません。ただ、下がっているのは stage 2 だけで、全データで学習した公式 RAP でも stage 2 は E2a より低いです。なので、今の時点で負の転移とまでは言えないと考えています。注意点が 2 つあって、E2a は学習ステップ数が E4 の 4 分の 1 くらいしかないことと、どれも seed が 1 つだけであることです。
そして一番大事な点ですが、これらの比較で変わっているのはデータセットで、rig ではありません。NAVSIM と nuScenes のカメラ配置はほぼ同じなので、rig そのものの影響はまだ分かっていません。

---

# Part 2　Problem, prior work, scope

## Slide 4｜Goal and scope

**Slide**
> **Goal: a planner that keeps working when the camera rig changes**
>
> **Two settings**
> - **A. Full camera-rig zero-shot** — the new rig is unknown at training time
>   e.g., Hashimoto et al. '26 → performance still not there; **out of scope** (10/9: needs depth → canonical 3D, beyond RAP)
> - **B. Full real-image zero-shot** — the new rig's geometry is known, but **0 real images** from it ← **our scope**
>
> **How to make training data for B**
> | Approach | Examples | Cost |
> |---|---|---|
> | 3D-box + map raster | RAP | **cheap** ✓ |
> | 3DGS / novel-view synthesis | CoIn3D's augmentation | high |
> | Simulator | CARLA, AlpaSim | high, sim-to-real gap |
>
> Evaluation: 0 / few (e.g., 1%, 10%) / 100% target real images (10/9)

**中文说明**
- 这页对应関川さん草稿里的 [やりたいこと] 和 [スコープ]：full camera-rig zero-shot（Hashimoto）→ 性能还不够；simulation → 计算成本高；所以落在 full real-image zero-shot + 便宜的 raster。
- "rig 几何已知"是 10/9 定的（DENSO/Toyota 场景下 rig 信息是已知输入）。
- zero / few / full 只用在"target 真实图有多少"这一轴上，不要再用来描述 rig。

**日本語発言稿**
ここから問題設定です。やりたいことは、カメラ rig が変わっても性能が落ちない planning です。
設定は大きく 2 つあります。1 つは、新しい rig が学習時に全く分からない設定で、Hashimoto らがこれを狙っていますが、まだ性能は十分ではありません。前回、これは RAP の枠を超える大きな課題なのでスコープ外、という話になりました。
もう 1 つは、新しい rig の幾何は分かっているけれど、その rig で撮った実画像は 1 枚もない、という設定です。私たちはこちらを扱います。学習データの作り方としては、RAP のような 3D ボックスと地図のラスタ、3DGS、シミュレータがありますが、後の 2 つは計算コストが高いので、安いラスタを使います。評価は前回決めたとおり、実画像 0、少量、100% の 3 段階です。

---

## Slide 5｜Prior work: what is solved, what is open

**Slide**
> **Prior work**
>
> | Work | Task | How it handles a new rig | Solved | Still open |
> |---|---|---|---|---|
> | Hashimoto et al. '26 (2604.00597) | Planning | No augmentation: depth-based 3D position embedding + 3D foundation-model features | Robust to pitch / height changes without new data | Smaller gains under **longitudinal translation**; open-loop benchmark |
> | VR-Drive (NeurIPS '25) | Planning | Novel-view images + viewpoint-consistent distillation | Viewpoint benchmark (pitch / height / depth) | Open-loop on nuScenes; closed loop only in CARLA |
> | CoIn3D (CVPR '26) | 3D detection | Camera-aware feature modulation (focal, ground depth, Plücker) + novel-view augmentation | Rig transfer across nuScenes / Waymo / Lyft | Detection, not planning |
> | Klinghoffer et al. (ICCV '23) | BEV segmentation | Re-render collected data to the target rig | Same motivation as ours: no new data collection | Perception only; CARLA evaluation |
> | RAP ('25) | Planning | Raster (boxes + map) from any viewpoint; raster ↔ real alignment | Cheap, scalable view synthesis | **Never tested across rigs** |
> | CARLA / AlpaSim | Simulation | Any rig | Full control | Cost; sim-to-real gap |
>
> **Gap (as far as I found):** planning on real images, with a known new rig and 0 real images from it, using only cheap raster data

**中文说明**
- 这页回答你要的"先行研究解决了什么、没解决什么"。
- 前 3 行和 RAP、模拟器来自関川さん草稿；**VR-Drive 和 Klinghoffer 是 10/14 补充调研的**，审稿人一定会提，尤其 Klinghoffer 的动机和我们几乎一样，主动摆出来比被指出好。
- Hashimoto 的"纵向平移增益小"是他们摘要里自己写的；他们用的是 VR-Drive 的 benchmark。
- CoIn3D 的增强方式摘要里写的是 training-free 的动态新视角合成，10/9 会上大家把它当作 3DGS 类；如果被追问细节，就说"增强部分需要再看原文"。
- 最后一行 Gap 加了"as far as I found"，不要说成"没有人做过"。

**日本語発言稿**
関連研究が何を解決していて、何が残っているかを表にまとめました。
Hashimoto らはデータ拡張なしで、深度から作った 3D 位置と 3D 基盤モデルの特徴を使っていて、pitch や高さの変化には強くなっていますが、前後方向の並進では効果が小さいと自分たちで書いています。VR-Drive は新視点画像と一貫性の蒸留を使っていて、評価は nuScenes 上の open-loop、closed-loop は CARLA だけです。CoIn3D はデータセットをまたいだ rig 転移をしていますが、タスクは 3D 検出です。Klinghoffer らは、集めたデータを target rig に描き直すという、私たちとほぼ同じ動機の研究ですが、これも認識タスクで、評価は CARLA です。RAP はどの視点からでも安くラスタを描けますが、rig をまたいだ評価はされていません。
私が調べた範囲では、実画像での planning を、安いラスタだけで新しい rig に合わせた例はまだありません。

---

## Slide 6｜Experiment matrix for our scope

**Slide**
> **What we have, what's missing** (revised from the Notion draft)
>
> | Run | Training data | Target real images | Target rig in training | v1 / v2 / navhard |
> |---|---|---|---|---|
> | Official RAP | NAVSIM 100% + RAP augmentation | 100% | real + rendered | 93.8 / 86.7 / 36.2 |
> | E2a | NAVSIM 10% | 10% | real + rendered | 82.8 / 76.9 / 36.1 |
> | E4 | NAVSIM 10% + nuScenes rendered in NAVSIM rig | 10% | real + rendered | 88.7 / 82.7 / 34.2 |
> | E3 | E4 + nuScenes (own rig) | 10% | real + rendered | 89.4 / 83.3 / 32.1 |
> | E1 | nuScenes (own rig) | 0 | no (calibration only at test) | 61.0* / 59.1* / 14.3* |
> | **missing** | nuScenes (own rig) + nuScenes rendered in NAVSIM rig | **0** | **rendered only** | — ← **the cell our scope needs** |
> | missing | k-shot curve: 0 / 1 / 10 / 100% | | | — |
>
> **Why the missing cell won't tell us much yet**
> - NAVSIM rig ≈ nuScenes rig: height 1.43–1.49 m vs. ~1.5 m; focal after resize 618 vs. 608 px; same yaw layout. Differences: rear FOV, distortion, calibration quality — and removing the rear camera changes nothing (T1)
>   → rendering "into the target rig" barely changes the data → **expected ≈ E1**
> - nuScenes has no nuPlan map → **the scorer cannot be trained** without target data

**中文说明**
- 这是 Notion 表的改写：拆成两个轴（target 真实图有多少 × 训练时用没用 target rig），已跑的格子填数字。
- 関川さん在草稿里批注过"P1 是 few-shot，和 full real-image zero-shot 的目标不一致"。这页把缺口具体化：真正对上 scope 的是"0 张真实图 + 只有渲染"那一格，而它现在还没跑。
- 可能被问："那为什么不先把这格跑了？" → 跑了大概率 ≈ E1，花二十多个 GPU 小时只能确认一个 ≈0；要先换 target rig 或解决 scorer 监督（→ Slide 15 的决定 ① ②）。
- E1 的 * 是 medoid 选法，scorer 没训练过。

**日本語発言稿**
このスコープで実験を整理し直したのがこの表です。Notion の表を、target の実画像がどれだけあるかと、学習中に target rig を使ったかどうかの 2 軸で書き直して、すでに回したものには数字を入れました。
足りないのは、実画像 0 で target rig のラスタだけを使う格子で、これがスコープのど真ん中です。関川さんにご指摘いただいた、提案 1 が few-shot でゴールとずれている、という点もここに対応します。
ただ、今のデータの組み合わせでこれを回しても、E1 とほとんど同じになると予想しています。理由は 2 つあって、NAVSIM と nuScenes の rig がほぼ同じなので、target rig で描き直してもデータの中身がほとんど変わらないこと。それから、nuScenes には nuPlan の地図がないので、scorer が学習できないことです。

---

## Slide 7｜Two stories, one shared premise

**Slide**
> **Two candidate stories** (10/9 draft with Sekikawa-san)
>
> **Story 1 — data**
> RAP gives cheap data augmentation and could adapt to a new rig.
> But naively adapting RAP to a new rig fails (zero-shot does not work).
> → Proposal P1 · issue: P1 is few-shot, which does not match the "full real-image zero-shot" goal
>
> **Story 2 — model**
> Like Hashimoto et al., avoid augmentation and retraining.
> But it does not generalize to translation, because [the 3D position embedding is tied to the camera position — to confirm].
> → Proposal: 3D voxel …
>
> **What each story assumes**
> - Story 1: **a rig change really hurts RAP.** Current evidence: E1 zero-shot 61.0 vs. 93.8 — but this mixes dataset shift, an untrained scorer, and an almost identical rig
> - Story 2: **translation is the hard case.** Hashimoto et al. report it; for RAP it is untested
>
> → Both stories need one fact first: **does a rig change hurt RAP, and where?** → T1

**中文说明**
- 两条 story 原样来自関川さん的草稿；我只加了最下面"各自的前提是什么、现在的证据是什么"。
- Story 2 的 [xxx] 我填的是推测：Hashimoto 摘要说纵向平移增益小；10/8 读到他们的反事实实验指向"依赖外参的显式 3D 位置编码"本身导致平移失效。**会上请関川さん确认这个填法**，所以页面上写了 "to confirm"。
- 这页是第二座桥：两条 story 都要先回答"rig 变化伤不伤 RAP、伤在哪"→ T1。

**日本語発言稿**
関川さんと整理したストーリーが 2 つあります。
ストーリー 1 は、RAP を使えば安くデータを増やせて、rig の変化にも対応できそうだ、ただし単純に RAP を別の rig に持っていくだけではうまくいかない、というものです。ただ、提案 1 は few-shot なので、実画像 0 というゴールとずれています。
ストーリー 2 は、Hashimoto らのようにデータ拡張や再学習を避けるアプローチですが、並進には汎化できない、だから 3D voxel で、という流れです。並進に弱い理由のところは、3D 位置埋め込みがカメラ位置に依存しているから、と仮に入れていますが、関川さんに確認させてください。
ここで、どちらのストーリーも「rig が変わると RAP が壊れる」ことを前提にしています。ストーリー 1 の根拠は今のところ E1 の zero-shot の結果ですが、これはデータセットの違いや、scorer が学習されていないことと混ざっています。ストーリー 2 の並進は、RAP ではまだ測っていません。そこで、まずこの前提が本当かどうかを確かめたのが、昨日の T1 です。

---

# Part 3　T1

## Slide 8｜T1: setup and raw results

**Slide**
> **T1: if the camera rotates, shifts, or zooms — how much does RAP lose, and why?**
>
> Setup
> - Official RAP weights, **no retraining**
> - NAVSIM navtest subset, 1,501 tokens (raw PDMS 93.8)
> - Exact transforms on **real images** (lens model kept); calibration updated to match → the model "knows" the camera changed
> - 95% CI: log-clustered bootstrap
>
> | Change | ΔPDMS |
> |---|---|
> | Pitch −10° / −5° / +5° | **−28.4** / −11.0 / −7.0 |
> | Image shifted up 272 px (= pitch −10° on screen) | −32.3 |
> | Focal ×0.8 / ×1.25 | −3.5 / −2.1 |
> | Principal point ±5% (lateral) | −2.1 / −3.2 |
> | Roll ±3°, side-camera yaw ±5° / ±10° | ≈ 0 |
> | Remove front cam F0 / side L0, R0 / rear B0 | **−48.9** / −0.4 / **0.0** |
>
> At first glance: **pitch looks like a big problem.**

**中文说明**
- "精确变换"：旋转和内参用单应变换，不需要 NVS，所以是真实图、几何精确。
- 去掉 B0 恰好 0（逐 token 相同）：轨迹 query 只在候选航点处采样，后视永远采不到。侧向 yaw 不敏感也是"不用"，不是"鲁棒"。
- 这页故意停在"看起来 pitch 是大问题"，下一页反转。

**日本語発言稿**
ここから T1 です。知りたかったのは、カメラが回ったり、ずれたり、ズームしたりしたときに、RAP がどれくらい落ちるのか、そしてなぜ落ちるのか、です。
再学習はせず、公式の RAP の重みをそのまま使いました。NAVSIM の実画像に回転や内部パラメータの変化を正確にかけて、キャリブレーションも同じだけ更新しています。つまり、モデルにはカメラが変わったことがちゃんと伝わっています。
素の結果だけ見ると、pitch を −10 度にすると 28 点落ち、前方カメラを消すと 49 点落ちます。逆に、roll や横のカメラの yaw はほとんど効かず、後方カメラは消しても全く変わりません。これだけ見ると、pitch が大問題に見えます。

---

## Slide 9｜It's the black border

**Slide**
> **Most of the pitch drop is a black-border artifact**
>
> 1. **Control:** keep exactly the visible content, at its original position → Δ ≈ 0 (|Δ| ≤ 0.3)
>    → the loss comes from *where content sits*, not from content lost
> 2. **Where RAP looks:** in round 1, ~**94%** of the front-camera attention is placed **outside the image** on purpose — it adds 0, a way of "not reading". Only ~6% lands on content
> 3. When the image shifts, that "outside" band becomes **black fill inside the image** → the model now reads black pixels (15% of attention at pitch −10°)
> 4. **Test:** count samples on black fill as outside (img_oob)
>
> | | before | after |
> |---|---|---|
> | Pitch −10° | −28.4 | **−0.3** |
> | Pitch −5° | −11.0 | −0.9 |
> | Shift up 272 px | −32.3 | −0.7 |
> | Remove F0 | −48.9 | −18.7 |
> | Focal ×0.8 | −3.5 | −4.2 (not a border effect) |
>
> → **~97% of the pitch drop is the black border**

**配图**
- `eval/t1_results/figs/sca_points/example_ppy_m7r_1dd4683469e457c5.webp`（三联图：raw 时采样点大量在图外下方；平移/pitch 后同一批点落进红框黑带）
- `eval/t1_results/figs/t1p_official/overview.webp`（红 = 原始掉分，蓝 = img_oob 后；一眼能看出 pitch/纵向平移被救回，focal 和缺 F0 没被救回）

**中文说明**
- 这是 T1 最重要的发现，我们内部叫"几何注意力汇点"：模型学会把采样偏移推到图外，用恒为 0 的区域实现"不读"。
- 缺 F0 救回 30 分，剩下 −18.7 是真实的信息损失。
- 焦距的掉分 img_oob 救不回，说明那是尺度效应，不是黑边。
- 可能被问："真实的新 rig 没有黑边，那这个结论有什么用？" → 正是下一个问题：新 rig 下这些"汇点"区域会出现真实内容（例如 pitch 往下时图下方是真实地面），这时有没有害？T1 回答不了，要靠 T2。

**日本語発言稿**
でも、中身を見るとそうではありませんでした。
まず対照実験として、見えている内容はそのままで、位置も元のままにすると、ほぼ 0 点です。つまり、内容が失われたのではなく、内容の位置がずれたことが原因です。
次に、RAP がどこを見ているかを調べると、1 回目の読み込みでは、前方カメラの注意の約 94% がわざと画像の外に置かれていました。画像の外は値がゼロなので、「読まない」ための場所として使っているようです。画像がずれると、この「外」の帯が画像の中の黒い埋め込み領域になって、モデルが黒い画素を読んでしまいます。
そこで、黒い領域に落ちたサンプルを画像の外として扱うと、pitch −10 度の 28 点の低下が 0.3 点まで戻りました。pitch の低下の 97% くらいは、この黒い縁による見かけの効果です。焦点距離の変化だけは、これでは戻らないので、別の原因です。

---

## Slide 10｜Is DINOv3 the problem? (revised Conclusion 1)

**Slide**
> **Draft claim:** "DINOv3 is robust enough to rig changes"
> **Revised:** "**The drop does not come from DINOv3**"
>
> Test: feed the decoder the features an ideal shift-equivariant backbone would give, and compare with the real backbone
>
> | Shift | Total Δ | Backbone share |
> |---|---|---|
> | Vertical (8 sizes) | −1.3 to −32.3 | −0.7 to +0.2 (mostly n.s.) |
> | Lateral principal point ±5% | −2.1 / −3.2 | −1.4 / −2.3 |
>
> - DINOv3 features **move with the content**: cosine 0.94–0.99 after alignment (0.37–0.84 without)
> - The divergence starts at the **first trajectory-query read**, not in the backbone
> - Limits: lateral share may partly be interpolation error (2.4-patch shift; to confirm) · focal (scale) not decomposed · camera translation not tested
>
> → "**Not the bottleneck**", rather than "robust"

**配图**
`eval/t1_results/figs/t1p_official/ladder.webp`（ViT/FPN 的余弦都还在 0.95 左右，曲线是从 query r1 才掉下去的；"equivariant backbone" 那条和真实主干几乎重合）

**中文说明**
- **为什么改说法**：
  1. "鲁棒"一般指特征不变；我们测到的是"平移等变"——特征跟着内容一起移动，这对下游按标定投影采样恰好是对的；
  2. 只测了纯平移，焦距（尺度）没有拆，相机平移没测，所以不能推广成"对 rig 扰动鲁棒"；
  3. 横向主点那里主干占了大头（−2.3 / −3.2），只是总量小。
- 可能被问："横向为什么主干份额大？" → ±5% 是 2.4 个 patch 的非整数平移，"理想特征"要靠双线性插值近似，误差可能被算进主干份额；需要用整 patch 平移再确认。

**日本語発言稿**
次に、草稿の結論 1「DINOv3 は rig の変化に十分頑健」を、データで確かめました。
理想的な、平行移動にそのままついてくる主干だったら出てくるはずの特徴をデコーダに直接入れて、本物の主干と比べました。縦方向のずれでは、主干の取り分は最大でも 0.7 点で、ほとんど有意ではありません。DINOv3 の特徴は、ちゃんと内容と一緒に動いていて、ずれが始まるのは主干ではなく、軌道クエリが最初に画像を読むところです。
横方向の主点のずれだけは、主干の取り分が 1.4〜2.3 点ありますが、補間の誤差かもしれないので確認中です。また、焦点距離のような拡大縮小と、カメラの並進はまだ分解していません。
なので、「頑健」と言うより、「低下の原因は DINOv3 ではない」と言うのが正確だと思います。

---

## Slide 11｜Generation vs. selection (revised Conclusion 2)

**Slide**
> **Which side breaks: generating candidates or selecting one?**
>
> Test: keep the original 64 candidates; only the scorer re-ranks them with the new images (= selection-only damage)
>
> | Change | Total Δ | Selection-only Δ | Best-of-64 Δ | Avg-of-64 Δ | Rank corr. (raw 0.54) |
> |---|---|---|---|---|---|
> | Pitch −10° | −28.4 | −8.6 (30%) | −12.4 | −37.2 | **−0.29** |
> | Pitch −5° | −11.0 | −3.2 (29%) | −1.3 | −15.3 | 0.22 |
> | Focal ×0.8 | −3.5 | −0.8 (23%) | −0.1 | −5.4 | 0.45 |
> | Focal ×1.25 | −2.1 | −1.1 (52%) | −0.3 | −2.1 | 0.44 |
> | Remove F0 | −48.9 | −11.6 (24%) | −15.0 | −51.6 | −0.22 |
>
> - **Most of the loss (~70%) comes with regenerated candidates:** the pool gets worse on average
> - **The scorer also breaks:** with the same candidates it loses up to 8.6 pts, and its ranking turns negative
> - Caveat: these big drops are triggered by the black border. After the fix, pitch costs ≤ 1 pt in total, so this split describes *how RAP fails when it fails*

**中文说明**
- **你的结论 2 基本成立**，我补了数字和前提。
- 最后一列是"固定候选时 scorer 分数和真实 PDMS 的 Spearman"。
- "选择端也会坏"要说清楚：不是 scorer 天生更脆弱，而是它读到的特征同样被黑边污染。焦距 ×1.25 时选择端约占一半，但数字小、CI 宽，别过度解读。
- 可能被问："Best-of-64 才掉 12 分，为什么总分掉 28？" → 候选里还有好的，但平均质量掉得多（−37），scorer 在更差的池子里更难挑对。

**日本語発言稿**
結論 2、生成と選択のどちらが壊れるか、です。
元の 64 本の候補軌道はそのままにして、scorer だけに新しい画像で並べ替えさせると、選択だけの損失が分かります。pitch −10 度では、全体 28 点の低下のうち、選択だけで 8.6 点、だいたい 3 割です。残りの 7 割は、候補を作り直したときに候補全体の質が落ちることから来ています。ただ scorer も壊れていて、順位の相関は 0.54 からマイナスになります。
注意点として、これは黒い縁のせいで大きく落ちたときの内訳です。黒い縁を処理すると、pitch による低下は全体で 1 点以下になるので、この内訳は、RAP が実際に失敗したときにどう失敗するか、という話です。

---

## Slide 12｜What T1 tells us

**Slide**
> **T1 takeaways**
> - Within what we can measure exactly (rotation, intrinsics), RAP with known calibration loses **≤ 1 pt** for pitch / vertical shift and **1–4 pts** for focal / lateral shift, once invalid regions are handled
> - The backbone is not the bottleneck; **the issue is where the decoder samples**
>
> **Not answered yet**
> 1. A new rig puts **real content** where RAP learned "not to read" — harmful? → **T2** (raster with full view vs. same rig with black fill)
> 2. **Camera translation** (height, forward) — needs raster or novel-view synthesis
>
> **For the stories**
> - Story 1: the premise "a rig change breaks RAP" is **not supported for rotation / intrinsics**. It has to come from T2 or translation
> - Story 2: translation is **exactly the axis we have not tested** → still open

**中文说明**
- 全场最关键的一页：数据告诉我们 Story 1 的前提在旋转/内参上站不住，Story 2 的平移轴还空着。
- 用词要稳："not supported for rotation / intrinsics"，不要说"Story 1 不成立"。T2 如果显示真实内容进入汇点区有害，Story 1 的前提可以换一个更具体、更新的版本回来。
- 这是第三座桥：如果瓶颈不在特征，那"更好的表示"（ViRA）该放在哪？

**日本語発言稿**
T1 のまとめです。正確に測れる範囲、つまり回転と内部パラメータについては、無効な領域をきちんと扱えば、キャリブレーションが分かっている RAP は、pitch や縦のずれで 1 点以下、焦点距離や横ずれで 1〜4 点しか落ちません。問題は主干の特徴ではなく、デコーダがどこからサンプルするかにあります。
まだ答えていないことが 2 つあります。1 つは、新しい rig では、RAP が「読まない」と決めていた場所に本物の内容が入ってくるので、それが害になるかどうか。これを T2 で調べます。もう 1 つはカメラの並進で、これはラスタか新視点合成が必要です。
ストーリーとの関係で言うと、ストーリー 1 の「rig が変わると RAP が壊れる」という前提は、回転と内部パラメータについては今のところ支持されていません。逆に、ストーリー 2 の並進は、まさに私たちがまだ測っていない軸です。

---

# Part 4　ViRA

## Slide 13｜ViRA (arXiv 2610.09695)

**Slide**
> **ViRA — "Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?"**
> - **Training only:** align the planner encoder's intermediate features to a frozen VFM (default DVGT; also VGGT, DA3, DINOv3, SAM3). Two losses: spatial dependency (KL over attention) + scene semantics (contrastive). Inference unchanged
> - Results (NAVSIM v2 EPDMS)
>   - RAP\* (their re-implementation: ResNet-34, no RAP augmentation): 72.8 → **83.7 (+10.9)**; navhard 32.8 → 49.2
>   - RAP\* with a DINOv3-L backbone: 83.8 → **86.3 (+2.5)** with VGGT target — about official RAP (86.7)
>   - Random-init target: 71.1 → the target matters
> - **No camera-rig experiments.** Zero-shot closed loop on HUGSIM (trained on NAVSIM only): 7.3 → 10.3 HD-Score; camera setup not stated (to check)
> - Their navhard uses the SimScale two-stage protocol → check the split before comparing with our numbers

**中文说明**
- 我们的 RAP 用的是冻结的 DINOv3-H+，所以可参考的增益是 +2.5，不是 +10.9。佐藤先生如果看到 +10.9 很可能会问，这里先讲清楚。
- HUGSIM 的场景是从其他数据集重建的，可能隐含跨数据集甚至跨相机，但论文没写相机设定，我还没核对，不要当成 rig 证据。

**日本語発言稿**
ViRA についてです。学習のときだけ、planner の中間特徴を凍結した VFM の特徴に合わせる手法で、推論は何も変わりません。
RAP を再実装した RAP\* では 10.9 点上がっていますが、この RAP\* は ResNet-34 で、RAP 本来のデータ拡張も使っていません。主干を DINOv3-L にすると、改善は 2.5 点で、ほぼ公式 RAP の値になります。私たちの RAP は凍結した DINOv3 を使っているので、参考にすべきなのは 10.9 点ではなく 2.5 点の方です。
rig を変えた実験はありません。HUGSIM での zero-shot 評価はありますが、カメラ設定が書かれていないので、確認が必要です。

---

## Slide 14｜Where ViRA fits

**Slide**
> **Positioning**
>
> | | Single rig | Across rigs |
> |---|---|---|
> | **Change the data** | RAP raster augmentation | Klinghoffer '23 (re-render), CoIn3D augmentation, **Ours P1** (external logs rendered in the target rig) |
> | **Change the model / representation** | **ViRA** | Hashimoto '26, VR-Drive '25, CoIn3D feature modulation |
>
> - ViRA = **representation axis, single rig**. It improves data efficiency, not rig transfer (never tested)
> - T1: the rig problem is not in the features but in **where the decoder samples** → better features alone are unlikely to fix it
> - **Role in our paper:** a baseline P1 must beat ("external data vs. VFM alignment, same target data"), and something we can stack on P1
> - Open hypothesis: geometry VFM targets (VGGT / DVGT) might help with translation — untested

**中文说明**
- 回答你的问题"ViRA 放在哪个轴"：表示轴、单 rig、服务 few-shot 下的数据效率。它不是 rig 迁移的抓手，而是 P1 必须比的 baseline，也可以叠加。
- 和 T1 串起来：T1 说问题在"读哪里"而不在"特征好不好"，所以单靠更好的表示不太可能解决 rig 问题。
- 留一个开放假设（几何 VFM 可能对平移有用），给 Story 2 留口子，但明确说没测过。

**日本語発言稿**
ViRA をどの軸に置くかです。データを変えるのかモデルや表現を変えるのか、そして単一の rig か rig をまたぐのか、の 2×2 で整理しました。
ViRA は表現の軸で、単一 rig の手法です。改善しているのはデータ効率で、rig の転移は試されていません。T1 でも、rig の問題は特徴そのものではなく、どこを読むかにあると分かったので、特徴を良くするだけでは解決しにくいと思います。
私たちの論文では、提案 1 が必ず比べるべきベースライン、つまり「外部データを足すのと VFM に合わせるのと、同じ target データでどちらが効くか」の比較相手で、提案 1 と組み合わせることもできます。ただ、VGGT のような幾何の基盤モデルを先生役にすると並進に効くかもしれない、という可能性は残っていて、これは仮説です。

---

# Part 5　Discussion

## Slide 15｜Decisions and next steps

**Slide**
> **① Zero-shot definition**
> - Option a: no NAVSIM logs at all (current Notion draft)
> - Option b: target-domain logs and maps allowed, **no target-rig images**
>   - Having a log ≠ having images: ~12.7k NAVSIM logs (297k samples) have no camera images but do have maps → the scorer can be trained; 13× the nuScenes data
>   - Same cities anyway: nuScenes (Boston, Singapore) ⊂ nuPlan's 4 cities
>
> **② Test bed** (NAVSIM rig ≈ nuScenes rig)
> - a. Synthetic target rig on NAVSIM real images — exact for rotation / intrinsics; T1 code ready
> - b. Translation: raster for training + novel-view synthesis for evaluation
> - c. A source dataset with a really different rig
>
> **③ Story**
> - Story 1 needs its premise from T2 or translation
> - Story 2's axis (translation) is the one we have not tested
>
> **Next (proposal):** T2 (top priority) · the 0-real-image cell · E2a with matched steps · ≥ 3 seeds for key runs

**中文说明**
- ① 我写成了 a / b 两个选项，没有替你表态。如果你同意 b，可以在讲的时候说"我倾向于 b"。b 的理由：部署场景（新车型新 rig、旧车队的 log 和地图）里 target 域的 log 本来就有，缺的是新 rig 的图像；而且不用 NAVSIM log 也换不来城市级 zero-shot。
- ② 的 a 最省事（T1 的 rig_transform 可以直接用），但只能覆盖旋转和内参，而 T1 已经显示这部分掉分小；平移要靠 b。
- 可能被问："T2 要多久？" → 按你实际的排期回答，这里我没有写时间。

**日本語発言稿**
最後にご相談です。
1 つ目は zero-shot の定義です。ログがあることと実画像があることは別で、NAVSIM には画像のないログが約 1.3 万本、30 万サンプルあり、地図もあるので scorer を学習できます。nuScenes のデータの 13 倍の量です。それに nuScenes の都市も nuPlan の 4 都市に含まれているので、ログを使わなくても都市レベルの zero-shot にはなりません。target 側のログと地図を使ってよいことにするかどうか、ご意見をいただきたいです。
2 つ目はテスト環境です。NAVSIM と nuScenes の rig はほぼ同じなので、rig の違いを測るには、NAVSIM の実画像に正確な変換をかけた合成の target rig を使うか、並進をラスタと新視点合成で評価するか、rig が本当に違うデータセットを使うか、のどれかだと考えています。
3 つ目はストーリーです。ストーリー 1 は T2 か並進の結果で前提を固める必要があり、ストーリー 2 の並進はまだ検証していない軸です。
次の実験としては、T2 を最優先にして、実画像 0 の格子、ステップ数を揃えた E2a、主要な実験の複数 seed を考えています。以上です。

---

# Backup slides（只放图表，不需要讲稿）

- **B1** T1 完整表格：`eval/t1_results/t1_summary_official.md`（几何 / 位置分解）
- **B2** T1 探针：`eval/t1_results/t1_probe_official.md`（8a 生成/选择、8b 余弦、8c 主干份额、img_oob）
- **B3** 注意力汇点图：`eval/t1_results/figs/sca_points/density_ppy_m7r_pitch_up_m10.webp` 和其余 example 图
- **B4** 掉分随扰动大小的变化：`eval/t1_results/figs/t1p_official/dose_response.webp`
- **B5** 术语表：PDMS（v1）、EPDMS（v2）、navhard stage 1 / 2、best-of-64（oracle@64）、avg-of-64（mean64）、capture、img_oob
