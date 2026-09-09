# 8/26 汇报 — 上下文交接

用途：新开对话窗口整理 slides 时的唯一上下文来源。所有数字均已核对到冻结产物。

汇报内容 = **Alignment 实验** + **DriveWeave v2 提案**。核心难题是让二者听起来是一件事。

---

## 1. 主线（整场汇报的脊椎）

> **RAP 已经证明：用目标数据集自己的 structured logs 造 recovery 监督是有效的。它没回答的是——这件事能不能跨数据集。我这一轮先把这条通道的「传输机制」和「测量精度」校准了一遍：传输机制本身不产生可检出的价值，测量噪声比预想大一个量级。所以 DriveWeave 把赌注押在货上——跨数据集构造 recovery 监督，而且 source 端一张图都不读。**

压缩版：**Alignment 实验回答「怎么测、能测多准、值不值得投在传输机制上」；DriveWeave 回答「recovery 监督能不能跨数据集扩展」。**

比喻：**桥与货。** RAP 的闭环增益来自 raster-only recovery 增强样本（货，v2 EPDMS 32.5→36.9），alignment 只是让这些没有真图的样本能喂给真实 planner 的机制（桥）。我的受控实验刻意卸掉了货，只称桥本身的重量。

---

## 2. 模块一：Alignment 实验

### 2.1 状态（必须首页声明）

- `status = incomplete`，`designation = pipeline_rehearsal_v1_only`
- 三项实质偏离：候选池仅 21%；400 optimizer steps ≈ 官方 1/100；v2.2 evaluator 未接入，co-primary family 不完整
- `EPDMS = not_tested`；PDMS 的 p 值标注 `descriptive_non_confirmatory`
- **协议 `forbidden_use` 禁止**：对 alignment 是否改善 planning performance 作任何方向性结论；作 Stage-B 效应量先导；对外称官方 Stage-A 结果

### 2.2 配置

2,498 train / 464 val｜20 epochs｜400 steps｜global batch 128（4 devices × 32）｜3 seeds × 2 conditions = 6 runs｜AdamW lr 1e-4｜cosine + 1 warmup epoch｜dropout 0｜16-mixed｜final-step checkpoint

**convergence gate FAILED 两条判据**：6 个 run 的 `argmin(val_loss)` 全在 epoch 19；末段斜率 `[-0.048 … -0.065]`，超阈值 2–3 倍 → 全部结论带 **under-trained regime** 标注

### 2.3 结果

| | seed0 | seed1 | seed2 | mean | sd |
|---|---:|---:|---:|---:|---:|
| NoAlign PDMS | 76.02 | 74.11 | 77.32 | **75.82** | 1.61 |
| FullAlign PDMS | 76.90 | 76.35 | 74.15 | **75.80** | 1.46 |

参照系：constant-velocity **20.65**｜RAP 论文 **≈93.8**｜human **94.8**

**主视觉对比**：换掉 alignment **−0.02 分** vs 换一个 seed **最多 3.21 分**（NoAlign s1 74.11 → s2 77.32）→ **174 倍**

- paired delta PDMS mean `-0.000184`，CI `[-0.00321, +0.00280]` 跨 0，p=0.9096，Cohen's dz `-0.0015`
- per-seed delta **+0.88 / +2.24 / −3.18**（符号翻转）
- 逐场景：13.6% 完全未变，39.6% 变好，46.8% 变差，总和 −2.24 分
- 5 个离散 sub-score 全部 seed 间翻方向

sub-scores（NoAlign）：NC 97.56｜DAC 93.26｜TTC 91.35｜C 97.32｜DDC 97.85｜**EP 61.37**
→ **「不撞、不越线、不违规，就是不走」**。75.8 与 93.8 的 18 分缺口主要是进展，不是安全。

**不能上台的观察**：FullAlign 在 7 个指标里 6 个 seed 方差更小（正则化的形状）。n=3、未预注册 → **只能作为 Stage-B / A6 的预注册假设提出**。

### 2.4 数据侧（讲法要收着）

real 三相机覆盖率 **20.887%**，raster 100%；32,974/32,975 个排除项是 real 图像缺失；一个 val log 159/159 全缺；P.1 verdict = **`data_bug`**（可修复），human override 于 2026-08-19 接受降格。

⚠️ **DriveWeave v2 已自我收窄**：原文写「不能把 10:1 的 log-hour 比例写成『丢弃 90% 的独立驾驶知识』」。所以 **21% 只作排期/资源事实（Part 5），不进科学论证**。

### 2.5 评测器可信度（这是真成果）

- G1–G7 关系式门禁全过
- RAP fork vs official v1.1（`0811876c…`）唯一实质差异 = `batch_lqr_utils.py` 伪逆；12,146 场景中 9 个非零，max |ΔPDMS| **3.35e-12**
- **constant-velocity 复现论文 Table 1 的 20.6 → 20.6517（差 0.05 pp）**，一次性验证 metric cache + 地图版本 + 数据集版本 + LQR 仿真 + PDMScorer 整条链路

### 2.6 已知错误（别上 slide）

报告 §4.1 那句「DINO backbone 实际 lr = 2e-5」**是错的**。`rap_agent.py` 中 `img_backbone` 全部参数 `requires_grad=False`，vit 参数组被注释掉。`summary_protocol_v3.json` 正在更正此条，**尚未冻结**。

---

## 3. 新增分析：Scene-intrinsic vs Run-stochastic（Part 2.5）

脚本：`scripts/alignment/scene_vs_run_decomposition.py` + `visualization.py`
产物：`outputs/alignment_stage_a/summary/scene_vs_run_decomposition.json` + `figs/fig_{a,b,c}*.png`

**设计要点**：6 次运行当成同一批场景的 6 次重复观测，**从不比较两个条件** → 合法绕开 `forbidden_use`。

| 现象 | 实测 | 随机预期 | 倍数 |
|---|---:|---:|---:|
| 六次全部硬失败（`NC×DAC==0`） | **242**（1.99%） | 0.0045 | **~54,000×** |
| 六次全部落在 EP 最低十分位 | **207** | 0.0034 | ~60,000× |
| 六次全部「安全满分 + EP<0.30」 | **31** | 3e-7 | — |

- 观测分布 `[9683, 947, 500, 314, 264, 196, 242]` vs 独立性预期 `[6999, 4055, 964, 120, 8.3, 0.30, 0.0045]` → **两个尾巴都比随机厚**
- 逐次运行硬失败率 **5.71%–11.72%**（2.05 倍摆动），但 242 个场景一次不落
- **铁核构成：197 个 DAC=0，44 个 NC=0** → 几何/路形问题，不是交互问题
- ICC(1) = **0.601**；**排除硬失败后 = 0.761**（更高，故不是 0/1 退化假象）
- 跨运行 Spearman 0.776；失败集 pairwise Jaccard 0.374（独立预期 0.046）
- safe & EP<0.30 逐 run：413/142/86/365/178/224（4.8 倍摆动）

**MDE**：seed sd **2.82 分** → 3 seeds 检出下限 **4.56 分**。RAP 自己的 recovery 效应是 **+4.4 分** → **现配置连已发表效应都测不出**。需要 seed 数：4.4→**4**，3.0→7，2.0→16。

**constant-velocity 交叉验证**（`evaluator_validation/batch_lqr_ab/rap_run1.csv`，PDMS 20.6517 确认身份）：
CV 全体通过率 **33.1%**，铁核内 **7.9%**（19/242，独立预期 80）→ 铁核对不含学习的规则策略也难 **4.2 倍**。
⚠️ 「CV 能过而 planner 全败」的叙事**不成立**（19 低于预期 80），别用。

### 三张图的分工

- **fig_a 热力图**：提出 —— 横向条纹，分数由行（场景）决定不由列（运行）决定。定性
- **fig_b 重合度柱状图**：证明 —— 54,000 倍。但只在硬失败一个维度
- **fig_c EP 直方图**：加固 + 扩展 —— 换连续指标、且场景**全无硬失败**，现象照旧；同时暴露 seed 方差在 EP 上最大

fig_c 的存在意义是堵住「你那 60% ICC 只是撞车/不撞车的二值分裂」这个反驳。

### 3.5 铁核画像 —— 元数据 + 真实图像（**已完成**）

任务书 `docs/rap-alignment-navtest-subset-profile.md`。脚本 `navtest_scene_labels.py` / `subset_metadata_profile.py` / `subset_contact_sheets.py`。

**验收**：endpoint 自校验 2,000 条 max |Δ| = **1.42e-14**（阈值 1e-3）；图像覆盖 **12,146/12,146 = 100%**；5 子集 × 9 标签均含 10,000 次 log-cluster bootstrap + 2,000 次随机零分布。
图像阈值取自 navtest 自身分位数（对 spec §5「只从 train split 计算」的**有意偏离**，因 train 侧真图仅 21% 可用）：illumination P20/P80 = 75.2401 / 106.2114｜contrast P10 = 42.4948｜sharpness P10 = 307.8803。

#### log 集中度 —— 决定哪些画像能用

| 子集 | logs | top-3 | 判定 |
|---|---:|---:|---|
| **hard_core (242)** | 64 | **17.4%** | ✅ 可解释为**场景类型**效应 |
| ep_bottom (207) | 42 | 44.9% | ❌ `log_confounded`，**画像作废** |
| safe_stalled (31) | 16 | 45.2% | ❌ `log_confounded`，**画像作废** |
| failed_once (947) | 115 | 13.5% | ✅ |
| never_failed (9683) | 136 | 12.2% | ✅ |

⚠️ **画像作废 ≠ 重合度结论作废**。207 / 31 的「六次全部重合」讲的是**跨运行可复现性**，与 log 集中无关，**继续用**；只是不给它们配元数据画像。

#### hard_core 的 TVD ÷ 随机零分布 p95

| 标签 | TVD | null p95 | 倍数 |
|---|---:|---:|---:|
| **endpoint_dy** | .3426 | .0798 | **4.29** |
| **map_location** | .2525 | .0772 | **3.27** |
| actor_density | .1271 | .0776 | 1.64 |
| endpoint_dyaw | .0697 | .0573 | 1.22 |
| route_command | .0629 | .0595 | **1.06（无）** |
| endpoint_dx | .0169 | .0233 | 0.73 |
| **illumination** | .0530 | .0668 | **0.79** |
| **contrast** | .0422 | .0405 | **1.04** |
| **blur_like** | .0363 | .0363 | **1.00** |

**图像三项全部落在零分布内。**

⚠️ 陷阱：`never_failed` 的 route TVD .0320 / null .0044 = 7.3 倍看似极显著，但绝对 TVD 仅 0.03——n=9,683 让零分布极小。**比值大 ≠ 重要，两个数必须一起看。**

#### 单调性（最强论据）

| | never_failed | failed_once | hard_core |
|---|---:|---:|---:|
| endpoint_dy TVD | .0644 | .2407 | **.3426** |
| map_location TVD | .0302 | .1181 | **.2525** |
| actor_density TVD | .0266 | .0850 | **.1271** |

失败频次越高偏离越大，三个标签全部单调。`route_command` **不**单调（.0320→.1322→.0629），佐证它本就是噪声。

#### 方向

**dy 符号已由数据确认：`+y = 左`**（TURN_LEFT 88.6% 落 `>2`，TURN_RIGHT 83.2% 落 `<-2`）。

`map_location`（spec §5 禁令：只能说 **domain bundle**，不得解释为道路风格/天气）：

| | 全体 | hard_core | 倍数 |
|---|---:|---:|---:|
| us-ma-boston | .307 | .459 | 1.50 |
| us-pa-pittsburgh-hazelwood | .201 | .302 | 1.50 |
| **us-nv-las-vegas-strip** | .335 | **.099** | **0.30** |
| sg-one-north | .158 | .140 | 0.89 |

**新加坡基本不动 → 左侧通行那条链不成立，别用。**

`actor_density`：0-5 辆 .356 → **.483**（1.36×）；>20 辆 .118 → **.050**（0.42×）。**车更少，不是更多。**

#### 去混杂 —— 两个效应各自独立

每格 hard-core 率（全局 **1.99%**）：

| | 0-5 | 6-10 | 11-20 | >20 |
|---|---:|---:|---:|---:|
| boston | **4.37%** | 3.04% | 2.91% | 1.60% |
| pittsburgh | **4.30%** | 1.84% | 1.06% | 3.70%¹ |
| sg-one-north | **2.05%** | 0.50% | 0.00% | 0.00% |
| vegas | 0.56% | 0.93% | 0.53% | 0.34% |

¹ Pittsburgh `>20` 仅占该城 1.1% 场景，n 极小，**不引用**

- **「车少更难」成立**：4 城中 3 城单调下降（Boston 4.37%→1.60%，2.7 倍）；Vegas 平坦但整城贴地板
- **「地理」也成立且更强**：同为 0-5 档，Boston 4.37% 是 Vegas 0.56% 的 **7.8 倍**；Vegas 每一格都 <1%

#### 决定性发现：**导航指令看不见几何需求**

hard-core 率 by route：**GO_STRAIGHT 2.18%｜TURN_LEFT 1.96%｜TURN_RIGHT 1.08%**（全局 1.99%）→ 指令几乎无区分度。

route 占比：hard_core `{直 .727, 左 .202, 右 .070}` vs 全体 `{直 .664, 左 .206, 右 .130}`
→ **`TURN_LEFT` 占比完全没变**；唯一 route 级变化是 `TURN_RIGHT` 减半。

**`GO_STRAIGHT` 内部**（占 hard_core 72.7%，n=176）：

| dy | 全体 GO_STRAIGHT | hard_core GO_STRAIGHT | 倍数 |
|---|---:|---:|---:|
| `>2`（左） | .147 | **.432** | **2.94** |
| `0.5~2`（左） | .138 | **.295** | **2.14** |
| `-0.5~0.5`（真直行） | .529 | **.199** | **0.38** |
| `<-2`（右） | .101 | .045 | 0.45 |

聚合：实际向左横移 28.5% → **72.7%**；实际基本直行 52.9% → **19.9%**。

> **在导航指令说「直行」的场景里，被稳定做坏的，是专家实际上要向左横移 2 米以上的那些。指令看不见这个需求。**

⚠️ `TURN_RIGHT` 行 n 仅 **17**，数字乱跳是噪声，**不要引用**。`GO_STRAIGHT` n=176、`TURN_LEFT` n=49 可用。

#### 合起来的读法

1. 197/242 是驶出可行驶区，仅 44 是碰撞
2. 周围车**更少**（0-5 档 1.36×，>20 档 0.42×），去混杂后同城内仍单调
3. 需要的横向位移显著更大，且集中在**被标为直行**的场景
4. 亮度/对比度/清晰度**与全体无差异**
5. 转向指令**无区分度**

> **planner 稳定失败在「指令说直行、实际需要大幅横移」的空旷场景，失败方式是横着开出路面。不是交互问题，不是看不清的问题。缺的是几何/行为层面的监督。**

左右不对称（向右机动反而更易：TURN_RIGHT 减半 + GO_STRAIGHT 内向右横移 .101→.045）**照实报告，不做因果归因**；并注明新加坡左侧通行混在池中（15.8%），对应 DriveWeave §6.1 已列的 traffic-rule conflict。

#### contact sheets（待人工看）

`contact_hard_core.png`（24 张最差）｜`contact_never_failed.png`（24 张对照，固定 seed）｜`contact_blind.png`（48 张混排，已核验唯一且 24/24 平衡）+ `blind_key.json`。

**看法：先看盲评片逐张记判断，48 张看完再对答案。**
猜对率显著高于 50% → 可讲「人眼能看出差别」并描述差别；**接近 50% 也是结果**，且与「图像统计无差异」一致——**这批场景的难在单帧图像上肉眼看不出来**。第二种情况对 DriveWeave 更有利，直接讲出来，别当失败。

---

## 3.6 Alignment 模块到底起了什么作用（问答备料）

「它有没有用」协议禁止回答，数据也答不了（效应比 seed 噪声小 175 倍）。但以下五件事是钉死的，且比方向性结论更可操作。

### 它实际影响了什么

1. **它在运行且接线正确。** 三 seed 逐行验证 passed：NoAlign 权重逐行为零、FullAlign 逐行 0.002/0.1、GRL 逐行合公式、total loss 可逐行重构。**「没效果是因为有 bug」被排除。**
2. **它改变了 86.4% 场景的输出**（39.6% 变好 / 46.8% 变差 / 13.6% 未变）→ 不是数值惰性项。
3. **幅度**：逐场景 \|Δ\| q95 **0.26** vs 换 seed **0.74–0.80**（≈1/3）；总分 **0.018** vs 最多 **3.21**（≈1/175）。按协议措辞：**总量效应不可与 seed 噪声区分。**
4. **它物理上够不到视觉编码器**（本节最硬的一条，来自代码非分数，不受协议约束）：

```
rap_agent.py:571-572    img_backbone 全部参数 requires_grad = False
rap_agent.py:576        vit 参数组被注释掉
rap_model.py:123,150-161  alignment 与 domain classifier 作用在 image_feature 上
image_encoder.py:105    img_backbone = DINOv3（冻结）
```

> **alignment 梯度到达 BEV 聚合层与 DINO 之后的投影，永远到不了 DINOv3 本身。它不是在「让图像编码器更好地看见几何」，而是在重塑「冻结 DINO 特征的 BEV 聚合方式」。DINO 没编码的东西它加不进去。**

5. **只探了损失权重空间的一个点**：`λ_spatial = 0.002`（论文默认）、`λ_global = 0.1`。

### 它没解决什么 / 带来了什么

1. **没触及铁核。** 242 个场景的定义就是六次全败，含全部 3 个 FullAlign run → 该集合零 rescue（按观测构造）。⚠️ 但 400 steps 下**什么都没触及它们**，不能外推到全预算。
2. **它留下的失败模式已刻画**：197/242 驶出可行驶区、车更少、`GO_STRAIGHT` 内向左横移 28.5%→72.7%、图像三项统计与全体无差异 → **几何/行为类失败，发生在无可测量感知缺陷的场景里。**
3. **代价侧已量化，结论是「也在噪声内」**：DAC 合并 930 regression / 932 rescue（约 1,860 个场景被搬动）曾看似是代价，但补上同条件离散零分布后——**五个离散指标的 cross \|dominance\| 全部低于 null，churn 也全部不超过 null**（见下）。**所以「alignment 制造了净破坏」这个说法不成立，别用。**
4. **不能用但该预注册**：FullAlign 在 7 个指标里 6 个 seed 方差更小（n=3、未预注册）。

### 对改进 Alignment 的直接启示（按杠杆排序）

| # | 启示 | 具体动作 |
|---|---|---|
| ① | **先修测量** | MDE 4.56 分 > RAP 自己的 4.4 分效应 → **seed 数 ≥4**；补离散噪声地板（已有脚本，1 小时、零 GPU） |
| ② | **换评测目标救不了它** | 曾以为问题是「aggregate PDMS 有 79.7% 场景饱和，应改评在 2,221 个争议区」。翻转集分析推翻了这个说法：**模块的足迹本来就完全在那个池子里，而该池在任何扰动下都随机翻转**。正确表述：**机制够不到你在意的失败集，先改机制再谈评测。**（分层评测仍需独立参照 run 集定义，否则循环） |
| ③ | **冻结 backbone 是硬天花板（首要假设）** | 现在是「足迹为何被限制」的自然机制解释。二选一并说清：**(a)** 部分解冻（末 N block / LoRA），backbone lr 当一等超参；**(b)** 承认它是 BEV 聚合正则项，停止称其为 representation transfer。← DriveWeave **A5** 已把这条列为消融轴 |
| ④ | **λ 扫描 = ③ 的诊断，不是性能搜索** | 可证伪预测：若足迹受限源于**冻结 backbone** → 加大 λ **不**扩展范围，只放大边界池随机 churn；若源于**权重太小** → 会推到新区域。最小方案 `{0.002, 0.02, 0.2} × ≥4 seeds`，判据是**足迹是否越出边界池**，不是分数 |
| ⑤ | **损失形式与失败模式不匹配** | 现状 = 拍平 MSE(λ=0.002) + 全局域对抗(λ=0.1)，两者都不保留空间结构；而 81% 残余失败是 drivable-area 几何。旁证 TerraTransfer Table 6（0.319 / 0.307 / **0.490**）；DriveWeave 的 `L_struct` 已指定 batch-relational low-rank。→ **在保留空间结构的 BEV 特征图上对齐（或按可行驶区加权）**，并与朴素 MSE 同协议对照 |
| ⑥ | **paired-only 是零载配置** | 重设计要么加回 raster-only 增强（载荷），要么换 source（DriveWeave）。再跑一次 paired-only 只配当 λ 扫描的对照组 |

**一句话**：
> **当前配置下的 alignment，是一个作用在「冻结 DINO 特征的 BEV 聚合」上的、权重 0.002 的弱正则项，被放在一个 80% 饱和的指标上评测，而评测的检出下限高于它要追的效应量。这次没测到它的「价值」；测到的是它的「作用范围被结构性限制」和「评测被系统性欠功率」——两件都能直接动手修。**

### 离散噪声地板（补报告 §12.1 的洞）

脚本 `scripts/alignment/discrete_noise_floor.py` → `summary/discrete_noise_floor.json` + `figs/fig_e_discrete_noise_floor.png`

```bash
python scripts/alignment/discrete_noise_floor.py
```

对 NC / DAC / TTC / C / DDC 各算全转移表、regression、rescue、**paired dominance**，比较：

- **cross-condition**：`NoAlign_s vs FullAlign_s`（3 对，**同 seed**）
- **same-condition null**：各条件内 seed 两两配对（**6 对**）

⚠️ **不对称性必须声明**：cross 对共享 init checkpoint、batch 顺序、augmentation RNG，只差 alignment 梯度；null 对三者全不同。**null 携带的变异源更多** → cross 落在 null 区间内是有意义的；cross **低于** null 区间是共享 seed 的必然结果，**不构成任何证据**。

解释规则逐字继承 `summary_protocol_v2.json` 的 `noise_floor.interpretation_rule`：cross \|dominance\| 不超过 null 最大 \|dominance\| → 必须写「**效应不可与 seed 噪声区分**」，**不得**写「无效应」或「有微小效应」。

图 e 读法：灰带 = seed 噪声区间，灰点 = 6 个 null 对，蓝菱 = 3 个 cross 对。**蓝菱落在灰带内 = 不可区分。**

#### 实测结果（2026-08-25）

| | cross \|dom\| | null \|dom\| | 比 | cross churn | null churn | 比 |
|---|---:|---:|---:|---:|---:|---:|
| NC | .0114 | .0173 | 0.66 | 2.88% | 3.70% | 0.78 |
| **DAC** | .0312 | .0499 | 0.63 | **7.04%** | **7.15%** | **0.98** |
| TTC | .0138 | .0312 | 0.44 | 6.57% | 7.88% | 0.83 |
| C | .0203 | .0264 | 0.77 | 2.99% | 3.30% | 0.91 |
| DDC | .0118 | .0336 | 0.35 | 3.90% | 4.21% | 0.93 |

`overall_verdict = all_discrete_metrics_indistinguishable_from_seed_noise`
**五个指标的 cross dominance 全部 seed 间翻符号。**

**可以说的**：
> 「报告 §12.1 记了一个洞：离散指标没有零分布可比。我补上了。**五个离散 sub-score 的跨条件方向性成分只有种子噪声的 0.35–0.77 倍，而且五个全部在 seed 之间翻符号。按协议措辞：效应不可与 seed 噪声区分。**」

**churn 那一列曾有两种解释，现已分开** → 见下节。

### 翻转集重合度：alignment 的作用范围（补 churn 的歧义）

脚本 `scripts/alignment/flip_set_overlap.py` → `summary/flip_set_overlap.json` + `figs/fig_f_flip_set_overlap.png`

**问题**：churn 的大小分不开「(a) 边缘场景饱和」与「(b) alignment 特异扰动」，但翻转集的**成员身份**可以。

两个独立测量：
1. **逐对 Jaccard**（3×6 / 15 / 3，不做并集，无数量偏倚）：`J(c,n) ≈ J(n,n)` → (a)；`J(c,n) ≈ J indep` → (b)
2. **边缘度剖面**（决定性）：按「6 个 null 对里有几个翻了它」给场景打分（0–6），问 alignment 的翻转有多少落在**从未被任何 null 对翻过**的场景上

#### 实测结果（2026-08-25）

| | J(c,n) | J(n,n) | J indep | 比独立性高 | exclusive share |
|---|---:|---:|---:|---:|---:|
| NC | .2560 | .2791 | .0131 | 19.5× | 0.36% |
| DAC | .2580 | .2748 | .0292 | 8.8× | 0.81% |
| TTC | .2591 | .2766 | .0308 | 8.4× | 1.51% |
| C | .2856 | .2102 | .0127 | 22.5× | 0.96% |
| DDC | .2858 | .3172 | .0160 | 17.9× | **0.00%** |

`overall_verdict = a_marginal_scene_saturation`（五个指标一致）

> **alignment 梯度制造的每一次翻转，98.5–100% 落在种子变异也能触及的场景上。它的翻转集与另一个种子的翻转集彼此吻合的程度，等同于两个种子之间彼此吻合的程度（0.90–1.36 倍），且两者都比独立性高 8–22 倍。**

⚠️ 测量 2 天生偏向 (a)（null 有 6 次机会 × 3 个变异源，「从未被翻」是苛刻集合）。**但量级已贴地板，且不带该偏倚的测量 1 独立给出同一答案。**

#### 与其余证据串起来

1. alignment 只能触及**冻结 DINO 特征的 BEV 聚合**（代码事实）
2. 总量效应是 seed 噪声的 **1/175**
3. 逐场景足迹**完全落在种子也会翻的边界池内**（本测量）
4. 242 个铁核中 **197 个在六次运行里 DAC 恒为 0** —— 位于「从不翻转」档，而 alignment 在该档的翻转率仅为其总体的 **0.94%**

> **模块的作用范围，和残余失败集，是不相交的。**

⚠️ 400 steps / under-trained 下测得；全预算下作用范围是否扩展，本实验答不了。

---

## 4. 模块二：DriveWeave v2

**Source-Sensor-Free Cross-Dataset Recovery Scaling for End-to-End Driving**
*Scale recovery knowledge across datasets; learn target-domain vision without forgetting nominal driving.*

目标：$\max_\theta \text{EPDMS}_{final}$ s.t. $S_1 \ge S_1(\theta_0)-\varepsilon$，$S_2 \ge S_2(\theta_0)+\delta$。**ε 由多随机种子的实验波动确定。**

- **O0**：$P_{nominal} \neq P_{recovery}$，加更多正常驾驶数据不改善 Stage 2
- **O1**：1,200h 标注 vs 120h 图像（v2 定性为 opportunity，**不是** gap）
- **O2**：用 source RGB 需要 sensor reconciliation
- **O3**：naive 异构混训可能显著变差（HEAT：LTF 83.8→55.6，LAW 84.6→72.3，HEAT 83.2→85.2）
- **O4**：structured teacher 缺 brake lights / hand signals
- 链条：`O1 →(O2,O3)→ O4`

方法：RAP repo 为基座；canonical ego-centric vectorized BEV，`[-32,32]×[-32,96]m`；字段 map/agents/ego/route/traffic light/**schema availability mask**（`UNKNOWN ≠ NONE`）。
**分工关键**：**source 侧必须无相机（canonical）；target 侧沿用 RAP perspective raster**（部署相机几何已知，合法）。

`L_P2 = L_GT^nom + λ_d·L_dec + λ_s·L_struct + λ_v·L_visual-rec`

矩阵 S0–S3 / C0–C3。关键比较：`S1−S0`（target-only recovery 增益）、`S3−S1`（external 的结构化增量）、`C3−C1`（最终可部署增量）、`C3−S3`（**迁移保持率，不是 external 的因果效应**）。

**A6 = alignment loss 消融**（`L_GT`/`L_dec`/`L_struct` 组合）← 我的实验就住在这里。
**R7 identifiability failure**：若 source shuffle 仍获同等 gain，必须改口报 regularization effect。
识别性控制：source shuffle、label shuffle、matched-volume duplicate-target、ego-status masked、leave-overlap-city-out。
成功分层：Failure / Weak / Strong / Ideal。

---

## 5. 四条连接链（汇报的核心资产）

1. **模块同一**：我测的那个 loss 项就是 DriveWeave Phase 2 的 `L_struct`；**A6 就是把我这个实验放大到正确预算下重做一遍**。我不是做完 alignment 就丢掉它。
2. **ε 是我测的**：提案里 ε 定义为「多随机种子的实验波动」，这在提案里是待填的空，我填上了（≥3.2 分，需 ≥4 seed）。
3. **负控制**：我这轮是纯 regularization、零新信息的测量 → 预先削减 R7 的捷径解释。
4. **指标选择**：v1 对 recovery 不敏感（RAP Table 6：92.5→92.5 vs 32.5→36.9），加上我训出的「高 safety / 低 EP」活标本 → **EPDMS 必须是 official final metric，且 Stage 2 gain 必须同时查 EP**（正是 v2 §6.2 的规则）。

---

## 6. 汇报大纲（~22 min 正文）

| 段 | 时长 | Takeaway |
|---|---|---|
| **0 总纲** | 1 min | 三个框：RAP 已证明（同源）→ 我校准了机制与精度 → DriveWeave 问能否跨数据集 |
| **1 引言** | 3 min | 我关心 off-nominal recovery 不是 nominal 精度；recovery 监督必须显式构造 |
| **2 Alignment 实验** | 7 min | 本轮没有 alignment 的科学结论（看分数前就写进协议了）；交付的是受控测量装置、噪声尺子、保守 planner 活标本 |
| **2.5 场景 vs 运行分解** | 3 min | 随机性搬得动 3.2 分，搬不动那 242 个场景一分 |
| **3 过渡桥梁** | 4 min | 三个问题各答一半：该投在哪、多大才算数、用哪个指标 |
| **4 DriveWeave** | 7 min | 不赌「多数据集更好」（HEAT 已证明可能更差），赌一个更窄、可证伪的命题 |
| **5 下一步** | 3 min | Week 1 的 canonical 无损性门禁**不需要图像**，今天就能开始 |

附录页（备查不讲）：报告 §12 的 10 条不一致 + MISSING 测试项；Prior Work 边界表；Claims to Avoid。

### Part 5 的排期表

| 任务 | 需要 | 现在能做 |
|---|---|---|
| `canonicalize_navsim.py` + canonical planner 无损性门禁（PDMS>90） | **零张图像** | ✅ 今天 |
| 补 sensor blobs | 纯 IO/带宽 | ✅ 后台挂着 |
| RAP 93.8 复现 | 完整 sensor blobs | ⏳ 阻塞 |
| Stage 1/2/Final EPDMS | 接 v2.2 evaluator + `navhard_two_stage` | ⏳ 阻塞 |

---

## 7. 四个接缝的话术

设计要点：**242 分析是主链的枢纽，不是独立一拍。** 它把 −0.02 从「需要辩解的 null」变成「优化侧干预应有的量级」。转向的依据因此是**实证消除**，不是信息论先验推导。

### 接缝 A｜Part 2 结尾 → 2.5（把 null 变成问题）

> 「所以这一轮 alignment 的效应是 **−0.02 分**，而换个 seed 是 **3.2 分**。协议禁止我从这里得出方向性结论，我也不打算得。
> 但它留下一个我**可以**回答的问题：**−0.02，到底是因为 alignment 没用、因为我测不出来、还是因为那个位置本来就没多少东西可动？**
> 这三种解释里，第三种我用现有数据就能查。」

### 接缝 B｜2.5 → Part 3（消除法，**全场转折点**）

> 「答案是第三种。
> 把 6 次运行当成同一批 12,146 个场景的 6 次重复观测——**全程不比较两个条件**，所以这个分析在协议禁令之外。
> **242 个场景，六次运行全部失败。如果失败是随机的，期望 0.0045 个。差 54,000 倍。** 逐次失败率在 5.7%–11.7% 之间摆动两倍，但那 242 个一次不落。
> 而且这不是「撞车/不撞车」那个二值分裂造成的假象：在**完全没有硬失败**的 9,683 个场景里，PDMS 剩余方差仍有 **76%** 由场景决定；另有 **207** 个场景六次全部落在进展指标最低十分位，随机预期 0.003。
> 所以：**分数的大部分由场景决定，只有一小部分由训练随机性决定。任何只改变优化路径的干预——换 seed、换正则、换 alignment loss——都只在那一小部分里竞争。**
> **−0.02 不是一个我需要辩解的空结果，它就是优化侧干预应有的量级。我现在知道，那个位置本来就没多少东西可动。**」

### 接缝 C｜Part 3 → Part 4（从消除到 DriveWeave）

> 「那么被稳定做坏的到底是什么？我给这 242 个场景做了元数据和真实图像的画像。
> **第一，失败方式：197 个是驶出可行驶区，只有 44 个是撞车。**
> **第二，它们周围的车更少，不是更多**——0-5 辆档从 35.6% 升到 48.3%，>20 辆档从 11.8% 降到 5.0%。去混杂后同一城市内部仍然单调（Boston 4.4% → 1.6%）。所以这不是拥挤交互问题。
> **第三，它们在图像上看不出区别**——亮度、对比度、清晰度三项全部落在随机零分布内。不是看不清的问题。
> **第四，也是最关键的：导航指令完全预测不了失败**，直行 2.18%、左转 1.96%、右转 1.08%，全局 1.99%。但几何预测得了——而且是在「直行」内部：占铁核 72.7% 的 GO_STRAIGHT 场景里，专家实际向左横移超过半米的从 28.5% 涨到 **72.7%**，真正直行的从 52.9% 掉到 **19.9%**。
> **所以：planner 稳定失败在『指令说直行、实际需要大幅横移』的空旷场景，失败方式是横着开出路面。**
> 这恰好是 NAVSIM-v2 Stage 2 要测的那一类，也恰好是 DriveWeave 里 recovery 监督的 structured scorer 用来筛候选的两条准则：drivable-area / lane-direction compliance，和 route progress。而 recovery 扰动 `δ=(Δx, Δy, Δθ, Δv)` 施加的正是横向这个自由度——**planner 最弱的自由度，恰好是 recovery 监督直接作用的那个。**
> 但我要划清楚：**navtest 是 nominal 起点，所以这不等于『恢复失败』。我的分析排除的是优化侧，不是替数据侧选定了答案。** 数据侧还剩三个分支：更多正常驾驶数据、目标数据集自己的 recovery 数据、还是跨数据集的 recovery 数据？
> **这正是 DriveWeave 的 S0 / S1 / S2 / S3 四格矩阵要分开的三件事。我划掉了一个分支，那个矩阵因此从一个设计选择，变成了必要步骤。**」

### 接缝 D｜Part 4 内部 → canonicalization

**不要**从「让 alignment 从 regularizer 变成 knowledge carrier」起手——那是空想推理，且与接缝 B/C 重复。改从**扩展需求**起手：

> 「假设 recovery 监督确实有用——那是 Week 2 的 `S1−S0` 要先过的门。接下来的问题就变成：**recovery 的场景多样性从哪来？**
> 目标数据集自己的日志能造出的 recovery 状态，受限于它自己跑过的那些路。要扩，只能上更多真实数据集。而一旦跨数据集，source 就不能再带相机——HEAT 要给 nuScenes 补两张空白图凑 8 路，RAP 混训要统一相机顺序、resize 到 576×1024、旋转缩放标定矩阵。这些是妥协，不是方法。而且 naive 混合会掉点，HEAT 那张表 LTF 从 83.8 掉到 55.6。
> 唯一干净的出口，是让 source 端的扩展完全发生在 canonical structured space 里。
> **注意分工：去相机是对 source 的要求，不是对整个 pipeline 的洁癖。** 部署目标的相机几何本来就是已知的，所以 Phase 2 我照样用 RAP 的 perspective raster。」

### 附带产出（压成两句，放 Part 3 收尾或 Part 4 开头，**不占主链**）

> 「顺带两件事。**第一**，3 个 seed 的检出下限是 4.56 分，而 RAP 自己报告的 recovery 效应是 4.4 分——**用现在这个配置，连已发表的效应都测不出来**。提案成功判据里那个 ε 定义为『多随机种子的实验波动』，是个待填的空，现在有数了：至少 4 个 seed。
> **第二**，我这轮跑的是 v1 PDMS，而按 RAP 自己的 Table 6，v1 对 recovery 完全不敏感（92.5→92.5），v2 才动（32.5→36.9）。加上我训出来这个『安全 93–98、EP 只有 61』的模型——**EPDMS 必须是最终指标，而且 Stage 2 的涨幅必须同时查 EP。**」

---

## 8. 表述红线

| ❌ | ✅ |
|---|---|
| 「实验表明 alignment 是纯正则化/没有增益」 | 「效应不可与 seed 噪声区分；协议禁止方向性结论」 |
| 「FullAlign 方差更小说明它在正则化」 | n=3、未预注册 → 只作为下一轮的**预注册假设**提出 |
| 「Week 1 的 RAP 复现已经在手」 | 手上是评测链路+受控装置+CV 锚点复现；**93.8 没做到**（当前 75.8/残缺子集） |
| 「我的 21% 证明数据集图像稀缺」 | 那是本地可用性（`data_bug`），只作排期事实 |
| 「这些场景永远解不了」 | 「在这个能力水平上，它们对训练随机性完全不敏感」 |
| 「197 个 DAC 失败 = recovery 失败」 | navtest 是 **nominal 起点**；只是落在 recovery 监督针对的同一类几何约束上 |
| 「CV 能过而 planner 全败」 | 19 < 独立预期 80，**该叙事不成立**；改用「铁核对 CV 也难 4.2 倍」 |
| 「首个多数据集 planning / structured-to-vision transfer」 | 已被 UniPlanner / TerraTransfer 占用 |
| 「sensor-agnostic」「canonicalization 解决了数据集异构」 | 用 **source-sensor-free**；canonicalization 只负责 common interface |
| 「DINO backbone lr = 2e-5」 | backbone 是**冻结的** |
| 「铁核偏向左转」 | **错**。`TURN_LEFT` 占比与失败率均为平均水平；左倾来自**被标成直行、却要向左大幅横移**的场景 |
| 「波士顿街道窄所以难」 | spec §5 禁令：`map_location` 只能说 **geography / domain bundle** |
| 「图像三项无差异 = 视觉不是问题」 | 只说明**这三个全局统计量**解释不了失败；遮挡/反光/物体类别它们测不到 |
| 引用 `TURN_RIGHT` 那一行 | n 仅 **17**，是噪声 |
| 「车少更难」不加限定 | 已去混杂（4 城中 3 城同城内单调），但需注明 Vegas 整城贴地板 |
| 「新加坡左侧通行导致左倾」 | sg 占比 .158→.140（0.89×）**基本不动**，该链不成立 |
| 「alignment 搬动了 1,860 个 DAC 场景，是净破坏」 | 离散零分布显示 **null 搬动得一样多甚至更多**；cross \|dom\| 只有 null 的 0.35–0.77 倍 → 仍是「不可与 seed 噪声区分」 |

**必须主动说的五句**：① under-trained（75.8 vs 93.8），铁核有一部分会在全量训练后消失；② navtest 是 **nominal 起点**，横移失败 ≠ recovery 失败；③ CV 的一致性有机械成分（直行策略必然在弯道出界）；④ MDE 在 v1 PDMS 上测，EPDMS 方差方向未知；⑤ **画像标签是在看到分数之后生成的 → descriptive / exploratory，非 confirmatory；图像阈值取自 navtest 自身，是对 spec §5 的有意偏离**。

---

## 9. 预判问答

1. **「你的 null 否定了 DriveWeave 吗？」** → ①那是 no-result 不是 null（gate 两条全 FAILED）；②即便当真，它测的是空载的桥，理论本就预测没增益，与 bridge/cargo 模型**一致**；③正因如此我把它放进 A6 而非主线。
2. **「为什么不直接把 nuScenes 标注按 NAVSIM 相机渲成 perspective raster？」** → ①把 source 知识重新绑回目标传感器，换相机就要全部重渲重训；②投影有损；③**最致命：raster 无法表达「不可观测」**——nuScenes 没有红绿灯标注，「没标注」和「没红灯」渲成同一张图，planner 会把缺失当 negative evidence 学。这是 A8 的 `UNKNOWN ≠ NONE`，像素空间做不到。（诚实补充：A3 只消融了 vector vs **BEV** raster，没覆盖 perspective 这一档，上述是原理判断。）
3. **「HEAT 已证明多数据集会掉点，你凭什么不掉？」** → 不假设不掉。O3 写在提案里，R5 有诊断与 mitigation，成功判据分四级，做 leave-overlap-city-out 与 driving-side conditioning，**胜利条件是持平不是超越**。
4. **「怎么保证 gain 来自 external knowledge 而非正则化？」** → 五个负控制；若 source shuffle 同等 gain 则收窄因果声明改报 regularization effect（已写进 R7）。
5. **「花一个月跑别人的 baseline 值得吗？」** → pilot 的职责是在花真钱前找出会杀死你的东西。它找出了三个：图像侧数据物流、seed 噪声地板、evaluator provenance，成本是 1/100 预算。
6. **「你有没有看过 alignment 在哪类场景上起作用？」**（大概率被问）→ 「看了，而且这正是**不能**回答的那个问题。**seed 0 的 DAC 是 rescue 254 / regression 107，seed 2 是 rescue 238 / regression 617**——同一份数据、同一个初始化、同一个 batch 顺序，方向反了。5 个离散 sub-score 全部这样。所以我把分析建在**从不比较两个条件**的地方：6 次运行当 6 次重复观测。这不是我没去看，是这个问题在当前预算下没有答案。」
7. **「242 个场景会不会只是某几段路？」** → 「查过。分布在 136 个 log 里的 **64** 个，top-3 只占 **17.4%**，未触发混杂标记，所以可以解释为场景类型效应。同一批分析里的另外两个子集（207 / 31）top-3 达 45%，**它们的画像我作废了**，只保留跨运行重合度的结论。」

---

## 10. 待完成 / 风险

- ~~navtest 画像任务赶不上~~ → **已完成并验收**（§3.5）。log 集中度这一关 **hard_core 通过**（64 logs / top-3 17.4%），所以「某类场景难开」的说法站得住；`ep_bottom` 与 `safe_stalled` 未通过，**画像作废但重合度结论保留**。
- **唯一待办：人工看 `contact_blind.png`**，逐张记判断后再对 `blind_key.json`。猜对率决定 Part 2.5 补充页怎么写（见 §3.5 末）。
- `summary_protocol_v3.json` 未冻结（`amended_by` 为空），引用时注意。
- ~~离散噪声地板未做~~ → **已完成**（§3.6 末）。5 个离散指标全部 `indistinguishable_from_seed_noise`，报告 §12.1 的洞已补。
- ~~churn 的两种解释未分开~~ → **已分开**（§3.6）。`flip_set_overlap.py` 五个指标一致判为 **(a) 边缘场景饱和**；由此得出「模块作用范围与残余失败集不相交」，并修正了设计启示 ②④。
- **数据侧分析到此收工。** 剩余全部是 slides 制作。
- **绝不能做**：把「FullAlign 赢 / NoAlign 赢」的场景集合拿去配元数据画像。三条理由——① 直接触犯 `forbidden_use` 第 1 条；② 分层切片是 spec §5/§6 明写的 Stage-B only，且要求标签在看到预测**之前**冻结；③ 那两个集合本身不稳定，5 个离散 sub-score **全部**在 seed 间翻方向。真被问到，用「逐 seed 画像互相矛盾」当**防守**（见 §9）。

---

## 11. 关键文件索引

```
docs/rap-alignment-experiment-plan.md              spec of record
docs/rap-alignment-stage-a-report.md               最终报告（结论、caveat、§12 已知不一致）
docs/rap-alignment-stage-a-step5-v1-only.md        v1-only 评测的替代指令
docs/rap-alignment-navtest-subset-profile.md       Codex 任务书（已完成）
~/Downloads/DriveWeave/DriveWeave_Proposal_v2.pdf  提案 v2（11 页）

outputs/alignment_stage_a/summary/
  scene_vs_run_decomposition.json   Part 2.5 全部数字
  per_run_subscore_means.json       绝对分数（descriptive，非冻结统计产物）
  summary_protocol_v2.json          统治性协议（run_scope / forbidden_use）
  navtest_scene_labels.csv          12,146 行 × 9 标签（§3.5）
  navtest_scene_labels_meta.json    阈值、覆盖率、endpoint 校验、crop 来源
  subset_metadata_profile.json      5 子集 × 9 标签的 TVD / CI / 零分布 / log 集中度
  {hard_core,ep_bottom_decile_all_six,safe_but_stalled_all_six}_tokens.txt
  {failed_exactly_once,never_failed}_tokens.txt
  figs/fig_{a,b,c}*.png             场景 vs 运行分解三张图
  discrete_noise_floor.json         §3.6 离散零分布（补报告 §12.1 的洞）
  flip_set_overlap.json             §3.6 作用范围（(a)/(b) 判别）
  figs/fig_d_subset_profiles.png    子集画像
  figs/fig_e_discrete_noise_floor.png
  figs/fig_f_flip_set_overlap.png
  figs/contact_{hard_core,never_failed,blind}.png + blind_key.json

scripts/alignment/scene_vs_run_decomposition.py
scripts/alignment/visualization.py
scripts/alignment/navtest_scene_labels.py
scripts/alignment/subset_metadata_profile.py
scripts/alignment/subset_contact_sheets.py
scripts/alignment/discrete_noise_floor.py
scripts/alignment/flip_set_overlap.py
```

关键代码位置（§3.6 第 4 条的依据）：
```
navsim/agents/rap_dino/rap_agent.py:567-576              backbone 冻结 / vit 参数组被注释
navsim/agents/rap_dino/rap_model.py:123,150-161          image_feature 与 alignment / GRL
navsim/agents/rap_dino/bevformer/image_encoder.py:105    img_backbone = DINOv3
```
