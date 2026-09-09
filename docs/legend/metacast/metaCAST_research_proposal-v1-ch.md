# Research Proposal: metaCAST

## 一、拟定题目（Working Titles）

**metaCAST: Metadata-based Camera-Agnostic Structured Transfer for Scalable End-to-End Autonomous Driving**

- **Metadata-based**：利用结构化场景数据学习 planning knowledge；
- **Camera-Agnostic**：外部训练数据不受 camera intrinsics、extrinsics 和 camera layout 限制；
- **Structured Transfer**：将 structured-domain 中学到的规划能力迁移至 image-only policy。

---

## 二、研究背景与动机（Motivation）

### 2.1 Vision-based end-to-end driving 的数据扩展受到 sensor configuration 限制

当前的 vision-based end-to-end autonomous driving 通常直接从多相机图像预测未来 ego trajectory。该范式具有较强的系统简洁性和数据扩展潜力，但其训练数据与具体的 camera rig 高度绑定。

不同自动驾驶数据集通常具有不同的：

- camera 数量与排列方式；
- intrinsics 与 extrinsics；
- field of view 与图像分辨率；
- temporal sampling rate；
- map representation；
- annotation taxonomy；
- trajectory horizon 与 coordinate convention。

[UniDrive](https://arxiv.org/abs/2410.13864) 已经表明，视觉模型容易受到 camera configuration 变化的影响，并通过将原始图像映射到统一 virtual camera space 来缓解跨 camera rig 的 perception generalization 问题。该方法仍然需要 source images 和 camera calibration，并主要针对 3D perception，而非跨数据集的 planning knowledge transfer。

因此，简单地将多个数据集的图像直接混合训练，通常需要繁重的 camera normalization、calibration conversion 和 architecture adaptation。[RAP](https://arxiv.org/abs/2510.04333) 在联合不同数据域时也需要统一相机顺序、图像分辨率、坐标系和 calibration matrices，说明视觉空间中的 dataset fusion 本身存在较高的几何适配成本。

### 2.2 大量结构化驾驶数据尚未被 camera-only planner 充分利用

与图像相比，许多公开数据集已经提供较完整的结构化驾驶信息，包括：

- ego and agent trajectories；
- agent class、position、heading、velocity；
- lane graph 与 drivable area；
- traffic-light state；
- local map topology；
- interaction-rich driving scenarios。

例如，[Waymo Open Motion Dataset](https://arxiv.org/abs/2104.10133) 提供大规模真实交通交互场景，[Argoverse 2](https://arxiv.org/abs/2301.00493) 也重点覆盖复杂运动预测与多城市道路结构。

[ScenarioNet](https://arxiv.org/abs/2306.12241) 已经能够将 Waymo、nuScenes、Lyft L5 和 nuPlan 等异构数据转换为统一的 scenario representation；[UniTraj](https://arxiv.org/abs/2403.15098) 也证明，增加数据规模和 domain diversity 可以改善 trajectory prediction 的泛化性能。

然而，这些工作主要解决：

\[
\text{heterogeneous datasets}
\rightarrow
\text{unified structured representation}
\rightarrow
\text{structured prediction/planning}
\]

尚未系统解决：

\[
\text{heterogeneous structured datasets}
\rightarrow
\text{shared planning competence}
\rightarrow
\text{target-domain camera-only policy}.
\]

换言之，现有 structured-data unification 尚未真正转化为 vision-based end-to-end planner 的数据扩展机制。

### 2.3 Existing structured-to-visual transfer 仍以单数据集或仿真策略迁移为主

[RAP](https://arxiv.org/abs/2510.04333) 将 driving annotations rasterize 成透视视角输入，利用大量 raster-only samples 训练 planner，并通过 spatial and global feature alignment 将知识迁移至真实图像。它证明了 lightweight structured rendering 可以支持 planning augmentation，但 raster representation 仍然与 perspective camera geometry 相关。

[DistillDrive](https://arxiv.org/abs/2508.05402) 使用 structured scene planning model 作为 teacher，为 image-based end-to-end model 提供 planning-oriented supervision；因此，“structured teacher teaches visual student”本身已不能作为 metaCAST 的主要 novelty。

更近期的 [TerraTransfer](https://arxiv.org/abs/2606.17386) 进一步提出将 **learning to drive** 与 **learning to see** 解耦：先在 vectorized simulator 中通过 self-play 学习策略，再使用 paired image–scene-state data，将 policy transfer 至视觉模型。

因此，metaCAST 的核心研究价值不应被表述为普通的 teacher–student distillation，而应被严格定位为：

> **利用 camera-configuration-independent canonical scene representation，将多个异构真实驾驶数据集转化为共享的 planning pretraining corpus，并将其规划知识迁移至仅使用目标域图像的端到端驾驶模型。**

其核心思想是：

> **Decouple planning-data scaling from sensor-data scaling.**

外部数据集负责扩大模型见过的道路结构、交通参与者行为和交互模式；目标域 image–metadata paired data 只负责解决“如何从目标 camera rig 恢复 planner 所需的信息”。

---

## 三、核心问题定义（Problem Statement）

### 3.1 Dataset setting

设存在 \(K\) 个 heterogeneous driving datasets：

\[
\mathcal{D}_k
=
\left\{
\left(
s_i^{k},
y_i^{k}
\right)
\right\}_{i=1}^{N_k},
\qquad k=1,\dots,K,
\]

其中：

- \(s_i^k\) 表示结构化场景，包括 map、agent histories、ego state 和可用的 navigation context；
- \(y_i^k\) 表示 logged ego future trajectory 或 planning target；
- 不要求数据集 \(\mathcal{D}_k\) 提供图像；
- 不要求不同数据集共享相同 camera configuration。

另有目标视觉数据集：

\[
\mathcal{D}_{T}
=
\left\{
\left(
x_i^{T},
s_i^{T},
y_i^{T}
\right)
\right\}_{i=1}^{N_T},
\]

其中 \(x_i^T\) 是目标 camera rig 采集的图像，并与 structured scene \(s_i^T\) 配对。

每个数据集拥有不同的数据 schema 和 domain：

\[
\Omega_k =
\left(
\mathcal{F}_k,
\mathcal{C}_k,
\Delta t_k,
\mathcal{M}_k,
\mathcal{Y}_k
\right),
\]

分别对应 feature schema、class taxonomy、sampling rate、map representation 和 trajectory definition。

### 3.2 Learning objective

首先构建 dataset-specific canonicalizer：

\[
C_k:s^k\rightarrow z^k,
\]

将不同数据集转换至统一的 ego-centric canonical structured space：

\[
z^k=
\left\{
z_{\mathrm{ego}},
z_{\mathrm{agent}},
z_{\mathrm{map}},
z_{\mathrm{route}},
m_{\mathrm{avail}}
\right\}.
\]

其中 \(m_{\mathrm{avail}}\) 是 attribute availability mask，用于避免将某数据集缺失的属性错误地解释为零值。

最终目标是学习 image-only policy：

\[
\pi_{\mathrm{img}}
:
(x^T,r^T)
\rightarrow
\hat y^T,
\]

使其在目标域上的规划风险最小：

\[
\min_{\pi_{\mathrm{img}}}
\mathcal{R}_{T}(\pi_{\mathrm{img}})
=
\mathbb{E}_{(x^T,y^T)\sim\mathcal{D}_T}
\left[
\ell_{\mathrm{plan}}
\left(
\pi_{\mathrm{img}}(x^T),y^T
\right)
\right],
\]

同时允许训练阶段利用：

\[
\bigcup_{k=1}^{K}\mathcal{D}_k,
\]

但满足部署约束：

\[
\pi_{\mathrm{img}}^{\mathrm{test}}
\text{ does not access }
s^T,
\text{ source images, or source calibrations.}
\]

### 3.3 Research questions

#### RQ1: Cross-dataset planning transfer

在目标 image data 数量保持不变时，加入来自其他数据集的 structured driving data，能否提高目标 camera-only planner 的性能？

\[
\mathcal{R}_T
\left(
\pi_{\mathrm{img}}^{\mathrm{multi}}
\right)
<
\mathcal{R}_T
\left(
\pi_{\mathrm{img}}^{\mathrm{target}}
\right).
\]

#### RQ2: Data diversity or merely data volume

性能增益究竟来自数据数量增加，还是来自跨城市、跨道路结构和跨 interaction pattern 的 scenario diversity？

#### RQ3: Visual data efficiency

multi-dataset structured pretraining 是否可以降低目标域所需的 image–metadata paired data 数量？

即，在 \(N_T'\ll N_T\) 时，是否仍能满足：

\[
\mathcal{R}_{T}
\left(
\pi_{\mathrm{img}}^{\mathrm{multi}},N_T'
\right)
\leq
\mathcal{R}_{T}
\left(
\pi_{\mathrm{img}}^{\mathrm{target}},N_T
\right)?
\]

### Central hypothesis

> A planner can acquire transferable driving competence from camera-independent structured scenes, while only the visual grounding stage needs to be specialized to the target camera configuration.

---

## 四、核心方法（Methodology: metaCAST）

### 4.1 Overall architecture

metaCAST 由两个 modality-specific encoders、两个 planner adapters 和一个共享 planner 构成：

\[
q_{\mathrm{meta}}
=
G_{\mathrm{meta}}
\left(
E_{\mathrm{meta}}(z)
\right),
\]

\[
q_{\mathrm{img}}
=
G_{\mathrm{img}}
\left(
B_{\mathrm{vis}}(x)
\right),
\]

\[
\hat y=P(q,r).
\]

其中：

- \(E_{\mathrm{meta}}\)：Canonical Structured Encoder；
- \(B_{\mathrm{vis}}\)：target-domain visual backbone；
- \(G_{\mathrm{meta}}\)、\(G_{\mathrm{img}}\)：将不同 modality 映射到统一 planner query space；
- \(P\)：共享的 planning decoder；
- \(r\)：navigation command 或 route context。

系统包含两个主要训练阶段。

---

### Step 1. Canonical Structured Scene Construction

首先对不同数据集定义统一的 canonical scenario schema。

#### 1. Ego-centric spatial normalization

所有 map 与 agent states 被转换至当前 ego coordinate：

\[
\tilde p_t
=
R(\theta_{\mathrm{ego}})^{\top}
(p_t-p_{\mathrm{ego}}).
\]

统一：

- 坐标轴方向；
- 单位；
- heading definition；
- trajectory horizon；
- map range；
- agent-centered versus ego-centered convention。

#### 2. Temporal normalization

将不同采样率的数据重采样至统一时间间隔：

\[
\tilde s_{t-j}
=
\operatorname{Interp}
\left(
s,
t-j\Delta t_{\mathrm{canonical}}
\right).
\]

同时显式提供 timestamp 或 time-offset embedding，避免将插值后的样本误认为原始同步观测。

#### 3. Structured tokenization

将 canonical scene 编码为：

\[
Z=
[
Z_{\mathrm{ego}};
Z_{\mathrm{agents}};
Z_{\mathrm{lanes}};
Z_{\mathrm{crosswalks}};
Z_{\mathrm{lights}};
Z_{\mathrm{route}}
].
\]

每个 dynamic-agent token 可包含：

\[
z_j^{\mathrm{agent}}
=
[
x,y,\cos\theta,\sin\theta,
v_x,v_y,a_x,a_y,
l,w,
c,
m_{\mathrm{avail}}
].
\]

Map elements 使用 polyline tokens 或 vectorized lane tokens 表示。

#### 4. Heterogeneous annotation handling

由于各数据集的 annotation richness 不一致，metaCAST 不强制所有数据拥有完整字段，而是采用：

- attribute availability mask；
- unified coarse taxonomy；
- dataset-specific input adapter；
- optional dataset embedding；
- masked reconstruction 或 masked attention。

这样可以避免 converter 中缺失字段所引入的虚假 supervision。

#### Representation ablation

核心方法使用 **direct structured tokens**。另设置两种 representation baseline：

1. canonical BEV raster；
2. camera-perspective raster，作为 RAP-style representation。

该设计用于回答：性能提升究竟来自更多数据，还是来自 camera-independent structured representation。

---

### Step 2. Multi-Dataset Structured Planner Pretraining

在所有 structured datasets 上训练 teacher：

\[
T_{\mathrm{meta}}
=
P\circ G_{\mathrm{meta}}\circ E_{\mathrm{meta}}.
\]

对于来自数据集 \(k\) 的样本：

\[
\hat y_i^k
=
P
\left(
G_{\mathrm{meta}}
\left(
E_{\mathrm{meta}}(C_k(s_i^k))
\right)
\right).
\]

训练目标为：

\[
\mathcal{L}_{\mathrm{meta}}
=
\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}
+
\lambda_{\mathrm{score}}\mathcal{L}_{\mathrm{score}}
+
\lambda_{\mathrm{safe}}\mathcal{L}_{\mathrm{safe}}.
\]

其中：

#### Trajectory imitation loss

\[
\mathcal{L}_{\mathrm{traj}}
=
\sum_t
w_t
\operatorname{SmoothL1}
(\hat y_t,y_t).
\]

对于 multi-modal trajectory head，可采用 mode classification 与 best-mode regression。

#### Planning-aware score supervision

若数据集支持 planner scorer，则进一步学习：

- collision risk；
- drivable-area compliance；
- route progress；
- comfort；
- time-to-collision。

这可以避免 structured planner 仅拟合 logged trajectory，而没有形成 planning-oriented representation。

#### Dataset-balanced optimization

为防止大型数据集完全主导训练，采用：

\[
\mathcal{L}_{\mathrm{multi}}
=
\sum_{k=1}^{K}
\alpha_k
\mathbb{E}_{\mathcal{D}_k}
[
\mathcal{L}_{\mathrm{meta}}
],
\]

其中 \(\alpha_k\) 可依据数据集规模、scenario rarity 或 validation transfer gain 设定。

训练完成后，\(P\) 应学习相对 modality-independent 的 planning function，而 \(E_{\mathrm{meta}}\) 负责把 canonical structured scene 映射为 planner-consumable queries。

---

### Step 3. Target-Domain Visual Grounding

随后使用目标数据集中的 paired samples：

\[
(x_i^T,s_i^T,y_i^T)
\]

训练 visual branch：

\[
S_{\mathrm{img}}
=
P\circ G_{\mathrm{img}}\circ B_{\mathrm{vis}}.
\]

Structured teacher 生成：

\[
q_i^{\mathrm{meta}},
\quad
h_{i,l}^{\mathrm{meta}},
\quad
p_i^{\mathrm{meta}}(y),
\]

visual student 生成：

\[
q_i^{\mathrm{img}},
\quad
h_{i,l}^{\mathrm{img}},
\quad
p_i^{\mathrm{img}}(y).
\]

初始阶段冻结：

\[
E_{\mathrm{meta}},
G_{\mathrm{meta}},
P,
\]

仅优化：

\[
B_{\mathrm{vis}},
G_{\mathrm{img}}.
\]

这使 visual branch 学习的目标不再是重建所有 metadata，而是产生足以驱动既有 planner 的 representation。

---

### Step 4. Hierarchical Planning-Knowledge Transfer

metaCAST 不仅在单一 feature layer 上进行 alignment，而是在 planner 的多个语义层级迁移知识。

#### 4.1 Planner-query alignment

\[
\mathcal{L}_{q}
=
\sum_j
w_j
d
\left(
q_j^{\mathrm{img}},
\operatorname{sg}
(q_j^{\mathrm{meta}})
\right).
\]

其中 \(d\) 可采用 normalized Smooth-L1、cosine distance 或 relational distance。

#### 4.2 Planner-state alignment

\[
\mathcal{L}_{h}
=
\sum_{l\in\mathcal{S}}
\beta_l
d
\left(
h_l^{\mathrm{img}},
\operatorname{sg}
(h_l^{\mathrm{meta}})
\right).
\]

相比只对齐 encoder feature，planner-state alignment 更直接地约束对决策有用的信息。

#### 4.3 Planning-output distillation

对于 multi-modal trajectory distribution：

\[
\mathcal{L}_{\mathrm{out}}
=
D_{\mathrm{KL}}
\left(
p^{\mathrm{meta}}(y)
\parallel
p^{\mathrm{img}}(y)
\right)
+
\operatorname{SmoothL1}
\left(
\hat y^{\mathrm{img}},
\operatorname{sg}
(\hat y^{\mathrm{meta}})
\right).
\]

TerraTransfer 同样采用 action-distribution alignment 与 latent structural alignment，说明 output-level 和 relation-level supervision 是 state-to-vision policy transfer 的合理基础；metaCAST 的区别在于其 teacher knowledge 来自多个异构真实驾驶数据集，而非单一 self-play simulator。

#### 4.4 Ground-truth planning loss

\[
\mathcal{L}_{\mathrm{GT}}
=
\mathcal{L}_{\mathrm{plan}}
(\hat y^{\mathrm{img}},y^T).
\]

总体目标为：

\[
\mathcal{L}_{\mathrm{transfer}}
=
\lambda_q\mathcal{L}_{q}
+
\lambda_h\mathcal{L}_{h}
+
\lambda_o\mathcal{L}_{\mathrm{out}}
+
\lambda_g\mathcal{L}_{\mathrm{GT}}.
\]

---

### Step 5. Visibility-Aware Distillation

Structured teacher 可能访问 camera 无法观察的对象，例如：

- 被前车完全遮挡的行人；
- 视野外的 agent；
- annotation 中存在但图像证据不足的交通参与者；
- 比视觉模型更准确的速度和 acceleration。

若强制 visual feature 与完整 structured feature 一致，student 可能被迫学习不可实现的 hallucination。

因此，为每个 structured element 构建 observability weight：

\[
v_j\in[0,1].
\]

局部 query alignment 修改为：

\[
\mathcal{L}_{q}^{\mathrm{vis}}
=
\sum_j
v_j
d
\left(
q_j^{\mathrm{img}},
q_j^{\mathrm{meta}}
\right).
\]

对于不可见信息，不进行直接 token-level imitation，而只保留：

- teacher uncertainty-aware output distillation；
- trajectory-level supervision；
- scene-level relational supervision。

这一区分了：

1. **visually recoverable scene knowledge**；
2. **privileged but non-observable information**。

---

### Step 6. Target-Domain Joint Refinement

在 visual grounding 稳定后，可采用低学习率解冻 planner 的上层模块：

\[
\theta_P
\leftarrow
\theta_P
-
\eta_{\mathrm{small}}
\nabla_{\theta_P}
\mathcal{L}_{\mathrm{transfer}}.
\]

该阶段用于缓解 structured-domain planner 与 image-derived noisy representation 之间的 distribution gap。

训练完成后的部署模型为：

\[
\pi_{\mathrm{deploy}}
=
P\circ G_{\mathrm{img}}\circ B_{\mathrm{vis}},
\]

推理阶段不需要：

- GT metadata；
- structured teacher；
- source images；
- source camera calibration；
- rasterizer。

---

## 五、实验验证方案（Evaluation Plan）

### 5.1 Dataset configuration

#### Primary target: NAVSIM / nuPlan camera domain

[NAVSIM](https://arxiv.org/abs/2406.15349) 基于真实驾驶数据构建 scalable planning evaluation，并使用安全、进度与舒适性相关指标评价规划结果。NAVSIM v2 进一步通过 counterfactual observations 评估车辆偏离 logged state 后的 recovery ability。

目标训练数据：

- NAVSIM target-domain images；
- 与图像配对的 nuPlan structured annotations；
- target-domain ego trajectories。

#### Structured source datasets

建议按以下顺序加入：

1. **Additional nuPlan structured-only logs**  
   用于验证 same-dataset structured scaling。

2. **Waymo Open Motion Dataset**  
   用于增加 interaction-rich US driving scenarios。

3. **Argoverse 2 Motion Forecasting Dataset**  
   用于增加多城市地图结构及长尾交互。

4. **nuScenes structured scenes**  
   用于增加 Singapore/Boston domains，并测试不同 annotation schema。

ScenarioNet 或 UniTraj converters 可作为 canonicalization 的初始工程基础，但需重新定义适合 ego planning 的 route、map 和 target schema。

#### Secondary benchmark: Bench2Drive

[Bench2Drive](https://arxiv.org/abs/2406.03877) 提供 CARLA closed-loop evaluation，包含 44 类交互场景、23 种天气、12 个 towns 和 220 条测试路线，适合验证模型是否真正获得 closed-loop driving competence，而不仅是降低 trajectory error。

---

### 5.2 Controlled comparison groups

所有方法应使用相同 visual backbone、planner capacity、target image data 和 optimization budget。

| Group | Structured training data | Source images | Transfer mechanism |
|---|---|---:|---|
| A. Image-only | None | No | GT trajectory only |
| B. Target teacher | Target metadata only | No | Structured-to-image distillation |
| C. In-domain scaling | All nuPlan metadata | No | Same-dataset transfer |
| D. Cross-dataset, single source | Target + one external dataset | No | metaCAST |
| E. Cross-dataset, multi-source | Target + multiple external datasets | No | full metaCAST |
| F. Raster baseline | Rasterized structured data | No | RAP-style R2R alignment |
| G. Image fusion baseline | Multi-dataset images | Yes | camera/virtual-rig normalization |

该分组可以区分：

- teacher distillation 的收益；
- same-dataset extra data 的收益；
- cross-dataset structured knowledge 的额外收益；
- direct tokens 相比 raster representation 的收益；
- camera-free transfer 相比 image-level dataset fusion 的收益。

---

### 5.3 Baselines

#### End-to-end planning baselines

- camera-only target planner；
- iPad or DiffusionDrive camera-only；
- RAP；
- 与 metaCAST 使用相同 backbone 的 target-only implementation。

RAP 应作为最重要的直接对比，因为它已经证明 raster-only augmentation 和 Raster-to-Real alignment 可以改善 NAVSIM v1/v2 performance。

#### Knowledge-transfer baselines

- target-only structured teacher distillation；
- DistillDrive-style structured teacher；
- query-only alignment；
- output-only distillation；
- TerraTransfer-style action KL + relational structural loss。

#### Camera normalization baseline

在 source images 可用的子集上，加入 UniDrive-style virtual camera normalization，用于比较：

\[
\text{normalize source images}
\quad \text{versus} \quad
\text{discard source images and transfer planning knowledge}.
\]

---

### 5.4 Main metrics

#### NAVSIM v1

报告：

- Planning Driving Metric Score, PDMS；
- No At-Fault Collision, NC；
- Drivable Area Compliance, DAC；
- Time-to-Collision, TTC；
- Ego Progress, EP；
- Comfort。

#### NAVSIM v2

报告：

- two-stage EPDMS；
- Stage-1 performance；
- Stage-2 counterfactual performance；
- collision and recovery-related submetrics；
- Traffic-Light Compliance；
- Driving-Direction Compliance；
- Lane Keeping；
- Extended Comfort。

NAVSIM v2 的 Stage 2 尤其重要，因为它能够检验 external structured data 是否提高了 planner 在偏离 expert trajectory 后的恢复能力，而非仅改善 logged-state imitation。

#### Bench2Drive

报告：

- Driving Score；
- Success Rate；
- Route Completion；
- Infraction Score；
- Efficiency；
- Comfortness；
- per-scenario-category performance。

#### Diagnostic metrics

- ADE/FDE 或 MinADE；
- teacher–student output consistency；
- collision rate；
- target-image data efficiency；
- inference latency；
- parameter count；
- source-data preprocessing cost。

---

### 5.5 Critical ablation studies

#### A. Source-data scaling

\[
\text{Target}
\rightarrow
\text{Target+A}
\rightarrow
\text{Target+A+B}
\rightarrow
\text{Target+A+B+C}.
\]

验证 performance 是否随着 structured scenario diversity 持续增长。

#### B. Target-image data efficiency

将 target paired image data 设置为：

\[
10\%,25\%,50\%,100\%.
\]

核心结果应展示：

> multi-dataset metadata teacher 能否让使用较少 target images 的模型达到 target-only full-data baseline。

#### C. Data volume versus diversity

控制总样本数量，对比：

- 更多 target-domain duplicated/subsampled data；
- 单一 source dataset；
- 多个 source datasets；
- domain-diverse but size-matched data。

只有 multi-domain data 在等量条件下仍有优势，才能证明增益不只是“more samples”。

#### D. Representation

对比：

- direct metadata tokens；
- canonical BEV raster；
- perspective raster；
- virtual-rig images。

#### E. Transfer location

分别移除：

- query alignment；
- planner-state alignment；
- output distillation；
- GT trajectory loss；
- visibility-aware weighting。

#### F. Negative-transfer analysis

进行 leave-one-dataset-out 实验，并按以下属性分组：

- city；
- road topology；
- interaction type；
- agent density；
- weather/time；
- maneuver category。

分析哪些 source domains 对 target planning 有正迁移，哪些会产生负迁移。

---

### 5.6 Success criteria

metaCAST 至少应满足以下三个条件，才能支持主要 research claim：

1. 在 target image data 相同的条件下，multi-dataset structured teacher 显著优于 target-only structured teacher；
2. 在 source images 与 source camera calibration 完全不使用的条件下，外部 structured datasets 仍带来稳定收益；
3. 收益在 NAVSIM v2 Stage 2 或 Bench2Drive closed-loop evaluation 中依然成立，而不只是降低 open-loop displacement error。

---

## 六、预期贡献（Expected Contributions）

### Contribution 1: New problem formulation

提出 **Camera-Agnostic Cross-Dataset Planning Transfer** 问题，将 autonomous-driving data scaling 分解为：

\[
\text{planning-data scaling}
\neq
\text{sensor-data scaling}.
\]

该问题允许来自不同 vehicle platforms、camera rigs，甚至不含图像的数据集，共同为一个 target-domain camera-only policy 提供规划知识。

### Contribution 2: metaCAST framework

提出 metaCAST，一种由以下模块组成的统一框架：

- canonical structured scene encoding；
- multi-dataset planner pretraining；
- shared planner interface；
- hierarchical structured-to-visual distillation；
- visibility-aware knowledge transfer。

与 RAP 的 perspective raster transfer 不同，metaCAST 的 source-stage representation 不依赖任何 source camera geometry；与一般 privileged distillation 不同，metaCAST 明确研究跨真实数据集的 planning knowledge aggregation。

### Contribution 3: Empirical scaling study

建立系统的 multi-dataset planning transfer evaluation protocol，量化：

- source dataset 数量；
- source scenario diversity；
- target paired-image ratio；
- representation choice；
- cross-dataset domain mismatch；

对最终 camera-only planning performance 的影响。

该研究将回答一个此前尚未被充分验证的问题：

> **How much can structured driving data from sensor-incompatible datasets contribute to a deployable vision-based planner?**

---

## 七、潜在风险与降级预案（Limitations & Mitigations）

### Risk 1: Dataset mismatch may cause negative transfer

不同数据集在以下方面存在系统差异：

- traffic rules；
- map semantics；
- vehicle dynamics；
- annotation quality；
- sampling rate；
- route availability；
- logged driver style。

因此，简单混合数据可能导致 teacher 学习相互冲突的 driving behaviors。

#### Mitigations

1. 使用 dataset-specific canonical adapters 和 availability masks；
2. 采用 dataset-balanced sampling，避免最大数据集主导训练；
3. 加入 dataset-conditioned normalization 或 lightweight domain adapters；
4. 使用 leave-one-dataset-out validation 估计每个 source domain 的 transfer value；
5. 当 full multi-dataset training 发生负迁移时，降级为：
   - target + selected source datasets；
   - structured pretraining followed by target metadata finetuning；
   - mixture-of-experts structured encoder with shared planner。

---

### Risk 2: Structured teacher possesses information unavailable to cameras

Structured teacher 能够直接访问准确 agent states、遮挡对象和地图属性，而 visual student 受到 occlusion、有限 field of view 和 perception ambiguity 的限制。

若对全部 structured features 进行强制 pointwise alignment，student 可能：

- 学习不可实现的 representation；
- 产生虚假 certainty；
- 过度依赖 teacher 的 privileged information；
- 在真实图像上出现 optimization conflict。

#### Mitigations

1. 使用 visibility-aware token masking；
2. 将直接 feature imitation 限于 visually observable information；
3. 对不可见信息只使用 uncertainty-weighted output distillation；
4. 保留 GT trajectory supervision，避免 student 完全复制 teacher error；
5. 将 pointwise feature matching 降级为 relational or planner-output alignment。

若 feature-level alignment 不稳定，最小可行版本可采用：

\[
\mathcal{L}
=
\lambda_{\mathrm{out}}\mathcal{L}_{\mathrm{out}}
+
\lambda_{\mathrm{GT}}\mathcal{L}_{\mathrm{GT}},
\]

仅迁移 teacher 的 planning distribution，而不要求 image feature 重建完整 metadata latent。

---

## Proposal Positioning Statement

metaCAST 不应被描述为简单的 “metadata fusion”，也不应将 novelty 建立在“structured teacher can supervise a visual student”之上。

更准确的定位是：

> **metaCAST is a camera-agnostic planning-data scaling framework that learns a shared planner from canonicalized structured scenes pooled across heterogeneous real-world driving datasets, and transfers the acquired planning competence to a target camera-only policy using paired data exclusively from the target sensor domain.**

其最关键、也最需要实验验证的 claim 是：

> **External driving datasets can improve a target image-only planner even when their images and camera calibrations are never used.**

---

## References

1. Li, Y., Zheng, W., Huang, X., & Keutzer, K. **UniDrive: Towards Universal Driving Perception Across Camera Configurations.** ICLR 2025. <https://arxiv.org/abs/2410.13864>
2. Li, Q., et al. **ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling.** NeurIPS 2023 Datasets and Benchmarks. <https://arxiv.org/abs/2306.12241>
3. Feng, L., et al. **UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction.** ECCV 2024. <https://arxiv.org/abs/2403.15098>
4. Feng, L., et al. **RAP: 3D Rasterization Augmented End-to-End Planning.** <https://arxiv.org/abs/2510.04333>
5. Yu, R., et al. **DistillDrive: End-to-End Multi-Mode Autonomous Driving Distillation by Isomorphic Hetero-Source Planning Model.** <https://arxiv.org/abs/2508.05402>
6. Xiong, Z., et al. **TerraTransfer: Learning End-to-End Driving Policies Without Expert Demonstrations.** <https://arxiv.org/abs/2606.17386>
7. Dauner, D., et al. **NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking.** NeurIPS 2024. <https://arxiv.org/abs/2406.15349>
8. Jia, X., et al. **Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving.** NeurIPS 2024 Datasets and Benchmarks. <https://arxiv.org/abs/2406.03877>
9. Ettinger, S., et al. **Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset.** ICCV 2021. <https://arxiv.org/abs/2104.10133>
10. Wilson, B., et al. **Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting.** NeurIPS 2021 Datasets and Benchmarks. <https://arxiv.org/abs/2301.00493>
