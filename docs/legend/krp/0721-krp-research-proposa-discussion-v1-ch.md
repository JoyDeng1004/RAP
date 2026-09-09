# 研究提案 —— Kinematic Recovery Perturbation (KRP)

**题目（暂定）**：*KRP: Policy-Aware, Kinematically-Consistent Recovery Perturbations for Camera-Based End-to-End Planners*

**摘要**：RAP 的 recovery 扰动是**人工设计、单帧瞬移**的，物理上不可实现；我们把它改成**运动学可行的控制序列**，并用**梯度**而非人工来选择扰动——而这条梯度链只有在 RAP 的**光栅化**观测下才存在。

---

## 1. 问题：RAP 的 recovery 扰动在物理上不可实现

- RAP 的前提：模仿学习的规划器在**闭环**中失败，因为它从未见过恢复状态。
- RAP 的对策：用光栅化廉价地渲染脱离日志的恢复状态。
- **但这些状态是怎么产生的？** 已发布代码 `process_data/create_openscene_metadata_purturbed.py`（`get_ego_params` L246–262，逐帧门控 L330–336）中，14 帧窗口里**只有 `batch[3]` 一帧**被扰动：

$$\textbf{(E1)}\qquad s_0 \leftarrow s_0 + \xi,\qquad \xi\sim\mathcal U(\Xi),\qquad t\neq 0 \text{ 时 } s_t\ \text{不变}$$

$$\Delta x,\Delta y\sim\mathcal U(-0.5,0.5)\,\text{m},\quad \Delta\psi\sim\mathcal U(-15^\circ,15^\circ),\quad \Delta v = \mathcal U(-0.2,0.2)\cdot v$$

### 三个后果

**① 运动学不可行。** `NAVSIM_INTERVAL_LENGTH = 0.5` s，代入：

$$\omega=\frac{0.26\ \text{rad}}{0.5\ \text{s}}=0.52\ \text{rad/s}\ \Longrightarrow\ a_{\text{lat}}=v\omega\Big|_{v=20}\approx 10.5\ \text{m/s}^2$$

- 超出附着极限 → **训练状态在闭环中根本无法到达**。
- 这是一行力学计算，且位于核心假设而非实现细节。

**② 制造了一条捷径。** 历史帧未被扰动，所以偏移量可直接从帧间不一致读出：

$$\hat\xi \;\approx\; s_0 - \kappa(s_{-1},a_{-1})$$

- 网络会学"检测跳变、反向抵消"，而不是"从图像理解车道在哪"。
- 属于 copycat / 因果混淆（de Haan 2019；Codevilla 2019），只是这次由增强手段自己制造。

**③ 静止时扰动失效。** 速度扰动是乘性的，$v=0\Rightarrow\Delta v=0$ —— 停车/起步场景（inertia problem 高发区）完全没有有效扰动。

---

## 2. 参考方法 KING，以及它为什么不能直接搬过来

### 2.1 KING 是什么

- KING（ECCV 2022）做的是**安全关键场景生成**：给定一个固定的自车策略，去优化**背景车**的行为，使其撞上自车。
- 它的关键主张：**不要手工编辑状态，要编辑动作**，并让可微运动学模型去产生状态：

$$\textbf{(E2)}\qquad s_{t+1}=\kappa\big(s_t,\ a_t+w_t\big)$$

- 好处：这样产生的状态**在构造上就是物理可达的**，而且可以用梯度去优化。
- **这正是 RAP 的 recovery 扰动所缺的东西**——场景生成领域从"人工设参"到"运动学约束优化"的转折发生在 KING，但**自车侧的恢复扰动至今仍停留在 KING 之前**。

### 2.2 KING 的梯度分两条路

$$\frac{d\mathcal C}{d\theta}
=\underbrace{\frac{d\mathcal C}{d\theta}\bigg|_{\text{direct}}}_{\text{背景车自身的运动}}
+\underbrace{\frac{d\mathcal C}{d\theta}\bigg|_{\text{indirect}}}_{\text{经由自车的反应}}$$

- **直接路**：$a_t^{i>0}\to s_{t+1}^{i>0}\to\mathcal C$，即

$$\frac{d\mathcal C}{da_t^{i>0}}\bigg|_{\text{direct}}
=\frac{\partial\mathcal C}{\partial s_{t+1}^{i>0}}\cdot\frac{\partial\kappa(s_t^{i>0},a_t^{i>0})}{\partial a_t^{i>0}}$$

  两个因子都是闭式的（距离公式求导 + 自行车模型求导）。**观测 $o_t$ 从未出现。**

- **间接路**：$a_t^{i>0}\to s_{t+1}^{i>0}\to \boxed{o_{t+1}}\to \pi(o_{t+1})\to s^0_{t+2}\to\mathcal C$
  需要 $\partial o/\partial s$ —— CARLA 渲染器是黑箱，**算不出来**。

- **KING 的做法：直接丢掉间接路。** 剩下的梯度有偏但非零、方向正确（"把车往自车将到的位置挪"），所以方法照样成立。
- **它能这么做的前提：代价 $\mathcal C$ 是几何量**（碰撞距离），所以直接路不碰观测就能抵达代价。

### 2.3 但这个前提在自车恢复问题上不成立

- 我们的代价是"**学习器恢复不了**"：

$$\mathcal L_{\text{rec}}=\mathcal L\Big(\pi_\theta\big(\,o(s^0(w))\,\big)\Big)$$

- 枚举从 $w$ 到 $\mathcal L_{\text{rec}}$ 的路线：
  - 存在一个"直接"项（$t=0$ 的横向偏移是 $w$ 的显式函数），**但它毫无信息量**——它只是复述了我们自己注入的扰动，对*这个规划器能否恢复*只字未提。
  - **所有携带学习器信息的路线都经过 $\pi_\theta$，因而都经过 $o$。**

$$\boxed{\ \frac{d\mathcal L_{\text{rec}}}{dw}\bigg|_{\text{direct}}\equiv 0\ }$$

- **KING 有本钱删掉观测路；我们没有——删掉之后什么都不剩。**

### 2.4 图 1

> **【图片位置：图 1 — KING 与 KRP 的梯度路径对比】**
>
> `![图 1](./figs/fig1_gradient_paths.png)`
>
> **图注**：(a) KING 的代价是几何量，蓝色 direct path 不经过渲染器因而存活；红色 indirect path 在渲染器处被剪断并丢弃。(b) KRP 不存在 direct path，唯一的红色路径必须穿过渲染器；可微光栅化正是使这条路径可通行的东西。

---

## 3. 方法：KRP

### 3.1 与两个先行工作的并列对比

| | 优化变量 | 约束 | 代价 | 选择方式 |
|---|---|---|---|---|
| **RAP** | 状态偏移 $\xi$ | 无 | 无 | $\xi\sim\mathcal U(\Xi)$，人工固定 |
| **KING** | 背景车动作 $\theta$ | $s^{i>0}_{t+1}=\kappa(\cdot)$ | **几何量**（碰撞距离） | $\nabla_\theta\mathcal C$ |
| **KRP（本文）** | **自车扰动 $w_{-K:0}$** | $s^{0}_{t+1}=\kappa(s^{0}_t,a^{\text{log}}_t\!+\!w_t)$ | **感知驱动** | $\nabla_w\mathcal L_{\text{rec}}$ |

- 前两列：KRP 继承 KING 的**参数化**。
- 第三列：难点所在——代价从几何变成感知驱动，于是必须算出下面这一项。

### 3.2 我们必须计算的梯度

$$\frac{\partial\mathcal L_{\text{rec}}}{\partial w}
=\sum_t
\underbrace{\frac{\partial\mathcal L_{\text{rec}}}{\partial\pi_\theta}\frac{\partial\pi_\theta}{\partial o_t}}_{\text{(i) 规划器：自动微分，免费}}
\;\underbrace{\frac{\partial o_t}{\partial s_t}}_{\text{(ii) \textbf{观测模型}}}
\;\underbrace{\frac{\partial s_t}{\partial w}}_{\text{(iii) 自行车模型：闭式}}$$

- (i) 和 (iii) 都是免费的。**(ii) 就是全部难点。**

### 3.3 双向锁死：为什么必须是 RAP 的光栅化

| 观测来源 | $\partial o/\partial s$ | 原因 |
|---|---|---|
| 记录的真实视频 | **不存在** | 车没去过那个位姿，那张图就不存在——不是算不出，是没有对象可算 |
| CARLA / 游戏引擎 | 不可微 | 黑箱；这正是 KING 当年被迫丢掉间接路的原因 |
| NeRF / 3DGS | 学出来的、外插下退化、慢 $10^2$–$10^3$ 倍 | 而扰动状态恰恰**住在外插区**；放不进内层循环 |
| **光栅化 primitive（RAP）** | **解析、精确、廉价** | 场景是已知几何体；$s\mapsto o$ 是闭式链 $T(\delta)\to$ 投影 $\to$（软）光栅化 |

$$\text{KING 用于自车}\Longrightarrow\text{需要 }\tfrac{\partial o}{\partial s}\Longrightarrow\text{需要光栅化观测}$$
$$\text{RAP 恢复增强}\Longrightarrow\text{需要物理可实现}\Longrightarrow\text{需要运动学参数化}$$

- 两个方向都不是修辞：前者由实验 3 检验，后者由实验 1 检验。

### 3.4 优化问题

$$\textbf{(E3)}\qquad
w^{*}_{-K:0}=\arg\min_{w}\|w\|_{W}
\quad\text{s.t.}\quad
\begin{cases}
s_{t+1}=\kappa\!\left(s_t,\ a^{\text{log}}_t+w_t\right) & \text{构造上可行}\\[2pt]
\mathcal L_{\text{rec}}\!\left(\pi_\theta,\,o(s_{-K:T})\right)>\tau & \text{学习器失败}\\[2pt]
w_t\in\mathcal W(v_t) & \text{执行器限制}
\end{cases}$$

- 直观：**找到当前规划器无法恢复的、最小的物理可行扰动。**
- 最后一条约束来自非完整性：$\omega=\dfrac{v\tan\delta_{\text{steer}}}{L}\ \Longrightarrow\ v=0\Rightarrow\omega=0$
  → **静止时航向根本无法被扰动**；当前实现违反了这一点，我们的形式化强制满足它，因此扰动幅度必然变成**速度相关**的。

### 3.5 两处代码级修改

- **M1** —— 把"状态改写"换成"控制空间扰动经 $\kappa$ 前向演化"，且**覆盖整个历史窗口**。
- **M2** —— 把光栅化器换成**可微（软）光栅化器**，primitive 不变。

> **【图片位置：图 2 — KRP pipeline】**
>
> `![图 2](./figs/fig2_pipeline.png)`

```
a^log        w_{-K:0}
  │             │
  └──► κ（可微自行车模型）──► s_{-K:0}
                                  │
      标注 3D primitive ──────────┤
                                  ▼
              可微光栅化器（7 相机 × K 帧）
                                  ▼
                    规划器 π_θ（+ R2R 对齐）
                                  ▼
                              L_rec
                                  │
   ◄────────── ∂L_rec/∂w（全解析）┘
```

---

## 4. 验证方案与风险

### 4.1 三个研究问题

- **RQ1｜参数化重要吗？** (E1) 允许捷径 $\hat\xi\approx s_0-\kappa(s_{-1},a_{-1})$，换成 (E2) 能否消除？
  预测信号：**开环恢复指标降、闭环 EPDMS 升**——与 §1 的旁证一致。
- **RQ2｜同预算下策略感知划算吗？** $\nabla_w$ 挖掘 vs 随机 vs DART vs 覆盖最大化。对抗训练文献表明**最难的样本可能损害泛化**，这是真开放问题。
- **RQ3｜可微光栅化是必要的吗？** 按梯度一致性 / 外插退化 / 耗时，横比各观测模型的 $\partial o/\partial s$。

### 4.2 平台与 leaderboard

- **主榜：NAVSIM v2 `navhard_two_stage`（指标 EPDMS）**
  - 其 Stage 2 **本身就是"从扰动状态初始化并按物理演化"**（77 场景，450 真实 + 5,462 合成帧）。
  - 是唯一一个评测协议**就是**恢复测试的公开 benchmark。
  - ⚠️ 必须写明：我们在**训练侧**做的，正是 Stage 2 在**测试侧**评的——否则会被指过拟合 benchmark。
- 正常性检查：`navtest` EPDMS
- 真闭环：**HUGSIM**（可选 Bench2Drive）

### 4.3 基线（同模型、同渲染器、同样本预算，只换扰动生成器）

| 编号 | 参数化 | 时序 | 选择 | 等价于 |
|---|---|---|---|---|
| B0 | — | — | — | 无增强 |
| **B1** | 状态 $\xi$ | 单帧 | 均匀 | **RAP 现版** |
| **B2** | 控制 $\kappa$ | $K$ 帧 | 均匀 | **DART**（2017） |
| B3 | 控制 $\kappa$ | $K$ 帧 | 覆盖最大化 | — |
| **B4** | 轨迹级，3DGS | 一致 | recovery 伪专家 | **SimScale**（CVPR 2026 Oral） |
| **KRP** | 控制 $\kappa$ | $K$ 帧 | $\nabla_w$ 挖掘 | 本文 |

- **B1→B2 隔离参数化（RQ1）；B2/B3→KRP 隔离策略感知（RQ2）；B4 是最强外部竞争者。**

### 4.4 指标

- **EPDMS 十个子指标必须全部拆开报**：NC、DAC、DDC、TLC、EP、TTC、LK、HC、EC、C
  - **EP 单列**：它是闭环性能最强的单项预测因子（强于 NC）；恢复训练的典型副作用是变保守 → NC 升、EP 降、聚合分不变但闭环变差。
  - **LK 单列**：直接惩罚长时间偏离中心线，是恢复最直接的代理（路口禁用）。
- **本文 headline 指标**（不是 EPDMS）：
  - 训练前后最小致失扰动 $\|w^*\|_W$ 的分布
  - 可恢复域体积与方向半径 $\rho(\theta)$
  - 梯度保真度 $\cos(\nabla^{\text{raster}},\nabla^{\text{FD}})$
  - 达到解所需渲染次数 vs $\dim(w)$，raster vs 3DGS
- ⚠️ **navhard 上最好的非特权结果（≈56.3）距特权 PDM-Closed（≈56.6）只剩 0.3 分。EPDMS 只能当"不退化检查"，不能当 headline。**

### 4.5 三个主要风险与降级预案

| 风险 | 应对 | 降级后还剩什么 |
|---|---|---|
| **梯度可能没信息量**；且 $K{=}3$、2 Hz 时 $\dim(w)\approx6$，有限差分只要 12 次渲染 | ① P1 先测 $\cos(\nabla^{\text{raster}},\nabla^{\text{FD}})$，不过关就不投工程；② 把 $w$ 细分到 10 Hz（$\dim(w)\approx30$）让梯度变成必需 | 用 CMA-ES/二分替代梯度：**RQ1、RQ3 完整存活**，只有 RQ2 的机制变化 |
| **SimScale 已做过大规模 recovery 增强**（CVPR 2026 Oral，跨三种架构） | 差异在三个可检验的轴：参数化（无容许控制约束 vs 构造上可行）、选择（覆盖 vs 失败前沿）、观测模型（§3.3 对 3DGS 不适用）。由 B4 正面对比量化 | 重新定位为**正交可组合**：KRP 出"选哪些"，SimScale 出"怎么渲"，组合成为 headline |
| **可微神经渲染做对抗 2023 年已有**（arXiv:2309.15770） | 四点区别写进 Introduction：扰动变量（环境 vs **自车控制序列**）、目标（攻击 vs **最小范数失败**）、观测模型（NeRF vs **光栅化**）、规模与用途（逐场景/测试 vs **$10^5$ 场景/训练**） | 可扩展性差异源自观测模型，即 §3.3 同一论证 |

### 4.6 里程碑

| 阶段 | 交付 | 关卡 |
|---|---|---|
| P1（2–3 周） | 受控场景上的梯度验证 | **梯度方案 Go/No-Go** |
| P2（3–4 周） | 实现 M1；B1 vs B2 | 回答 RQ1 |
| P3（4–6 周） | 实现 M2；KRP 挖掘循环；B2/B3/KRP | 回答 RQ2 |
| P4（2–3 周） | 观测模型研究 + B4 对比；HUGSIM 闭环 | 回答 RQ3；成稿 |

- **明确排除**（留待后续）：对抗性背景车、逐相机边界归因、等变性目标、world model rollout、跨架构迁移。

---

**参考**：RAP (2025) · KING (ECCV'22) · DART (CoRL'17) · de Haan et al. (NeurIPS'19) · Codevilla et al. (ICCV'19) · NAVSIM (NeurIPS'24) · SimScale (CVPR'26 Oral) · arXiv:2309.15770 · HUGSIM (2024) · Soft Rasterizer (ICCV'19) · nvdiffrast (SIGGRAPH Asia'20)
