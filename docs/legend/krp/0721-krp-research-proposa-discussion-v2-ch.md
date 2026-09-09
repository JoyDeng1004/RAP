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

> ①②两条在**图 2(a) 左半**画在了一起：三个历史帧原封不动压在车道中心线上，当前帧被一道"teleport"箭头瞬移到 0.5 m / 15° 的位置。

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

- 我们把恢复代价定义为**规划器输出轨迹** $\tau$ 的函数：

$$\mathcal L_{\text{rec}}=\mathcal L\big(\tau\big),\qquad \tau=\pi_\theta\big(\,o(s^0(w))\,\big)$$

- 在这个定义下，$w$ **只能**经由 $o$ 影响 $\mathcal L_{\text{rec}}$，因此

$$\boxed{\ \frac{d\mathcal L_{\text{rec}}}{dw}\bigg|_{\text{direct}}\equiv 0\qquad\text{（严格成立）}\ }$$

- 对照 KING：它的代价 $\mathcal C(s)$ 是几何量，直接路不碰观测就能抵达；我们的代价是**策略行为**的函数，**观测路径是唯一路径**。
- **KING 有本钱删掉观测路；我们没有——删掉之后什么都不剩。**

### 2.4 图 1

![图 1 — KING 与 KRP 的梯度路径对比](./figs/fig1_gradient_paths.png)

**图 1｜(a) KING：** 代价 $\mathcal C$ 是几何量，蓝色 direct path $a_t^{i>0}\!\to\!\kappa\!\to\! s_{t+1}\!\to\!\mathcal C_{t+1}$ **不经过渲染器**因而存活；红色 indirect path 在渲染器处被剪断并丢弃，方法照样成立。
**(b) KRP：** **不存在 direct path**（右上灰框，$\equiv 0$）。唯一的红色路径 $\mathcal L_{\text{rec}}\!\to\!\tau\!\to\!\pi_\theta\!\to\! o\!\to\!\mathcal R_{\text{diff}}\!\to\! s^0\!\to\!\kappa\!\to\! w_t$ **必须穿过渲染器**。链上四个 Jacobian 因子中，$\partial\pi_\theta/\partial o$ 由自动微分给出、$\partial s/\partial w$ 是闭式，唯一的瓶颈是 $\partial o/\partial s$——而**可微光栅化正是使它存在的东西**。

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
\underbrace{\frac{\partial\mathcal L_{\text{rec}}}{\partial\tau}}_{\text{(i)}}
\;\underbrace{\frac{\partial\pi_\theta}{\partial o_t}}_{\text{(ii) 自动微分}}
\;\underbrace{\frac{\partial o_t}{\partial s_t}}_{\text{(iii) 观测模型}}
\;\underbrace{\frac{\partial s_t}{\partial w}}_{\text{(iv) 自行车模型：闭式}}$$

- 这四项**逐一对应图 1(b) 底部的四个红色花括号**。
- (i)(ii) 由自动微分给出，(iv) 是闭式。**只有 (iii) 是瓶颈。**

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
\mathcal L_{\text{rec}}\!\left(\pi_\theta,\,o(s_{-K:T})\right)>\varepsilon & \text{学习器失败}\\[2pt]
w_t\in\mathcal W(v_t) & \text{执行器限制}
\end{cases}$$

- 直观：**找到当前规划器无法恢复的、最小的物理可行扰动。**
- 注意目标是**最小范数**：不是"让规划器尽可能崩"（那是梯度上升），而是"刚好越过失败阈值 $\varepsilon$ 的最小扰动"。
- 最后一条约束来自非完整性：$\omega=\dfrac{v\tan\delta_{\text{steer}}}{L}\ \Longrightarrow\ v=0\Rightarrow\omega=0$
  → **静止时航向根本无法被扰动**；当前实现违反了这一点，我们的形式化强制满足它，因此扰动幅度必然变成**速度相关**的。

### 3.5 两处代码级修改

- **M1｜扰动参数化** —— 把"状态改写"换成"控制空间扰动经 $\kappa$ 前向演化"，且**覆盖整个历史窗口**。
  → 对应**图 2(a) 右半**（同起点、同终点，但整段轨迹连续）与**图 2(b) 的 $\kappa$ 节点**。
- **M2｜可微渲染** —— 把光栅化器换成**可微（软）光栅化器**，primitive 不变。
  → 对应**图 2(b) 的 $\mathcal R_{\text{diff}}$**，即 §3.2 中唯一的瓶颈因子 $\partial o/\partial s$。

这两处修改合起来使得从 $w$ 到 $\mathcal L_{\text{rec}}$ 的整条链路**全解析可微**，于是 (E3) 可以用一阶方法求解。

### 3.6 挖掘循环与训练用法

- **挖掘**：对每个场景，从 $w=0$ 出发，沿 $\partial\mathcal L_{\text{rec}}/\partial w$ 迭代，直到满足 $\mathcal L_{\text{rec}}>\varepsilon$ 且 $\|w\|_W$ 最小，得到 $w^*$。
- **训练**：$w^*$ 定义一段扰动 clip → 在 **7 相机 × $K$ 帧**上重新渲染 → 由**特权专家**给出恢复目标 → 以**固定预算**混入训练集。
- 固定预算是必要的：所有基线（B0–B4）必须消耗同样多的增强样本，否则 RQ2 的对比不成立。

### 3.7 图 2

![图 2 — KRP 的扰动几何与挖掘循环](./figs/fig2_krp.png)

**图 2｜(a) 扰动几何。** 左：RAP 现版——三个历史帧原封不动留在车道中心线上，当前帧被一道 teleport 瞬移到终点；由此 ① 该状态在运动学上不可达（$a_{\text{lat}}\approx10.5\ \text{m/s}^2$），② 偏移量可由帧间不一致 $\hat\xi\approx s_0-\kappa(s_{-1},a_{-1})$ 直接读出，构成捷径。右：KRP——**起点与终点和左图完全相同**，但整段轨迹由 $s_{k+1}=\kappa(s_k,a_k^{\log}+w_k)$ 演化而来，因而构造上可行，且历史帧被一致地重新渲染。**同一个目标偏移，只有右边这条路是车能走的。**

**(b) 挖掘循环与训练用法。** 扰动 $w_{-K:0}$ 与日志动作 $a^{\log}$ 相加后经 $\kappa$ 得到状态序列，由 $\mathcal R_{\text{diff}}$（结合标注 3D primitive 与来自日志、保持不变的背景车状态 $s^{i>0}$）渲染出 7 相机 × $K$ 帧观测，送入规划器 $\pi_\theta$ 得到轨迹 $\tau$ 与恢复代价 $\mathcal L_{\text{rec}}$。红色虚线是全解析的梯度回流 $\partial\mathcal L_{\text{rec}}/\partial w$；据此迭代求解 $\min_w\|w\|_W$ s.t. $\mathcal L_{\text{rec}}>\varepsilon$。求得的 $w^*$ 用于重渲染 clip、配以特权专家的恢复目标，按固定预算并入训练集。**橙色标记的两处（$w$ 的控制空间参数化与 $\mathcal R_{\text{diff}}$）即 M1 与 M2。**

