# metaCAST — 首次讨论版

**Metadata-based Camera-Agnostic Structured Transfer for Scalable End-to-End Autonomous Driving**

<style>
@media print {
  body { font-size: 9.4pt; line-height: 1.35; }
  h1 { font-size: 17pt; margin-bottom: 8px; }
  h2 { font-size: 13pt; margin-top: 14px; margin-bottom: 8px; }
  h3 { font-size: 10.8pt; margin-top: 11px; margin-bottom: 5px; }
  p, ul, ol { margin-top: 5px; margin-bottom: 6px; }
  li { margin-bottom: 2px; }
  blockquote { margin: 7px 0; padding: 1px 10px; }
}
</style>

> **本次讨论目标**：先对齐 metaCAST 的 **problem formulation、system principle 与 validation logic**。具体数据集、scene representation 和 transfer loss 留待后续 survey 与实验决定。

---

## 1. Problem — 我们究竟要解决什么问题？

### 1.1 核心研究问题

视觉端到端驾驶的数据扩展通常与 camera rig 绑定；不同数据集的相机数量、内外参、视场角和时序配置不同，直接复用 source images 需要额外的 sensor normalization。

metaCAST 尝试绕开这一依赖：

> **Can external driving datasets improve a target image-only planner even when their images and camera calibrations are never used?**

换言之，核心主张不是统一所有 source sensors，而是：

> **Decouple planning-data scaling from sensor-data scaling.**

外部数据集用 camera-independent structured scenes 扩大 planner 接触到的道路结构、交通行为与交互模式；只有目标域 paired data 负责把这些 competence grounding 到目标 camera rig。

### 1.2 问题设定

- Source structured datasets：$\mathcal{D}_k=\{(s_i^k,y_i^k)\}_{i=1}^{N_k}$，不要求提供图像或共享 camera configuration。
- Target paired dataset：$\mathcal{D}_T=\{(x_i^T,s_i^T,y_i^T)\}_{i=1}^{N_T}$。
- 训练阶段可使用 $\bigcup_k\mathcal{D}_k$；部署时 policy 只访问 target images 与必要的 route command：

\[
\pi_{\mathrm{img}}:(x^T,r^T)\rightarrow \hat y^T,
\qquad
\pi_{\mathrm{img}}^{\mathrm{test}}\ \text{does not access}\ s^T,\ \text{source images, or source calibrations}.
\]

### 1.3 Novelty boundary

- [UniDrive](https://arxiv.org/abs/2410.13864) 统一不同 camera configurations 下的视觉观测，仍使用 source images/calibration，主要面向 perception generalization。
- [ScenarioNet](https://arxiv.org/abs/2306.12241) 已统一异构 structured scenario schema，因此 canonicalization 只能作为 infrastructure。
- [RAP](https://arxiv.org/abs/2510.04333) 的 source raster 不使用真实图像，但 representation 仍与 perspective geometry 相关。
- [DistillDrive](https://arxiv.org/abs/2508.05402) / [TerraTransfer](https://arxiv.org/abs/2606.17386) 已建立 structured/state policy 到 visual policy 的迁移，故 teacher–student transfer 本身也不是 novelty。

> **metaCAST 不是凭 canonicalization 或 distillation 单个组件主张 novelty，而是提出并验证 camera-agnostic, cross-dataset planning-data scaling 这一研究问题。**

<div style="break-after: page;"></div>

## 2. Principle — 系统应如何分工？

### 2.1 核心 decomposition

当前假设是把 **learning to drive** 与 **learning to see** 分开：

\[
\begin{aligned}
z=C_k(s^k)
&\xrightarrow{E_{\mathrm{meta}},\,G_{\mathrm{meta}}}
q_{\mathrm{meta}}
\xrightarrow{P}
\hat y,\\
x^T
&\xrightarrow{B_{\mathrm{vis}},\,G_{\mathrm{img}}}
q_{\mathrm{img}}
\xrightarrow{P}
\hat y .
\end{aligned}
\]

- $C_k$：把各数据集转换为 ego-centric canonical structured scene $z$；
- $E_{\mathrm{meta}}$：从 structured scenes 学习场景与交互表示；
- $P$：预期成为 camera-agnostic、可复用的 shared planner；
- $B_{\mathrm{vis}}$：只针对 target camera rig 学习 visual perception/grounding；
- $q$：两个模态与 planner 之间的 planner-facing interface。

这一 decomposition 需要在讨论中确认。若真正共享的只是 planning behavior，方法不必强制共用 decoder；若 shared planner 是核心，后续架构与实验必须围绕它设计。

### 2.2 最小训练流程

**Stage A — Structured planning pretraining**

\[
\bigcup_k \mathcal{D}_k
\xrightarrow[\text{no source images/calibration}]{\text{canonical structured scenes}}
E_{\mathrm{meta}}+G_{\mathrm{meta}}+P .
\]

目标是验证跨数据集 structured training 是否形成可迁移至 target domain 的 planning competence。

**Stage B — Target visual grounding**

只使用 $\mathcal{D}_T$ 中的 paired images 与 structured scenes，使视觉分支接入同一 planner。MVP 从最弱假设开始：

\[
\mathcal{L}_{\mathrm{MVP}}
=
\lambda_{\mathrm{GT}}\mathcal{L}_{\mathrm{GT}}
+
\lambda_{\mathrm{out}}\mathcal{L}_{\mathrm{out}} .
\]

仅当 output-level transfer 不足时，再考虑 planner-query / relational alignment、metadata noise/dropout、visibility-aware transfer 或 joint refinement；这些只是候选 engineering mechanisms。

### 2.3 当前保留的两个关键风险

1. **Planning semantics mismatch**：统一 schema 不等于统一 planning task。若 source data 缺少 route/command，logged ego future 可能更接近 behavior prediction，而非条件化 planning。
2. **Privileged-information mismatch**：structured teacher 可能使用 camera 看不到的精确状态或遮挡目标，导致 visual student 无法复现 teacher representation。

两者都需在 survey 与 MVP 中审计，但本次不展开具体 dataset 或 loss。

<div style="break-after: page;"></div>

## 3. Evidence — 什么结果才说明方向成立？

### 3.1 两个因果 checkpoint

**Checkpoint A：planning knowledge 能否跨数据集积累？**

在相同 architecture、训练预算和 target evaluation 下比较：

\[
\text{target-only structured planner}
\quad\text{vs.}\quad
\text{target + external structured-data planner}.
\]

若后者没有稳定提升，应先检查 planning compatibility、canonicalization 与 negative transfer，而非增加 visual transfer 机制。

**Checkpoint B：这种优势能否保留到 image-only policy？**

固定相同 target images、visual architecture、grounding procedure 与计算预算：

\[
\begin{aligned}
\text{target-only teacher} &\rightarrow \text{visual student A},\\
\text{multi-dataset teacher} &\rightarrow \text{visual student B}.
\end{aligned}
\]

只有 student B 在 target-domain planning evaluation 中稳定优于 student A，才能直接支持核心主张。最终证据应来自 planning-aware / closed-loop evaluation；open-loop error 只用于调试。

### 3.2 首次讨论只聚焦三个问题

1. **Problem formulation**

   > After rereading UniDrive, I understand the goal as using sensor-incompatible structured datasets to scale planning knowledge, rather than normalizing their images. **Is this the central problem you originally had in mind?**

2. **System principle**

   > I assume the planner is the reusable camera-agnostic component, while the target visual encoder learns its interface. **Is this separation essential, or do you imagine the two branches being trained more jointly?**

3. **Validation logic**

   > I propose testing cross-dataset structured transfer first, then whether the gain survives visual grounding under the same target images. **Does this two-checkpoint logic provide the right evidence?**

### 3.3 希望本次讨论得到的结论

- 确认 metaCAST 的主问题是否是 **camera-agnostic planning-data scaling**；
- 确认 shared planner + target-specific visual grounding 是否是必要的系统原则；
- 确认是否接受 **structured transfer → image-only transfer** 的两阶段验证逻辑。

在这三个 high-level 判断对齐后，再独立完成 source-dataset audit、representation survey，并定义第一个 go/no-go experiment。
