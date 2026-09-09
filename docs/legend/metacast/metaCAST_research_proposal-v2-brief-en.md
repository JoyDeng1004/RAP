# metaCAST - Initial Discussion Draft

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

> **Goal of this discussion:** First align on metaCAST's **problem formulation, system principle, and validation logic**. Specific datasets, scene representations, and transfer losses will be determined later through a survey and experiments.

---

## 1. Problem - What problem are we trying to solve?

### 1.1 Core research question

Scaling data for vision-based end-to-end driving is usually tied to a specific camera rig. Datasets differ in the number and arrangement of cameras, intrinsics and extrinsics, fields of view, and temporal configurations. Reusing source images directly therefore requires additional sensor normalization.

metaCAST attempts to bypass this dependency:

> **Can external driving datasets improve a target image-only planner even when their images and camera calibrations are never used?**

In other words, the core claim is not that all source sensors should be unified, but that we can:

> **Decouple planning-data scaling from sensor-data scaling.**

External datasets provide camera-independent structured scenes that expand the road structures, traffic behaviors, and interaction patterns observed by the planner. Only paired data from the target domain is used to ground this competence in the target camera rig.

### 1.2 Problem setting

- Source structured datasets: $\mathcal{D}_k=\{(s_i^k,y_i^k)\}_{i=1}^{N_k}$, which are not required to provide images or share a camera configuration.
- Target paired dataset: $\mathcal{D}_T=\{(x_i^T,s_i^T,y_i^T)\}_{i=1}^{N_T}$.
- Training may use $\bigcup_k\mathcal{D}_k$. At deployment time, the policy accesses only target images and the required route command:

\[
\pi_{\mathrm{img}}:(x^T,r^T)\rightarrow \hat y^T,
\qquad
\pi_{\mathrm{img}}^{\mathrm{test}}\ \text{does not access}\ s^T,\ \text{source images, or source calibrations}.
\]

### 1.3 Novelty boundary

- [UniDrive](https://arxiv.org/abs/2410.13864) unifies visual observations across different camera configurations. It still uses source images and calibration and primarily targets perception generalization.
- [ScenarioNet](https://arxiv.org/abs/2306.12241) already unifies heterogeneous structured scenario schemas. Canonicalization should therefore be treated as infrastructure rather than the main contribution.
- [RAP](https://arxiv.org/abs/2510.04333) does not require real source images for its source raster data, but its representation remains tied to perspective geometry.
- [DistillDrive](https://arxiv.org/abs/2508.05402) and [TerraTransfer](https://arxiv.org/abs/2606.17386) have established the transfer of a structured or state-based policy to a visual policy. Teacher-student transfer alone is therefore not novel.

> **metaCAST does not claim novelty from canonicalization or distillation as individual components. It proposes and tests camera-agnostic, cross-dataset planning-data scaling as the central research question.**

<div style="break-after: page;"></div>

## 2. Principle - How should the system be decomposed?

### 2.1 Core decomposition

The current hypothesis separates **learning to drive** from **learning to see**:

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

- $C_k$ converts each dataset into an ego-centric canonical structured scene $z$.
- $E_{\mathrm{meta}}$ learns scene and interaction representations from structured scenes.
- $P$ is intended to become a reusable, camera-agnostic shared planner.
- $B_{\mathrm{vis}}$ learns visual perception and grounding specifically for the target camera rig.
- $q$ is the planner-facing interface shared by the two modalities.

This decomposition must be confirmed in the discussion. If only the final planning behavior needs to be shared, the method does not have to enforce a physically shared decoder. If the shared planner itself is central to the idea, the subsequent architecture and experiments must be designed around it.

### 2.2 Minimal training procedure

**Stage A - Structured planning pretraining**

\[
\bigcup_k \mathcal{D}_k
\xrightarrow[\text{no source images/calibration}]{\text{canonical structured scenes}}
E_{\mathrm{meta}}+G_{\mathrm{meta}}+P .
\]

The goal is to determine whether cross-dataset structured training produces planning competence that transfers to the target domain.

**Stage B - Target visual grounding**

Using only the paired images and structured scenes in $\mathcal{D}_T$, the visual branch is connected to the same planner. The MVP starts with the weakest assumption:

\[
\mathcal{L}_{\mathrm{MVP}}
=
\lambda_{\mathrm{GT}}\mathcal{L}_{\mathrm{GT}}
+
\lambda_{\mathrm{out}}\mathcal{L}_{\mathrm{out}} .
\]

Planner-query or relational alignment, metadata noise or dropout, visibility-aware transfer, and joint refinement should be considered only if output-level transfer proves insufficient. These are candidate engineering mechanisms rather than predefined core contributions.

### 2.3 Two key risks

1. **Planning-semantics mismatch:** A unified schema does not imply a unified planning task. If a source dataset lacks a route or command, its logged ego future may be closer to behavior prediction than conditional planning.
2. **Privileged-information mismatch:** The structured teacher may rely on precise state information or occluded agents that cannot be observed by the cameras, making its representation impossible for the visual student to reproduce.

Both risks must be audited in the survey and MVP, but this initial discussion will not expand into specific dataset or loss choices.

<div style="break-after: page;"></div>

## 3. Evidence - What results would support the direction?

### 3.1 Two causal checkpoints

**Checkpoint A: Can planning knowledge accumulate across datasets?**

Under the same architecture, training budget, and target evaluation, compare:

\[
\text{target-only structured planner}
\quad\text{vs.}\quad
\text{target + external structured-data planner}.
\]

If the latter does not provide a stable improvement, planning compatibility, canonicalization, and negative transfer should be investigated before adding visual-transfer mechanisms.

**Checkpoint B: Does this advantage survive in an image-only policy?**

Fix the target images, visual architecture, grounding procedure, and compute budget:

\[
\begin{aligned}
\text{target-only teacher} &\rightarrow \text{visual student A},\\
\text{multi-dataset teacher} &\rightarrow \text{visual student B}.
\end{aligned}
\]

The core claim is directly supported only if student B consistently outperforms student A in target-domain planning evaluation. The final evidence should come from planning-aware or closed-loop evaluation; open-loop error should be used only for debugging.

### 3.2 Three questions for the initial discussion

1. **Problem formulation**

   > After rereading UniDrive, I understand the goal as using sensor-incompatible structured datasets to scale planning knowledge, rather than normalizing their images. **Is this the central problem you originally had in mind?**

2. **System principle**

   > I assume the planner is the reusable camera-agnostic component, while the target visual encoder learns its interface. **Is this separation essential, or do you imagine the two branches being trained more jointly?**

3. **Validation logic**

   > I propose testing cross-dataset structured transfer first, then whether the gain survives visual grounding under the same target images. **Does this two-checkpoint logic provide the right evidence?**

### 3.3 Intended outcomes of the discussion

- Confirm whether metaCAST's main problem is **camera-agnostic planning-data scaling**.
- Confirm whether a shared planner with target-specific visual grounding is an essential system principle.
- Confirm whether the two-stage validation logic of **structured transfer followed by image-only transfer** is appropriate.

After aligning on these three high-level judgments, independently complete the source-dataset audit and representation survey, and define the first go/no-go experiment.
