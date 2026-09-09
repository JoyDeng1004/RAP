# Research Proposal — Kinematic Recovery Perturbation (KRP)

**Author:** Joy Deng · **Date:** 2026-07-21 · **Status:** For advisor review

---

## 0. The Proposal in Three Equations

**What RAP does today** — perturb the ego by overwriting its state at a single frame:

$$\textbf{(E1)}\qquad s_0 \leftarrow s_0 + \xi,\quad \xi\sim\mathcal U(\Xi),\qquad s_t\ \text{unchanged for }t\neq 0$$

*In words: teleport the car sideways by a random amount, only at the current frame. History frames stay on the original path.*

**What KING taught us** — never edit states; edit **actions**, and let the dynamics produce the states:

$$\textbf{(E2)}\qquad s_{t+1}=\kappa\big(s_t,\ a_t+w_t\big)$$

*In words: apply a disturbance to the control input and roll it through a bicycle model. Whatever state comes out is reachable by construction.*

**What we propose** — apply (E2) to the **ego**, and choose $w$ by gradient rather than by hand:

$$\textbf{(E3)}\qquad w^{*}=\arg\min_{w}\ \|w\|_W\quad\text{s.t.}\quad s_{t+1}=\kappa(s_t,a_t^{\text{log}}\!+\!w_t),\ \ \mathcal L_{\text{rec}}\big(\pi_\theta\big)>\tau$$

*In words: find the smallest physically-admissible disturbance that the current planner cannot recover from.*

**The one obstacle, and the one thing that removes it.** Solving (E3) by gradient requires

$$\frac{\partial o}{\partial s}\quad\text{(how the image changes when the ego pose changes)}$$

This derivative **does not exist** for logged video, is **learned and unreliable** for NeRF/3DGS, and is **exact and cheap** for RAP's rasterization of annotated primitives. Hence the title of this proposal: KING's idea is only executable on ego state *if the observation is a rasterized scene*, and RAP's recovery augmentation is only physically valid *if the perturbation is kinematic*. The two lock together.

---

## 1. Working Titles

**T1 (preferred)** — *Driven, Not Designed: Kinematics-Gradient Recovery Perturbation via Differentiable Rasterization for End-to-End Driving*

**T2** — *Closing the Gradient Loop for Ego-State Recovery: Differentiable Rasterization Enables Kinematically-Feasible Perturbation Mining*

**T3** — *KRP: Policy-Aware, Kinematically-Consistent Recovery Perturbations for Camera-Based End-to-End Planners*

---

## 2. Motivation

### 2.1 A contradiction inside RAP itself

RAP's premise: an imitation planner fails **in closed loop** because it has never seen recovery states. Its remedy: render off-log recovery states cheaply via rasterization.

But look at how those states are produced. In the released code (`process_data/create_openscene_metadata_purturbed.py`, `get_ego_params` L246–262, per-frame gate L330–336), for a 14-frame window only `batch[3]` is perturbed:

$$\Delta x,\Delta y\sim\mathcal U(-0.5,0.5)\ \text{m},\qquad \Delta\psi\sim\mathcal U(-15^\circ,15^\circ),\qquad \Delta v = \mathcal U(-0.2,0.2)\cdot v$$

with `NAVSIM_INTERVAL_LENGTH = 0.5` s. Check whether a vehicle can do this:

$$\omega=\frac{\Delta\psi}{\Delta t}=\frac{0.26\ \text{rad}}{0.5\ \text{s}}=0.52\ \text{rad/s}
\quad\Longrightarrow\quad
a_{\text{lat}}=v\,\omega\Big|_{v=20\,\text{m/s}}\approx 10.5\ \text{m/s}^2$$

Beyond the friction limit. **The training states are unreachable in the very closed loop the method exists to fix.** This is a one-line calculation, and it sits in the core assumption rather than in an implementation detail.

Two further consequences of the same design:

**(a) It creates a shortcut.** Since history frames are *not* perturbed, the offset is directly readable from the frame-to-frame inconsistency:

$$\hat\xi \;\approx\; s_0 - \kappa(s_{-1},a_{-1})$$

The network can learn "detect the jump, invert it" instead of "understand from the image where the lane is." Gradient descent will prefer the easier route — the textbook copycat / causal-confusion failure mode (de Haan et al. 2019; Codevilla et al. 2019), here manufactured by the augmentation itself.

**(b) The velocity perturbation vanishes at standstill.** Because it is multiplicative, $v=0\Rightarrow\Delta v=0$. Stop-and-go scenes — exactly where the inertia problem lives — receive no effective perturbation.

### 2.2 A structural gap in the literature

| Era | Representative | How perturbations are obtained |
|---|---|---|
| early | manual scenario editing | hand-set parameters |
| **turning point** | **KING (ECCV 2022)** | **optimization over kinematically-constrained action sequences** |
| after | ReGentS, SaFeR, AlignADV, ADV-0 | refinements along KING's line |

Every entry concerns **background agents**. Ego-side recovery perturbation is still pre-KING: hand-set parameters, state rewrite. The gap is structural rather than accidental — §3.2 shows it requires a condition that only became available with RAP.

### 2.3 An empirical anomaly we can already explain

In our own `ref2d` experiments, open-loop recovery validation ranked models **inversely** to closed-loop EPDMS (`recovery_only` best open-loop, worst closed-loop at 0.181 vs. baseline 0.341). The shortcut hypothesis in §2.1(a) explains this: open-loop evaluation contains the same inconsistency cue and therefore rewards shortcut-taking models; closed loop does not.

*A hypothesis that explains an anomaly already in hand is stronger motivation than a hypothesis that predicts one.*

### 2.4 Train/test physics mismatch

`navhard_two_stage` Stage 2 initializes synthetic scenes **from perturbed ego states and evolves them physically**. Our training-side perturbation teleports. The same task is governed by two different physics on the two sides.

### 2.5 Why now

Three prerequisites matured simultaneously: (i) RAP's primitive-based scene representation makes $\partial o/\partial s$ analytic; (ii) differentiable rasterization is mature (Soft Rasterizer, nvdiffrast); (iii) NAVSIM v2 provides an official recovery-from-offset evaluation protocol. Remove any one and the project is infeasible.

---

## 3. Core Idea

### 3.1 Three formulations side by side

| | Optimization variable | Constraint | Cost | Selection |
|---|---|---|---|---|
| **RAP** | state offset $\xi$ | none | none | $\xi\sim\mathcal U(\Xi)$, fixed |
| **KING** | background actions $\theta=\{a^{i>0}_t\}$ | $s^{i>0}_{t+1}=\kappa(s^{i>0}_t,a^{i>0}_t)$ | $\mathcal C$ = **geometric** (collision distance) | $\nabla_\theta\mathcal C$ |
| **KRP (ours)** | ego disturbance $w_{-K:0}$ | $s^{0}_{t+1}=\kappa(s^{0}_t,a^{\text{log}}_t\!+\!w_t)$ | $\mathcal L_{\text{rec}}$ = **perception-driven** | $\nabla_w\mathcal L_{\text{rec}}$ |

The first two columns show KRP inherits KING's *parametrization*. **The third column is where the difficulty is**, and it is the subject of the next subsection.

### 3.2 The gradient-path argument (central to the paper)

> **Figure 1** will reproduce KING's computation graph (their Fig. 2) beside ours, with the direct path in blue and the indirect path in red — see §7 for the figure specification.

**KING.** Perturbing a background action propagates to the cost along two routes:

$$\frac{d\mathcal C}{d\theta}
=\underbrace{\frac{d\mathcal C}{d\theta}\bigg|_{\text{direct}}}_{\text{the adversary's own motion}}
+\underbrace{\frac{d\mathcal C}{d\theta}\bigg|_{\text{indirect}}}_{\text{via the ego's reaction}}$$

The **direct** route is

$$a_t^{i>0}\ \longrightarrow\ s_{t+1}^{i>0}\ \longrightarrow\ \mathcal C,
\qquad
\frac{d\mathcal C}{da_t^{i>0}}\bigg|_{\text{direct}}
=\frac{\partial\mathcal C}{\partial s_{t+1}^{i>0}}\,
\frac{\partial\kappa(s_t^{i>0},a_t^{i>0})}{\partial a_t^{i>0}}$$

*Both factors are closed-form: the first is the derivative of a distance formula, the second of a bicycle model.* **The observation $o_t$ never appears.**

The **indirect** route is

$$a_t^{i>0}\to s_{t+1}^{i>0}\to \boxed{o_{t+1}}\to a^0_{t+1}=\pi(o_{t+1})\to s^0_{t+2}\to\mathcal C$$

which requires $\partial o/\partial s$ — unavailable, since CARLA's renderer is a black box. **KING simply discards it.** The remaining gradient is biased but nonzero and directionally correct ("move this car toward where the ego will be"), so the method works.

**The enabling condition is that $\mathcal C$ is geometric**, so the direct route reaches the cost without touching the observation.

**KRP.** Now the cost is "the learner fails to recover." Where does that quantity live?

$$\mathcal L_{\text{rec}}=\mathcal L\Big(\underbrace{\pi_\theta\big(\,\underbrace{o(s^0(w))}_{\text{observation}}\,\big)}_{\text{planner output}}\Big)$$

Enumerate the routes from $w$ to $\mathcal L_{\text{rec}}$:

- A "direct" term does exist — e.g. lateral offset at $t=0$ is an explicit function of $w$. **But it is uninformative**: it only restates the perturbation we ourselves injected, and says nothing about whether *this planner* can recover.
- Every route carrying information about the learner passes through $\pi_\theta$, hence through $o$.

$$\boxed{\ \frac{d\mathcal L_{\text{rec}}}{dw}\bigg|_{\text{direct}}\equiv 0\ \ \text{(w.r.t. learner-relevant information)}\ }$$

**KING can afford to delete the observation route; we cannot — deleting it leaves nothing.** This is the precise sense in which KING's structural property fails for ego recovery.

The full gradient we must therefore compute is

$$\frac{\partial\mathcal L_{\text{rec}}}{\partial w}
=\sum_t
\underbrace{\frac{\partial\mathcal L_{\text{rec}}}{\partial\pi_\theta}\frac{\partial\pi_\theta}{\partial o_t}}_{\text{(i) planner — autodiff, free}}
\;\underbrace{\frac{\partial o_t}{\partial s_t}}_{\text{(ii) \textbf{observation model}}}
\;\underbrace{\frac{\partial s_t}{\partial w}}_{\text{(iii) bicycle model — closed form}}$$

Factors (i) and (iii) are free. **Factor (ii) is the entire problem.**

### 3.3 Where factor (ii) exists — the bidirectional lock

| Observation source | $\partial o/\partial s$ | Why |
|---|---|---|
| logged real video | **does not exist** | the car never occupied that pose, so the image does not exist — not "hard to compute" but "no object to compute" |
| CARLA / game engine | non-differentiable | black box — precisely what forced KING to drop the indirect path |
| NeRF / 3DGS | learned, degrades under extrapolation, $10^2$–$10^3\times$ slower | and $\delta$ lives *in* the extrapolation regime; unusable inside an inner loop |
| **rasterized primitives (RAP)** | **analytic, exact, cheap** | the scene is a known geometric object; $s\mapsto o$ is the closed chain $T(\delta)\to$ projection $\to$ (soft) rasterization |

$$\text{KING-on-ego}\ \Longrightarrow\ \text{needs }\tfrac{\partial o}{\partial s}\ \Longrightarrow\ \text{needs rasterized observation}$$
$$\text{RAP recovery augmentation}\ \Longrightarrow\ \text{needs physical realizability}\ \Longrightarrow\ \text{needs kinematic parametrization}$$

Neither direction is rhetorical; the first is tested in Experiment 3, the second in Experiment 1.

### 3.4 The optimization problem and the system

$$
w^{*}_{-K:0}=\arg\min_{w}\|w\|_{W}
\quad\text{s.t.}\quad
\begin{cases}
s_{t+1}=\kappa\!\left(s_t,\ a^{\text{log}}_t+w_t\right) & \text{feasible by construction}\\[2pt]
\mathcal L_{\text{rec}}\!\left(\pi_\theta,\,o(s_{-K:T})\right)>\tau & \text{learner fails}\\[2pt]
w_t\in\mathcal W(v_t) & \text{actuation limits}
\end{cases}
$$

The last constraint follows from non-holonomy:

$$\omega=\frac{v\tan\delta_{\text{steer}}}{L}\quad\Longrightarrow\quad v=0\ \Rightarrow\ \omega=0$$

*Heading cannot be perturbed at standstill. The current implementation violates this; ours enforces it, which is why perturbation magnitude necessarily becomes speed-dependent.*

**Two code-level modifications:**

- **M1** — replace state rewrite with control-space disturbance rolled through $\kappa$ across the whole history window.
- **M2** — replace the rasterizer with a differentiable (soft) rasterizer over the same primitives.

```
a^log        w_{-K:0}
  │             │
  └──► κ (differentiable bicycle model) ──► s_{-K:0}
                                              │
        annotated 3D primitives ──────────────┤
                                              ▼
                       differentiable rasterizer (7 cams × K frames)
                                              │
                                              ▼
                              planner π_θ  (+ R2R alignment)
                                              ▼
                                          L_rec
                                              │
       ◄────────── ∂L_rec/∂w  (fully analytic) ┘
```

Mined $w^*$ defines a perturbed clip, re-rendered across all cameras and history frames, paired with a privileged-expert recovery target, and mixed into training at fixed budget.

---

## 4. Research Questions

**RQ1 — Does the parametrization matter?**
Hypothesis: (E1) admits the shortcut $\hat\xi\approx s_0-\kappa(s_{-1},a_{-1})$; replacing it with (E2) removes it. Predicted signature: **open-loop recovery metrics decrease, closed-loop EPDMS increases** — consistent with the anomaly in §2.3.

**RQ2 — Does policy-awareness pay at fixed budget?**
Does $\nabla_w$-mined $w^*$ beat (i) uniform random $w$, (ii) learner-error-matched noise (DART), (iii) coverage-maximizing sampling? The adversarial-training literature shows the *hardest* samples can harm generalization, so this is genuinely open.

**RQ3 — Is differentiable rasterization necessary?**
Quantify $\partial o/\partial s$ across observation models by gradient agreement, extrapolation degradation, and wall-clock cost.

---

## 5. Evaluation Plan

### 5.1 Platforms

| Purpose | Platform |
|---|---|
| training / primary evaluation | **NAVSIM v2** (OpenScene/nuPlan): `navtrain` → `navtest`, `navhard_two_stage` |
| closed-loop verification | **HUGSIM** (3DGS closed loop); optionally **Bench2Drive** |
| gradient validation | synthetic scenes with ground-truth geometry |

`navhard_two_stage` is the primary leaderboard because its Stage 2 initializes from perturbed ego states (77 scenarios; 450 real + 5,462 synthetic frames; Stage 1 = 4 s real open loop) — the only public benchmark whose protocol *is* a recovery-from-offset test. **This alignment must be stated explicitly** to preempt an overfitting objection: we do on the training side what Stage 2 evaluates on the test side.

### 5.2 Baselines (identical model, renderer, and sample budget; only the perturbation generator differs)

| ID | Parametrization | Temporal | Selection | Equivalent to |
|---|---|---|---|---|
| B0 | — | — | — | no augmentation |
| **B1** | state, $\xi$ | single frame | uniform | **RAP as released** |
| **B2** | control, $\kappa$ | $K$-frame | uniform | **DART** (2017) |
| B3 | control, $\kappa$ | $K$-frame | coverage-max | — |
| **B4** | trajectory-level, 3DGS | consistent | recovery pseudo-expert | **SimScale** (CVPR 2026 Oral) |
| **KRP** | control, $\kappa$ | $K$-frame | $\nabla_w$-mined | ours |

**B1→B2 isolates parametrization (RQ1). B2/B3→KRP isolates policy-awareness (RQ2). B4 is the strongest external competitor.**
Secondary: gradient vs. finite difference vs. CMA-ES as a function of $\dim(w)$.

### 5.3 Metrics

**Benchmark (credibility check).** `navhard_two_stage` EPDMS **with all ten sub-metrics disaggregated**: NC, DAC, **DDC**, **TLC**, EP, TTC, **LK**, **HC**, **EC**, C.
- **EP separately** — it is the strongest single predictor of closed-loop performance, above NC; recovery training characteristically raises NC and lowers EP, leaving the aggregate unchanged while closed-loop driving worsens.
- **LK separately** — it penalizes sustained centerline deviation, the most direct proxy for recovery (disabled at intersections).
- `navtest` EPDMS — evidence of no nominal degradation.
- HUGSIM closed-loop score.

**Method-specific (headline).**
- distribution of $\|w^*\|_W$ before vs. after training
- recoverable-region volume and directional radius $\rho(\theta)$; $\mathcal R_{\text{after}}$ vs. $\mathcal R_{\text{before}}$
- $\cos\big(\nabla^{\text{raster}}_w,\ \nabla^{\text{FD}}_w\big)$ — gradient fidelity
- renders-per-solution vs. $\dim(w)$, raster vs. 3DGS
- fraction of raster-mined $w^*$ still failure-inducing on the real-image branch after R2R

**Expected magnitudes.** On `navhard`, the best non-privileged results (≈56.3) sit within 0.3 of privileged PDM-Closed (≈56.6). **EPDMS is a non-degradation check, not the headline.** The primary claim rests on $\|w^*\|$ and recoverable-region geometry.

---

## 6. Risks and Contingencies

### R1 — "The gradient may be uninformative, and finite differences suffice at low dimension."

**Most serious risk; resolved first.** Soft rasterization is a relaxation, and the gradient traverses a large vision backbone. Moreover at $K=3$, 2 Hz, $\dim(w)\approx6$, finite differences cost only $2\dim(w)\approx12$ renders.

**Response.** (i) Gate everything on Experiment 2: measure $\cos(\nabla^{\text{raster}},\nabla^{\text{FD}})$ on controlled scenes before any further engineering. (ii) Make the gradient necessary by design: subdivide $w$ to 10 Hz over the 1.5 s window ($\dim(w)\approx30$) and report the crossover, since one backward pass beats $2\dim(w)$ forward passes only above some dimension.

**Fallback (graceful, not catastrophic).** If the gradient is unusable, the paper becomes (a) a diagnosis of the shortcut, (b) kinematic reparametrization as the fix, (c) the observation-model analysis of §3.3, with CMA-ES/bisection replacing the gradient. **RQ1 and RQ3 survive intact; only RQ2's mechanism changes.**

### R2 — "SimScale (CVPR 2026 Oral) already does recovery augmentation at scale across three architectures."

**Legitimate.** SimScale perturbs ego trajectories, renders multi-view via 3DGS in a reactive environment, produces recovery- and planner-based pseudo-experts, and validates sim–real co-training on LTF / DiffusionDrive / GTRS-Dense.

**Response, on three testable axes.** (1) *Parametrization* — trajectory-level without explicit admissible-control constraints vs. feasibility by construction with speed-dependent limits. (2) *Selection* — coverage vs. the learner's failure frontier. (3) *Observation model* — §3.3 is inapplicable to 3DGS. Quantified by B4-vs-KRP at equal budget, not asserted in related work.

**Fallback.** If KRP does not beat B4, reposition as **orthogonal and composable**: KRP supplies *which* disturbances, SimScale *how* to render them. The combination becomes the headline.

### R3 — "Differentiating through a neural renderer for adversarial purposes was published in 2023."

**Valid** — arXiv:2309.15770 builds a differentiable NeRF simulator, differentiates the sensor model for $\partial o/\partial s$, and solves via implicit differentiation.

| Axis | arXiv:2309.15770 | KRP |
|---|---|---|
| perturbed variable | environment / appearance | **ego state via admissible control sequence** |
| objective | maximize deviation (attack) | **minimum-norm failure under recoverability** |
| observation model | NeRF (learned; extrapolation-degrading) | **rasterized primitives (analytic; exact)** |
| scale & use | per-scene optimization; *testing* | **$\sim10^5$ scenes; *training*** |

The scalability distinction follows from the observation model, so it is the same argument as §3.3 rather than an incidental difference. **State this in the introduction, not in related work.**

### Additional

- *"Does $w^*$ mined in raster space transfer to the real-image branch?"* → cross-branch transfer metric (§5.3). If weak, the claim is scoped to "recovery boundary under RAP rendering," and R2R becomes an explicit object of study.
- *"Fixing the shortcut may make results worse."* → expected, and predicted by RQ1. The "open-loop down, closed-loop up" signature **is** the result.
- *"Non-reactive background traffic biases the recoverable region."* → acknowledged scoping assumption; quantified as a sensitivity analysis on a subset, not resolved by building a reactive simulator.

---

## 7. Figure 1 Specification (comparison figure)

A two-panel computation graph in the style of KING Fig. 2.

**(a) KING.** Variable $a^{i>0}_{t}$. **Blue** direct path $a^{i>0}_{t}\!\to\!\kappa\!\to\! s_{t+1}\!\to\!\mathcal C_{t+1}$ (survives). **Red** indirect path through $\mathcal R\!\to\! o_t\!\to\!\pi_\omega\!\to\! a^0_t$, drawn **cut**, with the renderer marked non-differentiable.

**(b) KRP.** Variable $w_t$ entering $\kappa(s^0_t,\ a^{\text{log}}_t\!+\!w_t)$. **No blue path exists.** The single red path runs $w_t\!\to\!\kappa\!\to\! s^0_{t+1}\!\to\!\mathcal R_{\text{diff}}\!\to\! o_{t+1}\!\to\!\pi_\theta\!\to\!\mathcal L_{\text{rec}}$, drawn **unbroken**, with $\mathcal R_{\text{diff}}$ highlighted as the contribution.

Caption should read: *KING can discard the observation route because its cost is geometric. For ego recovery the observation route is the only route; differentiable rasterization is what makes it traversable.*

---

## 8. Milestones

| Phase | Deliverable | Gate |
|---|---|---|
| P1 (2–3 wk) | gradient validation: $\cos(\nabla^{\text{raster}},\nabla^{\text{FD}})$ on controlled scenes | **Go/No-Go on the gradient formulation** |
| P2 (3–4 wk) | M1 implemented; B1 vs. B2 trained and evaluated | RQ1 answered |
| P3 (4–6 wk) | M2 implemented; KRP mining loop; B2/B3/KRP at matched budget | RQ2 answered |
| P4 (2–3 wk) | observation-model study + B4 head-to-head; HUGSIM closed loop | RQ3 answered; paper assembled |

**Explicitly out of scope** (deferred): adversarial background agents, per-camera boundary attribution, equivariance objectives, world-model rollout, cross-architecture transfer.

---

## References (indicative)

Feng et al., *RAP: 3D Rasterization Augmented End-to-End Planning*, 2025 · Hanselmann et al., *KING*, ECCV 2022 · Laskey et al., *DART*, CoRL 2017 · de Haan et al., *Causal Confusion in Imitation Learning*, NeurIPS 2019 · Codevilla et al., *Exploring the Limitations of Behavior Cloning*, ICCV 2019 · Dauner et al., *NAVSIM*, NeurIPS 2024 · OpenDriveLab, *SimScale*, CVPR 2026 (Oral) · *Transferable Adversarial Simulation via Neural Rendering*, arXiv:2309.15770 · Zhou et al., *HUGSIM*, 2024 · Liu et al., *Soft Rasterizer*, ICCV 2019 · Laine et al., *nvdiffrast*, SIGGRAPH Asia 2020
