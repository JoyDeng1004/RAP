# KRP Proposal — Notation Table

Companion reference for *Research Proposal — Kinematic Recovery Perturbation (KRP)*.
Symbols are grouped by role. Superscript $0$ denotes the ego vehicle; superscript $i>0$ denotes background vehicles.

---

## 1. State and perturbation (§1, §3.4)

| Symbol | Meaning |
|---|---|
| $s_t$ | State at time $t$ (position $x,y$, heading $\psi$, speed $v$) |
| $s^0_t$ | **Ego** state (superscript $0$ = ego) |
| $s^{i>0}_t$ | **Background-vehicle** state (superscript $i>0$ = non-ego) |
| $s_0$ | State at the current frame ($t=0$) — note: **subscript** $0$ = frame index, not ego |
| $\xi$ | RAP's **state offset** perturbation, $\xi\sim\mathcal U(\Xi)$ |
| $\Xi$ | Sampling range of $\xi$ |
| $\hat\xi$ | Offset **read off** from frame-to-frame inconsistency (the shortcut), $\hat\xi\approx s_0-\kappa(s_{-1},a_{-1})$ |
| $\Delta x,\ \Delta y$ | Lateral / longitudinal position offset |
| $\Delta\psi$ | Heading offset |
| $\Delta v$ | Speed offset |

## 2. Kinematics and physics (§1, §3.4)

| Symbol | Meaning |
|---|---|
| $\kappa$ | Differentiable **kinematic bicycle model** (state transition) |
| $\omega$ | Yaw rate |
| $a_{\mathrm{lat}}$ | Lateral acceleration |
| $v$ | Speed |
| $\Delta t$ | Inter-frame interval ($=0.5$ s in NAVSIM) |
| $\delta_{\mathrm{steer}}$ | Front-wheel steering angle |
| $L$ | Wheelbase |

## 3. Actions and optimization variables

| Symbol | Meaning |
|---|---|
| $a_t$ | Generic action fed to $\kappa$; in **E2 (KING)** it is the free optimization variable |
| $a^0_t$ | Ego action; $a^{i>0}_t$ background-vehicle action |
| $a^{\log}_t$ | Ego's **logged** action from the dataset (a known constant) |
| $w_t,\ w_{-K:0}$ | Ego control **residual disturbance** — the KRP optimization variable |
| $w^{*}$ | The optimized (minimum-norm failure-inducing) disturbance |
| $\varepsilon$ | Failure threshold; $\mathcal L_{\mathrm{rec}}>\varepsilon$ marks "learner fails" |
| $\|\cdot\|_W$ | Weighted norm; $W$ is the weighting matrix |
| $\mathcal W(v_t)$ | **Speed-dependent** actuation-limit set (calligraphic $\mathcal W$, distinct from the weight $W$) |
| $K$ | History-window length (number of frames back) |
| $-K{:}0,\ -K{:}T$ | Time-index ranges |

> **Note (E3).** The applied ego action is $a^{\log}_t + w_t$ = **logged nominal (fixed)** + **disturbance (optimized)**. Unlike KING — which optimizes the full background action $a_t$ — the ego already has a real logged trajectory, so KRP optimizes only the residual $w_t$. This is what makes the minimum-norm objective $\min_w\|w\|_W$ meaningful (smallest deviation from natural driving) and keeps perturbed clips near the data manifold. It follows the DART principle of injecting disturbances into the supervisor's control stream.

## 4. Cost, policy, and observation (§2, §3)

| Symbol | Meaning |
|---|---|
| $\mathcal C$ | KING's cost (**geometric**: collision distance) |
| $\mathcal L_{\mathrm{rec}}$ | KRP's **recovery cost** (perception-driven) |
| $\tau$ | Planner **output trajectory** |
| $\pi_\omega$ | Policy in KING (parameters $\omega$) |
| $\pi_\theta$ | Learner / planner in KRP (parameters $\theta$) |
| $o_t$ | Observation (rendered image / BEV) |
| $\mathcal R$ | Renderer / rasterizer |
| $\mathcal R_{\mathrm{diff}}$ | **Differentiable** rasterizer (modification M2; our contribution) |
| $\mathcal M$ | Map (static scene) |
| $x_{\mathrm{goal}}$ | Goal location |
| $\rho(\theta)$ | Directional radius of the recoverable region (evaluation metric) |

## 5. Gradient factors (§3.2)

| Symbol | Meaning |
|---|---|
| $\dfrac{\partial\mathcal L_{\mathrm{rec}}}{\partial\tau}$ | Cost w.r.t. trajectory |
| $\dfrac{\partial\pi_\theta}{\partial o}$ | Planner w.r.t. observation (automatic differentiation) |
| $\dfrac{\partial o}{\partial s}$ | Observation w.r.t. state — **the sole bottleneck**, available only under rasterization |
| $\dfrac{\partial s}{\partial w}$ | State w.r.t. disturbance (bicycle model, closed form) |
| $T(\delta)$ | Coordinate transform induced by pose offset $\delta$; first link of the $s\mapsto o$ chain |

## 6. Index conventions

| Convention | Meaning |
|---|---|
| Superscript $0$ | Ego vehicle (e.g. $s^0_t,\ a^0_t$) |
| Superscript $i>0$ | Background vehicle |
| Subscript $t$ | Time step |
| Subscript $0$ | The current frame ($t=0$) |
| $\log$ superscript | Quantity taken from the driving log |

---

## Overloaded symbols — read with care

| Symbol | Meaning A | Meaning B |
|---|---|---|
| $\theta$ | In **E2 / KING**: the optimization variable $\{a^{i>0}_t\}$ (background actions) | In $\pi_\theta$: the **planner's network parameters** |
| $0$ | **Superscript** $s^0$: ego vehicle | **Subscript** $s_0$: frame at $t=0$ |
| $W$ | Weight matrix in the norm $\|w\|_W$ | Calligraphic $\mathcal W(v_t)$: actuation-limit **set** |

> Recommendation: in the KING paragraph, avoid introducing $\theta$ for the optimization variable — write $\{a^{i>0}_t\}$ directly — so that $\theta$ unambiguously denotes the planner parameters $\pi_\theta$ throughout.
