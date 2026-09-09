# 2026-09-09 → 09-11｜48h 实验冲刺计划（Sprint Plan）

- **计划起点**：2026-09-09（当前日期）
- **硬截止**：2026-09-11 13:00 JST 多人会议（池畑 / 佐藤 / 関川 / Joy）
- **可用窗口**：约 49 小时（含睡眠与 PPT 制作）
- **上游文档**：
  - 会议 Memo：`context/2026-09-08_佐藤1v1_完整版会议Memo.md`
  - 9/2 池畑 1v1 转录(context/20260902_ikehata.txt)、9/8 佐藤 1v1 转录(context/20260908_sato.txt)
- **本文档角色**：Research Spec → **Implementation Spec / Execution Plan** 的落地层。
  它**不新增科学主张**，只做规格化、排程、资源分配与卡点设置。

---

## 0. 工作流合规声明

本计划严格按下述链路推进，**任何阶段不得跳跃**：

```
Problem → Hypothesis → Research Spec → Implementation Spec
       → Environment Check → Smoke Test → Full Experiment
       → Measurement → Scientific Interpretation
```

| 阶段 | 状态 | 载体 |
|---|---|---|
| Problem | ✅ 已定义 | proposal §1；9/8 Memo 一、1 |
| Hypothesis | ✅ H0–H5 已冻结 | proposal §4 |
| Research Spec | ✅ 已冻结（§7.3 列出不可事后更改项） | proposal §5–§7 |
| **Implementation Spec** | 🔶 **本文档 §2–§5** | 本文件 |
| Environment Check | ⬜ 待执行（T-1） | §6 时间表 |
| Smoke Test | ⬜ 待执行（T-2） | §6 |
| Full Experiment | ⬜ 待执行（受 CP-2 门控） | §6 |
| Measurement | ⬜ | §7 |
| Scientific Interpretation | ⬜ 仅允许 `Supported` / `Refuted` / `Inconclusive` | §7.4 |

**分工边界**：Human（Joy）负责研究问题定义、SD-\* 科学决策、最终判断；AI 负责规格化、代码生成、验证、测量与监控。遇 SD-\* 一律停机等待。

---

## 1. 熔断清单：必须由人类裁决的科学决策

> 以下 7 项**在被裁决前，相关代码一行都不写、相关实验一次都不跑**。
> 每项给出「问题 / 冲突来源 / 影响面 / 我的建议（仅供参考，非默认执行）」。

### `[Scientific Decision Required] SD-1｜方案B 违反已冻结的 H0 access boundary` 🔴 最高优先

- **冲突**：
  - proposal §6.2：nuScenes 作为 source，`Explicitly prohibited input: all source RGB and source-camera calibration`
  - proposal §6.3 access audit：`source RGB file reads, decoded source RGB tensors, source intrinsics reads, source extrinsics reads all equal zero`
  - proposal §5.3：`The renderer accepts C_d as its only camera input. Source-camera files are outside its loader allowlist.`
  - **但**：池畑 9/2 的 core novelty（活用 external real image）与佐藤 9/8 有条件认可的「external real ↔ external-rig raster alignment」**必须**读 external RGB **和** external 相机内外参（否则无法渲染 external-rig raster）。
- **影响面**：这不是加一个 loss 开关。它需要 ① 修改 renderer 使其接受第二套 camera config ② 修改 loader allowlist ③ 重写 H0 access audit 的判定标准 ④ 修改 proposal §1.5 的 non-claim 边界与 §3.6 的 novelty boundary（原文把「no source pixels/calibration」列为对抗 R10「reviewer 认为这就是 RAP + 另一个数据集」的核心防线）。
- **我的建议（待裁决）**：把 X 系列（见 §2.2）注册为**协议修订版 P2**，与现行 P1 并存而非替换。P1 保持「zero source pixel」的干净主张，P2 作为独立分支主张。**不要**在 P1 内部偷偷放开 allowlist——那会同时毁掉两个主张。
- **裁决时机**：**CP-1（9/9 13:00 前）**。未裁决 → X 系列全部不启动，冲刺退化为 F 系列 pilot（这本身仍是合格交付）。

### `[Scientific Decision Required] SD-2｜9/9–9/11 的 run 如何归类`

- **冲突**：proposal §6.6 规定 10% 实验用 3 subsets × 2 seeds = 6 paired runs 的 crossed design，且「若 F0-only power audit 显示 power < 80%，需在**查看 F1 结果之前**增加 subset」。48h 内只跑得起 1 subset × 1 seed。若直接用 confirmatory subset 跑并查看结果，等于提前开箱，损伤预注册效力。
- **我的建议（待裁决）**：新建一个 **development subset（seed=DEV）**，与 3 个 confirmatory subsets 互斥，本次冲刺全部 run 落在 `runs/pilot/` 下，明确标注 `non-confirmatory, excluded from H1–H5 analysis`。9/11 PPT 上所有数字标 **preliminary / single subset / single seed**。
- **裁决时机**：**CP-1**。

### `[Scientific Decision Required] SD-3｜target budget 的对外口径`

- **冲突**：proposal §5.7 已冻结 `b* = 10% of legal target training scenes`（单位是 **scene/log**，不是 frame，不是分钟），`b_min = 100 scenes`。但 9/8 你向佐藤陈述的是「1000 小时 external vs 10 分钟 target ≈ 6000:1」，9/2 与池畑也在「10 分钟 / NAVSIM 10%」之间来回。Memo 已注明这是「假设性例子，并非今天确定的训练预算」。
- **影响面**：这个数字是你 external-alignment 论证的**唯一定量支点**（"target pair 太少所以 alignment 学不出来"）。口径不统一，9/11 现场会被直接拆掉。
- **我的建议（待裁决）**：统一到 **b\* = 10% navtrain scenes + b_min = 100 scenes**，并在 PPT 上同时给出绝对 scene 数 / frame 数 / 等效时长三个值（proposal §1.5 本来就要求 b_min 必须与绝对计数一同报告）。「1:6000」只作为 motivation 插图，不作为实验预算。
- **裁决时机**：**CP-1**（零成本，但不定则 PPT 无法写）。

### `[Scientific Decision Required] SD-4｜X 系列的 alignment 规格未定`

未定项（**不得由我猜测补全**）：
1. external raster 用哪套 rig 渲染（原生 nuScenes rig？还是简化后的 canonical rig？）
2. alignment 在哪一层做（`B` 输出？`P_R/P_d` 输出 `F^R/F^I`？planning representation `G` 之后？）
3. 哪些模块跨 rig 共享、哪些按 rig 分叉（池畑提的 rig-dependent embedding 要不要引入）
4. 距离度量与权重（判别器 / 余弦 / MSE；λ 初值）
5. external alignment 与已有的 target R2R loss（proposal §6.9 的 A0/A1 轴）如何共存

- **裁决时机**：**CP-3（9/9 18:00 前）**，且必须在写 X 系列代码之前。

### `[Scientific Decision Required] SD-5｜external-rig planning loss 做不做`

- **冲突**：池畑 9/2 §44 提出 Joint Training（两个 planner + 共享 encoder + rig-dependent embedding），**其前提就是在 external rig 上算 planning loss**；佐藤 9/8 10:08–10:11 明确判断「アライメントはいいと思うんですけど、プラナンは多分だめだと思う」「性能には寄与しない」。Memo §五记录准确状态为「保留为可检验的消融，不能写成已被证明无效」。
- **两位老师互不知道对方的表态。** 这是 9/11 会上第一议题。
- **我的建议（待裁决）**：本次冲刺**不跑** X3，只带**实验设计**上会，让两位老师当面裁决后再实现。理由见 §3.3。
- **裁决时机**：9/11 会上（会前只需准备 slide）。

### `[Scientific Decision Required] SD-6｜冲刺期的评测指标`

- proposal §6.6 冻结主指标为官方 **NAVSIM-v2 Final EPDMS**（含 Stage1/Stage2 及全部分项），open-loop 轨迹误差**仅为 diagnostic**。官方评测的墙钟耗时本地未知。
- **问题**：若 9/10 晚官方评测跑不完，是否允许 PPT 上只放 open-loop 代理指标？
- **我的建议（待裁决）**：允许，但必须①标注为 diagnostic-only ②同时给出 EPDMS 的 ETA ③**不得**用 open-loop 差异下任何 Supported/Refuted 结论（只能 Inconclusive）。
- **裁决时机**：**CP-4（9/10 11:00）**。

### `[Scientific Decision Required] SD-7｜Simulator/RL 路线的冲刺投入上限`

- 佐藤 9/8 新提，认为「可能研究价值更高」，甚至可各成一篇；池畑不知情。
- **我的建议（待裁决）**：本次冲刺**硬时间盒 90 分钟**，只做候选表 + 一次接口可行性判定，**不写任何训练代码**。
- **裁决时机**：**CP-1**。

---

## 2. 实验臂定义与命名对齐

### 2.1 命名冲突警告 ⚠️

Memo 里的 `E0–E3` 与 proposal 里的 `F0/F1/F2/A0/A1` **不是同一套坐标**，且存在一个危险的同名异义：

| Memo | proposal | 是否等价 | 说明 |
|---|---|---|---|
| E0 | **F0** | ✅ 等价 | target-only @ b\* |
| E1 | **F1** | ✅ 等价 | + source metadata → **target-rig** raster planning |
| E2 | ❌ **不存在** | — | target-only + **external-rig** real/raster alignment |
| E3 | ❌ **不存在** | — | F1 + external-rig alignment（= 方案B） |
| — | **A0 / A1** | ⚠️ **易混** | A0/A1 切换的是 **target R2R alignment（λ_A=0）**，**不是** external alignment |

> **9/11 现场如果把 A1 说成"关掉 external alignment"，佐藤会立刻发现表和 spec 对不上。**
> 本文档一律使用 proposal 的 F/A 命名；新增臂统一冠以 **X**（e**X**ternal-pixel，视觉上提示"越过了 H0 access boundary"）。

### 2.2 本次冲刺涉及的全部臂

**P1 协议内（无需 SD-1，可立即执行）**

| 臂 | Phase-1 辅助流 | target R2R | 数据读取边界 | 冲刺内状态 |
|---|---|:--:|---|---|
| **F0** | 无 | on | 仅 target 预算内 + source 结构化 | ✅ 必跑 |
| **F1** | source raster under `C_d` | on | 同上，**零 source 像素/标定** | ✅ 必跑 |
| **F2** | target-raster replay（同 b 场景） | on | 同上 | 🔶 有余力则跑 |
| **F3** | source raster + 标签置换 | on | 同上 | ⬜ 会后 |
| **A0/A1** | 无 / source raster | **off** | 同上 | ⬜ 会后（G2b） |

**P2 协议（受 SD-1 门控，未裁决则全部不启动）**

| 臂 | 定义 | 对应谁的主张 | 冲刺内状态 |
|---|---|---|---|
| **X1** | F0 + external real ↔ external-rig raster alignment | 池畑「不要丢掉 external real」的隔离版 | ⬜ 仅出 spec |
| **X2** | F1 + external real ↔ external-rig raster alignment | **方案 B**（佐藤有条件认可） | ⬜ 仅出 spec |
| **X3** | X2 + **external-rig planning loss** | 池畑 Joint Training ↔ 佐藤明确反对 | ⬜ 仅出 spec，SD-5 |

### 2.3 为什么 X 系列不进 48h 关键路径（工程判断，非科学判断）

X 系列不是 flag 翻转，它需要串行完成：

1. renderer 接受第二套 camera config（现状：`accepts C_d as its only camera input`）
2. loader allowlist 放开 source RGB + source calib（现状：显式禁止，且有计数器断言）
3. 重写 H0 access audit（现状的断言会**直接 assert fail**）
4. 新建 external real/raster pair 构建 + 缓存
5. 新增 alignment 分支与损失接线（SD-4 五个未定项）
6. 走一遍 smoke test

在 H0 几何审计、manifest 冻结、F0/F1 训练、评测、PPT 全部同时要做的前提下，把 X2 强塞进 48h，最可能的结果是**得到一个没通过 H0 的数字**——而 proposal §6.3 明文写着 `A build that fails H0 is not used for any transfer conclusion`。那个数字上会只会造成伤害。

**因此 9/11 的 X 系列交付物是「规格 + 协议修订请求 + 需老师裁决的问题」，不是数字。** 这与 Memo §「9/11 最需要带去的材料」一致：*一份无歧义的问题定义、一张可核对的数据/rig/loss 表、一组最小对照的结果或明确进度、一份有依据的 RL 可行性判断*。

---

## 3. 两日实验冲刺矩阵

> **资源估算原则**：本地无 GPU 吞吐先验，因此**所有耗时都是 smoke test 实测后的函数**，不写死。
> `T_finish = (remaining_steps / measured_steps_per_hour) × 1.2`（20% 余量给评测、存档与波动）

### 3.1 冲刺假说表

每个实验必须绑定一个可证伪假说。以下 `SH-*` 是**冲刺级 pilot 假说**，**不替代** proposal 的 H0–H5 confirmatory 假说。

| ID | 假说（Hypothesis） | 对照（Baseline / Variables） | 主要终点（Metric） | 支持判据 | 反驳判据 | 关联臂 |
|---|---|---|---|---|---|---|
| **SH-0** | 当前 source→target-view 构建管线满足 proposal §6.3 的 access + geometry 全部门限 | 无（审计，非训练） | 违规读取计数；landmark 重投影误差；round-trip 误差；corruption 检出率 | 全部门限通过（读取=0；≤1 px；≤1e-3 m / 1e-4 rad；corruption 100% 检出） | 任一门限未过 | — |
| **SH-1** | 在 DEV subset @ b\*，F1 的 pilot 指标 **不低于** F0（无灾难性负迁移） | F0；变量 = source raster 辅助流的有无 | Final EPDMS（或 SD-6 批准的代理） | `M(F1) ≥ M(F0) − ε_noise` | `M(F1) ≪ M(F0)` 且可复现 | F0, F1 |
| **SH-2** | F1 相对 F0 的变化**不能**仅由额外训练曝光解释 | F2（同曝光的 target-raster replay） | 同上 | `M(F1) > M(F2)` | `M(F1) ≈ M(F2)` | F1, F2 |
| **SH-3** | 训练管线在 pilot 预算下**数值稳定**且可复现（工程假说） | 无 | loss 曲线无 NaN/发散；ckpt 可恢复；两次同 seed 结果一致 | 全部满足 | 任一不满足 → **Technical Failure**，不计入科学结论 | 全部 |
| **SH-R** | 存在一个可直接复用的简易 driving simulator，能①取状态②按 `C_d` 渲染 raster③执行 action④返回默认 reward | 无（可行性，非性能） | 四项接口的二值判定 + 缺口清单 | 四项全通 | 任一项需自研 > 1 周 | — |

> **SH-1 刻意设成弱假说**（"不低于"而非"显著优于"）。原因：单 subset × 单 seed 的 pilot **在统计上没有能力**支持 `δ_main = 1.0` 的效应判定（proposal §6.6 要求 3×2 crossed design）。把 pilot 写成"证明 F1 更好"就是伪科学，9/11 会被佐藤当场拆穿。

### 3.2 Day-1（9/9）矩阵

| 时段 | 任务 | 类型 | 假说 | 资源 | 产出 | 卡点 |
|---|---|---|---|---|---|---|
| 12:00–13:00 | SD-1/2/3/7 裁决；冻结 DEV manifest 定义 | 决策 | — | 人 | 1 页协议纪要 | **CP-1** |
| 13:00–15:00 | **Environment Check**：H0 access audit + geometry audit（含 corruption 注入） | 审计 | SH-0 | CPU | `h0_report.json` + 可视化 | **CP-2** |
| 13:00–15:00 | 并行：冻结 DEV budget manifest（scene 级分层随机，§5.7）+ 哈希归档 | 数据 | — | CPU | `manifests/target_budget/dev_b010_seedDEV.json` | — |
| 15:00–16:00 | **Smoke Test**：F0/F1 各 ~50 步；测 `steps/hour`；验证 tensor shape / loss 接线 / ckpt 存取 | 冒烟 | SH-3 | GPU×1 短占用 | `smoke_report.md` + 吞吐实测值 | **CP-2** |
| 16:00 | **调度决策门**（见 §3.4 决策表） | 决策 | — | 人 | 过夜任务序列 | **CP-2** |
| 16:00–18:00 | 启动 F0 / F1（GPU）；CPU 侧预生成评测缓存；人写 PPT p.1–p.4 | 训练 | SH-1 | GPU + CPU + 人 | 训练中 | — |
| 18:00–19:30 | **SD-4 规格化**：X 系列 Implementation Spec（不写代码） | 规格 | — | 人 | `X_arms_spec.md` | **CP-3** |
| 19:30–21:00 | Simulator 候选调查（硬时间盒 90 min） | 可行性 | SH-R | 人 | 候选表 + 四项判定 | — |
| 21:00–22:00 | 睡前监控：前 30–60 min loss 正常？ckpt 可恢复？评测命令已挂到训练完成后？磁盘/显存/NaN 告警已设？ | 运维 | SH-3 | 人 | 监控确认清单 | — |
| 22:00– | 过夜：**只跑已通过 smoke test 的配置** | 训练 | SH-1 | GPU | — | — |

**过夜禁止项**：当天下午新写的复杂分支；未做 smoke test 的任何臂；需要人工挑数据或处理异常的流程；X 系列任何内容。

### 3.3 Day-2（9/10）矩阵

| 时段 | 任务 | 类型 | 假说 | 产出 | 卡点 |
|---|---|---|---|---|---|
| 08:00–09:30 | 收 F0/F1；统一评测；核对样本数/更新步数/墙钟是否公平 | 测量 | SH-1 | `results/pilot_v1.csv` | — |
| 09:30–11:00 | 出三张核心材料：①F0/F1 指标表 ②学习曲线 ③3–5 个定性案例（刹车/横向偏移/交叉路口） | 测量 | — | 图表 | — |
| 11:00 | **SD-6 裁决 + 最后实验窗口决策门** | 决策 | — | — | **CP-4** |
| 11:00–13:00 | 按 CP-4 结果二选一：① 启动 F2（若 SH-1 有信号，需 SH-2 隔离曝光）② 复跑 F0/F1 换一个 seed（若 SH-1 差异极小，先确认噪声量级） | 训练 | SH-2 / SH-3 | 训练中 | — |
| 13:00–18:00 | **最后允许启动关键实验的时段**。优先级：修会改变结论的明确 bug > F2 > 一次稳定性复跑 > 定性案例扩充 | 训练 | — | — | — |
| **18:00** | **架构冻结线**：不再新增模块 / 不再改数据定义 / 不再重建大规模缓存 / 不再启动预计 9/11 08:00 前跑不完的训练 | 冻结 | — | — | **CP-5** |
| 18:00–23:00 | 完成 PPT（7–9 页，结构见 §8） | 汇报 | — | `slides.pdf` | — |
| 23:00–次日 08:00 | 仅允许：已 smoke 过的复跑 / 从可靠 ckpt 续训 / 自动评测 / 日志归档 | 训练 | — | — | — |

### 3.4 调度决策表（16:00 CP-2 使用）

| smoke 实测的单臂完整训练耗时 | 决策 |
|---|---|
| ≤ 6 h | 跑完整 F0 + F1，并预留 F2 槽位 |
| 6–10 h | 只跑 F0 + F1（完整） |
| 10–16 h | 全部改为**统一缩减 schedule**（同 step 数、同初始化、同采样规则），结论一律标 preliminary |
| > 16 h | 不启动完整训练；优先复用已有 ckpt 重新评测；或跑统一 proxy schedule |
| 未完成 smoke test | **禁止启动过夜任务**（无例外） |

### 3.5 会前明确砍掉

密集 scaling 曲线（H3）、≥3 seeds、多 backbone / 多分辨率 / 多 PE 设计、alignment 超参搜索、X1/X2/X3 的训练、F3、A0/A1/A1r、H4 多源、BEV / 3D feed-forward / 3DGS baselines、从零训 RL teacher、RAP planner 闭环 RL、新数据集接入、为出图临时重构训练框架、完整 related work。

> 这些**不是不重要**（池畑 9/2 明确说「比较不足可以一发 Reject」），而是 48h 内做不出可信版本。它们进 9/11 的「下一步计划」页，不进「结果」页。

---

## 4. 数据资产与路径规划

> ⚠️ **路径可信度声明**：本机（`/Users/joy/Downloads/DriveWeave`）**不含训练代码与数据集**，`RAP/` 下只有空的 `docs/`。真实的 NAVSIM/nuScenes 原始数据与 RAP 代码位于 TSUBAME，我**无法验证**其实际路径。
> 因此下表中所有 `${...}` 均为**待你填入的占位符**，目录树是**建议约定**而非现状描述。请在 CP-1 时把真实根路径填进 `RAP/docs/paths.env` 并提交。

### 4.1 数据源明细（每个臂需要什么）

| 臂 | 需要的 Dataset | 需要的字段 | Camera Rig 配置 | 明确禁止读取 |
|---|---|---|---|---|
| **F0** | NAVSIM v2 / OpenScene（**仅 DEV budget manifest 内的 scene**） | budgeted real images、structured records、planning labels | `C_d`（NAVSIM target rig） | budget 外的任何 target scene；`navtest` / `navhard_two_stage` / `warmup_two_stage` 全部评测侧数据 |
| **F1** | F0 全部 **+** nuScenes（S1） | nuScenes：maps、agents、ego/world state、future trajectory labels | 渲染**只用** `C_d` | **nuScenes RGB、nuScenes 相机内外参**（H0 断言=0） |
| **F2** | 同 F0（辅助流从同一 b 场景 replay） | 同 F0 | `C_d` | 同 F0；不得引入新 target scene 或新 target RGB |
| **X1/X2/X3** | F1 全部 **+** nuScenes RGB **+** nuScenes 相机标定 | 额外：source RGB、source `K_s, T_s` | `C_d` **和** `C_s` 两套 | 🔴 **当前协议下整体禁止 → 见 SD-1** |
| **REF** | NAVSIM 全量 navtrain | 全部 | `C_d` | 评测侧数据 |

**Rig 配置来源**：`C_d = {K_{d,v}, T_{d,v}, H_{d,v}, W_{d,v}}_{v=1..V_d}`（proposal §5.3）。必须以**冻结的 config + 哈希**形式落盘，每张生成的 raster 都要记录该哈希。

**允许 vs 禁止的边界（H0 计数器必须区分）**：
- ✅ 允许并单独记账：source 的 ego/world pose 与坐标变换（解释结构化场景所必需）
- ❌ 禁止：source 的 RGB 文件读取、解码张量、intrinsics 读取、extrinsics 读取

### 4.2 获取方式

| 资产 | 获取方式 | 备注 |
|---|---|---|
| NAVSIM v2 / OpenScene raw | 集群已有挂载 → 填入 `${DW_RAW_NAVSIM}` | 必须记录 release / commit（proposal §7.3 冻结项） |
| nuScenes 结构化 | 集群已有挂载 → `${DW_RAW_NUSCENES}` | **只经结构化 loader**，不经图像 loader |
| target budget manifest | 由 `tools/build_manifest.py` 生成（scene 级分层随机，§5.7） | 生成即冻结 + 哈希；`b_min ⊂ 1% ⊂ 5% ⊂ 10% ⊂ ...` 嵌套 |
| raster | 由确定性 renderer 生成，输入仅 `(U_x, C_d)` | 落盘时目录名带 `C_d` 哈希，避免不同 rig 混放 |
| 初始化权重 | 通用 DINO checkpoint（frozen backbone） | 🚫 **禁止**用 full-NAVSIM RAP ckpt 作 initializer / teacher / feature cache / pseudo-labeler / 超参 oracle（proposal §6.4） |

### 4.3 保存路径树（建议约定）

```
${DW_DATA_ROOT}/                              # 例：/gs/bs/<group>/<user>/driveweave
│
├── raw/                                      # 只读，永不写入
│   ├── navsim_v2/                            # target
│   ├── openscene/
│   └── nuscenes/                             # source S1
│       ├── structured/                       # ✅ 允许读
│       └── samples/                          # 🔴 图像；P1 协议下 loader 不得触碰
│
├── manifests/                                # 冻结即不可变，全部带 sha256
│   ├── target_budget/
│   │   ├── dev_b010_seedDEV.json             # ← 本次冲刺唯一使用（SD-2）
│   │   ├── dev_b010_seedDEV.sha256
│   │   ├── confirm_b010_subsetA_seed0.json   # ← 冲刺期间不得打开
│   │   ├── confirm_b010_subsetB_seed0.json
│   │   └── confirm_b010_subsetC_seed0.json
│   ├── source/
│   │   └── nuscenes_s1_train.json
│   └── REGISTRY.md                           # 每个 manifest 的生成命令 + 哈希 + 冻结时间
│
├── canonical/                                # 规范化结构化包 U_x（与相机无关）
│   ├── navsim/
│   └── nuscenes/
│       └── _schema_eligibility.json          # proposal §5.2 逐字段合格性矩阵
│
├── raster/                                   # 目录名显式编码「渲染 rig」，杜绝 Old/New 混淆
│   ├── rig=navsim_v2__cfg=<Cd_hash8>/        # ← 唯一合法的 P1 渲染 rig
│   │   ├── scene_src=navsim/                 #    target 场景 → target rig
│   │   └── scene_src=nuscenes/               #    external 场景 → target rig  ★F1 的关键资产
│   └── rig=nuscenes_orig__cfg=<Cs_hash8>/    # 🔴 X 系列专用，SD-1 未裁决前不得创建
│       └── scene_src=nuscenes/
│
├── pairs/                                    # real ↔ raster 配对索引（只存索引，不复制像素）
│   ├── target_real_raster__b010_seedDEV.jsonl
│   └── external_real_raster__nuscenes.jsonl  # 🔴 X 系列专用，同上
│
└── cache/
    ├── eval_gt/
    └── features/
```

> **命名铁律**：`raster/` 下第一层是**渲染 rig**，第二层是**场景来源**。
> 这正是 Memo §五「原比较表需要一起修正」指出的问题——`Old Synthetic Raster / New Synthetic Raster` 把两个维度混成一个，导致「external metadata 生成的大量 target-view raster」被误记成「少量 New Synthetic」。目录结构上强制拆开，表就不会再画错。

```
${DW_RUNS_ROOT}/
│
├── pilot/                                    # 本次冲刺全部落这里（SD-2）
│   └── 2026-09-09/
│       └── <run_id>/
│           ├── config.frozen.yaml            # 含全部超参 + 决策阈值
│           ├── hashes.json                   # code / config / manifest / renderer / init ckpt
│           ├── logs/{train.log,gpu.log}
│           ├── ckpt/{step_XXXX.pt,latest.pt}
│           ├── eval/{final_epdms.json,components.json,openloop.json}
│           ├── preds/                        # 固定案例的预测，供跨 run 对比
│           └── STATUS                        # queued|running|evaluated|failed|aborted
│
└── confirmatory/                             # 会后 G2 起用；冲刺期间禁止写入
```

**`run_id` 约定**：`{YYYYMMDD}_{arm}_{budget}_{subset}_{seed}_{git8}`
例：`20260909_F1_b010_dev_s0_a1b2c3d4`

### 4.4 实验账本（单一真相源）

落盘于 `RAP/docs/experiments/registry.csv`，**每次启动训练前先写一行**，图表程序**只读这张表**，禁止手工从日志复制数字。

| 字段 | 说明 |
|---|---|
| `run_id` / `arm` / `protocol` | protocol ∈ {P1, P2}，一眼看出是否越过 H0 边界 |
| `target_manifest` + `sha256` | |
| `source_manifest` + `sha256` | |
| `render_rig` + `Cd_hash` | |
| `losses` | 形如 `plan_tgt=1.0,r2r_tgt=0.1,plan_src=0.25,align_ext=OFF` |
| `init_ckpt` + `checksum` | 用于证明未使用 full-NAVSIM RAP |
| `steps` / `wall_clock` / `n_target_scenes` / `n_src_presentations` | 区分「数据量增加」与「训练更久」 |
| `seed` / `subset` | |
| `status` / `failure_type` | failure_type ∈ {none, **technical**, **scientific**} |
| `metric_final_epdms` / `metric_stage1` / `metric_stage2` / `openloop` | |
| `is_confirmatory` | 本次冲刺一律 `false` |

> proposal §6.12 要求：**失败、不稳定、负迁移的 run 一律留在 registry，不得静默过滤。**

---

## 5. 人工检查卡点（Human-in-the-loop Checkpoints）

| ID | 时间 | 触发条件 | 必须由人类检查什么 | 未通过的动作 |
|---|---|---|---|---|
| **CP-1** | 9/9 12:00–13:00 | 任何代码/训练开始**之前** | 裁决 SD-1 / SD-2 / SD-3 / SD-7；填写 `paths.env`；确认 DEV manifest 定义 | **全部停机**。未裁决 SD-1 → X 系列永久不启动；未裁决 SD-3 → PPT 无法写 |
| **CP-2** | 9/9 15:00–16:00 | H0 审计完成 + smoke test 完成 | ① 逐条读 `h0_report.json`：source RGB/calib 读取是否真为 0、landmark 重投影误差、corruption 是否 100% 检出 ② 肉眼看若干 raster 叠图 ③ 确认 smoke 的 loss 接线与实测吞吐 | **H0 任一门限未过 → 禁止启动任何训练**（proposal §6.3：failed build 不得用于任何 transfer 结论）。这是本计划最硬的门 |
| **CP-3** | 9/9 18:00–19:30 | 写 X 系列代码**之前** | 裁决 SD-4 的 5 个未定项；确认 X 系列 spec 与 P1 的隔离方式 | 未裁决 → X 系列只出文档，不进代码库 |
| **CP-4** | 9/10 11:00 | 第一批结果出来后 | ① 先只看 Measurement，**不做解释** ② 核对公平性（同 step / 同 target scene / 同评测器版本）③ 裁决 SD-6 ④ 决定最后窗口跑什么 | 若公平性核对不过 → 该结果标 `invalid`，不上 PPT |
| **CP-5** | 9/10 18:00 | 架构冻结线 | 确认无新模块、无数据定义变更、无新缓存、无跑不完的训练在排队 | 强制冻结 |
| **CP-6** | 9/11 09:30 | 结果硬冻结 | 逐条检查结论措辞与证据强度是否匹配；确认所有数字标了 `preliminary / single subset / single seed` | 09:30 后**不因任何新数字重写主叙事**，迟到结果进 appendix |

**Technical Failure vs Scientific Failure 判定规则（写进 registry 的 `failure_type`）**：

| 现象 | 分类 | 处置 |
|---|---|---|
| OOM、shape mismatch、dataloader 崩、NaN、ckpt 损坏 | **Technical** | 修复后重跑；**不产生任何科学结论**；不计入假说判定 |
| 训练正常收敛，但 `M(F1) < M(F0)` | **Scientific** | 记录为 negative transfer 观测，进 registry 与 PPT，**不得**因结果不理想而私改 protocol |
| 训练正常，但差异 < 噪声 | **Scientific** | 结论 = `Inconclusive`，**不得**改写成"趋势向好" |
| H0 审计不通过 | **既非** | 阻断门，整个 build 作废 |

> **铁律**：实验开始后不得因初期结果不理想而私自修改 Protocol。任何 protocol 变更必须开新 `run_id`、在 registry 记录变更原因、并明确该 run 与前序 run **不可配对比较**。

---

## 6. 时间表汇总（半天粒度）

```
9/9  上午        │ (已过) 
9/9  12:00–13:00 │ ■CP-1 决策         │ 人
9/9  13:00–16:00 │ H0审计 + manifest冻结 + smoke │ CPU + GPU短占  │ ■CP-2
9/9  16:00–18:00 │ 启动 F0/F1         │ GPU  ┆ CPU:评测缓存 ┆ 人:PPT p1-4
9/9  18:00–21:00 │ X系列spec(■CP-3) + simulator调查(90min盒) │ 人
9/9  21:00–22:00 │ 过夜监控确认清单    │ 人
9/9  22:00→8:00  │ F0/F1 过夜训练      │ GPU（仅已smoke配置）
─────────────────┼──────────────────────────────────────────
9/10 08:00–11:00 │ 收结果 + 统一评测 + 出图 │ 人 + CPU       │ ■CP-4
9/10 11:00–18:00 │ 最后实验窗口(F2 或 复跑) │ GPU ┆ 人:PPT结果页 │ ■CP-5 18:00冻结
9/10 18:00–23:00 │ PPT 定稿            │ 人
9/10 23:00→8:00  │ 仅稳定续训/评测/归档 │ GPU
─────────────────┼──────────────────────────────────────────
9/11 08:00–09:30 │ 收尾结果 + 更新图表  │ 人
9/11 09:30       │ ■CP-6 结果硬冻结
9/11 09:30–11:20 │ 措辞校准 + Q&A 彩排  │ 人
9/11 11:20–12:20 │ 导出PDF + 本地备份 + 40min缓冲 │ 人
9/11 13:00       │ 多人会议
```

**三线并行原则**：
| 流水线 | 常驻任务 |
|---|---|
| **GPU lane** | 训练、checkpoint、推理 |
| **CPU/Data lane** | raster 预生成、缓存、评测、结果导出 |
| **Human lane** | 决策、错误分析、画图、PPT、Q&A 准备 |

**调试硬时间盒**：数据/shape/配置问题 30 min；明确 bug 最多 90 min；原因不明的训练不稳定 → 停该分支保留日志；新数据格式接入 2 h 内做不出 smoke test 就砍。

---

## 7. 测量与科学解释规程

### 7.1 报告顺序（不可颠倒）

> **先 Measurement，后 Scientific Interpretation。**
> PPT 与口头汇报都必须是：`这是数字 → 这是公平性核对 → 这是我能下的结论 → 这是我不能下的结论`。

### 7.2 允许写进 PPT 的结论类型

仅三种：**`Supported` / `Refuted` / `Inconclusive`**。

由于本次全部为 pilot（单 subset × 单 seed），**默认结论应为 `Inconclusive`**，除非出现灾难性负迁移（可下 `Refuted`（SH-1））或 H0 门限未过（可下 `Refuted`（SH-0））。

### 7.3 结果解释对照表（沿用 proposal §6.7）

| 观测 | **允许**的表述 | **禁止**的表述 |
|---|---|---|
| F1 > F0 且 F1 > F2 | external structured content 在此 pilot 条件下有超出 replay 与辅助分支的正向信号 | "证明了 external knowledge 有效"；"H1 成立" |
| F1 > F0 但 F1 ≈ F2 | 增加的训练分支有帮助，但 external content 未被隔离 | "external 场景多样性带来增益" |
| F1 ≈ F0 | 在该预算与配方下未观测到效应，或功效不足 | "略有提升" |
| F1 < F0 | 在该配置下观测到负迁移 | 静默丢弃该 run |
| 只有 open-loop 指标 | 仅 diagnostic，`Inconclusive` | 任何 Supported/Refuted |

### 7.4 措辞审计（CP-6 逐条过）

- 每个数字后必须带：`b*`、subset ID、seed、step 数、是否 pilot
- 每个主张必须带范围限定：`on the evaluated NAVSIM target rig`（proposal R11）
- 出现 "证明 / 显著 / 最强 / 有效" 而无 95% 区间支撑 → 一律改写
- 「all-data 最强」在 9/11 之前**不得**作为结论出现（佐藤 9/8 明确要求；Memo 已确认状态为「值得开展比较实验；不预设 B 优于 A」）

---

## 8. 9/11 汇报结构（7–9 页）

按 Memo §八建议顺序：**目标与已知条件 → 两类知识及数据路径 → 最小对照结果 → 能/不能支持的判断 → RL 可行性初查 → 需要老师共同判断的问题**

| # | 页 | 关键内容 |
|---|---|---|
| 1 | 我希望老师判断什么 | 直接列 SD-1 / SD-5 / 3D baseline 排期 / RL 投入比例 |
| 2 | 问题定义与协议 | target 是谁、何时知道 calibration、zero/few-shot 数什么、b\*=10% scenes（SD-3 结果）+ 绝对计数 |
| 3 | 两类知识 × 数据路径 × loss | **场景来源 × 渲染 rig × 接受的 loss** 三维表（修正原表 Old/New Synthetic 混淆） |
| 4 | H0 审计通过证明 | 读取计数、几何误差、corruption 检出、raster 叠图 |
| 5 | 最小对照 pilot 结果 | F0/F1(/F2)；全部标 preliminary；含公平性核对表 |
| 6 | 我能下 / 不能下的结论 | 严格用 Supported / Refuted / Inconclusive |
| 7 | **X 系列（方案B）设计 + 协议冲突** | SD-1 全文；池畑主张 ↔ H0 边界的冲突；请老师裁决 |
| 8 | **external-rig planning（SD-5）** | 并列呈现池畑 Joint Training 与佐藤反对意见，请两位当面裁决 |
| 9 | RL 可行性 + 下一步 | SH-R 四项判定 + 缺口；3D/BEV baseline 排期 |

> 结果未完成的格子**不留空白**，写清：当前状态 / 已运行时长 / ETA / 能回答什么 / 暂时不能回答什么。

---

## 9. Git Worktree 并行操作指南（保姆级）

### 9.0 先决条件：本机当前没有 git 仓库

实测结果：

```
$ git -C /Users/joy/Downloads/DriveWeave/RAP rev-parse --is-inside-work-tree
fatal: not a git repository (or any of the parent directories): .git
```

`DriveWeave/` 和 `RAP/` 都不是 git 仓库。所以**第一步是确认你真正的 RAP 代码仓库在哪**：

```bash
# 在 TSUBAME 上（或本地放代码的地方）找到含 .git 的目录
find ~ -maxdepth 4 -type d -name ".git" 2>/dev/null | head
```

下面的指南默认你的代码仓库根目录是 `$REPO`（例如 `~/work/RAP`）。

---

### 9.1 一句话理解 worktree

> **worktree = 同一个 git 仓库，同时在磁盘上"摊开"成多个独立目录，每个目录停在不同分支。**

对比你现在可能在做的：

| 做法 | 问题 |
|---|---|
| `git checkout` 来回切分支 | 训练跑到一半切分支 → **正在读的代码被换掉**，实验直接污染 |
| `git stash` | 容易忘记 stash 了什么，且不能同时跑两份 |
| `cp -r` 整个仓库 | 两份 `.git` 各自演化，改动合不回去，磁盘翻倍 |
| **`git worktree`** | ✅ 共享一个 `.git`（省磁盘、历史统一），但工作目录彼此独立、互不干扰 |

**为什么这次需要它**：F0/F1 跑在 `exp/f1-source-raster` 分支上要好几个小时；同一时间你要开发 X 系列（`exp/x2-external-align`）。没有 worktree，你只能干等。

---

### 9.2 完整实操（复制粘贴即可）

**Step 1｜进入仓库，确认干净**

```bash
cd $REPO
git status
```

> 如果有未提交改动，先 `git add -A && git commit -m "wip"` 或 `git stash`。**worktree 不会帮你搬运未提交的改动。**

**Step 2｜看一眼现有 worktree（一开始只有主目录）**

```bash
git worktree list
```

**Step 3｜创建第一个并行工作树（跑实验用）**

```bash
git worktree add ../RAP-exp-f1 -b exp/f1-source-raster
```

含义：在**上一级目录**新建 `RAP-exp-f1/`，并在其中创建并切到新分支 `exp/f1-source-raster`。

**Step 4｜创建第二个并行工作树（开发 X 系列用）**

```bash
git worktree add ../RAP-dev-x2 -b exp/x2-external-align
```

**Step 5｜确认**

```bash
git worktree list
# /home/you/work/RAP           a1b2c3d [main]
# /home/you/work/RAP-exp-f1    a1b2c3d [exp/f1-source-raster]
# /home/you/work/RAP-dev-x2    a1b2c3d [exp/x2-external-align]
```

**Step 6｜各自干活**

```bash
# 终端 A：跑训练，绝不碰这个目录的代码
cd ~/work/RAP-exp-f1
python train.py --config configs/f1.yaml

# 终端 B：随便改代码，完全影响不到终端 A
cd ~/work/RAP-dev-x2
vim rap/losses/external_align.py
```

**Step 7｜提交与同步**（每个 worktree 就是普通仓库，命令完全一样）

```bash
cd ~/work/RAP-dev-x2
git add -A
git commit -m "spec: external-rig alignment branch (no allowlist change yet)"
git push -u origin exp/x2-external-align
```

**Step 8｜用完清理**

```bash
cd $REPO
git worktree remove ../RAP-dev-x2      # 目录干净时可直接删
git worktree remove --force ../RAP-dev-x2   # 有未提交改动时（会丢改动，慎用）
git worktree prune                     # 清理手工 rm 掉目录后残留的记录
```

---

### 9.3 新手最容易踩的 6 个坑

| # | 坑 | 现象 | 正确做法 |
|---|---|---|---|
| 1 | **同一分支不能同时被两个 worktree 检出** | `fatal: 'main' is already checked out at ...` | 每个 worktree 用**自己的分支**（`-b` 新建），或用 detached：`git worktree add ../tmp <commit>` |
| 2 | **worktree 不共享未提交改动** | 新目录里看不到你刚改的代码 | 先在原目录 commit，再 `git worktree add` |
| 3 | **Python 环境不会自动跟过去** | 新目录里 `import rap` 报错 | 每个 worktree 单独 `pip install -e .`；或用同一个 conda env 但确认 `PYTHONPATH` 指向当前 worktree |
| 4 | **数据/输出目录被复制或写乱** | 磁盘爆掉，或两个 run 互相覆盖 | **数据与 runs 永远放在仓库外**（`${DW_DATA_ROOT}` / `${DW_RUNS_ROOT}`），worktree 内只放符号链接：`ln -s ${DW_DATA_ROOT} data` |
| 5 | **TSUBAME 作业脚本路径写死** | 提交的 job 跑的是另一个 worktree 的代码 | job 脚本第一行显式 `cd /abs/path/to/RAP-exp-f1`；并在日志里打印 `git rev-parse HEAD` |
| 6 | **直接 `rm -rf` 掉 worktree 目录** | `git worktree list` 里留幽灵条目 | 用 `git worktree remove`；已经 rm 了就跑 `git worktree prune` |

---

### 9.4 本次冲刺的推荐布局

```
~/work/
├── RAP/                 [main]                        ← 只读参考，不在这里跑东西
├── RAP-exp-f1/          [exp/f1-source-raster]        ← 9/9 16:00 起长时间占用 GPU
└── RAP-dev-x2/          [exp/x2-external-align]       ← 9/9 18:00 起写 spec / 代码（SD-1 通过后才动 allowlist）
```

配套（每个 worktree 内执行一次）：

```bash
ln -s ${DW_DATA_ROOT}  data
ln -s ${DW_RUNS_ROOT}  runs
echo "data/"  >> .git/info/exclude
echo "runs/"  >> .git/info/exclude
```

在训练脚本开头加一行，保证 registry 里的 `git8` 与实际跑的代码一致：

```bash
echo "RUN_GIT_SHA=$(git rev-parse --short=8 HEAD)  WORKTREE=$(pwd)" | tee -a runs/pilot/$RUN_ID/hashes.json
```

---

## 10. 附录：与 9/2 池畑要求的对齐状态

| 池畑 9/2 的要求 | 本冲刺内 | 会后排期 |
|---|---|---|
| 明确 Problem Setting，拆细 rig 差异（count / extrinsics / FOV / focal / projection / distortion） | ✅ PPT p.2–3 | 受控 rig 变体实验（G6） |
| 先做 baseline，明确"要打倒的敌人" | ✅ F0 | REF_100（clean full-target reference） |
| 必须比较 BEV / 3D feed-forward / 3DGS / VGGT+Fisheye3R 式 adaptation，training data 完全相同 | ❌ 48h 内不可能 | **会后第一优先**；先做 3D feed-forward 的 zero-shot 可跑性检查（成本最低、收益最高） |
| 不要"先有 RAP 再找理由" | ✅ 用 F3 标签置换 + F2 曝光控制来证伪 | G4 |
| random sampling × 多 seed，不要精挑 few-shot 样本 | 🔶 proposal §5.7 用的是**分层随机**，并已注册 pure-random robustness check | 与池畑确认分层 vs 纯随机的取舍（proposal §5.7 已明写该选择会改变主张措辞） |
| 最终不能只有 NAVSIM 一个 target | ❌ | G6（proposal §7.3 开放问题 1） |

---

*本文档由 AI 依据已冻结的 Research Spec 与 9/2、9/8 两份 1v1 记录规格化生成。所有 `[Scientific Decision Required]` 项均未被自动补全。所有 `${...}` 路径为待人工填写的占位符。*
