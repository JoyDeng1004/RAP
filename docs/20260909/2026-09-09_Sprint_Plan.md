# 2026-09-09 → 09-11｜48h 实验冲刺计划（Sprint Plan）· **v2.2**

> v2 以 **`docs/20260909/` 下的现有文件 + RAP 开源代码库（本仓库）** 为唯一基准重写，补齐 TSUBAME4.0 执行架构、真实路径规范、数据集策略与 Git 并行方案。
> 所有涉及云端的断言均已降级为 **待审计项**，并附可执行的审计命令。
> v2.1 增补逐次 `CP-CODE` Human 检查、原子 commit / 小步迭代规则，以及进入 Smoke Test 前的 `A5` 训练配置与数据冻结门。
> v2.2 记录 Human 对 `SD-0` 选项 B 及 `SD-7/SD-8` 的裁决，并冻结当前 development pilot 的预算、实验臂、seed、指标、access boundary、checkpoint 规则、Simulator/RL 投入上限与脚本入库边界。

---

## 0. 基准、环境与角色

### 0.1 唯一基准

| 类型 | 位置 | 状态 |
|---|---|---|
| 会议 Memo | `context/2026-09-08_佐藤1v1_完整版会议Memo.md` | ✅ 在库 |
| 9/2 池畑 1v1 转录 | `context/20260902_ikehata.txt` | ✅ 在库 |
| 9/8 佐藤 1v1 转录 | `context/20260908_sato.txt` | ✅ 在库 |
| 代码库 | 本仓库 `main` 分支（`JoyDeng1004/RAP`，RAP 上游 fork） | ✅ 在库 |

### 0.2 时间与运行环境

- **计划起点**：2026-09-09
- **硬截止**：2026-09-11 13:00 JST 多人会议（池畑 / 佐藤 / 関川 / Joy）
- **可用窗口**：约 **40 小时**（含睡眠与 PPT 制作）

| 环节 | 位置 | 说明 |
|---|---|---|
| 开发 / 编写 | 本地 MacBook | 只编辑、只提交，**不训练** |
| 训练 / 评测 | **TSUBAME4.0** | H100 × 4（= 1 × `node_f`），作业调度器见 `EF-1` |
| 仓库根（云端） | `/gs/bs/tga-RLA/qdeng/RAP` | 即 `$RAP_ROOT` |
| 审计报告归档 | `/gs/bs/tga-RLA/qdeng/RAP/docs/audits/` | 本次 = `docs/audits/2026-09-09_asset_audit/` |

### 0.3 分工边界

Human（Joy）：研究问题定义、`SD-*` 科学决策、`EF-*` 环境事实确认、最终判断。
AI：规格化、代码生成、审计脚本、验证、测量与监控。遇 `SD-*` / `EF-*` 一律停机等待。

### 0.4 工作流合规

```
Problem → Hypothesis → Research Spec → Implementation Spec
       → Environment Check → Smoke Test → Full Experiment
       → Measurement → Scientific Interpretation
```

**禁止跳跃。** 本文件是 Implementation Spec 层；`Environment Check` 由 §3 的资产审计承担，未通过不得进入 Smoke Test。

| 阶段 | 状态 | 载体 |
|---|---|---|
| Problem | ✅ 已定义 | 9/8 Memo 一、1 |
| Hypothesis | ✅ F0/F1 development pilot 已冻结 | §2 `SD-0/1/2/3/6`、§5.4 |
| Research Spec | 🔶 P1 pilot 已冻结；P2 仍受 `SD-4/5` 影响 | §2、§5 |
| **Implementation Spec** | 🔶 **本文档** | 本文件 |
| Environment Check | ⬜ 待执行 | §1 `EF-*` + §3 资产审计 |
| Smoke Test | ⬜ 受 `CP-2b` 门控 | §3.6、§6 |
| Full Experiment | ⬜ 受 `CP-2b/2c` 门控 | §6 |
| Measurement | ⬜ | §10 |
| Scientific Interpretation | ⬜ 仅允许 `Supported` / `Refuted` / `Inconclusive` | §10.2 |

---

## 1. 环境事实待确认清单 `[EF-*]`

> 这些是**可以被查证的客观事实**，不是科学决策。查证前不得据此写作业脚本。

### `[EF-1]｜作业调度器：Slurm 还是 qsub？` 🔴 阻断所有作业脚本编写

- **任务简报声明**：TSUBAME4.0 采用 **Slurm**。
- **仓库实际证据**（`scripts/*.qsub`，共 8 个文件，本地工作树内）：

  ```bash
  #!/bin/bash
  #$ -S /bin/bash                 # ← Grid Engine 风格指令，不是 #SBATCH
  #$ -N rap-stage-a-02
  #$ -ar 8016                     # ← advance reservation
  #$ -l node_f=1                  # ← TSUBAME4 节点类型（node_f = 满节点 = H100×4）
  #$ -l h_rt=06:30:00
  #$ -j y
  #$ -o /gs/bs/tga-RLA/qdeng/RAP/outputs/alignment_stage_a/step_0_2_train_metric_cache.log
  # 提交方式：qsub -g tga-RLA <this-script>
  ```

- **冲突**：`#$` 指令 + `qsub -g` + `node_f` / `gpu_1` / `h_rt` / `-ar` 全部是 **qsub 体系**；Slurm 应为 `#SBATCH` + `sbatch` + `--gres` / `--time` / `-A`。
- **必须确认**：登录 TSUBAME 后执行

  ```bash
  which qsub sbatch squeue qstat 2>/dev/null
  qstat --version 2>/dev/null; sinfo --version 2>/dev/null
  ```

- **在确认前**：所有作业模板写成 **调度器无关**的两层结构 —— 业务逻辑放 `scripts/jobs/<name>.body.sh`（纯 bash，可直接本地跑），调度指令放 `.qsub` / `.sbatch` 两个薄壳。无论 `EF-1` 结论如何，业务代码零改动。
- **默认取值**：在明确证据出现前，**默认沿用仓库现存的 qsub 体系**（有 8 个可工作先例，风险最低）。

### `[EF-2]｜`scripts/` 被 gitignore，作业脚本从未进入版本控制` 🔴

`.gitignore` 第 6 行：`scripts/`
`git ls-tree -r main -- scripts/` 与 `git ls-tree -r exp/rap-alignment-regression -- scripts/` **均为 0 个文件**。

**后果**：
1. 8 个 TSUBAME 作业脚本**只存在于本地/云端磁盘**，丢失即不可恢复；
2. §8 的 branch / worktree 方案**无法携带它们**（worktree 只 checkout 被跟踪的文件）；
3. 每个 run 的 provenance 缺失最关键的一环——实际提交的作业脚本内容。

**处置（`SD-8` 已通过；仍须按 `CP-CODE` 获得 Human `PASS` 后执行并提交原子 commit）**：

```gitignore
scripts/*
!scripts/jobs/
!scripts/jobs/**
!scripts/audit/
!scripts/audit/**
```

### `[EF-3]｜`*.json` 被 gitignore，审计报告不能用 JSON 落库`

`.gitignore` 含 `*.json`、`*.png`、`outputs/`。因此：
- 审计报告主体用 **Markdown**（`.md`）；
- 机器可读副本用 `.jsonl` 或显式 `!docs/audits/**/*.json` 白名单；
- raster 可视化（`.png`）默认不入库，只存 `$RAP_ROOT/outputs/` 并在报告里记录路径 + sha256。

### `[EF-4]｜`main` 分支缺少冲刺所需的多项资产` 🔴

`git diff --stat main exp/rap-alignment-regression` → **89 files changed, +114,014**。`main` 上**不存在**：

| 缺失项 | 在 `exp/rap-alignment-regression` 上 | 冲刺是否需要 |
|---|---|---|
| `tools/render_nuscenes_camera_cross.py` | 484 行 | ⭐ **F1 臂的核心**（nuScenes 跨相机渲染） |
| `tools/render_navsim_scene.py` | 364 行 | ⭐ target-rig raster 渲染 |
| `tools/render_navsim_augmentations.py` | 439 行 | 🔶 |
| `tools/canonical_bev/` | — | 🔶 |
| `train_test_split/paired_trainval.yaml`、`scene_filter/paired_54log.yaml` | 2 个 | ⭐ paired 数据划分 |
| `navsim/planning/script/build_alignment_small_data.py` | 1 个 | ⭐ |
| `docs/20260909/` 本身 | 4 个文件 | ⭐ |

**`navsim/agents/rap_dino/` 在 `main` 上存在（22 个文件），这点没问题。**

**结论**：「基础代码库基于 `main` 且保持干净」这条约束**当前不可直接满足**——纯 `main` 跑不了 F1。必须先把上述资产**逐路径**并入一个 main-based 分支（见 §8.2 Step 0），否则整个冲刺没有起跑线。**这是今天的第一件事。**

---

## 2. 熔断清单 `[Scientific Decision Required]`

> 未裁决前，相关代码不写、相关实验不跑。

### `SD-0｜冻结规格不在基准内` ✅ Human 已裁决：选项 B

v1 的 `SD-1/2/3`、`F0/F1/F2/F3/A0/A1` 命名、`b*=10%`、`δ_main=1.0`、`Final EPDMS` 判据，**全部引自 `DriveWeave_proposal.md`**，而该文件不在本仓库、也不在 `docs/20260909/`。

**Human 决议（2026-09-09）**：选择 **选项 B**。`DriveWeave_proposal.md` 不作为本冲刺的约束来源；其中派生的 `b_min=100`、`δ_main=1.0`、`3 subsets × 2 seeds`、`Final EPDMS` 以及 F2/F3/A0/A1 的既有定义全部失效，不得继续当作冻结事实引用。

**替代冻结协议（Human 已确认）**：

| 项目 | 本冲刺冻结值 |
|---|---|
| Target budget | NAVSIM 合法训练集的 `scene/log`；比例 = **10%**；无最小 scene 数。绝对 scene/frame/image 数由 A1/A5 实测 |
| 实验臂 | 当前只运行 **F0/F1**；X1/X2/X3 仅保留为 P2 规格；F2/F3/A0/A1 不属于本冲刺冻结协议 |
| Run 身份 | 全部为 `development pilot`、`non-confirmatory`、`preliminary` |
| Subset / seed / 重复 | 一个随机 DEV subset；`manifest_seed=20260909`；`training_seed=20260909`；F0/F1 各 1 次 |
| 主指标 | NAVSIM **PDMS**；其既有分项为 secondary metrics；open-loop 仅为 diagnostic |
| 效应阈值 | 当前 pilot **不设数值型 confirmatory threshold**；不得据单 subset / 单 seed 下 `Supported` 或 `Refuted`，科学结论统一为 `Inconclusive` |
| Source RGB / calibration | P1（F0/F1）禁止读取；P2（X1/X2/X3）仅可为 external real/raster alignment 读取，且须与 P1 隔离并先通过 `SD-4/CP-3` |
| Checkpoint | F0/F1 必须 from scratch；只允许同一 run 在 code/config/data hash 完全一致时续训；既有 checkpoint 仅作 A3 审计，不得作为 initializer、teacher、feature cache、pseudo-labeler 或超参数依据；P2 distillation 另行审批 |

上述值如需改变，必须由 Human 新开决议、生成新 `run_id` 并重做 A5；不得在查看结果后回改本表。

### `SD-1｜source-pixel / calibration 边界` ✅ P1 已裁决；P2 受 `SD-4/CP-3` 门控

**P1（本冲刺 F0/F1）**：nuScenes 只允许读取 maps、agents、ego/world state 与 future trajectory labels；source RGB、source intrinsics、source extrinsics 的读取/解码计数必须全部为 0。F1 raster 只使用已知 target rig `C_d`。

**P2（X1/X2/X3）**：允许读取 external RGB 与对应 source calibration，但仅用于同场景、source-rig `C_s` 下的 external real/raster alignment。P2 必须使用独立协议、代码路径、产物目录和 access audit；不得把 P2 产物或 cache 输入 P1。

`A4` 仍是硬门：若现有工具或产物读取过 source RGB/calibration，必须标为 P2；不能用于 F0/F1，也不能声称满足 P1。X 系列在 `SD-4/CP-3` 前只写规格，不写代码、不运行。

### `SD-2｜9/9–9/11 的 run 如何归类` ✅ 已裁决

本窗口全部 run 均为 **development pilot**，不是 confirmatory experiment。使用一个随机 DEV subset：

- Target sampling unit = 合法 NAVSIM training `scene/log`；抽取比例 = 10%
- `manifest_seed = 20260909`
- `training_seed = 20260909`
- F0/F1 使用同一份冻结 manifest、相同 seed，各运行 1 次
- manifest 生成一次后保存 sha256；后续只读取该文件，不得重新抽样
- 所有 run 落 `$RAP_ROOT/outputs/sprint_20260909/pilot/`，并标注 `non-confirmatory, preliminary, single subset, single seed`

未来 confirmatory protocol 的 subset、seed 和重复次数**尚未定义**，不得从已失效 proposal 恢复，也不得使用当前 pilot 结果反向选择。

### `SD-3｜target budget 对外口径` ✅ 已裁决

本冲刺 target budget 冻结为：**NAVSIM 合法训练集 scene/log 的 10%**。无 `b_min`；原 `b_min=100 scenes` 已随 `SD-0` 选项 B 失效。

- A1/A5 必须记录合法训练集总 scene/log 数、抽中数量、对应 frame/image/pair 数与 manifest sha256。
- frame、image、pair 和等效时长只作描述性统计，不作为抽样单位。
- 「1000h vs 10min ≈ 6000:1」只可作为会议中出现的假设性 motivation，不得作为实际预算或实测比例。

### `SD-4｜X 系列 alignment 规格未定`

未定项（**不得由 AI 猜测补全**）：
1. external raster 用哪套 rig 渲染（nuScenes 原生 rig？还是简化 canonical rig？）
2. alignment 在哪一层做（`B` 输出？`P_R/P_d` 输出 `F^R/F^I`？planning representation `G` 之后？）
3. 跨 rig 共享哪些模块、按 rig 分叉哪些（池畑提的 rig-dependent embedding 要不要引入）
4. 距离度量与权重（判别器 / 余弦 / MSE；λ 初值）
5. 与既有 target R2R loss 如何共存、是否需要单独的 on/off 消融

**裁决时机**：`CP-3`，且必须在写 X 系列代码之前。

### `SD-5｜external-rig planning loss 做不做`

池畑 9/2 §44 的 Joint Training（两个 planner + 共享 encoder + rig-dependent embedding）**前提就是在 external rig 上算 planning loss**；佐藤 9/8 10:08–10:11 明确判断「アライメントはいいと思うんですけど、プラナンは多分だめだと思う」「性能には寄与しない」。Memo §五记录的准确状态是「保留为可检验的消融，不能写成已被证明无效」。

**两位老师互不知道对方的表态。这是 9/11 会上第一议题。**

**建议**：本冲刺**不跑** X3，只带实验设计上会，当面裁决后再实现。

### `SD-6｜冲刺期评测指标` ✅ 已裁决

本冲刺主指标为 NAVSIM **PDMS**；当前评测器已有的 PDMS 分项作为 secondary metrics。open-loop 指标仅为 diagnostic，不可替代 PDMS。

当前 development pilot 不设置数值型 confirmatory effect threshold。无论 F1 相对 F0 的方向或幅度如何，科学结论均为 `Inconclusive`；只允许报告原始差值、方向、公平性核对与限制。`Final EPDMS`、`δ_main=1.0` 不再属于本冲刺协议。

### `SD-7｜Simulator/RL 投入上限` ✅ Human 已裁决

佐藤 9/8 新提，认为「可能研究价值更高」，甚至可各成一篇；池畑不知情。

**Human 决议（2026-09-09）**：接受 **90 分钟硬时间盒**；本冲刺只整理候选表并判断接口可行性，**不写训练代码、不启动 Simulator/RL 训练**。到达时间上限即停止，并记录已完成项、未决问题与后续建议，不得挤占 F0/F1、A1–A5、H0 或汇报准备的关键路径。

### `SD-8｜是否把 `scripts/` 纳入版本控制` ✅ Human 已裁决

**Human 决议（2026-09-09）**：允许将 `scripts/jobs/` 与 `scripts/audit/` 纳入版本控制；`scripts/` 下其他内容继续忽略。具体 `.gitignore` 修改与目录纳入按 `EF-2`、§8.0 和 `CP-CODE` 执行，不得把未检查的其他脚本一并加入。

---

## 3. 前置工程任务：TSUBAME 云端资产审计

> **这是 Environment Check 阶段，必须在任何训练之前完成。**
> 报告统一归档至 `/gs/bs/tga-RLA/qdeng/RAP/docs/audits/2026-09-09_asset_audit/`。
> 审计脚本落在 `scripts/audit/`（`SD-8` 已通过；实际入库仍受 `CP-CODE` 门控）。

### 3.0 通用规范

- 每项审计产出**两份**：`<name>.md`（人读，入库）+ `<name>.jsonl`（机读，入库需 `EF-3` 白名单）
- 每份报告首部记录：审计时间、执行主机、`git rev-parse HEAD`、`RAP_ROOT`、执行者
- **只读**：不得写入被审计目录，不得移动/删除任何文件
- 大目录统计用 `find ... | wc -l` 而非 `ls -R`；**超过 5 分钟的统计放计算节点跑**，避免 login 节点超时

### 3.1 `A1`｜RAP 数据增强/处理目录

**对象**：`$RAP_ROOT/dataset_aug`、`$RAP_ROOT/dataset_norm`、`$RAP_ROOT/dataset_perturbed`

**已知线索（来自本仓库，需验证）**：三者均出现在 `.gitignore`，路径 = `$RAP_ROOT/<name>`，说明是**仓库根下的生成物**。`process_data/` 下恰好有三个对应脚本：

| 目录 | 推测生成脚本 | 待验证 |
|---|---|---|
| `dataset_norm` | `process_data/create_openscene_metadata.py` | 是否为未扰动 baseline metadata |
| `dataset_aug` | `process_data/create_openscene_metadata_aug.py` | 增强类型、倍率 |
| `dataset_perturbed` | `process_data/create_openscene_metadata_purturbed.py`（上游拼写 `purturbed`；`env/tsubame` 分支已更名 `perturbed`） | 扰动参数分布 |

**必须回答的 4 个问题**：
1. **数据来源**：哪个脚本、哪个 commit、哪次作业生成？输入的 OpenScene/NAVSIM 原始路径？
2. **属于 NAVSIM 哪个子集**：`navtrain` / `navtest` / `trainval` / `mini` / `navhard_two_stage`？
3. **数据完整度**：log 数、token 数、缺失率；与官方 split 的交集/差集
4. **子目录文件结构**：到叶子节点的目录树 + 每层文件类型与数量

```bash
OUT=$RAP_ROOT/docs/audits/2026-09-09_asset_audit
mkdir -p "$OUT"

for D in dataset_aug dataset_norm dataset_perturbed; do
  P="$RAP_ROOT/$D"
  {
    echo "## $D"
    echo '```'
    echo "path: $P"
    echo "exists: $([ -d "$P" ] && echo yes || echo no)"
    echo "total_size: $(du -sh "$P" 2>/dev/null | cut -f1)"
    echo "n_files: $(find "$P" -type f 2>/dev/null | wc -l)"
    echo "n_dirs:  $(find "$P" -type d 2>/dev/null | wc -l)"
    echo "mtime_range: $(find "$P" -type f -printf '%T+\n' 2>/dev/null | sort | sed -n '1p;$p' | tr '\n' ' ')"
    echo; echo "-- depth-2 tree --";        find "$P" -maxdepth 2 -type d 2>/dev/null | head -40
    echo; echo "-- extension histogram --"; find "$P" -type f 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15
    echo; echo "-- 3 sample leaf files --"; find "$P" -type f 2>/dev/null | head -3
    echo '```'; echo
  } >> "$OUT/A1_rap_datasets.md"
done
```

**子集归属判定**（`scripts/audit/a1_split_membership.py`，规格如下，**目录结构确认前不实现**）：

```
1. 从 dataset_norm 解析出全部 token / log 名
2. 载入官方 split：navsim/planning/script/config/common/train_test_split/scene_filter/{navtrain,navtest,navmini}.yaml
3. 输出 |A∩navtrain| |A∩navtest| |A\(全部官方split)|
4. 硬门限：与 navtest / navhard_two_stage 的交集必须为 0，否则标 CONTAMINATED 并阻断训练
```

> ⚠️ **这是本次审计中唯一可能直接否决整个冲刺的门**：若 `dataset_*` 混有 `navtest` token，任何基于它们的结果都不可用。

### 3.2 `A2`｜nuScenes 原始数据集

**对象**：`/gs/bs/tga-RLA/qdeng/data/nuscenes`

**必须回答**：
1. 版本（`v1.0-trainval` / `v1.0-mini` / `v1.0-test`）与是否完整
2. 上下级目录结构：`maps/` `samples/` `sweeps/` `v1.0-*/` 各自存在与体量
3. **`samples/`（RGB）与 `v1.0-*/calibrated_sensor.json`（相机标定）是否存在** ← 直接决定 `SD-1` 的 X 系列在物理上是否可行
4. 与本仓库 `nuscenes-mini/` 本地副本的关系

```bash
P=/gs/bs/tga-RLA/qdeng/data/nuscenes
{
  echo "## nuScenes"; echo '```'
  echo "total_size: $(du -sh $P | cut -f1)"
  echo "-- top level --"; ls -la "$P"
  echo "-- version dirs --"; ls -d "$P"/v1.0-* 2>/dev/null
  for V in "$P"/v1.0-*; do [ -d "$V" ] || continue
    echo "-- $(basename $V) tables --"; ls -la "$V" | awk '{print $5, $9}'; done
  echo "-- sensor dirs under samples/ --"; ls "$P/samples" 2>/dev/null
  echo "-- per-sensor file counts --"
  for S in "$P"/samples/*; do [ -d "$S" ] && echo "$(basename $S): $(find "$S" -type f | wc -l)"; done
  echo "-- maps --"; ls "$P/maps" 2>/dev/null | head
  echo "-- calibration table present? --"
  for V in "$P"/v1.0-*; do [ -f "$V/calibrated_sensor.json" ] && echo "$V/calibrated_sensor.json  $(stat -c%s "$V/calibrated_sensor.json") bytes"; done
  echo '```'
} > "$OUT/A2_nuscenes.md"
```

**完整度判据**（`v1.0-trainval`）：`sample` 表约 34k 条、6 个相机目录各约 34k 张、`sweeps/` 远大于 `samples/`。报告须写明**实测值 vs 官方声明值**的差异。

### 3.3 `A3`｜模型 Checkpoint

**对象**：`$RAP_ROOT/ckpts`。**必须回答**：每个 ckpt 的**内部结构**与**加载方式**。

`scripts/audit/a3_probe_ckpt.py`：

```python
import sys, torch, hashlib, json
from pathlib import Path

p = Path(sys.argv[1])
sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
obj = torch.load(p, map_location="cpu", weights_only=False)

rec = {"file": str(p), "size_bytes": p.stat().st_size, "sha256_16": sha,
       "top_level_type": type(obj).__name__}

if isinstance(obj, dict):
    rec["top_level_keys"] = sorted(obj.keys())[:30]
    rec["is_lightning"] = all(k in obj for k in ("state_dict", "epoch", "global_step"))
    sd = obj.get("state_dict", obj)
    rec["n_tensors"] = sum(1 for v in sd.values() if hasattr(v, "shape"))
    prefixes = {}
    for k in sd:
        prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
    rec["module_prefixes"] = dict(sorted(prefixes.items(), key=lambda x: -x[1])[:20])
    rec["sample_shapes"] = {k: list(v.shape) for k, v in list(sd.items())[:8] if hasattr(v, "shape")}
    if "hyper_parameters" in obj:
        rec["hyper_parameters_keys"] = sorted(obj["hyper_parameters"].keys())[:30]

print(json.dumps(rec, ensure_ascii=False, indent=2))
```

**每个 ckpt 必须给出一行「加载方式」结论**：

| 类别 | 判定依据 | 加载方式 |
|---|---|---|
| Lightning 训练 ckpt | 含 `state_dict`+`epoch`+`global_step` | `AgentLightningModule.load_from_checkpoint()`，或手动取 `["state_dict"]` 后 strip 前缀 |
| 纯 `state_dict` | 顶层即 tensor 字典 | `model.load_state_dict(torch.load(...))` |
| HF / timm backbone | 前缀含 `backbone.` / `blocks.` / `patch_embed.` | 仅作架构与格式审计；F0/F1 不允许 load 既有权重 |
| 未知 | 其他 | 标 `UNKNOWN`，不得使用 |

**⚠️ Human 冻结的硬约束**：F0/F1 必须 from scratch；所有既有 checkpoint 均禁止作为 initializer、teacher、feature cache、pseudo-labeler 或超参数依据，审计报告统一标 `ALLOWED_AS_INIT: no`。只允许同一 `run_id` 在 code/config/data hash 完全一致时从该 run 自身 checkpoint 续训；P2 distillation 须另行审批。

### 3.4 `A4`｜source-pixel 边界现状审计 🆕（P1 F0/F1 的前置）

**对象**：`tools/render_nuscenes_camera_cross.py`(484 行)、`tools/canonical_bev/`

**必须回答**：
1. 是否读取 nuScenes 的 `calibrated_sensor.json` / `sensor.json`（= source 相机标定）？
2. 是否读取 `samples/CAM_*/*.jpg`（= source RGB）？
3. 输出的 raster 用的是 **nuScenes 原生 rig** 还是 **NAVSIM target rig**？
4. `outputs/poster_pairs/nuscenes_cross_camera/` 里已有的产物属于哪一种？

```bash
grep -nE "calibrated_sensor|sensor\.json|CAM_|samples/|\.jpg|intrinsic|extrinsic|translation|rotation" \
  tools/render_nuscenes_camera_cross.py | head -40
```

**判定后果**：
- **只读 metadata/标注、不读 RGB 与标定** → 可继续检查是否满足 P1 F1 的边界
- **读取 source RGB 或标定** → 代码与产物必须标为 P2；不得输入 F0/F1，也不得声称满足 P1

### 3.5 审计交付物清单

```
/gs/bs/tga-RLA/qdeng/RAP/docs/audits/2026-09-09_asset_audit/
├── README.md                  # 概要 + 每项 PASS/FAIL/BLOCKED + 阻断项清单
├── A1_rap_datasets.md
├── A1_split_membership.md     # ⭐ 与 navtrain/navtest 交集，含污染判定
├── A2_nuscenes.md
├── A3_ckpts.md                # 每个 ckpt 一节 + ALLOWED_AS_INIT
├── A4_source_pixel_boundary.md
├── A5_training_readiness.md   # ⭐ 训练配置、数据体量、初始化与复现条件的冻结审计
└── raw/                       # 原始 jsonl / 命令输出
```

### 3.6 `A5`｜训练前配置与数据冻结审计 🆕

> **这是进入 Smoke Test 之前的硬门。** A1–A4 只回答资产是否存在、是否完整、是否污染以及能否合法使用；它们不能替代对“这一次具体会怎样训练”的检查。每个拟运行的实验臂必须生成一份 `A5_training_readiness.md`，并在 `CP-2b` 获得 Human 明确 `PASS`。任何字段为 `UNKNOWN`、没有证据路径或尚未裁决，均标 `BLOCKED`；Agent 不得自行补值。

**每个实验臂必须逐项记录并冻结：**

1. **数据完整性**：实际输入根路径、split / manifest 及其 sha256；与评测侧的交集检查；缺失、损坏、不可读、重复记录与 real/raster pair 失配数量。A1/A2 的结论须在这里引用，不得只写“已检查”。
2. **数据体量**：Target / External 分别记录 scene / log / token / 多相机同步帧 / image / pair 的实测数量（不适用的单位写 `N/A`）；同时记录每个 epoch 或整个 schedule 的有效采样/呈现次数与采样比例，避免把数据增加与训练曝光增加混为一谈。
3. **训练 epoch / schedule**：明确 `epochs`、`max_steps`、每 epoch step 数、global/effective batch size、gradient accumulation、warmup、early stopping、checkpoint 保存频率，以及 resume 后剩余 epoch/step 的计算方式。若训练按 step 驱动，也必须给出等效 epoch；不得只写“沿用默认值”。
4. **初始化与恢复方式**：F0/F1 必须明确记录 `from scratch`。仅同一 `run_id` 在 code/config/data hash 完全一致时允许续训，并记录 checkpoint 路径、sha256、已完成 epoch/step 与剩余 schedule；既有 checkpoint 不得加载。P2 distillation 未另行获批前标 `BLOCKED`。
5. **与原论文设定的逐项对照**：至少对照 optimizer、learning rate、scheduler、warmup、weight decay、batch size、输入分辨率、augmentation、loss 及其权重、训练更新预算和 checkpoint 策略。每项必须写 `一致 / 有意偏离 / 无法核验`、证据位置和偏离理由；`无法核验` 或未获 Human 接受的偏离均为 `BLOCKED`。本目录没有给出的原论文参数值不得由 Agent 推测或补写。
6. **其他有效性与可复现性配置**：随机 seed、subset 生成规则、renderer / target rig 配置与 hash、预处理与缓存版本、各 loss 开关、数据混合/采样比例、precision、分布式训练方式、硬件节点、软件环境、评测器版本与指标、输出目录、`git_sha`、dirty 状态、`run_id`、失败与恢复策略。
7. **执行契约**：列出 H0、Smoke Test、Full Experiment 与评测各自的准确工作目录、入口文件、完整解析后命令/参数、scheduler wrapper、输入 manifest、预期产物、成功退出条件与失败告警。H0 还必须列出 fixture/sample manifest、坐标系约定以及三项误差/检出率的报告位置；只有“约 50 步”而没有可执行命令不算完成。

**冻结产物与判定：**

- `A5_training_readiness.md` 必须包含“字段 → 实测/解析值 → 证据路径 → 与基准的差异 → PASS/BLOCKED”表；只列配置文件名而不列最终解析值视为未完成。
- 每个 run 在提交前生成 `config.frozen.yaml`，其内容必须与 A5 表及 registry 计划行一致；Human 检查后对两者记录 sha256。检查后任何配置、manifest、代码或初始化变化都使原 `PASS` 失效，必须新建 `run_id`、重做 A5 与 `CP-CODE` / `CP-2b`。
- **A5 未 `PASS`：禁止 Smoke Test；Smoke Test 未通过：禁止 Full Experiment。**

---

## 4. 数据集策略与数据流向规范

### 4.1 角色固定

| 角色 | 数据集 | 阶段 | 允许读取 | 禁止读取 |
|---|---|---|---|---|
| **Target（目标域）** | **NAVSIM**（固定不变） | 全程 | budget 内 real images、structured records、planning labels、**target rig 标定 `C_d`** | budget 外 target scene；`navtest` / `navhard_two_stage` / `warmup_two_stage` 等全部评测侧数据 |
| **External / Source** | **nuScenes**（初期唯一） | P1 F1 | maps、agents、ego/world state、future trajectory labels | nuScenes RGB、nuScenes 相机内外参 |
| **External / Source** | **nuScenes**（初期唯一） | P2 X 系列 | 同场景 external RGB、对应 source-rig raster、source calibration；仅用于 external alignment | 在 `SD-4/CP-3` 前读取或用于训练；任何 P2 cache/产物流入 P1 |
| Source 扩展 | **KITTI、Waymo** | 后续 | 同上，且须先通过 schema 合格性审计 | 同上 |

> 术语一律用 **External / Target**，不用 Old / New（佐藤 9/8 明确要求：数据新旧不决定其研究角色）。

### 4.2 数据流向（三维正交，不得压成一维）

Memo §五点名的问题：原表把 `Old Synthetic Raster / New Synthetic Raster` 混成一个维度。正确的三维是：

```
维度 1  场景来源 (scene source)  ∈ {NAVSIM, nuScenes, KITTI, Waymo}
维度 2  渲染 rig  (render rig)   ∈ {C_d = NAVSIM target rig, C_s = source 原生 rig}
维度 3  接受的 loss              ∈ {plan_target, r2r_target, align_external, plan_external}
```

**P1 协议下的合法组合**：

| 场景来源 | 渲染 rig | alignment | planning | 说明 |
|---|---|---|---|---|
| NAVSIM (budget 内) | `C_d` | ✅ target R2R | ✅ target plan | 基础 |
| nuScenes | **`C_d`** | ➖ 不自动成 pair | ✅ **target-rig** plan | ⭐ F1 的关键资产 |
| nuScenes | `C_s` | 🔴 P2 专用 | 🔴 `SD-5` | X 系列 |

**关键论断（写进 PPT）**：「planning 只面向 target rig」**不等于**「planning 只能用 target dataset 的场景」。external 场景经 `C_d` 渲染后，其 planning 监督是完全合法的 target-rig 监督。

### 4.3 KITTI / Waymo 扩展的准入条件（后续阶段）

任一新 source 进入前须逐字段通过：拥有 map 几何、agent cuboid、ego/world pose、时间同步、**可构造未来轨迹标签**。缺任一项则排除该 source，**不得用图像或 source 相机标定去补全**。

仓库已有 `run_waymo_dataset_caching.py` / `run_waymo_qa_generation.py` / `run_waymo_submission.py`，Waymo 接入的工程起点低于 KITTI。**建议扩展顺序：nuScenes → Waymo → KITTI。**

---

## 5. 实验臂定义

### 5.1 命名对齐（⚠️ 防止 9/11 现场翻车）

| Memo 用法 | 本计划 | 是否等价 | 说明 |
|---|---|---|---|
| E0 | **F0** | ✅ | target-only @ 10% legal NAVSIM training scene/log |
| E1 | **F1** | ✅ | + external metadata → **`C_d`** raster planning |
| E2 | **X1** | 🆕 | F0 + external real ↔ **`C_s`** raster alignment |
| E3 | **X2** | 🆕 | F1 + 同上（= 会议中的 external-alignment 方案；不同于 `SD-0` 选项 B） |
| — | **X3** | 🆕 | X2 + external-rig planning；仅为未来消融 |

> 新增臂统一冠以 **X**（e**X**ternal-pixel，视觉上提示"越过了 access boundary"）。
> `F2/F3/A0/A1` 的旧定义来自已失效 proposal，不属于当前冻结协议；若未来需要曝光控制、标签置换或 target alignment on/off 消融，必须重新定义并由 Human 审批。

### 5.2 全部臂

**P1（边界已冻结；仍须 A1–A5、H0 与 smoke 通过）**

| 臂 | Phase-1 辅助流 | target R2R | 数据读取边界 | 冲刺内 |
|---|---|:--:|---|---|
| **F0** | 无 | on | 仅 10% DEV target manifest；无 external 输入 | ✅ 各 1 次 |
| **F1** | nuScenes raster under `C_d` | on | 与 F0 相同 target manifest + external 结构化；**零 source RGB/标定** | ✅ 各 1 次 |

**P2（access boundary 已定义；实现仍受 `SD-4/CP-3` 门控）**

| 臂 | 定义 | 对应主张 | 冲刺内 |
|---|---|---|---|
| **X1** | F0 + external alignment | 池畑主张的隔离版 | ⬜ 仅出 spec |
| **X2** | F1 + external alignment | 会议中的 external-alignment 方案（不同于 `SD-0` 选项 B） | ⬜ 仅出 spec |
| **X3** | X2 + **external-rig planning loss** | 池畑 Joint Training ↔ 佐藤明确反对 | ⬜ 仅出 spec（`SD-5`） |

### 5.3 为什么 X 系列不进关键路径（工程判断，非科学判断）

X 系列需串行完成：① renderer 接受第二套 camera config ② loader allowlist 放开 ③ 重写 access audit ④ 建 external real/raster pair 缓存 ⑤ alignment 接线（`SD-4` 五个未定项）⑥ smoke test。

叠加 `EF-4`（main 缺 tools/paired split）、A1–A4 审计、F0/F1 训练、评测、PPT，强塞 X2 最可能的结果是**拿到一个没通过 access audit 的数字**——而这类数字上会只会造成伤害。

> **9/11 的 X 系列交付物 = 规格 + 协议修订请求 + 待老师裁决的问题，不是数字。**
> 这与 Memo「9/11 最需要带去的材料」完全一致：*一份无歧义的问题定义、一张可核对的数据/rig/loss 表、一组最小对照的结果或明确进度、一份有依据的 RL 可行性判断*。

### 5.4 冲刺假说表

| ID | 假说 | 变量 / 对照 | 终点 | 支持 | 反驳 |
|---|---|---|---|---|---|
| **SH-A** | A1–A5 资产与训练配置满足使用前提，且 `dataset_*` 与 NAVSIM 评测侧 token 交集为 0 | 无（审计） | 污染 token 数；缺失率；ckpt 判定完备性；A5 配置冻结完备性 | 交集=0、无 UNKNOWN ckpt，且 A5 全部 PASS | 任一交集 > 0 或 A5 任一 BLOCKED → **阻断全部训练** |
| **SH-0** | source→`C_d` 渲染管线几何正确 | 无（审计） | landmark 重投影 ≤1px；world/ego round-trip ≤1e-3 m / 1e-4 rad；注入 corruption 100% 检出 | 全通过 | 任一不通过 |
| **SH-1** | @ DEV subset，观察加入 external structured raster 后 F1 相对 F0 的方向与幅度 | F0；变量 = nuScenes raster 辅助流有无 | PDMS + 既有分项 | `N/A`：pilot 不设 confirmatory threshold，只报告观测 | `N/A`：pilot 不据此 Refuted，结论为 `Inconclusive` |
| **SH-3** | 训练管线数值稳定且可复现（工程假说） | 无 | 无 NaN/发散；ckpt 可恢复；同 seed 可复现 | 全满足 | 任一不满足 → **Technical Failure** |
| **SH-R** | 存在可复用的简易 driving simulator（取状态 / 按 `C_d` 渲 raster / 执行 action / 返回默认 reward） | 无（可行性） | 四项二值判定 + 缺口清单 | 四项全通 | 任一项需自研 > 1 周 |

> **SH-1 是 development observation，不是 confirmatory hypothesis test。** 单 DEV subset × 单 seed × 每臂 1 次没有预注册的数值效应阈值，因此无论方向或幅度如何，科学结论均为 `Inconclusive`。未来 confirmatory protocol 必须在查看其结果前另行冻结。

---

## 6. 冲刺矩阵（TSUBAME 作业感知）

### 6.1 资源模型

- 1 × `node_f` = **H100 × 4**。以下耗时估算均以「1 node_f 独占」为单位。
- **不写死墙钟**：先测吞吐，再按 `T_finish = (remaining_steps / measured_steps_per_hour) × 1.2` 推。
- 队列等待**不可控**：① 尽早提交 ② 用 `-ar` 预约（现有脚本已有 `#$ -ar 8016` / `8176` 先例）③ 长作业拆 checkpoint 可续训。

### 6.2 Day-1（9/9，自当前时刻起）

| 时段 | 任务 | 类型 | 假说 | 位置 | 卡点 |
|---|---|---|---|---|---|
| 15:30–16:00 | 复核已冻结 `SD-0/1/2/3/6/7/8`；确认 `EF-1`（`which qsub sbatch`）；填写并复核 `env/sprint_20260909.env` | 决策 / Environment Check | — | TSUBAME + Human | **CP-1** |
| 16:00–17:00 | **`EF-4` 收敛**：逐路径把 `tools/`、`paired_*.yaml`、`build_alignment_small_data.py` 并入 main-based 分支 `exp/sprint-0909-base` | 工程 | — | Mac→push | **CP-2a** |
| 16:00–18:00 | **A1–A5 资产与训练准备审计**（A1–A4 只读；A5 冻结拟运行配置，可与上一行并行） | 审计 | SH-A | TSUBAME login + Human | **CP-2b** |
| 18:00–19:00 | **H0 几何审计** + **Smoke Test**（F0/F1 各 ~50 步，测 steps/hour） | 审计+冒烟 | SH-0, SH-3 | TSUBAME 1×node_f 短作业 | **CP-2c** |
| 19:00 | **调度决策门**（§6.4） | 决策 | — | 人 | — |
| 19:00–20:30 | 提交 F0 / F1 长作业；CPU 侧预生成评测缓存；写 PPT p.1–p.4 | 训练 | SH-1 | TSUBAME + 人 | — |
| 20:30–21:30 | `SD-4` 规格化：X 系列 Implementation Spec（**不写代码**） | 规格 | — | 人 | **CP-3** |
| 21:30–23:00 | Simulator 候选调查（硬时间盒 90 min） | 可行性 | SH-R | 人 | — |
| 23:00–23:30 | 睡前监控清单：loss 正常？ckpt 可存可恢复？评测已串在训练之后？磁盘/显存/NaN 告警已设？ | 运维 | SH-3 | 人 | — |
| 23:30→ | 过夜：**只跑已通过 smoke test 的配置** | 训练 | SH-1 | TSUBAME | — |

**过夜禁止**：当天新写的复杂分支、未 smoke 的任何臂、需人工介入的流程、X 系列任何内容。

> ⏱ 若 `CP-2b` 或 `CP-2c` 未能在 19:00 前通过，**放弃当晚长作业**，把窗口全部转给 PPT 与 X 系列 spec。带着"审计发现的阻断项 + 清晰的下一步"上会，比带一个来路不明的数字更有价值。

### 6.3 Day-2（9/10）

| 时段 | 任务 | 假说 | 卡点 |
|---|---|---|---|
| 08:00–09:30 | 收 F0/F1；统一评测；核对公平性（同 step / 同 target manifest+sha / 同 `manifest_seed` / 同 `training_seed` / 同评测器版本 / 同 `git_sha`） | SH-1 | — |
| 09:30–11:00 | 出三张核心材料：① F0/F1 指标表 ② 学习曲线 ③ 3–5 个定性案例（刹车 / 横向偏移 / 交叉路口） | — | — |
| 11:00 | 检查 PDMS 评测完成度、公平性与技术有效性；决定是否仍有时间完成已冻结的同 run 恢复/评测 | — | **CP-4** |
| 11:00–13:00 | 只允许完成 F0/F1 的同 run 恢复、统一评测、日志/registry/定性案例；不得新增 F2、换 seed 或重采样 | SH-1 / SH-3 | — |
| 13:00–18:00 | **最后允许提交关键作业的时段**。优先级：修会使 run 无效的明确 bug > F0/F1 同 run 恢复 > PDMS 评测 > 定性案例 | — | — |
| **18:00** | **架构冻结线**：不再新增模块 / 不改数据定义 / 不重建大规模缓存 / 不提交预计 9/11 08:00 前跑不完的作业 | — | **CP-5** |
| 18:00–23:00 | PPT 定稿（7–9 页，见 §11） | — | — |
| 23:00→次日 08:00 | 仅允许：已 smoke 过的同配置复跑 / 从同一 `run_id` 且 hash 一致的 checkpoint 续训 / 自动评测 / 归档 | — | — |

### 6.4 调度决策表（`CP-2c` 后使用）

| smoke 实测的单臂完整训练耗时（1×node_f） | 决策 |
|---|---|
| ≤ 6 h | 跑完整 F0 + F1；不新增实验臂 |
| 6–10 h | 只跑完整 F0 + F1 |
| 10–16 h | 全部改**统一缩减 schedule**（同 step 数、同初始化、同采样规则），结论一律标 preliminary |
| > 16 h | 不启动完整训练；经 A5 重新冻结后可选择统一的 from-scratch proxy schedule，否则只报告阻断与 ETA |
| 未完成 smoke test | **禁止提交过夜作业**（无例外） |
| 队列等待 > 4 h | 立即改用 `-ar` 预约或降级节点类型；同时把 PPT 前移 |

### 6.5 三线并行

| 流水线 | 常驻任务 |
|---|---|
| **TSUBAME lane** | 训练、评测、checkpoint |
| **本地 CPU lane** | 审计脚本、结果导出、画图、registry 维护 |
| **Human lane** | 决策、错误分析、PPT、Q&A 彩排 |

**调试硬时间盒**：数据/shape/配置问题 30 min；明确 bug 最多 90 min；原因不明的训练不稳定 → 停该分支保留日志；新数据格式接入 2 h 内做不出 smoke test 就砍。

### 6.6 会前明确砍掉

密集 scaling 曲线、多 seed / 多 subset、多 backbone / 多分辨率 / 多 PE 设计、alignment 超参搜索、X1/X2/X3 训练、F2/F3/A0/A1、KITTI/Waymo 接入、BEV / 3D feed-forward / 3DGS baselines、从零训 RL teacher、闭环 RL、为出图临时重构训练框架、完整 related work。

> 这些**不是不重要**（池畑 9/2 明确说「比较不足可以一发 Reject」），而是本窗口内做不出可信版本。它们进「下一步计划」页，不进「结果」页。

---

## 7. 路径规范

### 7.1 云端根路径（已由 `env/stage_a.env` 与现有 qsub 脚本证实）

| 变量 | 值 | 来源 |
|---|---|---|
| `$RAP_ROOT` | `/gs/bs/tga-RLA/qdeng/RAP` | `env/stage_a.env` |
| TSUBAME group | `tga-RLA` | `qsub -g tga-RLA` |
| 其他 | `$OPENSCENE_DATA_ROOT` `$NAVSIM_EXP_ROOT` `$NAVSIM_DEVKIT_ROOT` `$RASTER_SRC_ROOT` `$STAGE_A_OUT` `$NUPLAN_MAPS_ROOT` `$NAVHARD_TWO_STAGE_ROOT` `$HF_HOME` `$REF_DINO_CKPT` `$DINO_PRETRAINED_CKPT` | `env/stage_a.env`（**含敏感值，勿入库**） |

冲刺新增变量写进 `env/sprint_20260909.env`，**继承**而非复制：

```bash
source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env
export SPRINT_OUT=$RAP_ROOT/outputs/sprint_20260909
export SPRINT_AUDIT=$RAP_ROOT/docs/audits/2026-09-09_asset_audit
```

### 7.2 数据与产物目录树

```
/gs/bs/tga-RLA/qdeng/
├── data/
│   └── nuscenes/                          # ← A2 审计对象（source）
│       ├── maps/  samples/  sweeps/  v1.0-*/
│
└── RAP/                                   # = $RAP_ROOT（git 仓库根）
    ├── dataset_norm/                      # ← A1（gitignored）
    ├── dataset_aug/                       # ← A1
    ├── dataset_perturbed/                 # ← A1
    ├── ckpts/                             # ← A3（gitignored: *.ckpt）
    ├── env/                               # 环境变量，勿入库
    ├── docs/
    │   ├── 20260909/                      # 本计划 + context
    │   └── audits/
    │       └── 2026-09-09_asset_audit/    # ← 审计报告统一归档处
    └── outputs/                           # gitignored
        └── sprint_20260909/
            ├── raster/
            │   ├── rig=navsim_Cd__cfg=<Cd_hash8>/     # ★ 第一层 = 渲染 rig
            │   │   ├── scene_src=navsim/
            │   │   └── scene_src=nuscenes/            # ★ F1 的关键资产
            │   └── rig=nuscenes_Cs__cfg=<Cs_hash8>/   # 🔴 P2 X 系列专用，SD-4/CP-3 未过不得创建
            ├── manifests/
            │   ├── dev_b010_seed20260909.json + .sha256 # ← 本冲刺唯一使用；生成一次后冻结
            │   └── future_confirmatory/               # ← 协议尚未定义，不得创建 manifest
            ├── pairs/
            │   ├── target_real_raster__b010_seed20260909.jsonl
            │   └── external_real_raster__nuscenes.jsonl   # 🔴 X 系列专用
            └── pilot/
                └── <run_id>/
                    ├── config.frozen.yaml
                    ├── hashes.json        # code/config/manifest/renderer/init ckpt
                    ├── job.script         # ★ 实际提交的作业脚本原文副本
                    ├── logs/  ckpt/  eval/  preds/
                    └── STATUS             # queued|running|evaluated|failed|aborted
```

**命名铁律**：`raster/` 第一层是**渲染 rig**，第二层是**场景来源**。用目录结构强制修掉 Memo §五点名的 `Old/New Synthetic Raster` 二维混淆——表就再也画不错。

**`run_id`**：`{YYYYMMDD}_{arm}_{budget}_{subset}_{seed}_{git8}`，例 `20260909_F1_b010_DEV_s20260909_a1b2c3d4`

### 7.3 实验账本（单一真相源）

`docs/20260909/registry.csv`（报告只从这里取数，**禁止手工从日志复制**）：

`run_id, arm, protocol(P1|P2), branch, git_sha, job_id, node_type, target_manifest+sha, source_manifest+sha, render_rig, Cd_hash, losses, init_mode(from_scratch|same_run_resume), resume_ckpt+sha, steps, wall_clock, n_target_scenes, n_src_presentations, manifest_seed, training_seed, subset, status, failure_type(none|technical|pilot_observation), metric_pdms, metric_pdms_components, openloop_diagnostic, is_confirmatory(false)`

> **失败、不稳定、负迁移的 run 一律留在 registry，不得静默过滤。**

---

## 8. Git 策略：Feature Branch vs Worktree

### 8.0 小步迭代与原子提交（所有代码修改的强制规则） 🆕

**Agent 必须按“一个可独立检查的单一目的 = 一次最小修改 = 一个对应 commit”推进。** “代码修改”包括新增、删除或修改源码、训练/评测配置、数据处理与审计脚本、作业脚本；不得把多个无关目的打包到同一 commit，也不得积累多次修改后一次性提交。

每次循环的固定顺序如下，禁止跳步：

1. 只完成一个最小、可独立验证的修改；执行与该修改直接相关的静态检查或最小验证，但不得启动 Smoke Test 或训练。
2. 按 §9 的 `CP-CODE` 列出全部待检查文件及具体代码段，然后**暂停并等待 Human 检查**。
3. Human 明确 `PASS` 后，立即提交与该修改一一对应的 commit；commit message 必须准确说明该次修改的目的，commit body 记录关键文件、验证结果与对应 `CP-CODE`。Human 要求修改时，本轮仍保持暂停状态；修正后重新提交完整检查卡并再次等待 `PASS`。
4. 记录 commit SHA 后才可开始下一次修改。任何实验只允许使用已经 Human `PASS`、已经 commit 且 worktree clean 的代码。

**禁止**：`WIP` / `misc changes` 等无法对应修改目的的 message、事后补写 commit message、把 Human 未检查的追加改动塞入已通过的 commit、用一次“大提交”替代小步迭代。文档-only 修改可独立提交，但不得与实验代码混在同一 commit。

### 8.1 决策结论

**两者不是二选一，而是正交的两件事——必须同时用，各司其职。**

| 概念 | 回答的问题 | 本项目的用法 |
|---|---|---|
| **Branch** | *跑的是哪份代码？* | **每个会改动代码的实验族一个分支**。F0/F1 必须共享完全相同的代码（公平性控制），**共用一个分支、一个 commit** |
| **Worktree** | *这份代码物化在磁盘的哪里？* | **每个"同时在跑且代码不同"的实验族一个 worktree**，且只在 **TSUBAME** 上开 |

**关键判断：worktree 的价值在云端，不在 MacBook。**

- MacBook 只编辑不训练 → 没有"两份代码同时被占用"的问题 → **本地普通分支切换即可，不需要 worktree**。
- TSUBAME 上一个作业会 `cd $RAP_ROOT && python ...` 跑几小时。**如果这期间你在同一目录 `git checkout` 或改文件，正在跑的作业会读到被换掉的代码**——而且不报错，只静默污染实验。这才是 worktree 要解决的真问题。

**最小正确方案**：

```
TSUBAME:
/gs/bs/tga-RLA/qdeng/RAP            [main]                  ← 只读参考，不在这里提交作业
/gs/bs/tga-RLA/qdeng/RAP-p1         [exp/sprint-0909-base]  ← F0/F1 全部从这里提交
/gs/bs/tga-RLA/qdeng/RAP-p2         [exp/sprint-0909-x]     ← X 系列开发（SD-4/CP-3 通过后才动 allowlist）
```

只需 **2 个 worktree**。F0/F1 共用 `RAP-p1` 是**正确的**（它们必须同 commit）；拆开反而增加"代码不一致"的风险。

### 8.2 操作步骤

**Step 0｜先把 `main` 补齐（`EF-4`，今天必做）**

```bash
cd /Users/joy/pythonProject/RAP
git switch main
git pull --ff-only                       # 确保与 origin/main 一致

git switch -c exp/sprint-0909-base
```

随后严格按 §8.0 的循环门，**一次只带入一个可独立检查的逻辑单元**：

1. `tools/render_nuscenes_camera_cross.py`；
2. `tools/render_navsim_scene.py`；
3. 仅在已确认需要时，分别处理 `tools/render_navsim_augmentations.py` 与 `tools/canonical_bev/`，不得用整个 `tools/` 目录一次性覆盖；
4. `paired_trainval.yaml` 与 `paired_54log.yaml`；
5. `build_alignment_small_data.py`；
6. `docs/20260909/` 作为独立的 documentation-only 修改。

每个逻辑单元都必须执行：逐文件 `git checkout <source-branch> -- <exact-path>` → `git status` / `git diff` → `CP-CODE` 等待 Human `PASS` → 对应原子 commit → 记录 SHA；全部通过 `CP-2a` 后才 `git push -u origin exp/sprint-0909-base`。不得把上述六项压成一个 commit。

> ⚠️ **不要 `git merge exp/rap-alignment-regression`**——那是 89 files / +114k 行，会把大量与本冲刺无关的实验性改动一并带入，破坏"`main` 干净"的前提。**逐路径 checkout 是唯一安全做法。**

**Step 0.5｜把指定 `scripts/` 子目录纳入版本控制（`SD-8` 已通过；执行仍受 `CP-CODE` 门控）**

```bash
# .gitignore 改为：
#   scripts/*
#   !scripts/jobs/
#   !scripts/jobs/**
#   !scripts/audit/
#   !scripts/audit/**
mkdir -p scripts/jobs scripts/audit
git add -f scripts/jobs scripts/audit .gitignore
git commit -m "chore: version-control job and audit scripts (was fully gitignored)"
```

没有这一步，worktree 里**不会有任何作业脚本**——这是 `EF-2` 的直接后果。

**Step 1｜本地确认干净**

```bash
git status --porcelain          # 必须为空（或只剩 .DS_Store 之类）
```

> worktree **不会携带未提交改动**。当前工作树若有未提交内容，`git worktree add` 出来的是干净的旧代码——这是最常见、最难发现的坑。

**Step 2｜在 TSUBAME 上建 worktree**

```bash
ssh tsubame
cd /gs/bs/tga-RLA/qdeng/RAP
git fetch origin
git switch main && git pull --ff-only

git worktree add ../RAP-p1 exp/sprint-0909-base          # 跟踪已存在的远程分支
git worktree add ../RAP-p2 -b exp/sprint-0909-x main     # 新建 X 系列分支
git worktree list
```

**Step 3｜每个 worktree 一次性配置**

```bash
for W in ../RAP-p1 ../RAP-p2; do
  cd "$W"
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/dataset_norm      dataset_norm
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/dataset_aug       dataset_aug
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed dataset_perturbed
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/ckpts             ckpts
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/env               env
  ln -sfn /gs/bs/tga-RLA/qdeng/RAP/outputs           outputs
  printf 'dataset_*\nckpts\nenv\noutputs\n' >> .git/info/exclude
done
```

> **数据与产物永远在仓库外，worktree 内只放符号链接。** 否则每个 worktree 都会试图重建缓存，磁盘和时间都撑不住。

**Step 4｜作业脚本必须锁定 worktree 并自证代码状态**

```bash
set -euo pipefail
WORKTREE=/gs/bs/tga-RLA/qdeng/RAP-p1     # ← 显式绝对路径，不用 $PWD、不用 -cwd
cd "$WORKTREE"
source env/stage_a.env
export RAP_ROOT="$WORKTREE"              # ★ 覆盖 stage_a.env 里的 RAP_ROOT，指向本 worktree

# --- provenance 自证：不干净就拒绝跑 ---
GIT_SHA=$(git rev-parse --short=8 HEAD)
DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
  echo "REFUSING: worktree is dirty ($DIRTY files). Commit or stash first." >&2
  git status --porcelain >&2
  exit 1
fi
RUN_DIR="$SPRINT_OUT/pilot/${RUN_ID}"
mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/job.script"            # ★ 归档实际提交的脚本原文
echo "{\"git_sha\":\"$GIT_SHA\",\"worktree\":\"$WORKTREE\",\"job_id\":\"${JOB_ID:-NA}\"}" > "$RUN_DIR/hashes.json"
```

`export RAP_ROOT="$WORKTREE"` 是**必需的**：`env/stage_a.env` 里的 `RAP_ROOT` 指向主仓库，不覆盖就会出现「worktree 的代码 + 主仓库的配置」这种最难查的错配。

**Step 5｜清理**

```bash
cd /gs/bs/tga-RLA/qdeng/RAP
git worktree remove ../RAP-p2         # 目录干净时
git worktree remove --force ../RAP-p2 # 有未提交改动（会丢改动，慎用）
git worktree prune                    # 手工 rm 掉目录后清理残留记录
```

### 8.3 六个必踩的坑

| # | 坑 | 现象 | 正确做法 |
|---|---|---|---|
| 1 | 同一分支不能被两个 worktree 同时检出 | `fatal: 'main' is already checked out at ...` | 每个 worktree 用自己的分支；或 detached：`git worktree add ../tmp <commit>` |
| 2 | **worktree 不携带未提交改动** | 新目录看不到刚改的代码，作业跑的是旧逻辑 | 先 commit 再 `worktree add`；作业脚本用 Step 4 的 dirty 检查兜底 |
| 3 | Python 环境不跟过去 | `import navsim` 失败 | 每个 worktree 单独 `pip install -e .`；或共用 conda env 但确认 `PYTHONPATH` 指向当前 worktree |
| 4 | 数据/输出被复制或写乱 | 磁盘爆掉；两个 run 互相覆盖 | 数据与 outputs 放仓库外，worktree 内只放符号链接（Step 3） |
| 5 | 作业脚本路径写死或用 `-cwd` | 提交的作业跑了另一个 worktree 的代码 | 首行显式 `cd <绝对路径>` + 覆盖 `RAP_ROOT` + 打印 `git rev-parse HEAD` |
| 6 | 直接 `rm -rf` worktree 目录 | `git worktree list` 留幽灵条目 | 用 `git worktree remove`；已 rm 就 `git worktree prune` |

### 8.4 何时应该**不用** worktree

- 只改文档 / 只改 PPT → 直接在一个分支上做，别开 worktree
- 只跑一个实验族 → 一个工作目录够了，worktree 只增加认知负担
- 磁盘紧张 → worktree 共享 `.git`（省了历史），但**每个都会完整展开一份工作树文件**；本仓库工作树不大，可接受

---

## 9. 人工检查卡点

### 9.0 `CP-CODE`｜每次关键代码修改后的循环门 🆕

下列内容均视为**关键代码**：数据读取/split/manifest、renderer 与 camera calibration、real/raster pairing、model/encoder/projector/planner、loss、optimizer/scheduler/训练配置、checkpoint 初始化或恢复、评测指标、会影响门控结论的审计逻辑，以及实际提交的 job script。

每次修改或新增上述代码后，Agent 必须停止后续编辑与实验操作，向 Human 提交以下检查卡：

| 必填项 | 要求 |
|---|---|
| 修改目的 | 仅一个、可独立验收的目的；说明对应 `EF-*` / `SD-*` / SH / 实验臂 |
| 关键文件 | 列出全部相对路径；不得只写目录或“见 diff” |
| 具体代码段 | 对每个文件列出函数/类/配置 key、当前行号或 diff hunk，以及该段的起止语义 |
| 行为变化与风险 | 写清数据流、loss、初始化、训练或评测行为如何变化，以及不应变化的部分 |
| 验证证据 | 列出已执行的静态检查/最小验证及结果；不得以 Smoke Test 代替 Human 检查 |
| 待 Human 判断 | 明确给出 `PASS / 需要修改 / BLOCKED`；不得把沉默视为通过 |

**Human 明确 `PASS` 前，Agent 不得 commit 该修改、不得开始下一次关键代码修改、不得启动 Smoke Test 或 Full Experiment。** `PASS` 后按 §8.0 立即完成对应原子 commit；若 commit 前 diff 发生任何变化，`PASS` 自动失效，必须重新提交检查卡。

| ID | 时间 | 触发 | 人类必须检查什么 | 未通过 |
|---|---|---|---|---|
| **CP-1** | 15:30–16:00 | 任何代码/作业之前 | 复核已冻结 `SD-0/1/2/3/6/7/8`；确认 `EF-1`（`which qsub sbatch`）；填并复核 `env/sprint_20260909.env` | 任一未完成 → **全部停机** |
| **CP-2a** | 17:00 | `main` 补齐后 | 逐条 review `git status`，确认只带入必需资产，未混入 alignment regression 的实验性改动 | 回退，重做逐路径 checkout |
| **CP-2b** | 18:00 | A1–A5 审计完成、拟运行配置已冻结 | ⭐ **逐条读 `A1_split_membership.md`**：`dataset_*` 与 `navtest`/`navhard_two_stage` 交集是否为 0；A1/A2 的完整性与实测体量；`A3` 每个 ckpt 的 `ALLOWED_AS_INIT`；`A4` 的 source-pixel 判定；`A5_training_readiness.md` 中数据、epoch/schedule、from-scratch/ckpt、原论文超参数对照及复现配置是否全部有证据且 PASS | **交集 > 0、任一 UNKNOWN/BLOCKED、配置或 manifest 未冻结 → 禁止 Smoke Test 与任何训练**（最硬的门） |
| **CP-2c** | 19:00 | H0 几何审计 + smoke 完成 | landmark 重投影误差、round-trip 误差、corruption 检出率；肉眼看若干 raster 叠图；smoke 的 loss 接线与实测吞吐 | **H0 未过 → 禁止启动训练** |
| **CP-3** | 20:30–21:30 | 写 X 系列代码之前 | 裁决 `SD-4` 五项；确认 P2 与 P1 的隔离方式 | X 系列只出文档，不进代码库 |
| **CP-4** | 9/10 11:00 | 第一批结果出来 | ① **先只看 Measurement，不做解释** ② 核对公平性（同 step / 同 target manifest+sha / 同 `manifest_seed` / 同 `training_seed` / 同评测器版本 / 同 `git_sha`）③ 核对 PDMS 与分项是否完整 ④ 决定最后窗口 | 公平性不过 → 标 `invalid`，不上 PPT |
| **CP-5** | 9/10 18:00 | 架构冻结线 | 确认无新模块、无数据定义变更、无跑不完的作业在排队 | 强制冻结 |
| **CP-6** | 9/11 09:30 | 结果硬冻结 | 逐条检查结论措辞与证据强度是否匹配；确认所有数字标 `preliminary / single subset / single seed` | 09:30 后不因新数字重写主叙事，迟到结果进 appendix |

### Technical Failure vs Scientific Failure

| 现象 | 分类 | 处置 |
|---|---|---|
| OOM、shape mismatch、dataloader 崩、NaN、ckpt 损坏、作业被队列杀 | **Technical** | 修复后重跑；**不产生任何科学结论** |
| 训练正常收敛，但 `M(F1) < M(F0)` | **Pilot observation** | 记录方向与差值，进 registry 与 PPT；结论仍为 `Inconclusive`，不得私改 protocol |
| 训练正常，F1 与 F0 存在任意幅度差异 | **Pilot observation** | 原样报告 PDMS 与分项；没有 confirmatory threshold，不得写成 Supported/Refuted |
| A1 污染检出 / H0 未过 | **既非** | 阻断门，整个 build 作废 |

> **铁律**：实验开始后不得因初期结果不理想而私改 Protocol。任何 protocol 变更必须开新 `run_id`、在 registry 记录原因，并声明该 run 与前序 run **不可配对比较**。

---

## 10. 测量与科学解释规程

### 10.1 报告顺序（不可颠倒）

**先 Measurement，后 Scientific Interpretation。** PPT 与口头汇报都是：`这是数字 → 这是公平性核对 → 我能下的结论 → 我不能下的结论`。

### 10.2 结论类型

科学结论词汇仅三种：**`Supported` / `Refuted` / `Inconclusive`**。本次全部为 development pilot（单 DEV subset × 单 seed × 每臂 1 次），且没有数值型 confirmatory threshold，因此本次科学结论**只能是 `Inconclusive`**。

A1–A5、H0 或技术检查未通过时，写 `BLOCKED` / `INVALID` / `Technical Failure`，不得把工程门控失败写成科学上的 `Refuted`。

### 10.3 结果解释对照表

| 观测 | **允许**的表述 | **禁止**的表述 |
|---|---|---|
| F1 > F0 | 在该 DEV manifest 与 seed 下观测到正向差值；结论为 `Inconclusive` | "证明了 external knowledge 有效"、`Supported` |
| F1 ≈ F0 | 在该 DEV manifest 与 seed 下差值很小；结论为 `Inconclusive` | "没有效应"或"等价" |
| F1 < F0 | 在该 DEV manifest 与 seed 下观测到负向差值；结论为 `Inconclusive` | `Refuted`、静默丢弃该 run |
| 只有 open-loop 指标 | 仅 diagnostic，不能替代 PDMS；结论为 `Inconclusive` | 任何 Supported/Refuted 或 PDMS 主张 |

### 10.4 措辞审计（`CP-6` 逐条过）

- 每个数字必须带：10% scene/log budget、DEV manifest+sha、`manifest_seed=20260909`、`training_seed=20260909`、step 数、`git_sha`、是否 pilot
- 每个主张必须带范围限定：**on the evaluated NAVSIM target rig**
- 出现"证明 / 显著 / 最强 / 有效"而无置信区间支撑 → 一律改写
- **「all-data 最强」在 9/11 之前不得作为结论出现**（佐藤 9/8 明确要求；Memo 已确认状态为「值得开展比较实验；不预设 B 优于 A」）

---

## 11. 9/11 汇报结构（7–9 页）

顺序按 Memo §八：**目标与已知条件 → 两类知识及数据路径 → 最小对照结果 → 能/不能支持的判断 → RL 可行性初查 → 需要老师共同判断的问题**

| # | 页 | 内容 |
|---|---|---|
| 1 | 我希望老师判断什么 | 直接列 `SD-4` / `SD-5` / 3D baseline 排期 / RL 投入比例；已冻结 P1 access boundary 作为前提 |
| 2 | 问题定义与协议 | target 是谁、何时知道 calibration、zero/few-shot 数什么、10% scene/log budget + A1/A5 绝对计数 |
| 3 | 三维数据流表 | **场景来源 × 渲染 rig × 接受的 loss**（修正 Old/New Synthetic 混淆） |
| 4 | 资产、训练准备审计与 H0 通过证明 | A1 污染检查、A2 完整度、A3 ckpt 合规、A4 边界现状、A5 冻结配置；几何误差与 corruption 检出 |
| 5 | 最小对照 pilot 结果 | F0/F1；全部标 preliminary / single DEV subset / seeds=20260909；含公平性核对表 |
| 6 | 我能下 / 不能下的结论 | 科学结论统一 `Inconclusive`；并列原始差值、方向、限制与下一步 confirmatory 缺口 |
| 7 | **P2 X 系列设计 + P1 隔离** | 展示已冻结 access boundary、P2 不得流入 P1，以及 `SD-4` 待裁决项 |
| 8 | **external-rig planning（`SD-5`）** | 并列呈现池畑 Joint Training 与佐藤反对意见，请两位当面裁决 |
| 9 | RL 可行性 + 下一步 | SH-R 四项判定 + 缺口；3D/BEV baseline 与 Waymo/KITTI 扩展排期 |

> 结果未完成的格子**不留空白**，写清：当前状态 / 已运行时长 / ETA / 能回答什么 / 暂时不能回答什么。

---

## 12. 与 9/2 池畑要求的对齐状态

| 池畑 9/2 的要求 | 本冲刺 | 会后排期 |
|---|---|---|
| 明确 Problem Setting，拆细 rig 差异（count / extrinsics / FOV / focal / projection / distortion） | ✅ PPT p.2–3 | 受控 rig 变体实验 |
| 先做 baseline，明确"要打倒的敌人" | ✅ F0 | 全量 target 参考（REF） |
| 必须比较 BEV / 3D feed-forward / 3DGS / VGGT+Fisheye3R 式 adaptation，training data 完全相同 | ❌ 本窗口内不可能 | **会后第一优先**；先做 3D feed-forward 的 zero-shot 可跑性检查（成本最低、收益最高） |
| 不要"先有 RAP 再找理由" | 🔶 本冲刺只做 F0/F1 development observation，不声称已隔离全部因果 | 未来重新定义曝光控制、标签置换等对照并由 Human 预注册 |
| random sampling × 多 seed，不要精挑 few-shot 样本 | 🔶 当前为随机 10% DEV subset，`manifest_seed=20260909`，仅单 seed pilot | 未来 confirmatory protocol 预先定义多个 subset/seed 与重复次数 |
| 最终不能只有 NAVSIM 一个 target | ❌ | 第二 target rig |
| 多个 external dataset | ❌ | **nuScenes → Waymo → KITTI**（仓库已有 `run_waymo_*`，Waymo 工程起点低于 KITTI） |

---

*本文档以 `docs/20260909/` 现有文件与本仓库代码为唯一基准生成。Human 已于 2026-09-09 冻结 `SD-0/1/2/3/6/7/8`；`SD-4/5` 与全部 `[EF-*]` 仍须按各自门控处理。云端路径下的一切断言均标记为待审计，未经 §3 审计通过不得引用。*
