# 2026-09-09 → 09-11｜48h 实验冲刺计划（Sprint Plan）· **v2.1**

> v2 以 **`docs/20260909/` 下的现有文件 + RAP 开源代码库（本仓库）** 为唯一基准重写，补齐 TSUBAME4.0 执行架构、真实路径规范、数据集策略与 Git 并行方案。
> 所有涉及云端的断言均已降级为 **待审计项**，并附可执行的审计命令。
> v2.1 增补逐次 `CP-CODE` Human 检查、原子 commit / 小步迭代规则，以及进入 Smoke Test 前的 `A5` 训练配置与数据冻结门。

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
| Hypothesis | 🔶 受 `SD-0` 影响 | 见 `SD-0` |
| Research Spec | 🔶 受 `SD-0` 影响 | 见 `SD-0` |
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

**处置（`SD-8` 裁决后执行）**：

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

### `SD-0｜冻结规格不在基准内` 🆕 v2 新增

v1 的 `SD-1/2/3`、`F0/F1/F2/F3/A0/A1` 命名、`b*=10%`、`δ_main=1.0`、`Final EPDMS` 判据，**全部引自 `DriveWeave_proposal.md`**，而该文件不在本仓库、也不在 `docs/20260909/`。

**待裁决**：
- **选项 A（推荐）**：把 proposal 复制进 `docs/20260909/context/DriveWeave_proposal.md`，恢复其"冻结规格"地位，本计划的引用全部有效。
- **选项 B**：宣布 proposal 不再具约束力，则下列约束**全部失效并需重新定义**：预算口径、主指标、效应阈值、run 数设计、access boundary。

**在裁决前**，本文件把关键约束**原文内联复述**（不再靠章节号引用），使文档自包含。

### `SD-1｜方案 B 越过 source-pixel 边界` 🔴 最高优先

原 proposal 明文规定（内联复述）：nuScenes 作为 source 时，`所有 source RGB 与 source 相机标定` 为禁止输入；access audit 要求 `source RGB 读取数 / 解码张量数 / intrinsics 读取数 / extrinsics 读取数 全部等于 0`；renderer `只接受 C_d 作为唯一相机输入，source 相机文件在 loader allowlist 之外`。

而池畑 9/2 的 core novelty（活用 external real image）与佐藤 9/8 有条件认可的 external alignment，**必须读 nuScenes RGB + nuScenes 标定**。

**⚠️ v2 新证据**：`tools/render_nuscenes_camera_cross.py`（484 行）已存在于 `exp/rap-alignment-regression`。**若该工具读取 nuScenes 相机参数用于跨相机渲染，则 access boundary 可能已被现有代码越过。** 这是 `A4` 审计项，必须在裁决 `SD-1` 前查清。

**影响面**：不是加一个 loss 开关。需要 ① renderer 接受第二套 camera config ② loader allowlist 放开 ③ 重写 access audit 判定 ④ 修改 non-claim 边界与 novelty boundary（原文把「no source pixels/calibration」列为对抗「reviewer 认为这就是 RAP + 另一个数据集」的核心防线）。

**建议**：把 X 系列注册为并存协议 **P2**，不在 P1 内部放开 allowlist——那会同时毁掉两个主张。
**裁决时机**：`CP-1`。未裁决 → X 系列全部不启动，冲刺退化为 F 系列 pilot（这本身仍是合格交付）。

### `SD-2｜9/9–9/11 的 run 如何归类`

confirmatory 设计要求 3 subsets × 2 seeds = 6 paired runs，且「若 F0-only power audit 显示 power < 80%，需在**查看 F1 结果之前**增加 subset」。本窗口只跑得起 1 subset × 1 seed；直接用 confirmatory subset 并查看结果等于提前开箱，损伤预注册效力。

**建议**：新建 `seed=DEV` 的 development manifest，与 3 个 confirmatory subsets 互斥，所有 run 落 `$RAP_ROOT/outputs/sprint_20260909/pilot/`，标注 `non-confirmatory, excluded from confirmatory analysis`。PPT 上所有数字标 **preliminary / single subset / single seed**。

### `SD-3｜target budget 对外口径`

proposal 冻结 `b* = 10% of legal target training scenes`（单位 **scene/log**，不是 frame、不是分钟）、`b_min = 100 scenes`；而 9/8 你对佐藤讲的是「1000h external vs 10min target ≈ 6000:1」，9/2 与池畑也在「10 分钟 / NAVSIM 10%」之间来回。Memo 已注明后者是假设性例子。

**影响面**：这是 external-alignment 论证的**唯一定量支点**（"target pair 太少所以 alignment 学不出来"）。口径不统一，9/11 现场会被直接拆掉。

**建议**：统一到 **b\* = 10% navtrain scenes + b_min = 100 scenes**，PPT 上同时给出绝对 scene / frame / 等效时长三个值；「1:6000」只作 motivation 插图。

### `SD-4｜X 系列 alignment 规格未定`

未定项（**不得由 AI 猜测补全**）：
1. external raster 用哪套 rig 渲染（nuScenes 原生 rig？还是简化 canonical rig？）
2. alignment 在哪一层做（`B` 输出？`P_R/P_d` 输出 `F^R/F^I`？planning representation `G` 之后？）
3. 跨 rig 共享哪些模块、按 rig 分叉哪些（池畑提的 rig-dependent embedding 要不要引入）
4. 距离度量与权重（判别器 / 余弦 / MSE；λ 初值）
5. 与既有 target R2R loss（A0/A1 轴）如何共存

**裁决时机**：`CP-3`，且必须在写 X 系列代码之前。

### `SD-5｜external-rig planning loss 做不做`

池畑 9/2 §44 的 Joint Training（两个 planner + 共享 encoder + rig-dependent embedding）**前提就是在 external rig 上算 planning loss**；佐藤 9/8 10:08–10:11 明确判断「アライメントはいいと思うんですけど、プラナンは多分だめだと思う」「性能には寄与しない」。Memo §五记录的准确状态是「保留为可检验的消融，不能写成已被证明无效」。

**两位老师互不知道对方的表态。这是 9/11 会上第一议题。**

**建议**：本冲刺**不跑** X3，只带实验设计上会，当面裁决后再实现。

### `SD-6｜冲刺期评测指标`

主指标为官方 NAVSIM-v2 Final EPDMS（含 Stage1/Stage2 及全部分项），open-loop 仅 diagnostic。官方评测墙钟耗时未知。

**建议**：若 9/10 晚跑不完，允许只放代理指标，但必须 ① 标注 diagnostic-only ② 同时给出 EPDMS 的 ETA ③ **不得**据此下 Supported/Refuted（只能 Inconclusive）。**裁决时机**：`CP-4`。

### `SD-7｜Simulator/RL 投入上限`

佐藤 9/8 新提，认为「可能研究价值更高」，甚至可各成一篇；池畑不知情。
**建议**：硬时间盒 90 分钟，只做候选表 + 接口可行性判定，不写训练代码。

### `SD-8｜是否把 `scripts/` 纳入版本控制` 🆕

见 `EF-2`。**建议纳入**（只跟踪 `scripts/jobs/` 与 `scripts/audit/`）。不纳入则 §8 的并行方案无法成立。

---

## 3. 前置工程任务：TSUBAME 云端资产审计

> **这是 Environment Check 阶段，必须在任何训练之前完成。**
> 报告统一归档至 `/gs/bs/tga-RLA/qdeng/RAP/docs/audits/2026-09-09_asset_audit/`。
> 审计脚本落在 `scripts/audit/`（需 `SD-8` 通过才能入库）。

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
| HF / timm backbone | 前缀含 `backbone.` / `blocks.` / `patch_embed.` | 仅作架构参考；**是否允许 load 权重取决于 `SD-0`** |
| 未知 | 其他 | 标 `UNKNOWN`，不得使用 |

**⚠️ 硬约束（内联复述）**：`RAP_DINO_navsimv2.ckpt` 一类 **full-NAVSIM 训练 ckpt 禁止**作为 initializer / teacher / feature cache / pseudo-labeler / 超参 oracle。审计报告必须逐个 ckpt 打上 `ALLOWED_AS_INIT: yes/no`。

### 3.4 `A4`｜source-pixel 边界现状审计 🆕（`SD-1` 的前置）

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
- **只读 metadata/标注、不读 RGB 与标定** → F1 工程量大幅下降，`SD-1` 不受影响
- **已读 source 标定** → 现有产物已越过 access boundary，`SD-1` 必须优先裁决，且已有结果不能声称满足 zero-source-pixel

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
4. **初始化与恢复方式**：明确选择 **from scratch**、已有 checkpoint 初始化或 checkpoint 续训三者之一。使用 checkpoint 时必须记录路径、sha256、来源 run、用途、加载方式、已完成 epoch/step，并引用 A3 的 `ALLOWED_AS_INIT: yes`；无法证明合法或出现 `UNKNOWN` 时阻断。池畑 9/2 优先建议公平的 from-scratch 比较，因此任何非 from-scratch 方案都必须写明比较目的并由 Human 单独确认。
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
| **External / Source** | **nuScenes**（初期唯一） | 初期 | maps、agents、ego/world state、future trajectory labels | 🔴 **P1 协议下**：nuScenes RGB、nuScenes 相机内外参 |
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
| E0 | **F0** | ✅ | target-only @ b* |
| E1 | **F1** | ✅ | + external metadata → **`C_d`** raster planning |
| E2 | **X1** | 🆕 | F0 + external real ↔ **`C_s`** raster alignment |
| E3 | **X2** | 🆕 | F1 + 同上（= **方案 B**） |
| — | **A0 / A1** | ⚠️ **易混** | A0/A1 切换的是 **target R2R alignment**，**不是** external alignment |

> **9/11 现场如果把 A1 说成"关掉 external alignment"，佐藤会立刻发现表和 spec 对不上。**
> 新增臂统一冠以 **X**（e**X**ternal-pixel，视觉上提示"越过了 access boundary"）。

### 5.2 全部臂

**P1（无需 `SD-1`，审计通过即可跑）**

| 臂 | Phase-1 辅助流 | target R2R | 数据读取边界 | 冲刺内 |
|---|---|:--:|---|---|
| **F0** | 无 | on | 仅 target 预算内 + source 结构化 | ✅ 必跑 |
| **F1** | nuScenes raster under `C_d` | on | 同上，**零 source 像素/标定** | ✅ 必跑 |
| **F2** | target-raster replay（同 b 场景） | on | 同上 | 🔶 有余力 |
| **F3** | source raster + 标签置换 | on | 同上 | ⬜ 会后 |
| **A0/A1** | 无 / source raster | **off** | 同上 | ⬜ 会后 |

**P2（受 `SD-1` 门控，未裁决则全部不启动）**

| 臂 | 定义 | 对应主张 | 冲刺内 |
|---|---|---|---|
| **X1** | F0 + external alignment | 池畑主张的隔离版 | ⬜ 仅出 spec |
| **X2** | F1 + external alignment | **方案 B**（佐藤有条件认可） | ⬜ 仅出 spec |
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
| **SH-1** | @ DEV subset，F1 **不低于** F0（无灾难性负迁移） | F0；变量 = nuScenes raster 辅助流有无 | Final EPDMS（或 `SD-6` 批准的代理） | `M(F1) ≥ M(F0) − ε_noise` | `M(F1) ≪ M(F0)` 且可复现 |
| **SH-2** | F1 的变化不能仅由额外训练曝光解释 | F2（同曝光 target-raster replay） | 同上 | `M(F1) > M(F2)` | `M(F1) ≈ M(F2)` |
| **SH-3** | 训练管线数值稳定且可复现（工程假说） | 无 | 无 NaN/发散；ckpt 可恢复；同 seed 可复现 | 全满足 | 任一不满足 → **Technical Failure** |
| **SH-R** | 存在可复用的简易 driving simulator（取状态 / 按 `C_d` 渲 raster / 执行 action / 返回默认 reward） | 无（可行性） | 四项二值判定 + 缺口清单 | 四项全通 | 任一项需自研 > 1 周 |

> **SH-1 刻意写成弱假说**（"不低于"而非"显著优于"）。单 subset × 单 seed 在统计上没有能力支持 `δ_main = 1.0` 的效应判定（confirmatory 设计要求 3 subsets × 2 seeds）。把 pilot 包装成"证明 F1 更好"会被佐藤当场拆穿。

---

## 6. 冲刺矩阵（TSUBAME 作业感知）

### 6.1 资源模型

- 1 × `node_f` = **H100 × 4**。以下耗时估算均以「1 node_f 独占」为单位。
- **不写死墙钟**：先测吞吐，再按 `T_finish = (remaining_steps / measured_steps_per_hour) × 1.2` 推。
- 队列等待**不可控**：① 尽早提交 ② 用 `-ar` 预约（现有脚本已有 `#$ -ar 8016` / `8176` 先例）③ 长作业拆 checkpoint 可续训。

### 6.2 Day-1（9/9，自当前时刻起）

| 时段 | 任务 | 类型 | 假说 | 位置 | 卡点 |
|---|---|---|---|---|---|
| 15:30–16:00 | 裁决 `SD-0/1/2/3/7/8`；确认 `EF-1`（`which qsub sbatch`） | 决策 | — | 人 | **CP-1** |
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
| 08:00–09:30 | 收 F0/F1；统一评测；核对公平性（同 step / 同 target scene / 同评测器版本 / 同 `git_sha`） | SH-1 | — |
| 09:30–11:00 | 出三张核心材料：① F0/F1 指标表 ② 学习曲线 ③ 3–5 个定性案例（刹车 / 横向偏移 / 交叉路口） | — | — |
| 11:00 | 裁决 `SD-6` + 最后实验窗口决策 | — | **CP-4** |
| 11:00–13:00 | 二选一：① 提交 F2（若 SH-1 有信号，需 SH-2 隔离曝光）② 换 seed 复跑 F0/F1（若差异极小，先确认噪声量级） | SH-2 / SH-3 | — |
| 13:00–18:00 | **最后允许提交关键作业的时段**。优先级：修会改变结论的明确 bug > F2 > 稳定性复跑 > 定性案例 | — | — |
| **18:00** | **架构冻结线**：不再新增模块 / 不改数据定义 / 不重建大规模缓存 / 不提交预计 9/11 08:00 前跑不完的作业 | — | **CP-5** |
| 18:00–23:00 | PPT 定稿（7–9 页，见 §11） | — | — |
| 23:00→次日 08:00 | 仅允许：已 smoke 过的复跑 / 从可靠 ckpt 续训 / 自动评测 / 归档 | — | — |

### 6.4 调度决策表（`CP-2c` 后使用）

| smoke 实测的单臂完整训练耗时（1×node_f） | 决策 |
|---|---|
| ≤ 6 h | 跑完整 F0 + F1，预留 F2 槽位 |
| 6–10 h | 只跑完整 F0 + F1 |
| 10–16 h | 全部改**统一缩减 schedule**（同 step 数、同初始化、同采样规则），结论一律标 preliminary |
| > 16 h | 不启动完整训练；优先复用已有 ckpt 重新评测，或跑统一 proxy schedule |
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

密集 scaling 曲线、≥3 seeds、多 backbone / 多分辨率 / 多 PE 设计、alignment 超参搜索、X1/X2/X3 训练、F3、A0/A1、KITTI/Waymo 接入、BEV / 3D feed-forward / 3DGS baselines、从零训 RL teacher、闭环 RL、为出图临时重构训练框架、完整 related work。

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
            │   └── rig=nuscenes_Cs__cfg=<Cs_hash8>/   # 🔴 X 系列专用，SD-1 未过不得创建
            ├── manifests/
            │   ├── dev_b010_seedDEV.json  + .sha256   # ← 本冲刺唯一使用
            │   └── confirm_b010_subset{A,B,C}.json    # ← 冲刺期间不得打开
            ├── pairs/
            │   ├── target_real_raster__b010_seedDEV.jsonl
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

**`run_id`**：`{YYYYMMDD}_{arm}_{budget}_{subset}_{seed}_{git8}`，例 `20260909_F1_b010_dev_s0_a1b2c3d4`

### 7.3 实验账本（单一真相源）

`docs/20260909/registry.csv`（报告只从这里取数，**禁止手工从日志复制**）：

`run_id, arm, protocol(P1|P2), branch, git_sha, job_id, node_type, target_manifest+sha, source_manifest+sha, render_rig, Cd_hash, losses, init_ckpt+sha, steps, wall_clock, n_target_scenes, n_src_presentations, seed, subset, status, failure_type(none|technical|scientific), metric_final_epdms, metric_stage1, metric_stage2, openloop, is_confirmatory(false)`

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
| **Branch** | *跑的是哪份代码？* | **每个会改动代码的实验族一个分支**。F0/F1/F2 因必须共享完全相同的代码（公平性控制），**共用一个分支、一个 commit** |
| **Worktree** | *这份代码物化在磁盘的哪里？* | **每个"同时在跑且代码不同"的实验族一个 worktree**，且只在 **TSUBAME** 上开 |

**关键判断：worktree 的价值在云端，不在 MacBook。**

- MacBook 只编辑不训练 → 没有"两份代码同时被占用"的问题 → **本地普通分支切换即可，不需要 worktree**。
- TSUBAME 上一个作业会 `cd $RAP_ROOT && python ...` 跑几小时。**如果这期间你在同一目录 `git checkout` 或改文件，正在跑的作业会读到被换掉的代码**——而且不报错，只静默污染实验。这才是 worktree 要解决的真问题。

**最小正确方案**：

```
TSUBAME:
/gs/bs/tga-RLA/qdeng/RAP            [main]                  ← 只读参考，不在这里提交作业
/gs/bs/tga-RLA/qdeng/RAP-p1         [exp/sprint-0909-base]  ← F0/F1/F2 全部从这里提交
/gs/bs/tga-RLA/qdeng/RAP-p2         [exp/sprint-0909-x]     ← X 系列开发（SD-1 通过后才动 allowlist）
```

只需 **2 个 worktree**。F0/F1/F2 共用 `RAP-p1` 是**正确的**（它们必须同 commit）；拆成 3 个反而增加"代码不一致"的风险。

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

**Step 0.5｜把 `scripts/` 纳入版本控制（`SD-8` 通过后）**

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
| **CP-1** | 15:30–16:00 | 任何代码/作业之前 | 裁决 `SD-0/1/2/3/7/8`；确认 `EF-1`（`which qsub sbatch`）；填 `env/sprint_20260909.env` | **全部停机** |
| **CP-2a** | 17:00 | `main` 补齐后 | 逐条 review `git status`，确认只带入必需资产，未混入 alignment regression 的实验性改动 | 回退，重做逐路径 checkout |
| **CP-2b** | 18:00 | A1–A5 审计完成、拟运行配置已冻结 | ⭐ **逐条读 `A1_split_membership.md`**：`dataset_*` 与 `navtest`/`navhard_two_stage` 交集是否为 0；A1/A2 的完整性与实测体量；`A3` 每个 ckpt 的 `ALLOWED_AS_INIT`；`A4` 的 source-pixel 判定；`A5_training_readiness.md` 中数据、epoch/schedule、from-scratch/ckpt、原论文超参数对照及复现配置是否全部有证据且 PASS | **交集 > 0、任一 UNKNOWN/BLOCKED、配置或 manifest 未冻结 → 禁止 Smoke Test 与任何训练**（最硬的门） |
| **CP-2c** | 19:00 | H0 几何审计 + smoke 完成 | landmark 重投影误差、round-trip 误差、corruption 检出率；肉眼看若干 raster 叠图；smoke 的 loss 接线与实测吞吐 | **H0 未过 → 禁止启动训练** |
| **CP-3** | 20:30–21:30 | 写 X 系列代码之前 | 裁决 `SD-4` 五项；确认 P2 与 P1 的隔离方式 | X 系列只出文档，不进代码库 |
| **CP-4** | 9/10 11:00 | 第一批结果出来 | ① **先只看 Measurement，不做解释** ② 核对公平性（同 step / 同 target scene / 同评测器版本 / 同 `git_sha`）③ 裁决 `SD-6` ④ 决定最后窗口 | 公平性不过 → 标 `invalid`，不上 PPT |
| **CP-5** | 9/10 18:00 | 架构冻结线 | 确认无新模块、无数据定义变更、无跑不完的作业在排队 | 强制冻结 |
| **CP-6** | 9/11 09:30 | 结果硬冻结 | 逐条检查结论措辞与证据强度是否匹配；确认所有数字标 `preliminary / single subset / single seed` | 09:30 后不因新数字重写主叙事，迟到结果进 appendix |

### Technical Failure vs Scientific Failure

| 现象 | 分类 | 处置 |
|---|---|---|
| OOM、shape mismatch、dataloader 崩、NaN、ckpt 损坏、作业被队列杀 | **Technical** | 修复后重跑；**不产生任何科学结论** |
| 训练正常收敛，但 `M(F1) < M(F0)` | **Scientific** | 记录为 negative transfer 观测，进 registry 与 PPT，**不得因结果不理想私改 protocol** |
| 训练正常，差异 < 噪声 | **Scientific** | 结论 = `Inconclusive`，**不得**改写成"趋势向好" |
| A1 污染检出 / H0 未过 | **既非** | 阻断门，整个 build 作废 |

> **铁律**：实验开始后不得因初期结果不理想而私改 Protocol。任何 protocol 变更必须开新 `run_id`、在 registry 记录原因，并声明该 run 与前序 run **不可配对比较**。

---

## 10. 测量与科学解释规程

### 10.1 报告顺序（不可颠倒）

**先 Measurement，后 Scientific Interpretation。** PPT 与口头汇报都是：`这是数字 → 这是公平性核对 → 我能下的结论 → 我不能下的结论`。

### 10.2 结论类型

仅三种：**`Supported` / `Refuted` / `Inconclusive`**。

本次全部为 pilot（单 subset × 单 seed），**默认结论应为 `Inconclusive`**，除非出现灾难性负迁移（可下 `Refuted`(SH-1)）或审计门未过（可下 `Refuted`(SH-A/SH-0)）。

### 10.3 结果解释对照表

| 观测 | **允许**的表述 | **禁止**的表述 |
|---|---|---|
| F1 > F0 且 F1 > F2 | external structured content 在此 pilot 条件下有超出 replay 与辅助分支的正向信号 | "证明了 external knowledge 有效" |
| F1 > F0 但 F1 ≈ F2 | 增加的训练分支有帮助，但 external content 未被隔离 | "external 场景多样性带来增益" |
| F1 ≈ F0 | 在该预算与配方下未观测到效应，或功效不足 | "略有提升" |
| F1 < F0 | 在该配置下观测到负迁移 | 静默丢弃该 run |
| 只有 open-loop 指标 | 仅 diagnostic，`Inconclusive` | 任何 Supported/Refuted |

### 10.4 措辞审计（`CP-6` 逐条过）

- 每个数字必须带：`b*`、subset ID、seed、step 数、`git_sha`、是否 pilot
- 每个主张必须带范围限定：**on the evaluated NAVSIM target rig**
- 出现"证明 / 显著 / 最强 / 有效"而无置信区间支撑 → 一律改写
- **「all-data 最强」在 9/11 之前不得作为结论出现**（佐藤 9/8 明确要求；Memo 已确认状态为「值得开展比较实验；不预设 B 优于 A」）

---

## 11. 9/11 汇报结构（7–9 页）

顺序按 Memo §八：**目标与已知条件 → 两类知识及数据路径 → 最小对照结果 → 能/不能支持的判断 → RL 可行性初查 → 需要老师共同判断的问题**

| # | 页 | 内容 |
|---|---|---|
| 1 | 我希望老师判断什么 | 直接列 `SD-1` / `SD-5` / 3D baseline 排期 / RL 投入比例 |
| 2 | 问题定义与协议 | target 是谁、何时知道 calibration、zero/few-shot 数什么、`b*` 口径（`SD-3` 结果）+ 绝对计数 |
| 3 | 三维数据流表 | **场景来源 × 渲染 rig × 接受的 loss**（修正 Old/New Synthetic 混淆） |
| 4 | 资产、训练准备审计与 H0 通过证明 | A1 污染检查、A2 完整度、A3 ckpt 合规、A4 边界现状、A5 冻结配置；几何误差与 corruption 检出 |
| 5 | 最小对照 pilot 结果 | F0/F1(/F2)；全部标 preliminary；含公平性核对表 |
| 6 | 我能下 / 不能下的结论 | 严格用 Supported / Refuted / Inconclusive |
| 7 | **X 系列（方案 B）设计 + 协议冲突** | `SD-1` 全文；池畑主张 ↔ source-pixel 边界的冲突；请裁决 |
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
| 不要"先有 RAP 再找理由" | ✅ 用 F3 标签置换 + F2 曝光控制来证伪 | — |
| random sampling × 多 seed，不要精挑 few-shot 样本 | 🔶 当前用分层随机 | 与池畑确认分层 vs 纯随机的取舍 |
| 最终不能只有 NAVSIM 一个 target | ❌ | 第二 target rig |
| 多个 external dataset | ❌ | **nuScenes → Waymo → KITTI**（仓库已有 `run_waymo_*`，Waymo 工程起点低于 KITTI） |

---

*本文档以 `docs/20260909/` 现有文件与本仓库代码为唯一基准生成。所有 `[Scientific Decision Required]` 与 `[EF-*]` 均未被自动补全。云端路径下的一切断言均标记为待审计，未经 §3 审计通过不得引用。*
