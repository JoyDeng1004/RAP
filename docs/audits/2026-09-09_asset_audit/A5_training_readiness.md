# A5｜F0/F1 Training Readiness

- 审计时间：2026-09-09 17:40–17:54 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex
- 工作树：`DIRTY`

## 判定

**F0: BLOCKED。F1: BLOCKED。CP-2b 不可 PASS。**

没有生成 `config.frozen.yaml`：当前不存在可合法冻结的 manifest/config/run，且未知值不得由 Agent 补全。

## 字段 → 实测/解析值 → 证据 → 差异 → 状态

| 字段 | F0 实测/解析值 | F1 实测/解析值 | 证据路径 | 与基准的差异 | 状态 |
|---|---|---|---|---|---|
| Target root | 未冻结；现有候选 `dataset_norm` | 同 F0 | A1；`dataset_norm/` | 候选数据污染且 raster 不完整 | **BLOCKED** |
| 合法 training 总量 | 1,192 logs / 103,288 tokens（官方 YAML） | 同 F0 | `.../scene_filter/navtrain.yaml` | 只是合法全集，不是已选 manifest | PASS as reference only |
| 10% DEV manifest | 不存在 | 不存在 | `outputs/sprint_20260909/manifests/` 不存在 | 抽中 logs/frames/images、规则与 SHA 均未知 | **BLOCKED** |
| Evaluation overlap | `dataset_norm` 命中 navtest 10 logs / 1,365 tokens；命中 warmup 62 logs / 563 tokens；navhard UNKNOWN | 同 F0 | `A1_split_membership.md` | 必须全部为 0 | **BLOCKED** |
| 缺失/损坏/重复 | metadata 64/64 可读，0 duplicate token；主 raster 缺 259,287/414,936 paths | 同 F0 | `A1_rap_datasets.md` | 缺失率必须可接受且 manifest 级核验 | **BLOCKED** |
| Target scene/log/token/frame/image/pair | selected 全部 UNKNOWN | selected 全部 UNKNOWN | manifest 缺失 | A5 要求实测绝对数 | **BLOCKED** |
| External root | N/A | nuScenes root 存在 | `A2_nuscenes.md` | F1 source manifest 未定义 | F0 N/A；F1 **BLOCKED** |
| External scene/log/token/frame/image/pair | N/A | 850 trainval scenes / 34,149 sample frames 是资产总量；实际选择/呈现/pair UNKNOWN | `A2_nuscenes.md` | 资产总量不能代替训练 manifest | F1 **BLOCKED** |
| P1 source access | N/A | 当前 renderer 读取 RGB/calibration | `A4_source_pixel_boundary.md` | P1 要求读取/解码/内外参计数全 0 | F1 **BLOCKED** |
| Renderer / C_d hash | N/A | UNKNOWN | 当前无 P1-only renderer output/config | 必须冻结 hash | F1 **BLOCKED** |
| Epoch / max_steps | UNKNOWN；仓库默认 max_epochs=100 | UNKNOWN | `default_training.yaml` | 未有 arm-specific 最终解析值 | **BLOCKED** |
| Steps per epoch / equivalent epochs | UNKNOWN | UNKNOWN | selected dataset 与 devices 未冻结 | 必填 | **BLOCKED** |
| Batch size | 默认 per-loader 64；global/effective UNKNOWN | 同 F0 | `default_training.yaml` | devices 未指定，不能算 global/effective batch | **BLOCKED** |
| Gradient accumulation | 默认 1，未冻结 | 同 F0 | `default_training.yaml` | 不能写“沿用默认” | **BLOCKED** |
| Optimizer | 代码为 AdamW，lr 1e-4，weight_decay 1e-4；image backbone 被 freeze | 同 F0 | `rap_agent.yaml`；`RAPAgent.get_optimizers()` | 未形成 frozen config | **BLOCKED** |
| Scheduler / warmup | `WarmupCosLR(min_lr=1e-5, epochs=20, warmup_epochs=1)` | 同 F0 | `RAPAgent.get_optimizers()` | 与 trainer 默认 100 epochs 不一致；最终 schedule 未裁决 | **BLOCKED** |
| Early stopping | 未发现 callback；未冻结 | 同 F0 | `get_training_callbacks()` 仅 ModelCheckpoint | A5 必须明确 | **BLOCKED** |
| Checkpoint frequency | 默认 save_last + top3，monitor `val/score`；频率随 validation | 同 F0 | `RAPAgent.get_training_callbacks()` | 未冻结准确频率 | **BLOCKED** |
| Init mode | agent `checkpoint_path: ''` 表面为 scratch，但 trainer 固定 `ckpt_path='last'` | 同 F0 | `rap_agent.yaml`；`run_training.py` | 复用 output dir 时可能自动 resume，不能证明 from scratch | **BLOCKED** |
| Resume contract | run_id/config/data hash/remaining steps 均不存在 | 同 F0 | 无 registry/config/hashes | 必须同 run 且 hash 一致 | **BLOCKED** |
| Loss switches | 默认 `distill_feature: True`；base loss weights来自 RAPConfig | 同 F0 | `rap_agent.yaml`；`navsim_config.py` | F0/F1 具体 on/off 与 P1 合法性未冻结 | **BLOCKED** |
| Input resolution | 代码默认 1024×256，未冻结 | 同 F0 | `RAPConfig.camera_width/height` | 未证明与运行解析值一致 | **BLOCKED** |
| Augmentation | UNKNOWN | UNKNOWN | 无 frozen config/manifest | 必填 | **BLOCKED** |
| Seed | 计划冻结 20260909；代码默认仍为 0 | 同 F0 | Sprint Plan；`default_training.yaml` | 未有解析命令覆盖 `seed=20260909` | **BLOCKED** |
| Precision / distributed | 默认 `16-mixed` / `ddp_find_unused_parameters_true` / 1 node；devices UNKNOWN | 同 F0 | `default_training.yaml` | node/device/global batch 未冻结 | **BLOCKED** |
| Hardware | 计划要求 `node_f`，实际 job wrapper 不存在 | 同 F0 | Sprint Plan；`scripts/jobs/` 未建立 | 不得把计划当提交事实 | **BLOCKED** |
| Software env | 当前审计 shell可见 torch 2.1.0+cu121；完整环境 lock UNKNOWN | 同 F0 | A3 probe | 缺 Python/package/CUDA/driver freeze | **BLOCKED** |
| Metric | PDMS + existing components 已由 SD-6 冻结 | 同 F0 | Sprint Plan §SD-6 | evaluator version/command 未冻结 | **BLOCKED** |
| Output / run_id / registry | 目录、run_id、registry 均不存在 | 同 F0 | `outputs/sprint_20260909/`、`docs/20260909/registry.csv` 均不存在 | 必填 | **BLOCKED** |
| Git provenance | HEAD 已知，但工作树 dirty，renderer staged 未 commit | 同 F0 | `git status` | 实验要求 Human PASS + committed + clean | **BLOCKED** |
| Failure/recovery | UNKNOWN | UNKNOWN | 无执行契约 | 必填 | **BLOCKED** |

## 与原论文设定逐项对照

本仓库与 `docs/20260909/` 没有提供可逐项核验的原论文训练参数表。按计划不得从外部记忆或猜测补值。

| 项目 | 当前实现可见值 | 原论文值 | 对照 | 状态 |
|---|---|---|---|---|
| optimizer | AdamW | UNKNOWN | 无法核验 | **BLOCKED** |
| learning rate | 1e-4 | UNKNOWN | 无法核验 | **BLOCKED** |
| scheduler | WarmupCosLR | UNKNOWN | 无法核验 | **BLOCKED** |
| warmup | 1 epoch（代码 hard-code） | UNKNOWN | 无法核验 | **BLOCKED** |
| weight decay | 1e-4 | UNKNOWN | 无法核验 | **BLOCKED** |
| batch size | local loader 64；global UNKNOWN | UNKNOWN | 无法核验 | **BLOCKED** |
| input resolution | 1024×256 | UNKNOWN | 无法核验 | **BLOCKED** |
| augmentation | UNKNOWN | UNKNOWN | 无法核验 | **BLOCKED** |
| loss / weights | RAPConfig 当前默认值；arm-specific 未冻结 | UNKNOWN | 无法核验 | **BLOCKED** |
| training update budget | UNKNOWN；100 trainer epochs vs 20 scheduler epochs 冲突 | UNKNOWN | 无法核验 | **BLOCKED** |
| checkpoint policy | save_last/top3；trainer auto `last` | UNKNOWN | 无法核验 | **BLOCKED** |

## 执行契约

| 阶段 | 工作目录 | 入口/完整命令 | wrapper | manifest | 产物/成功条件/告警 | 状态 |
|---|---|---|---|---|---|---|
| H0 | UNKNOWN | UNKNOWN | UNKNOWN | fixture UNKNOWN | 坐标系、三项误差报告位置均 UNKNOWN | **BLOCKED** |
| Smoke F0/F1 | UNKNOWN | UNKNOWN | `scripts/jobs/` 不存在 | UNKNOWN | UNKNOWN | **BLOCKED** |
| Full F0/F1 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **BLOCKED** |
| PDMS eval | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | evaluator version与产物 UNKNOWN | **BLOCKED** |

因此不能生成或签署 F0/F1 `config.frozen.yaml`，也不能请求 CP-2b PASS。
