# A1 追加｜污染剔除后的干净可用池

- 审计时间：2026-09-10T14:14:18+09:00
- 执行主机：`r4n11`
- Git HEAD：`f5ae782afc25eda4b451fd742b87c8eb17856aa1`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：uq02279
- 脚本：`scripts/audit/a1_clean_subset.py`（只读）
- 相机集合（`C_d`，Sprint Plan §7.2）：CAM_B0, CAM_F0, CAM_L0, CAM_R0

## 方法

干净池按**加法**定义：

```
clean = (dataset tokens ∩ 已批准训练池 tokens) − 评测侧 tokens
```

不使用减法（全部 token 减去已知污染），因为 `A1_split_membership` 记录了 44,389 个不在任何官方 split token 列表内的 token，并明确警告不得把它们自动视为合法训练 token。

`SD-12` 未裁决，因此三个候选池全部计算，本报告仅作决策材料，不代替裁决。

## 数据集实测

| 项 | 值 |
|---|---:|
| metadata pickle | 64 |
| 不可读 pickle | 0 |
| records | 51867 |
| unique tokens | 51867 |
| unique logs | 64 |
| 缺 token 字段 | 0 |
| 缺 cams 字段 | 0 |
| `C_d` 相机条目缺失 | 0 |

## 评测侧 split

| split | 状态 | 官方 logs | 官方 tokens | ∩ logs | ∩ tokens |
|---|---|---:|---:|---:|---:|
| `navtest` | PRESENT | 136 | 12146 | 10 | 1365 |
| `navhard_two_stage` | **MISSING_SPLIT_DEFINITION** | — | — | — | — |
| `warmup_test_e2e` | PRESENT | 62 | 563 | 62 | 563 |
| `navmini` | PRESENT | 62 | 396 | 62 | 396 |

> 🔴 以下评测侧 split 定义缺失，其交集**无法核验**，本报告的干净池对它们不提供任何保证：`navhard_two_stage`

## 每个 `SD-12` 候选池的干净子集

### `B1_navtrain` — `navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml`

official NAVSIM training split; smallest, most conservative

池规模：1192 logs / 103288 tokens；`SD-3` 的 10% ≈ 119 logs

与 dataset 的交集：52 logs / 6104 tokens

| 剔除粒度 | 剩余 logs | 剩余 tokens | raster root | 该 root 内文件数 | `C_d` 四相机齐全的 token |
|---|---:|---:|---|---:|---:|
| **token 级** | 52 | 5658 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **1878** |
| **log 级** | 0 | 0 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **0** |

每路相机单独命中数：

- token 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 5658、CAM_L0 5658、CAM_R0 5658
- token 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 1878、CAM_F0 5658、CAM_L0 5658、CAM_R0 5658
- log 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 0、CAM_L0 0、CAM_R0 0
- log 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 0、CAM_F0 0、CAM_L0 0、CAM_R0 0

### `B2_navall` — `navsim/planning/script/config/common/train_test_split/scene_filter/navall.yaml`

full trainval pool; largest, needs its own navtest separation proof

池规模：14539 logs / 401300 tokens；`SD-3` 的 10% ≈ 1454 logs

与 dataset 的交集：54 logs / 6110 tokens

| 剔除粒度 | 剩余 logs | 剩余 tokens | raster root | 该 root 内文件数 | `C_d` 四相机齐全的 token |
|---|---:|---:|---|---:|---:|
| **token 级** | 53 | 5664 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **1878** |
| **log 级** | 2 | 6 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **0** |

每路相机单独命中数：

- token 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 5664、CAM_L0 5664、CAM_R0 5664
- token 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 1878、CAM_F0 5664、CAM_L0 5664、CAM_R0 5664
- log 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 6、CAM_L0 6、CAM_R0 6
- log 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 0、CAM_F0 6、CAM_L0 6、CAM_R0 6

### `B3_split_config` — `process_data/default_train_val_test_log_split.yaml`

the pool build_alignment_small_data.py actually consumes; LOG-LEVEL ONLY

池规模：13180 logs（**仅 log 级，无 token 列表**）；`SD-3` 的 10% ≈ 1318 logs

与 dataset 的交集：44 logs / 35890 tokens（token 由 log 反推，保证更弱）

| 剔除粒度 | 剩余 logs | 剩余 tokens | raster root | 该 root 内文件数 | `C_d` 四相机齐全的 token |
|---|---:|---:|---|---:|---:|
| **token 级** | 44 | 35520 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **2387** |
| **log 级** | 1 | 800 | `rendered_sensor_blobs` | 155748 | **0** |
| | | | `rendered_sensor_blobs_4cam_v1` | 133249 | **0** |

每路相机单独命中数：

- token 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 35520、CAM_L0 35520、CAM_R0 35520
- token 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 2387、CAM_F0 35520、CAM_L0 35520、CAM_R0 35520
- log 级 / `rendered_sensor_blobs`：CAM_B0 0、CAM_F0 800、CAM_L0 800、CAM_R0 800
- log 级 / `rendered_sensor_blobs_4cam_v1`：CAM_B0 0、CAM_F0 800、CAM_L0 800、CAM_R0 800

## 如何读这张表

1. **最后一列才是能否跑 F0 的答案。** token 数再多，若 `C_d` 四相机不齐，该 token 对 F0/F1 不可用。
2. **两个 raster root 不合并。** 依 `A1_rap_datasets.md`，`rendered_sensor_blobs_4cam_v1` 是独立根，未经 manifest/pair 审计不得用于补齐主根。
3. **token 级与 log 级剔除的差距本身就是一个待裁决问题。** Sprint Plan §3.1 只写了「交集必须为 0」，未指定粒度。若同一 driving log 的 token 分处训练与评测两侧是否构成泄漏，需 Human 以 `SD-*` 裁决。
4. 本报告不改变 `A1` 的 CONTAMINATED 判定，也不解除 `CP-2b`。

