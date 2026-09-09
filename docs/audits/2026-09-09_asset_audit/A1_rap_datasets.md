# A1｜RAP 数据增强/处理目录

- 审计时间：2026-09-09 16:55–17:54 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex

## 判定

**FAIL / CONTAMINATED。** 污染证据见 `A1_split_membership.md`。此外，目录内部存在明显的不完整和失配，不能直接作为本次 F0/F1 输入。

## 数据来源与 provenance

| 目录 | 当前代码中的对应脚本 | 当前代码能证明的行为 | 生成 commit / job / 实际原始输入 |
|---|---|---|---|
| `dataset_norm` | `process_data/create_openscene_metadata.py` | 写每 log 一个 pickle；把 `sensor_blobs` 路径替换为 `rendered_sensor_blobs` | **UNKNOWN**。产物未嵌入 git SHA/job id；`process_data/process.sh` 只有占位路径。 |
| `dataset_aug` | `process_data/create_openscene_metadata_aug.py` | 写带额外 16-hex suffix 的 pickle，并生成四路 augmented raster | **UNKNOWN**，理由同上。 |
| `dataset_perturbed` | `process_data/create_openscene_metadata_purturbed.py` | 文件名仍保留上游拼写 `purturbed`；当前代码生成 perturbed metadata/raster | **UNKNOWN**，理由同上。 |

`process_data/process.sh` 声明输入为 nuPlan v1.1 的 `${NUPLAN_PATH}/splits/${split}` 与 sensor/maps 路径，但实际值仍是 `/PATH_TO/...` 占位符。现有产物无法反推出实际原始路径或作业号，因此不得把模板值当生成事实。

## 目录体量与结构

| 根目录 | 文件数 | 目录数 | 已测字节 | 扩展名 | 结构摘要 |
|---|---:|---:|---:|---|---|
| `dataset_aug` | 4,524,750 | 210,459 | UNKNOWN | 4,507,451 jpg；17,299 pkl | `navsim_logs/mini/*.pkl`；`rendered_sensor_blobs_augmented/<log>_<id>/{CAM_F0,CAM_L0,CAM_R0,CAM_B0}/*.jpg` |
| `dataset_norm` | 289,061 | 571 | 36,855,331,706 | 288,993 jpg；64 pkl；4 png | `navsim_logs/mini/*.pkl`；主 raster；独立 `rendered_sensor_blobs_4cam_v1` |
| `dataset_perturbed` | 20,380 | 176 | 3,585,639,951 | 15,285 jpg；5,095 pkl | `navsim_logs/mini/*.pkl`；`rendered_sensor_blobs_perturbed/<log>/{CAM_F0,CAM_L0,CAM_R0}/*.jpg` |

`dataset_aug` 的全树 size/mtime scan 因逐文件 `stat` 超过 15 分钟且交互中断，未得到可信结果；随后不做 per-file stat 的完整目录项计数在 208 秒内完成。这里明确保留 `UNKNOWN`，不做体量外推。

### 叶级计数

| 路径 | 顶层样本/log 目录 | 文件分布 |
|---|---:|---|
| `dataset_aug/navsim_logs/mini` | N/A | 17,299 pkl |
| `dataset_aug/rendered_sensor_blobs_augmented` | 42,091 | CAM_F0 1,126,864；CAM_L0 1,126,863；CAM_R0 1,126,862；CAM_B0 1,126,862 |
| `dataset_norm/navsim_logs/mini` | N/A | 64 pkl |
| `dataset_norm/rendered_sensor_blobs` | 65 | CAM_F0 51,898；CAM_L0 51,898；CAM_R0 51,900；CAM_B0 48；根部 png 4 |
| `dataset_norm/rendered_sensor_blobs_4cam_v1` | 54 | CAM_F0/L0/R0 各 43,429；CAM_B0 2,962 |
| `dataset_perturbed/navsim_logs/mini` | N/A | 5,095 pkl |
| `dataset_perturbed/rendered_sensor_blobs_perturbed` | 43 | CAM_F0/L0/R0 各 5,095 |

## `dataset_norm` 完整性

- 64/64 pickle 可读。
- 51,867 records；51,867 unique tokens；0 duplicate token occurrence。
- 64 unique logs；pickle 文件名与记录内 `log_name` 全部一致。
- 每条 metadata 含 8 路 camera path，共 414,936 个唯一引用。
- 主 `rendered_sensor_blobs` 只有 155,748 files；与 metadata 路径集合比较：259,287 missing、99 extra，missing rate = **62.4884%**。
- 主 raster 基本只有 F0/L0/R0；B0 仅 48 张。`rendered_sensor_blobs_4cam_v1` 是另一套独立根，仅覆盖 54 个有文件的 logs，不能在没有 manifest/pair audit 时自动补齐主根。
- 主 raster 比 metadata 多出的顶层目录为 `missing_camera`。

## 子集归属

目录名虽然是 `mini`，内容并非一个可直接用于训练的纯合法 training subset：它同时与 `navtrain`、`navtest`、`navmini`、`warmup_test_e2e` 相交，并含 2 个不在当前可用 split 定义内的 logs。详见 `A1_split_membership.md`。
