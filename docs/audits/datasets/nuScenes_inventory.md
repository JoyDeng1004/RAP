# nuScenes inventory

## 当前结论

状态：**FAIL（full trainval/test 不完整）**。

- 主根有 `v1.0-trainval` 和 `v1.0-test` metadata，devkit 初始化都 PASS，但 official `assert_download.py` 都 FAIL。
- trainval 只缺 `sweeps/RADAR_BACK_RIGHT` 的 32,605 条引用；其他 samples/sweeps 全存在。
- test 的 samples 和 LIDAR sweeps 完整，但六路 camera sweeps 与五路 radar sweeps全缺，共 338,382 条。
- 仓库内 mini 副本缺全部 14,008 个 camera files 和 map mask；但发现了三个独立完整 mini 根，因此 mini 不需要重新下载。

“devkit 能初始化”只说明 metadata 能加载，不能证明 sensor blobs 完整。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| 深扫节点 | `r21n9`（主 deep）；`r13n3`（missing-by-channel）；`r12n11`（额外 mini roots） |
| Git SHA / 状态 | `b95916ffda6a08e603d43c20a8a3a684c53359f0` / `DIRTY`（盘点前 Human 资产，未覆盖） |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp` |

敏感环境变量未记录。主根和仓库 mini 都不是 symlink，均位于 Lustre filesystem ID `25c66b0a00000000`。

## 根与结构

| 根 | realpath | 版本 | bytes / files / dirs | root mtime | 判定 |
|---|---|---|---:|---|---|
| `/gs/bs/tga-RLA/qdeng/data/nuscenes` | 同左 | `v1.0-trainval`, `v1.0-test` | `498,108,321,792` / `4,377,830` / `36` | `2026-07-30 21:06:22 +0900`；全量 file mtime 范围见 raw filesystem snapshot | PARTIAL |
| `/gs/bs/tga-RLA/qdeng/RAP/nuscenes-mini/nuscenes` | 同左 | `v1.0-mini` | `7,557,971,697` / `25,052` / `33` | files `2026-09-08 15:47:32`–`15:49:28 +0900` | PARTIAL / unusable as-is |

主根存在 `samples/`、`sweeps/`、`maps/`、`can_bus/`、`v1.0-trainval/`、`v1.0-test/`。`maps/` 含 raster、`basemap/`、`expansion/`、`prediction/`。主根没有 `v1.0-mini/`、`lidarseg/`、`panoptic/`。

仓库 mini 存在同名目录壳和 metadata，但缺 camera files 和至少一个 devkit 初始化所需 map mask。

## Metadata 表和唯一 token

所有已加载表都满足 `count == unique token`，duplicate token occurrences 为 0，缺 token 行为 0。

| version | scene | sample | sample_data | sample_annotation | log | sensor | calibrated_sensor | ego_pose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1.0-trainval` | 850 | 34,149 | 2,631,083 | 1,166,187 | 68 | 12 | 10,200 | 2,631,083 |
| `v1.0-test` | 150 | 6,008 | 462,901 | 0 | 15 | 12 | 1,800 | 462,901 |
| `v1.0-mini`（repo copy） | 10 | 404 | 31,206 | 18,538 | 8 | 12 | 120 | 31,206 |

test 的 annotation 为 0 是官方 test 标签不公开的正常结构，不判缺失。

三类外键对每个版本都核验：`calibrated_sensor_token`、`ego_pose_token`、`sample_token` 的 missing 均为 0。

## 主根 sample_data 完整性

| version | sample_data | present nonempty | missing | zero/unreadable | missing rate | image/lidar/radar sample decode/read |
|---|---:|---:|---:|---:|---:|---|
| trainval | 2,631,083 | 2,598,478 | 32,605 | 0 | 1.2392% | 各抽 128，0 errors |
| test | 462,901 | 124,519 | 338,382 | 0 | 73.0992% | 从 present 集各抽 128，0 errors |
| repo mini | 31,206 | 17,198 | 14,008 | 0 | 44.8875% | image 0 可抽；lidar/radar 各 128，0 errors |

抽样无错误不能证明未抽样文件无损；全量内容 checksum 仍 UNKNOWN。

### trainval 按 channel

| channel | samples present/expected | sweeps present/expected | 状态 |
|---|---:|---:|---|
| CAM_FRONT | 34,149/34,149 | 164,166/164,166 | PRESENT |
| CAM_FRONT_LEFT | 34,149/34,149 | 164,274/164,274 | PRESENT |
| CAM_FRONT_RIGHT | 34,149/34,149 | 163,881/163,881 | PRESENT |
| CAM_BACK | 34,149/34,149 | 161,453/161,453 | PRESENT |
| CAM_BACK_LEFT | 34,149/34,149 | 160,856/160,856 | PRESENT |
| CAM_BACK_RIGHT | 34,149/34,149 | 164,266/164,266 | PRESENT |
| LIDAR_TOP | 34,149/34,149 | 297,737/297,737 | PRESENT |
| RADAR_FRONT | 34,149/34,149 | 188,562/188,562 | PRESENT |
| RADAR_FRONT_LEFT | 34,149/34,149 | 191,133/191,133 | PRESENT |
| RADAR_FRONT_RIGHT | 34,149/34,149 | 190,435/190,435 | PRESENT |
| RADAR_BACK_LEFT | 34,149/34,149 | 185,467/185,467 | PRESENT |
| RADAR_BACK_RIGHT | 34,149/34,149 | **156,460/189,065** | **PARTIAL：缺 32,605** |

### test 按 channel

每个 channel 的 6,008 个 `samples/` 都存在。`LIDAR_TOP` 的 52,423 个 sweeps 全存在。六路 camera sweeps 分别缺 28,916 / 28,894 / 28,818 / 28,452 / 28,315 / 28,887；五路 radar sweeps 分别缺 33,145 / 33,548 / 33,522 / 32,667 / 33,218。也就是 test 的这些 sweeps 全部缺失，而不是随机零散坏文件。

## Devkit 与 official assert

使用已安装的 `/home/9/uq02279/.local/lib/python3.9/site-packages/nuscenes/tests/assert_download.py`，没有安装、升级或联网：

| root/version | devkit init | official assert | 第一条失败证据 |
|---|---|---|---|
| main / trainval | PASS，850 scenes / 34,149 samples | FAIL | 缺 `sweeps/RADAR_BACK_RIGHT/n015-...1531883531584120.pcd` |
| main / test | PASS，150 scenes / 6,008 samples | FAIL | 缺 `sweeps/RADAR_FRONT/n008-...1533153858046583.pcd` |
| repo / mini | FAIL | 未越过初始化 | 缺 `maps/53992ee3023e5494b90c316c183be829.png` |

前两个 init PASS 与 assert FAIL 不冲突：`NuScenes(...)` 初始化只建 metadata 索引，assert 才逐条检查文件。

## 其他 mini 候选

目录身份没有被当成证据；以下都实际解析了 `v1.0-mini/sample_data.json` 并逐引用检查：

| root | sample_data present | bytes | files / dirs | 判定 |
|---|---:|---:|---:|---|
| `/gs/bs/tga-RLA/qdeng/BEVFormer/data/nuscenes` | 31,206/31,206 | 5,900,841,499 | 31,242 / 31 | PRESENT |
| `/gs/bs/tga-RLA/qdeng/data-mini/nuscenes` | 31,206/31,206 | 5,900,827,159 | 31,241 / 32 | PRESENT |
| `/gs/bs/tga-RLA/qdeng/GenAD-archive/data-mini/nuscenes` | 31,206/31,206 | 5,900,827,159 | 31,241 / 32 | PRESENT |

三者都不是 symlink。是否为 byte-identical copy、hardlink 或应保留哪份仍 UNKNOWN，本轮不删不并。

repo mini 的 404 sample tokens 全属于 trainval；更直接的是上面已有三份完整 mini。因此 mini 缺口应靠路径选择/canonical layer 解决，不应再下载。

## 核心、当前任务、可选扩展

| 组件 | 资产状态 | nuScenes core | 当前 F0/F1 | 判定 |
|---|---|---|---|---|
| metadata tables / annotations / ego poses | PRESENT | 核心 | F1 需要 structured data | P0 PRESENT |
| maps + map expansion | PRESENT | planning/rendering 核心 | F1 需要 maps | P0 PRESENT |
| camera images + calibration | trainval PRESENT；test sweeps PARTIAL | sensor core | P1 F1 **禁止读取**；P2 才可用 | 非 P0 |
| lidar/radar | trainval 仅 RADAR_BACK_RIGHT sweeps PARTIAL；test radar sweeps MISSING | sensor core | 当前 F1 structured contract 不需要 | P2/P3 gap |
| CAN bus | PRESENT | expansion | 是否进入最终 F1 manifest UNKNOWN | P3 / UNKNOWN need |
| lidarseg | MISSING | optional expansion | 不需要 | P3 |
| panoptic | MISSING | optional expansion | 不需要 | P3 |

## 检查命令与 UNKNOWN

检查命令：两份 env 的只读 `source`；`realpath`/`stat`/有限深度结构；逐 JSON table/token/FK；逐 `sample_data` 的 `exists + size`；按 channel/area 汇总；PIL camera 抽样；lidar/radar binary read 抽样；devkit init；official assert；`du`/file/mtime 长扫通过 UGE。

raw 证据：`raw/nuscenes_references_20260910T092936+0900.jsonl`。其前两条来自失败后被纠正的 `8621409`，与后续 canonical test root/table 两条重复；最终判断使用后续完整记录。详细 filesystem 树见 `raw/filesystem_20260910T092936+0900.txt`。

仍为 UNKNOWN：缺失 nuScenes files 对应官方 blob archive 的当前压缩/解压大小；三份完整 mini 是否 byte-identical/hardlinked；未抽样文件的可解码性；主根全目录 file mtime 最终范围（filesystem job 在该阶段触发 2 小时 hard limit）；CAN bus 是否进入最终 F1 manifest。
