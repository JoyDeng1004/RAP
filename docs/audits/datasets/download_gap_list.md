# Dataset 补全清单

## 当前结论

状态：**FAIL（存在 P0/P1 缺口）**。P0 不是“重下全部数据”：当前 `navtrain` 只缺 46 个 history camera 文件；更大的阻塞来自 RAP processed 数据污染、缺配对和来源不清。nuScenes trainval 只缺 32,605 个 `RADAR_BACK_RIGHT` sweep，当前 P1 F1 明确只读 structured metadata/maps/labels，因此这项不是 F1 的 P0。

本页只给清单和命令，**没有下载、安装、解压或覆盖任何文件**。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| Git SHA / 状态 | `b95916ffda6a08e603d43c20a8a3a684c53359f0` / `DIRTY`（盘点前资产，未覆盖） |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp`; `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |

敏感环境变量未写入。

## P0–P3 清单

| dataset | component | split/version | current_path | expected evidence | observed count/size | status | required_for | official source URL/script | estimated download size | estimated extracted size | license/account requirement | priority | notes |
|---|---|---|---|---|---:|---|---|---|---:|---:|---|---|---|
| NAVSIM | camera history blobs | navtrain / trainval | `$OPENSCENE_DATA_ROOT/sensor_blobs/trainval` | 103,288 YAML tokens；4-frame history 引用非空 | current `826,304/826,304`；history unique `1,219,936/1,219,982`；缺 46；另有 2 个 `missing_camera/*` 声明，不属于下载缺口 | PARTIAL | F0/F1 train | [NAVSIM splits](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md)；`.../download/download_navtrain_aws.sh` | 定向补 46：UNKNOWN；全包官方表为 445 GB with history / 300 GB without | UNKNOWN | NAVSIM/OpenScene license；是否需账号 UNKNOWN | **P0** | 46 个文件分布在 3 个 logs、8 camera channels；10 个 current tokens 产生 80 次引用，去重后排除 2 个声明缺图即 46 个物理缺口。archive shard UNKNOWN，先向 manifest/发布方定位，别盲下 445 GB。 |
| NAVSIM | clean legal RAP processed target | navtrain-derived | `RAP/dataset_norm` 等候选 | manifest 只命中 navtrain，评测交集 0；raster/token 全配对 | `dataset_norm` 命中 navtest 10 logs/1,365 tokens、v2.2 navhard 7/44、warmup 1/2；raster 缺 `259,287/414,936` | PARTIAL | F0/F1 train | N/A：应从合法 manifest 重跑 RAP processing | N/A | UNKNOWN | N/A | **P0** | **不是 raw 下载问题**；属于 split 污染 + RAP processing 未完成。 |
| NAVSIM | augmented metadata/raster | RAP processed | `RAP/dataset_aug` | 所有 pickle 可读；来源 manifest 和 token 映射可追溯 | 17,299 pickle，其中 20 个 EOF/truncated；14,589,551 records；camera jpg 4,507,451 | PARTIAL | F0/F1 train candidate，实际是否采用 UNKNOWN | N/A：修复/重跑 RAP processing | N/A | UNKNOWN | N/A | **P0** | **不是 raw 下载问题**；v2.2 split 交集按当前变换后 token 为 0，但原始来源映射缺失，不能据此宣布无污染。 |
| NAVSIM | maps | `nuplan-maps-v1.0` | `$NUPLAN_MAPS_ROOT` | version JSON + 四地 map roots | PRESENT；内容级全量校验 UNKNOWN | UNKNOWN | F0/F1 train、navtest、navhard | `.../download/download_maps.sh` | UNKNOWN | UNKNOWN | nuPlan map license/terms | **P0** | 路径和版本存在；当前没有官方 archive checksum 能证明全量完整。 |
| nuScenes | structured metadata/maps/labels | v1.0-trainval | `/gs/bs/tga-RLA/qdeng/data/nuscenes` | scene/sample/annotation/ego/maps；FK 0 missing | 850 scenes；34,149 samples；1,166,187 annotations；2,631,083 ego poses；FK 全 0 missing；maps present | PRESENT | F1 train | [nuScenes download](https://www.nuscenes.org/nuscenes#download) | 0 | 0 | 需 nuScenes 账号并接受 Terms | **P0** | 当前 Sprint 的 P1 F1 禁止读 source RGB/calibration，只允许 structured data。 |
| NAVSIM | raw test OpenScene | test/navtest | `$OPENSCENE_DATA_ROOT/{navsim_logs,sensor_blobs}/test` | navtest YAML 136 logs/12,146 tokens；metadata 引用非空 | raw test 147 pickle/75,122 tokens；600,976/600,976 camera refs；v2.2 navtest 136/12,146 全命中 | PRESENT | navtest | [OpenScene test script](https://github.com/autonomousvision/navsim/blob/main/download/download_test.sh) | 1 GB logs + 217 GB sensors（官方表） | UNKNOWN | OpenScene terms；账号要求 UNKNOWN | **P1** | 不得用于训练。 |
| NAVSIM | two-stage metadata + blobs | navhard_two_stage v2.2 | `$OPENSCENE_DATA_ROOT/navhard_two_stage` | v2.2 filter 76 logs、450 real tokens、5,462 synthetic tokens；refs 完整 | 2,731 pickle；13,655 records；109,240/109,240 refs；camera-only 1,423,089 files；sample decode 128/128 | UNKNOWN | navhard | [v2.2 script](https://github.com/autonomousvision/navsim/blob/main/download/download_navhard_two_stage.sh) | 892 MB logs + 31 GB sensors（官方表） | UNKNOWN | NAVSIM/OpenScene terms；账号要求 UNKNOWN | **P1** | 内容完整性 PASS；archive 已删除且官方未给本地内容 checksum，故“确为修复版 v2.2”仍 UNKNOWN。现有 metadata aggregate SHA-256 已记录。 |
| NAVSIM | raw private-test blobs 的路径接合 | private_test_hard | metadata 在 `$OPENSCENE_DATA_ROOT/navsim_logs/private_test_hard`；blob 在 `navsim_workspace/navsim/download/private_test_hard_navsim_sensor/private_test_hard` | 355 records × 8 camera refs | metadata 根单独看缺 2,840/2,840；对 staging blob 根则 2,840/2,840 present | PARTIAL | challenge | [private hard script](https://github.com/autonomousvision/navsim/blob/main/download/download_private_test_hard_two_stage.sh) | 0（数据已在别处） | 0 | 比赛/数据条款；账号要求 UNKNOWN | **P1** | **不是下载缺口**；属于 path layout 未接合。Human 批准前不建 symlink。不得训练读取。 |
| NAVSIM | two-stage metadata + blobs | warmup_two_stage v2.2 | `$OPENSCENE_DATA_ROOT/warmup_two_stage` | v2.2 filter 7 logs、16 real tokens、204 synthetic tokens | 102 pickle；510 records；4,080/4,080 refs；sample decode 128/128 | UNKNOWN | challenge/warmup | [v2.2 script](https://github.com/autonomousvision/navsim/blob/main/download/download_warmup_two_stage.sh) | 27 MB logs + 1.2 GB sensors（官方表） | UNKNOWN | NAVSIM/OpenScene terms；账号要求 UNKNOWN | **P2** | 内容完整；v2.2 provenance 仍 UNKNOWN。不得训练读取。 |
| NAVSIM | two-stage metadata + blobs | private_test_hard_two_stage v2.2 | `$OPENSCENE_DATA_ROOT/private_test_hard_two_stage` | v2.2 filter 140 real + 1,732 synthetic tokens | 866 pickle；4,330 records；34,640/34,640 refs；sample decode 128/128 | UNKNOWN | challenge/private test | [v2.2 script](https://github.com/autonomousvision/navsim/blob/main/download/download_private_test_hard_two_stage.sh) | 14 MB logs + 11 GB sensors（官方表） | UNKNOWN | 比赛/数据条款；账号要求 UNKNOWN | **P2** | 内容完整；v2.2 provenance 仍 UNKNOWN。不得训练读取。 |
| nuScenes | missing radar sweep blobs | v1.0-trainval | `/gs/bs/tga-RLA/qdeng/data/nuscenes/sweeps/RADAR_BACK_RIGHT` | 189,065 referenced sweeps | 156,460 present；32,605 missing；0 zero-byte | PARTIAL | full nuScenes / P2；当前 P1 F1 不需要 sensor blobs | [nuScenes download](https://www.nuscenes.org/nuscenes#download) | UNKNOWN | UNKNOWN | 需账号 + Terms | **P2** | official `assert_download.py` 因首个缺失 RADAR_BACK_RIGHT 文件 FAIL。应在新 staging 解压对应 trainval sensor-blob archive，不覆盖现根。 |
| nuScenes | source RGB/calibration path | v1.0-trainval | main root | 六 camera samples/sweeps 可读；calibration 存在 | camera samples/sweeps 引用全 present；抽样 128/128 可解码 | PRESENT | P2 X-series only | [nuScenes download](https://www.nuscenes.org/nuscenes#download) | 0 | 0 | 需账号 + Terms | **P2** | P1 F1 明确禁止读取；存在不等于获准使用。 |
| NAVSIM | complete OpenScene sensor archive | full trainval（不是 navtrain subset） | `$OPENSCENE_DATA_ROOT/sensor_blobs/trainval` | 全 1,320 logs 的 5,852,075 camera refs | 1,219,960 present；4,632,115 missing；coverage 20.8466% | PARTIAL | optional/full OpenScene | [OpenScene trainval script](https://github.com/autonomousvision/navsim/blob/main/download/download_trainval.sh) | >2,000 GB sensors + 14 GB logs（官方表） | UNKNOWN | OpenScene terms；账号要求 UNKNOWN | **P3** | 当前 F0/F1 只需 navtrain；不能拿 full-trainval 缺失率替代 navtrain 定向完整性。 |
| NAVSIM | raw mini | mini | `$OPENSCENE_DATA_ROOT/{navsim_logs,sensor_blobs}/mini` | metadata 引用完整 | 64 pickle/51,867 tokens；414,936/414,936 refs；sample decode 128/128 | PRESENT | optional | [OpenScene mini script](https://github.com/autonomousvision/navsim/blob/main/download/download_mini.sh) | 1 GB logs + 151 GB sensors（官方表） | UNKNOWN | OpenScene terms；账号要求 UNKNOWN | **P3** | 完整性来自 metadata 引用，不来自目录名。另有 162,321,792,361-byte legacy 副本候选。 |
| nuScenes | test camera/radar sweeps | v1.0-test | main root | 462,901 sample_data 全存在 | 124,519 present；338,382 missing：六 camera sweeps + 五 radar sweeps全缺；samples 和 LIDAR sweeps完整 | PARTIAL | optional nuScenes test | [nuScenes download](https://www.nuscenes.org/nuscenes#download) | UNKNOWN | UNKNOWN | 需账号 + Terms | **P3** | devkit init PASS 只证明 metadata 可加载；official assert FAIL。 |
| nuScenes | usable mini | v1.0-mini | repo copy不完整；另有 `data-mini/nuscenes`、`BEVFormer/data/nuscenes`、`GenAD-archive/data-mini/nuscenes` | 31,206 sample_data + maps | 三个候选均 31,206/31,206 nonempty；repo copy 17,198 present、14,008 camera files missing且 map mask缺失 | PRESENT | optional / renderer development | [nuScenes download](https://www.nuscenes.org/nuscenes#download) | 0（已有完整副本） | 0 | 已有数据仍受 Terms 约束 | **P3** | **不要重新下载**；先由 Human 选择 canonical 完整副本。repo 默认路径问题属于 path/config，不是 raw 缺失。 |
| nuScenes | lidarseg | v1.0-trainval/test/mini | main root | `lidarseg/` + lidarseg metadata | 未发现 | MISSING | optional；当前 F0/F1 不需要 | [nuScenes lidarseg](https://www.nuscenes.org/nuscenes#download) | UNKNOWN | UNKNOWN | 需账号 + Terms | **P3** | 语义分割扩展，不是 core detection/planning metadata。 |
| nuScenes | panoptic | v1.0-trainval/mini | main root | `panoptic/` + panoptic metadata | 未发现 | MISSING | optional；当前 F0/F1 不需要 | [nuScenes panoptic](https://www.nuscenes.org/nuscenes#download) | UNKNOWN | UNKNOWN | 需账号 + Terms | **P3** | 可选扩展。 |
| nuScenes | CAN bus | expansion | main root `can_bus/` | CAN JSON/README | PRESENT | PRESENT | optional；当前 F1 是否消费 UNKNOWN | [nuScenes CAN bus](https://www.nuscenes.org/nuscenes#download) | 0 | 0 | 需账号 + Terms | **P3** | “目录存在”只证明资产在；配置未冻结，任务需要性保留 UNKNOWN。 |

## 仅供批准后执行的下载命令

所有命令都必须在新 staging 目录运行，不能在现有数据根运行。下面只记录，不执行：

```bash
# NAVSIM navtrain：官方脚本会下载、解压并删除 archive。
# 先复制/审查脚本并把工作目录改到新的 staging；不要直接指向现有根。
bash /gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/download_navtrain_aws.sh

# 完整 OpenScene split（P3，体量 >2 TB；本轮不建议执行）
bash /gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/download_trainval.sh

# v2.2 two-stage，仅当 checksum/provenance 复核要求重取时使用
bash /gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/download_navhard_two_stage.sh
bash /gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/download_warmup_two_stage.sh
bash /gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/download_private_test_hard_two_stage.sh
```

nuScenes 没有可安全公开的匿名直链：先登录官方 download page、接受 Terms，人工选择缺失版本对应的 sensor blob archive，再下载到新 staging。下载大小和解压大小当前均为 UNKNOWN，不伪造数值。

## 检查命令与 UNKNOWN

检查用了 metadata/YAML/token 解析、逐引用 `exists + size`、official `assert_download.py`、devkit init、camera/lidar/radar 128-file 抽样、`find`/`realpath` 和 UGE 深扫。没有升级环境或联网安装。

仍为 UNKNOWN：46 个 NAVSIM history 文件属于 `navtrain_history_{1..4}.tgz` 的哪个 shard；two-stage 解压物是否可由保留 archive checksum 复证；nuScenes 缺失 archive 的官方当前压缩/解压大小；CAN bus 是否进入最终 F1 manifest。
