# A2｜nuScenes 原始数据集

- 审计时间：2026-09-09 17:15–17:31 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex
- 对象：`/gs/bs/tga-RLA/qdeng/data/nuscenes`

## 判定

**PASS_WITH_LIMITATION。** trainval/test 的核心 metadata、RGB samples、calibration、maps 与 sweeps 都存在，关键数量一致。全目录 total bytes/mtime 的逐文件扫描运行超过 10 分钟后主动终止，因此不报告虚假的总大小。

## 版本与表

| 版本 | samples | scenes | logs | calibrated sensors | sensors |
|---|---:|---:|---:|---:|---:|
| `v1.0-trainval` | 34,149 | 850 | 68 | 10,200 | 12 |
| `v1.0-test` | 6,008 | 150 | 15 | 1,800 | 12 |

trainval 的 34,149 samples 与 Sprint Plan 引用的官方约 34k 一致。

## 顶层结构

存在：`maps/`、`samples/`、`sweeps/`、`v1.0-trainval/`、`v1.0-test/`、`can_bus/`，以及若干预生成 info/database 文件。`v1.0-mini/` 不在此根目录。

`maps/` 递归 14 files，含四个 map raster、`basemap/`、`expansion/`、`prediction/`。

## 传感器文件数

`samples/` 是 trainval + test 共用目录。六路相机和 LIDAR_TOP 都是 40,157 files，恰好等于 34,149 + 6,008。

| Sensor | samples | sweeps |
|---|---:|---:|
| CAM_FRONT | 40,157 | 164,166 |
| CAM_FRONT_LEFT | 40,157 | 164,274 |
| CAM_FRONT_RIGHT | 40,157 | 163,881 |
| CAM_BACK | 40,157 | 161,453 |
| CAM_BACK_LEFT | 40,157 | 160,856 |
| CAM_BACK_RIGHT | 40,157 | 164,266 |
| LIDAR_TOP | 40,157 | 350,160 |
| RADAR_FRONT | 40,154 | 188,562 |
| RADAR_FRONT_LEFT | 40,155 | 191,133 |
| RADAR_FRONT_RIGHT | 40,155 | 190,435 |
| RADAR_BACK_LEFT | 40,155 | 185,467 |
| RADAR_BACK_RIGHT | 40,155 | 156,460 |

`sweeps/` 明显大于 `samples/`，符合完整 trainval/test 数据的结构预期。

## RGB 与 calibration

- `samples/CAM_*` 六路存在且各 40,157 images。
- `v1.0-trainval/calibrated_sensor.json` 存在，3,268,133 bytes / 10,200 records。
- `v1.0-test/calibrated_sensor.json` 存在，577,170 bytes / 1,800 records。
- `sensor.json` 两版本均存在，12 records。

因此 P2 在物理资产层面可以读取 source RGB/calibration；这不等于获准用于 P1。

## 与仓库内 `nuscenes-mini/` 的关系

`/gs/bs/tga-RLA/qdeng/RAP/nuscenes-mini/nuscenes` 是独立目录，不是 symlink。其 `v1.0-mini/sample.json` 有 404 records，这 404 个 sample token 全部包含在 full trainval 中；但本地 mini 六路 `samples/CAM_*` 与 `sweeps/CAM_*` 均为 **0 files**。它只有 lidar/radar 数据，不能作为现有 A4 renderer 默认路径所需的 RGB mini。
