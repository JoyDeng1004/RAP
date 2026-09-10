# Dataset path layout 方案

## 当前结论

状态：**PASS（方案可审核）/ BLOCKED（未获 Human 路径批准）**。

第一轮只建立逻辑视图最稳妥：保留现有数据原位，在 `/gs/bs/tga-RLA/qdeng/datasets/` 建 canonical symlink layer。**本轮没有执行 `mv`、`cp`、`rm`，也没有创建或替换任何数据 symlink。** symlink 方案不复制数据，新增空间只包含目录项，预计低于 1 MiB；真实 inode/配额开销由建链时文件系统决定。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| Git SHA / 状态 | `b95916ffda6a08e603d43c20a8a3a684c53359f0` / `DIRTY`（盘点前资产，未覆盖） |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp`; `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |

敏感环境变量未写入。所有对象当前位于 Lustre filesystem ID `25c66b0a00000000`；这不代表未来 canonical 根一定仍在同一 filesystem，建链前必须复查。

## existing path → canonical path

| existing path | canonical path | 类型 | 完整性 | 建议 | 风险 | 预计新增空间 | 回滚 |
|---|---|---|---|---|---|---:|---|
| `/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps` | `datasets/raw/navsim/maps` | raw maps | PRESENT；版本值 `nuplan-maps-v1.0`，内容级完整性仍 UNKNOWN | keep + symlink | 版本文件存在不等于全部 map blob 完整 | symlink，约 0 data bytes | 删除新 symlink；原目录不变 |
| `.../dataset/navsim_logs/{mini,trainval,test}` | `datasets/raw/navsim/openscene/navsim_logs/{mini,trainval,test}` | raw OpenScene metadata | 三个 split metadata 存在；身份由 pickle/YAML 判定，不靠目录名 | keep + per-split symlink | `trainval` sensor 实体只覆盖约 20.85% 全量引用 | 同上 | 同上 |
| `.../dataset/sensor_blobs/{mini,trainval,test}` | `datasets/raw/navsim/openscene/sensor_blobs/{mini,trainval,test}` | raw OpenScene blobs | mini/test 对 metadata 引用完整；trainval PARTIAL | keep + per-split symlink | 把 PARTIAL 根标成 full trainval 会误导 | 同上 | 同上 |
| `.../dataset/navhard_two_stage` | `datasets/raw/navsim/two_stage/navhard_two_stage` | official filtered eval | 引用完整；v2.2 provenance 仍需 archive/checksum 证实 | keep + symlink，名称不改 | 评测数据误入训练；旧 metadata 风险 | 同上 | 同上 |
| `.../dataset/warmup_two_stage` | `datasets/raw/navsim/two_stage/warmup_two_stage` | official filtered challenge | 引用完整；v2.2 provenance 仍需证实 | keep + symlink | 同上 | 同上 | 同上 |
| `.../dataset/private_test_hard_two_stage` | `datasets/raw/navsim/two_stage/private_test_hard_two_stage` | official filtered challenge | 引用完整性等 deep 结果；版本 provenance UNKNOWN | keep + symlink | 许可证/比赛数据隔离；绝不能训练读取 | 同上 | 同上 |
| `/gs/bs/tga-RLA/qdeng/data/nuscenes` | `datasets/raw/nuscenes` | raw nuScenes | v1.0-trainval/test metadata 存在，但 sample_data 均 PARTIAL | keep + symlink | devkit init 可过，不能误写成完整下载 | 同上 | 同上 |
| `/gs/bs/tga-RLA/qdeng/data-mini/nuscenes` | `datasets/raw/nuscenes-mini`（若 Human 选择保留独立 mini） | raw nuScenes mini | PRESENT：31,206/31,206 sample_data 非空 | keep；是否建额外 symlink 待 Human 决定 | 与 canonical full root 分叉；三份独立副本浪费空间 | 同上 | 同上 |
| `/gs/bs/tga-RLA/qdeng/BEVFormer/data/nuscenes` | 不进入 canonical；登记 manifest | raw + processed metadata 混合 | mini 引用 PRESENT；独立副本 | keep | 与 `data-mini` 内容接近但不是 symlink，误删风险 | 0 | 无操作 |
| `/gs/bs/tga-RLA/qdeng/GenAD-archive/data-mini/nuscenes` | 不进入 canonical；登记 manifest | raw + processed metadata 混合/archive workspace | mini 引用 PRESENT；独立副本 | keep | archive workspace 归属和保留期 UNKNOWN | 0 | 无操作 |
| `/gs/bs/tga-RLA/qdeng/RAP/nuscenes-mini/nuscenes` | 不进入 canonical；标记 incomplete fixture | raw 副本/fixture | PARTIAL：metadata 有 31,206 引用，但 required blobs/maps 缺失 | keep | 当前工具默认指向这里，会得到错误路径 | 0 | 无操作 |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm` | `datasets/processed/navsim/rap/norm` | RAP processed derivative | PARTIAL / CONTAMINATED | keep + symlink 仅供审计；训练默认不得注册 | 评测 split 污染、raster 引用缺失 | symlink，约 0 data bytes | 删除新 symlink |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_aug` | `datasets/processed/navsim/rap/aug` | RAP processed derivative | split 交集和配对完整性由本次 deep 报告判定 | keep + symlink 仅供审计 | 数百万文件；来源/provenance 不完备 | 同上 | 同上 |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed` | `datasets/processed/navsim/rap/perturbed` | RAP processed derivative | split 交集和配对完整性由本次 deep 报告判定 | keep + symlink 仅供审计 | 同上 | 同上 | 同上 |
| 目前未确认统一根 | `datasets/processed/nuscenes/rap/{raster,cache}` | processed/cache | MISSING/UNKNOWN | 只预留逻辑目录；本轮不创建 | processed 缺失不能靠重下 raw 修复 | 0 | 无操作 |
| `/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/openscene_sensor_test_camera_1.tgz.1` | `datasets/archives/navsim/` | incomplete archive fragment | PARTIAL，3,370,355,258 bytes；不是 navtrain archive | keep；登记但不建 canonical 链 | 文件名 `.tgz.1` 暗示未完成，不能当完整 archive | 0 | 无操作 |
| 未来 manifest | `datasets/manifests/` | manifest | MISSING | 获批后创建文本 manifest，不搬数据 | manifest 与物理数据漂移 | 很小 | Git/快照恢复 |

`/gs/bs/tga-RLA/qdeng/nuplan_dataset` 是独立 nuPlan raw 根，不等同于 NAVSIM/OpenScene。本方案保留原位，不把它塞进 `raw/navsim/openscene`。

`/gs/bs/tga-RLA/qdeng/RAP/navsim_one_scene/dataset` 只有 3 files、0 pickle，属于 523,774,224-byte fixture/cache 候选，不作为 raw split。

## 建议逻辑结构

```text
/gs/bs/tga-RLA/qdeng/datasets/
├── raw/
│   ├── navsim/
│   │   ├── maps -> existing maps
│   │   ├── openscene/
│   │   │   ├── navsim_logs/{mini,trainval,test} -> existing split dirs
│   │   │   └── sensor_blobs/{mini,trainval,test} -> existing split dirs
│   │   └── two_stage/{navhard_two_stage,warmup_two_stage,private_test_hard_two_stage} -> existing dirs
│   └── nuscenes -> /gs/bs/tga-RLA/qdeng/data/nuscenes
├── processed/
│   ├── navsim/rap/{norm,aug,perturbed} -> RAP/dataset_*
│   └── nuscenes/rap/{raster,cache}
├── manifests/
└── archives/
```

## 执行门和 UNKNOWN

Human 批准前不创建目录或 symlink。批准后也应先生成 manifest，逐条核对目标不存在，再只用 `ln -s` 建新链接。任何现有 canonical target 已存在时必须停止，不能替换。

当前 UNKNOWN：canonical 根的 ownership/ACL/quota；三份完整 mini 副本是否允许合并；two-stage 解压内容是否可由保留 archive 的官方 checksum 复证；processed nuScenes 的真实规划路径；各 symlink 最终 owner/group。UNKNOWN 不阻止评审，但阻止执行。

## 检查命令

只读命令包括：`source` 两份 env、`find /gs/bs/tga-RLA/qdeng ...`、`realpath`、`stat`、`find -maxdepth 3`、UGE 上的 `du -sb`/file-dir-mtime scan、metadata/token 解析。没有运行下载、移动、复制、删除、symlink、H0、Smoke Test、训练或评测。
