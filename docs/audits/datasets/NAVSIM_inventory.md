# NAVSIM inventory

## 当前结论

状态：**FAIL（F0/F1 navtrain history 缺 46 个真实 camera 文件；processed 候选不满足 clean manifest）**。

- raw OpenScene `mini`、`trainval`、`test` 都有 metadata。逐 pickle/token 判定后，`mini` 和 `test` 的当前帧 camera 引用完整；full `trainval` 仅有 `1,219,960/5,852,075` 个引用存在，不能称为完整 trainval。
- 当前项目配置明确使用 `num_history_frames: 4`。官方 v2.2 `navtrain` 的 103,288 tokens 全能在 trainval metadata 中定位；当前帧 `826,304/826,304` 引用存在，4 帧历史去重后 `1,219,936/1,219,982` 存在。排除官方 `missing_camera/*.png` 的 2 个声明缺图后，实际缺 46 个文件。
- raw `test` 覆盖 v2.2 `navtest` 的 136 logs / 12,146 tokens，也包含 navhard/warmup 的 real tokens。训练不得读取它。
- `navhard_two_stage`、`warmup_two_stage`、`private_test_hard_two_stage` 的现有 metadata→blob 引用都完整。但本地 archive 已不在，官方也没有给解压内容 checksum，所以“确定为修复后的 v2.2 archive 内容”仍是 **UNKNOWN**，不能仅凭目录 mtime 宣布有效。
- `dataset_norm`、`dataset_aug`、`dataset_perturbed` 是 RAP processed derivative，不是 raw。`dataset_norm` 明确混入评测侧 tokens；`dataset_aug` 有 20 个不可读 pickle 且缺来源映射；`dataset_perturbed` 命中 navtrain/navmini，不能只靠目录名判断用途。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| 深扫节点 | `r14n1`, `r6n3`, `r5n6`, `r5n9` |
| Git SHA / 状态 | `b95916ffda6a08e603d43c20a8a3a684c53359f0` / `DIRTY`（盘点前 Human 资产未覆盖；本轮只新增报告） |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境来源 | 依次只读 `env/stage_a.env`、`env/sprint_20260909.env` |
| 安全环境值 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp`; `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |

敏感环境变量未记录。所有显式检查根均在 Lustre filesystem ID `25c66b0a00000000`；以下 realpath 与绝对路径相同，显式根都不是 symlink。

## 根、占用与结构

| 根 | 分类 | bytes / files / dirs | file mtime 范围 | 状态 |
|---|---|---:|---|---|
| `/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset` | NAVSIM/OpenScene raw + official filtered | `1,617,888,706,560` / `4,619,729` / `23,292` | `2023-02-01 17:56:38`–`2026-05-27 16:52:19 +0900` | PARTIAL |
| `/gs/bs/tga-RLA/qdeng/nuplan_dataset` | 独立 nuPlan raw；不是 NAVSIM split | `1,202,896,064,512` / `2,594,741` / `659` | `2022-05-05`–`2024-01-31` | PRESENT，内容级完整性 UNKNOWN |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm` | RAP processed derivative | `36,855,331,706` / `289,061` / `570`（dirs 不含 root） | `2026-05-27 16:27:59`–`2026-08-18 18:39:58 +0900` | PARTIAL / CONTAMINATED |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_aug` | RAP processed derivative | 当前 bytes/mtime UNKNOWN；既有盘点为 `4,524,750` files / `210,459` dirs | UNKNOWN：独立 UGE scan 取消前未完成该根 | PARTIAL |
| `/gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed` | RAP processed derivative | `3,585,639,951` bytes / `20,380` files / `176` dirs（既有审计值） | 当前 snapshot 的全量 mtime UNKNOWN | PARTIAL |

主 raw 根的深度 3 结构摘要：

```text
dataset/
├── maps/
├── mini_sensor_blobs/mini/                 # legacy duplicate candidate
├── navsim_logs/{mini,trainval,test,private_test_hard}/
├── sensor_blobs/{mini,trainval,test,private_test_e2e}/
├── navhard_two_stage/{openscene_meta_datas,sensor_blobs}/
├── warmup_two_stage/{openscene_meta_datas,sensor_blobs}/
└── private_test_hard_two_stage/{openscene_meta_datas,sensor_blobs}/
```

完整有限深度树和候选路径在 `raw/filesystem_20260910T092936+0900.txt`。候选发现也命中过 conda package/test fixture；这些没有被当数据根。`RAP/navsim_one_scene/dataset` 只有 3 files、0 pickle，是 fixture/cache，不是 raw split。

## raw downloaded split 完整性

以下身份来自实际 pickle/log/token，不来自目录名。每个有效 record 期望 8 路 camera；duplicate token/reference occurrences 都是 0，所有存在文件均非空。

| split | pickle / records | unique logs / tokens | metadata refs present/missing | missing rate | 实际 camera files | 额外文件 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| mini | `64 / 51,867` | `64 / 51,867` | `414,936 / 0` | `0%` | 每路 `51,867` | `51,867` 个 `MergedPointCloud/*.pcd` | PRESENT |
| trainval | `1,320 / 731,510` | `1,320 / 731,510` | `1,219,960 / 4,632,115` | `79.1534%` | 每路 `152,495` | `152,495` 个 PCD | PARTIAL |
| test | `147 / 75,122` | `147 / 75,122` | `600,976 / 0` | `0%` | 每路 `75,122` | `75,122` 个 PCD | PRESENT |
| private_test_hard metadata + 默认 blob root | `1 / 355` | `71 / 355` | `0 / 2,840` | `100%` | 0 | 0 | 路径不接合 |

`private_test_hard` 不是下载缺失：metadata 声明的 2,840 个引用在 `/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download/private_test_hard_navsim_sensor/private_test_hard` 全部存在且非空。它属于 path layout 问题；Human 批准前不建 symlink。

camera 解码只对 `.jpg/.png` 做抽样，避免把 `MergedPointCloud/*.pcd` 误交给 PIL。raw mini/trainval/test、三套 two-stage 以及三个 processed 根各抽 128 个实际 camera 文件，均为 0 decode errors。抽样 PASS 不代表全量内容 checksum PASS。

### 实际 camera files 按 channel

| source | B0 | F0 | L0 | L1 | L2 | R0 | R1 | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw mini | 51,867 | 51,867 | 51,867 | 51,867 | 51,867 | 51,867 | 51,867 | 51,867 |
| raw trainval | 152,495 | 152,495 | 152,495 | 152,495 | 152,495 | 152,495 | 152,495 | 152,495 |
| raw test | 75,122 | 75,122 | 75,122 | 75,122 | 75,122 | 75,122 | 75,122 | 75,122 |
| navhard_two_stage blob pool | 177,830 | 178,577 | 178,278 | 177,830 | 177,830 | 177,830 | 177,710 | 177,204 |
| warmup_two_stage | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 |
| private_test_hard_two_stage | 4,330 | 4,330 | 4,330 | 4,330 | 4,330 | 4,330 | 4,330 | 4,330 |
| dataset_norm 主 rendered root | 48 | 51,898 | 51,898 | 0 | 0 | 51,900 | 0 | 0 |
| dataset_aug | UNKNOWN | UNKNOWN | UNKNOWN | N/A（4-camera derivative） | N/A | UNKNOWN | N/A | N/A |
| dataset_perturbed | UNKNOWN | UNKNOWN | UNKNOWN | N/A（4-camera derivative） | N/A | UNKNOWN | N/A | N/A |

`dataset_norm` 另有 `rendered_sensor_blobs_4cam_v1`，所以 processed 根的总 JPG 是 288,993，不能把主 rendered root 的四路数直接当全根总数。aug/perturbed 的按 channel 全量计数随 combined deep timeout 保留 UNKNOWN；现有总 JPG 分别为 4,507,451 和 15,285。

## official v2.2 scene-filter 交集

官方对照来自 `/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim`，Git `a5f7110...`（`v2.2-17-ga5f7110`）。RAP 内的 `navtrain/navtest/navmini` membership 与该 checkout 相等；RAP 的 `warmup_test_e2e.yaml` 和 `private_test_e2e.yaml` 是旧定义，不拿它们代替 v2.2 two-stage 真值。

| source | navtrain logs/tokens | navtest | navmini | navhard_two_stage | warmup_two_stage | private hard two-stage |
|---|---:|---:|---:|---:|---:|---:|
| raw mini | `52/6,104` | `10/1,365` | `62/396` | `7/44` | `1/2` | `0/0` |
| raw trainval | `1,192/103,288` | `10/1,365` | `62/396` | `7/44` | `1/2` | `0/0` |
| raw test | `0/0` | `136/12,146` | `10/0` | `76/450` | `7/16` | `0/0` |
| raw private_test_hard | `0/0` | `0/0` | `0/0` | `0/0` | `0/0` | `0/140` |
| navhard_two_stage filtered | `0/0` | `0/0` | `0/0` | `0/5,462` | `0/204` | `0/0` |
| warmup_two_stage filtered | `0/0` | `0/0` | `0/0` | `0/204` | `0/204` | `0/0` |
| private_test_hard_two_stage filtered | `0/0` | `0/0` | `0/0` | `0/0` | `0/0` | `0/1,732` |
| processed norm | `52/6,104` | `10/1,365` | `62/396` | `7/44` | `1/2` | `0/0` |
| processed aug | `0/0` | `0/0` | `0/0` | `0/0` | `0/0` | `0/0` |
| processed perturbed | `0/5,095` | `0/0` | `0/320` | `0/0` | `0/0` | `0/0` |

v2.2 filter 本身的规模：navtrain `1,192 logs / 103,288 tokens`；navtest `136/12,146`；navmini `62/396`；navhard `76 real logs / 450 real + 5,462 synthetic tokens`；warmup `7 / 16 + 204`；private hard `0 / 140 real + 1,732 synthetic`。

`processed_aug` 的当前变换后 token 与官方 filter 交集为 0，但其 source-token provenance 不存在，因此不能推论“无 split 污染”。

## official filtered split 引用完整性与 v2.2 风险

| filtered root | pickle / records / unique tokens | refs present/missing | 实际 camera files | duplicate / zero | metadata aggregate SHA-256 | 判定 |
|---|---:|---:|---:|---:|---|---|
| navhard_two_stage | `2,731 / 13,655 / 13,655` | `109,240 / 0` | `1,423,089` | `0 / 0` | `4210f441b1d0376fc878cbfdced2ec0ad1d01f66f5126e0526d5cb0bb29a005a` | 引用 PASS；v2.2 provenance UNKNOWN |
| warmup_two_stage | `102 / 510 / 510` | `4,080 / 0` | `4,080` | `0 / 0` | `b02bf7caeca1bdfa83a4b3a7c6e3daedfa02c05d49ca577d0dadb33a9d146716` | 引用 PASS；v2.2 provenance UNKNOWN |
| private_test_hard_two_stage | `866 / 4,330 / 4,330` | `34,640 / 0` | `34,640` | `0 / 0` | `3db3832ed7dc2071a0031799ea69b09598176aee9e44ad434a5ecd1f3638f470` | 引用 PASS；v2.2 provenance UNKNOWN |

navhard blob pool 比 filtered metadata 大：1,423,089 camera files 中只有 109,240 被当前 metadata 引用，另有 164,848 个非 camera files。这里的 `extra` 表示“未被该 filtered manifest 引用”，不直接等价于损坏。

官方 v2.2 changelog 明确修复 navhard/warmup metadata。本地只保留解压内容，未找到对应 archive；上表 hash 只能作为今后对比基线，不能与官方 archive MD5 等价。

## navtrain history 需求与补全方案

配置搜索确认当前相关 scene filters 和 loader 使用 `num_history_frames: 4`，所以本项目需要历史帧，不是 UNKNOWN。

| 方案 | 证据 | 缺口 | 本轮建议 |
|---|---|---:|---|
| with history | 103,288 target tokens 全命中；unique refs `1,219,982` | 46 个真实 camera files；3 logs、8 channels | **有效方案**：先定位缺文件所属官方 shard，再在新 staging 定向补全；不覆盖现根 |
| without history | current-only refs `826,304/826,304` | 0 | 仅数据上可满足；但会改变已确认的模型语义/配置，本轮无权采用 |

46 个物理缺口来自 80 次历史引用，涉及 10 个 current tokens、6 个 history tokens 和 3 个 logs：

- `2021.06.08.12.54.54_veh-26_04262_04732.pkl`
- `2021.06.08.14.35.24_veh-26_02555_03004.pkl`
- `2021.06.14.17.26.26_veh-38_04544_04920.pkl`

按 channel 的真实缺口：`CAM_B0/F0/L1/L2/R0/R1` 各 6，`CAM_L0/R2` 各 5。另有两条 `missing_camera/*.png` 是 metadata 主动声明缺图，不列为下载缺口。具体属于 `navtrain_history_{1..4}.tgz` 哪个 shard仍是 UNKNOWN。

## maps 与 archive

`NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`，realpath 同左，不是 symlink；`2,853,234,984` bytes、14 files、19 dirs。存在根级和子目录级 `nuplan-maps-v1.0.json`，环境值为 `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`。路径/版本 PASS；没有官方 manifest 对全部 map 内容逐项复核，所以内容级完整性 UNKNOWN。

没有发现 `navtrain_current_{1..4}.tgz` 或 `navtrain_history_{1..4}.tgz`。因此官方 MD5 检查状态是 **NOT_AVAILABLE**，绝不对解压目录伪造 archive MD5 结论。发现的 `openscene_sensor_test_camera_1.tgz.1` 只有 `3,370,355,258` bytes，是不完整 test archive fragment，不是 navtrain。

官方 `download_navtrain_aws.sh` 中的期望 MD5 仅作为未来 staging 校验依据：

| archive | expected MD5 | archive | expected MD5 |
|---|---|---|---|
| current_1 | `6f92f38d5f03ed852da7872a7122bdd2` | history_1 | `dc46ed34d92d5ab9cc1464d67b72fbf6` |
| current_2 | `7a72f0a758b5df6cbe4c565920a4869f` | history_2 | `fab177bdb79c0c9536da1566d13e5995` |
| current_3 | `b083fce1428308abb5682a1a150cc1af` | history_3 | `71ed9a2387edc3849921861d7873c7f0` |
| current_4 | `68354ac2c993aa1ebbfac59547fdb840` | history_4 | `2cc13aced2f458e50fe4cc2f26d18e07` |

这些值没有与任何本地 navtrain archive 比对，因为对应 archive 不存在。

## RAP processed derivative

| root | pickle / records | unique logs / tokens | raster/files | 引用或来源判定 |
|---|---:|---:|---:|---|
| dataset_norm | `64 / 51,867` | `64 / 51,867` | `288,993 jpg`；metadata 对主 rendered root `155,649 present / 259,287 missing` | missing rate `62.4884%`；明确命中 navtest/navhard/warmup，CONTAMINATED |
| dataset_aug | `17,299 / 14,589,551` | `17,277 / 936,674` | `4,507,451 jpg` | 20 pickle EOF/truncated；source provenance UNKNOWN；全量 metadata→raster missing/duplicate/extra 因 deep job 超时仍 UNKNOWN |
| dataset_perturbed | `5,095 / 71,330` | `5,095 / 13,138` | `15,285 jpg` | 命中 navtrain 5,095、navmini 320 tokens；全量 metadata→raster missing/duplicate/extra仍 UNKNOWN |

这些目录内容没有被修改。processed 缺配对、污染或处理未完成时，修复动作应是冻结合法 manifest 后重跑 RAP processing；不能写成“重新下载 raw 就能修好”。

## 检查命令、长扫与 UNKNOWN

检查命令类型：两份 env 的只读 `source`；`realpath/stat/find -maxdepth`；pickle/YAML/token 全量解析；camera path `exists+size`；filter 交集；PIL 128-file 抽样；maps/version；archive 定向搜索；UGE 上 `du -sb`、file/dir/mtime 与引用深扫。没有运行 H0、Smoke Test、训练、评测、下载、解压、移动、复制、删除或 symlink。

raw 证据：`raw/navsim_references_20260910T092936+0900.jsonl` 和 `raw/filesystem_20260910T092936+0900.txt`。主 NAVSIM deep job `8621392` 在 2 小时 hard limit 被杀，已完成到 `dataset_norm` 的结果有效；`dataset_aug/perturbed` 未完成的全量引用指标严格保留 UNKNOWN。

仍为 UNKNOWN：46 个 history 文件的 archive shard；two-stage 解压物的官方 v2.2 provenance；maps 的官方全量内容 manifest；legacy mini pool 与 active mini 是否 byte-identical/hardlinked；processed aug/perturbed 的全量引用 missing/duplicate/extra 与 source-token provenance；`8623456` 取消前未产出的 aug 当前 bytes/mtime。
