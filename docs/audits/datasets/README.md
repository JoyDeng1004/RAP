# NAVSIM / nuScenes dataset audit — 2026-09-10

## 当前结论

总状态：**FAIL（存在 P0 数据/processed 合规缺口）**。只读盘点和重构方案已完成；没有下载、解压、移动、复制、删除、覆盖、建 symlink、改训练配置，也没有运行 H0、Smoke Test、训练或评测。

| 数据集 | 实际拥有 | 明确完整 | 明确缺失/不完整 | UNKNOWN |
|---|---|---|---|---|
| NAVSIM/OpenScene raw | metadata `mini/trainval/test`；blobs `mini/trainval/test`；private challenge 数据分居两根 | mini 当前帧 `414,936/414,936`；test `600,976/600,976`；v2.2 navtrain current-only `826,304/826,304` | full trainval 缺 `4,632,115/5,852,075` refs；当前 4-history navtrain 缺 46 个真实 camera files | 46 files 的 archive shard；two-stage 是否可复证为 v2.2；maps 内容级 completeness |
| NAVSIM official filtered | `navhard_two_stage`, `warmup_two_stage`, `private_test_hard_two_stage` | 三者 metadata→blob 分别 `109,240/109,240`、`4,080/4,080`、`34,640/34,640` | 无已证实引用缺失 | archive 不在，无法把本地 hash 对上官方 archive |
| RAP processed NAVSIM | `dataset_norm`, `dataset_aug`, `dataset_perturbed` | 只确认若干已有 raster 可解码抽样 PASS | norm 污染且缺 `259,287/414,936` 主 rendered refs；aug 20 corrupt pickle | aug/perturbed 全量配对和来源 provenance |
| nuScenes main | `v1.0-trainval`, `v1.0-test`，maps、can_bus | trainval structured tables/FK/maps；test metadata；devkit init 两版本 PASS | trainval 缺 32,605 RADAR_BACK_RIGHT sweeps；test 缺 338,382 camera/radar sweeps；official assert 两版本 FAIL | 缺失 archive 的当前下载/解压大小；未抽样文件可解码性 |
| nuScenes mini | repo copy不完整；另有 3 个完整独立根 | 三个候选各 `31,206/31,206` refs | repo copy 缺 14,008 camera files和 map mask | 三份是否 byte-identical/hardlinked、最终 canonical 选择 |

目录名从未被当作 split 身份。NAVSIM 用 pickle/YAML 的 log/token 交集判定，nuScenes 用 version JSON tables、sample_data 引用和 token 外键判定。raw download、official filtered split、RAP processed derivative 在报告里分开记录。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| Git SHA | `b95916ffda6a08e603d43c20a8a3a684c53359f0` |
| Git 状态 | `DIRTY`；盘点前已有 Human 资产，详见下文；本轮只新增/更新本审计目录 |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境来源 | 依次只读取 `env/stage_a.env`、`env/sprint_20260909.env` |
| 安全环境值 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp`; `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |

敏感环境变量没有写进报告或 raw snapshot。

盘点开始前的 dirty 项：

```text
 M docs/20260909/2026-09-09_Sprint_Plan.md
A  tools/render_nuscenes_camera_cross.py
?? .DS_Store
?? .pnpm-store/
?? ckpts/
?? docs/.DS_Store
?? docs/slides/
?? env/
?? exp0-tl-audit.o8260654
?? exp0-tl-audit.o8260684
?? nuscenes-mini/
?? tools/md2pdf/
```

这些均视为 Human/既有资产，没有被覆盖或清理。

## 报告导航

- [NAVSIM_inventory.md](NAVSIM_inventory.md)：raw、official filtered、processed、v2.2 split 交集、history、maps、archive。
- [nuScenes_inventory.md](nuScenes_inventory.md)：主 trainval/test、repo mini、三个完整 mini 候选、sample_data/FK/channel/devkit/assert。
- [download_gap_list.md](download_gap_list.md)：按 P0–P3 的补全清单、官方来源和只记录不执行的命令。
- [path_layout.md](path_layout.md)：existing → canonical 映射、风险、空间和回滚；未执行任何路径变更。
- [maintenance.md](maintenance.md)：fast/deep 维护契约和 `CP-CODE-DS-01` review card；脚本尚未创建。
- [raw/inventory_20260910T092936+0900.jsonl](raw/inventory_20260910T092936+0900.jsonl)：结论级机器可读 snapshot。
- [raw/filesystem_20260910T092936+0900.txt](raw/filesystem_20260910T092936+0900.txt)：路径发现、有限树和 filesystem 长扫原始输出；该 job 超时，产物为明确标注的 partial snapshot。
- [raw/navsim_references_20260910T092936+0900.jsonl](raw/navsim_references_20260910T092936+0900.jsonl)：NAVSIM metadata/ref 与补充结论。
- [raw/nuscenes_references_20260910T092936+0900.jsonl](raw/nuscenes_references_20260910T092936+0900.jsonl)：nuScenes tables/FK/ref/devkit/assert。

## P0–P3 摘要

| priority | 结论 |
|---|---|
| P0 | NAVSIM 4-history navtrain 定向补 46 个真实 camera files；冻结只含合法 navtrain 的 processed manifest 并重跑/修复 RAP processing；maps 路径存在但内容级证明仍 UNKNOWN。nuScenes structured trainval/maps 已 PRESENT。 |
| P1 | NAVSIM raw test/navtest 已 PRESENT；navhard 引用完整但 v2.2 provenance UNKNOWN；private hard raw blobs 已在 staging，只需未来获批后的路径接合，不需下载。 |
| P2 | warmup/private two-stage 引用完整但 provenance UNKNOWN；nuScenes trainval 的 32,605 个 radar sweeps 只影响 full sensor 使用，不是当前 structured F1 P0。 |
| P3 | full OpenScene trainval 大缺口、nuScenes test sensor sweeps、lidarseg/panoptic/CAN bus、独立 mini 整理都属于可选或后续。 |

逐行字段、官方脚本/URL、估算大小、许可证和 notes 在 `download_gap_list.md`。所有下载命令仅作为获批后的 staging 示例，没有运行。

## 路径方案

建议保留现有数据原位，在已有且非空的 `/gs/bs/tga-RLA/qdeng/datasets/` 下追加 canonical symlink layer。该根当前已有 `Bench2Drive-Base` 和 `Bench2Drive-mini`，不得覆盖。第一轮只建议 `keep + symlink`，新增 data bytes 约 0；Human 批准前不创建任何目录或链接。

主要逻辑目标：

```text
datasets/
├── raw/navsim/{maps,openscene,two_stage}/
├── raw/nuscenes -> existing main root
├── processed/navsim/rap/{norm,aug,perturbed}/
├── processed/nuscenes/rap/{raster,cache}/
├── manifests/
└── archives/
```

`/gs/bs/tga-RLA/qdeng/nuplan_dataset` 继续作为独立 nuPlan raw 根；`dataset_norm/aug/perturbed` 只放 processed 层；不允许把它们伪装成 OpenScene raw。

## 长扫描 job 账本

`scripts/agent_tools/job_status.py --pbs` 在 TSUBAME UGE 上调用 PBS `qstat -x`，因此对这些 UGE job 返回 `UNKNOWN`。最终状态依据 UGE accounting、exit status和产物交叉确认；这个工具兼容性问题本身保留记录。

### 有效执行

| job ID | 节点 | 状态 | 作用 | 日志/产物 |
|---:|---|---|---|---|
| 8621138 | r18n2 | PASS, exit 0 | NAVSIM raw test 引用 | `outputs/dataset_audit_20260910T092936+0900/nav_test.log`, `nav_test/` |
| 8621345 | r5n9 | PASS, exit 0 | mini/NAVSIM quick batch；其中 nuScenes assert 失败被后续 deep 单独确认 | `outputs/dataset_audit_20260910T092936+0900/quick_batch.log`, `nav_mini/` |
| 8621346 | r4n9 | FAIL, exit 1 | trainval batch；NAVSIM 产物有效，nuScenes assert 导致 exit 1 | `outputs/dataset_audit_20260910T092936+0900/trainval_batch.log`, `nav_trainval/` |
| 8621347 | r4n9 | TIMEOUT, exit 137 / failed 44 | filesystem/du/mtime/candidate；2h hard limit，保留已完成的 partial raw | `docs/audits/datasets/raw/filesystem_20260910T092936+0900.txt` |
| 8621392 | r14n1 | TIMEOUT, exit 137 / failed 44 | NAVSIM refs；完成 through dataset_norm，aug/perturbed refs UNKNOWN | `docs/audits/datasets/raw/navsim_references_20260910T092936+0900.jsonl`, `outputs/.../navsim_deep.err` |
| 8621409 | r19n8 | FAIL, exit 1 | nuScenes deep 初版，代码 typo；不采用其重复前缀 | `docs/audits/datasets/raw/nuscenes_references_20260910T092936+0900.jsonl`, `outputs/.../nuscenes_deep.err` |
| 8621414 | r21n9 | PASS, exit 0 | 修正后的 nuScenes 主 deep | 同上 raw JSONL |
| 8621420 | r6n3 | PASS, exit 0 | navtrain 4-history coverage | `outputs/.../navtrain_history.jsonl` |
| 8621449 | r5n9 | PASS, exit 0 | camera-only decode samples | `outputs/.../navsim_image_decode.jsonl` |
| 8621450 | r12n11 | PASS, exit 0 | 三个额外 nuScenes mini 根 | 汇总进 `raw/nuscenes_references_20260910T092936+0900.jsonl` |
| 8621481 | r5n6 | PASS, exit 0 | official v2.2 filter intersections | `outputs/.../navsim_v22_intersections.jsonl` |
| 8621498 | r6n3 | PASS, exit 0 | navtrain 46 个物理缺口归并 | `outputs/.../navtrain_missing_refs.jsonl` |
| 8621500 | r13n3 | PASS, exit 0 | nuScenes missing-by-channel | `outputs/.../nuscenes_missing_by_channel.jsonl` |
| 8621516 | r5n9 | PASS, exit 0 | NAVSIM staging/legacy pool | `outputs/.../navsim_download_staging.jsonl` |
| 8623456 | r3n10 | CANCELLED_AFTER_PARTIAL, exit 137 / failed 100 | processed roots bytes/files/mtime；38m15s 后仍只有 norm 一行，为避免 Human review 后遗留后台任务而取消 | `outputs/.../processed_stats.jsonl` |

`8623456` 没有被当作成功。它在 `14:19:05`–`14:57:21` 运行 `2,295.494s`，只读且没有修改数据；保留 norm partial evidence，aug/perturbed 未产出的 bytes/mtime继续标 UNKNOWN。取消后没有遗留本轮后台 job。

### 未形成审计结论的 job

- `8621048`, `8621049`, `8621050`, `8621079`, `8621056`：早期 submission/setup 无效，没有可采用产物，标 `INVALID_SETUP`。
- `8621099`：参数检查 PASS，不是数据深扫。
- `8621137`, `8621139`, `8621142`, `8621150`, `8621160`, `8621171`, `8621172`, `8621173`, `8621174`, `8621241`, `8621282`, `8621289`, `8621293`, `8621306`, `8621318`：重复或错误资源请求，均在实际扫描前取消，标 `CANCELLED_NO_SCAN`，无有效日志。

## 检查命令

实际检查包含：

- `source env/stage_a.env`，随后 `source env/sprint_20260909.env`；只记录上面的非敏感路径变量。
- `git rev-parse HEAD`、`git status --short`、`hostname`、`realpath`、`stat`。
- 用受限 `find -maxdepth` / `safe_find.py` 做浅层结构和 `/gs/bs/tga-RLA/qdeng` 范围内的定向候选发现。
- 在 UGE 计算节点做 `du -sb`、file/dir/mtime、NAVSIM pickle/YAML/token/reference/filter intersection、nuScenes JSON/token/FK/sample_data、camera/lidar/radar抽样。
- nuScenes 使用现有 devkit 初始化 `v1.0-trainval`/`v1.0-test` 并运行已有 `nuscenes/tests/assert_download.py`；没有升级环境或联网安装。
- 对 metadata aggregate 计算 SHA-256；没有对大 blob 逐文件做 SHA-256。navtrain archive 不存在，MD5 明确记为 `NOT_AVAILABLE`。

## PASS / FAIL / BLOCKED / UNKNOWN

- PASS：只读边界遵守；raw/processed 分层；mini/test 引用判定；nuScenes table/FK 解析；所有报告和 raw 文件均按固定 snapshot 命名。
- FAIL：navtrain history 46 个物理文件缺失；full trainval 不完整；processed clean/配对条件不满足；nuScenes official assert 对 trainval/test 都失败。
- BLOCKED：canonical symlink 实施等待 Human 路径批准；审计脚本实现和 atomic commit 等待 `PASS CP-CODE-DS-01`；下载受许可证、账号、空间和本轮禁令阻塞。
- UNKNOWN：任何未能由 metadata/manifest/checksum/成功深扫证明的项继续保留 UNKNOWN，原因写在各分报告；没有用推测补值。

## Human 检查门

请先审核四件事：P0 的 46-file 定向补全策略、processed clean manifest 边界、canonical symlink 映射、`maintenance.md` 的 CP-CODE review card。收到明确的 `PASS CP-CODE-DS-01` 前，不创建 `scripts/audit/datasets/inventory_datasets.py` 或 `update_dataset_inventory.body.sh`，也不提交 commit。
