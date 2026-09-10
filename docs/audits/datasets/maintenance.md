# Dataset inventory 维护方案

## 当前结论

状态：**PASS（维护方案已写明）/ BLOCKED（脚本实现等待 Human PASS）**。

本轮只给出运行契约和 CP-CODE review card。没有创建审计脚本，没有部署 cron、常驻 watcher 或定时任务。`deep` 只应通过 qsub/UGE 在计算节点运行。

## 审计信息

| 字段 | 值 |
|---|---|
| 盘点时间 | `2026-09-10T09:29:36+09:00`（snapshot `20260910T092936+0900`） |
| 发起节点 | `r4n11` |
| Git SHA | `b95916ffda6a08e603d43c20a8a3a684c53359f0` |
| Git 状态 | `DIRTY`；均视为盘点前已有或 Human 资产，本轮不覆盖 |
| 执行者 | `Codex`（UGE owner `uq02279`） |
| 环境来源 | 依次只读取 `env/stage_a.env`、`env/sprint_20260909.env` |
| 安全环境值 | `RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset`; `NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps`; `NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/RAP`; `NAVSIM_EXP_ROOT=/gs/bs/tga-RLA/qdeng/RAP/exp`; `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |

敏感环境变量未记录。当前 UNKNOWN：脚本尚未获准创建，因此运行时间、缓存命中率和失败恢复行为尚未实测。

## 获批后的运行方式

数据下载、解压或 RAP processing 完成后手动运行 `fast`：

```bash
python scripts/audit/datasets/inventory_datasets.py \
  --mode fast \
  --output-dir docs/audits/datasets \
  --navsim-root /gs/bs/tga-RLA/qdeng/navsim_workspace/dataset \
  --nuscenes-root /gs/bs/tga-RLA/qdeng/data/nuscenes
```

重要实验前通过 UGE 运行 `deep`。示例只申请 CPU 资源，不含定时部署：

```bash
qsub -g tga-RLA \
  -N dataset-inventory-deep \
  -l cpu_16=1 \
  -l h_rt=04:00:00 \
  -j y \
  -o outputs/dataset_inventory_deep.log \
  scripts/audit/datasets/update_dataset_inventory.body.sh deep
```

`fast` 只收路径、版本、浅层计数和最近更新时间，目标是数分钟内结束。`deep` 才做全量引用完整性、文件数、实际字节和 split 交集。两种模式都先写同目录临时文件；所有步骤成功后才用同一文件系统上的 atomic rename 更新 Markdown、`latest-success.jsonl` 和 timestamped snapshot。失败时保留上一份成功报告。

## 比较和恢复

比较两次 JSONL snapshot：

```bash
diff -u \
  docs/audits/datasets/raw/inventory_OLD.jsonl \
  docs/audits/datasets/raw/inventory_NEW.jsonl
```

恢复上一份成功报告时，不改数据。先从 `latest-success.jsonl` 读取上一成功 snapshot 的路径，人工核对 timestamp 和 Git SHA，再把对应 timestamped Markdown 设为当前展示版本。实现必须用临时文件加 atomic rename，不能在失败路径覆盖现有报告。

hash 策略：大文件默认缓存 `realpath/path/size/mtime`，不逐文件算 SHA-256；metadata、manifest 和保留的下载 archive 可以算 SHA-256/MD5。缓存键必须包含上述字段，任一变化就重算。

## CP-CODE review card：`CP-CODE-DS-01`

**状态：WAITING_HUMAN_PASS。** 这是下一轮唯一建议的代码修改。本轮没有创建下面两个文件，也没有 commit。

| 文件 | 具体代码段 | 行为 |
|---|---|---|
| `scripts/audit/datasets/inventory_datasets.py` | `parse_args()` | 增加 `--mode {fast,deep}`、`--output-dir`、`--navsim-root`、`--nuscenes-root` 四组参数；所有 path 先 `resolve(strict=False)`，不写数据根。 |
| 同上 | `collect_fast()` | 只读 path/type/symlink/filesystem、版本目录、浅层计数和 mtime；输出带 schema version 的 JSONL records。 |
| 同上 | `collect_deep()` | 解析 NAVSIM pickle/YAML 与 nuScenes JSON 外键；统计 missing/zero/duplicate/extra、channel、split 交集；超过 login-node 范围时要求 UGE 环境标记，否则拒绝 deep。 |
| 同上 | `SnapshotWriter` | 写 `*.tmp`，`fsync` 后 `os.replace()`；成功才更新 timestamped snapshot、Markdown 与 `latest-success.jsonl`；失败清理本次临时文件，不碰上次成功产物。 |
| 同上 | `MetadataCache` | 大 blob 用 path/size/mtime 缓存；metadata/manifest/archive 可 hash；UNKNOWN 原样序列化并附 reason。 |
| `scripts/audit/datasets/update_dataset_inventory.body.sh` | 参数和环境检查段 | `set -euo pipefail`；只接受 `fast`/`deep`；解析两个 env 文件；只把上述非敏感路径变量传给 Python。 |
| 同上 | qsub body | 记录 job ID/hostname/Git 状态/log；`deep` 校验 `JOB_ID`/UGE 上下文；不含下载、移动、删除、symlink、cron 或实验命令。 |

建议原子 commit 范围仅含这两个新脚本及对应的脚本级测试。Human 审核时请重点检查：

1. deep 的 login-node 拒绝门是否足够严格；
2. atomic publish 是否能保证失败不覆盖；
3. split 判定是否只来自 metadata/token/log，不读取目录名作为身份；
4. raw、official filtered、RAP processed 三类是否在 schema 中强制分开；
5. UNKNOWN 是否始终保留 reason。

批准口令：`PASS CP-CODE-DS-01`。收到 PASS 前不实现、不提交。

## 检查命令

本方案依据以下只读检查：`source env/stage_a.env`、`source env/sprint_20260909.env`、`rg` 配置调用、`find -maxdepth`、`git status --short`、`qstat -j` 和 `qacct -j`。没有运行 H0、Smoke Test、训练或评测。
