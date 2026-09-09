# Codex 任务书 — sensor_blobs 覆盖率审计与补下载

审计脚本：`scripts/data_audit/sensor_blob_coverage.py`（已就绪，无需修改）

---

## 复制给 Codex 的 prompt

```
# 任务

在 TSUBAME 上审计 NAVSIM sensor_blobs 的本地覆盖率，定位缺失的 log，然后只补下载缺失的部分。

背景：Stage-A 实验记录的 real 三相机覆盖率是 20.887%，已判定为 `data_bug`（本地可用性问题，可修复），
而不是数据集特征。本任务要先用证据确认这个判定，再补齐。

# 环境

source /Users/joy/pythonProject/RAP/env/stage_a.env  # 路径以集群上的实际位置为准
  OPENSCENE_DATA_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset
  RAP_ROOT=/gs/bs/tga-RLA/qdeng/RAP

数据布局（已从 navsim/common/dataclasses.py 与 navsim/planning/script/build_4cam_root.py 核实）：
  元数据   $OPENSCENE_DATA_ROOT/navsim_logs/<split>/<log_name>.pkl   （list[frame dict]）
  图像     $OPENSCENE_DATA_ROOT/sensor_blobs/<split>/<log_dir>/<CAMERA>/<file>.jpg
  解析规则 image_path = sensor_blobs_path / frame["cams"][CAMERA]["data_path"]
  相机     CAM_F0 / CAM_L0 / CAM_R0（Stage-A 用的三摄）
  split    navtrain -> data_split=trainval；navtest -> data_split=test

# 第 1 步：审计（先做，不要跳过）

对 trainval 和 test 各跑一次：

  python scripts/data_audit/sensor_blob_coverage.py \
    --data-split trainval \
    --out-dir $RAP_ROOT/outputs/data_audit/trainval \
    --dump-missing-files --decode-sample 500

  python scripts/data_audit/sensor_blob_coverage.py \
    --data-split test \
    --out-dir $RAP_ROOT/outputs/data_audit/test \
    --dump-missing-files --decode-sample 500

产物：sensor_blob_coverage.json / logs_absent.txt / logs_partial.txt /
      redownload_logs.txt / missing_files.txt

# 第 2 步：判读并回报（在下载任何东西之前）

按 split 分别报告：

  1. 应有 log 数 / 实有 log 数 / 完整 / 部分缺失 / 完全缺失
  2. 文件覆盖率，并与 20.887% 对照
  3. declared_missing_paths（data_path 以 "missing_camera/" 开头）—— 这部分是数据集特征，
     补下载解决不了，必须单独列出、不计入"要修的量"
  4. files_zero_byte 与 decode_check.corrupt —— 传输中断的证据
  5. coverage_by_date —— 如果缺失按日期块聚集，说明是整块 chunk 没下全；
     如果均匀分散，说明是别的问题，此时停下来先报告，不要开始下载
  6. 结论：缺失是"整条 log 级别"还是"log 内零星帧"

这一步的输出决定要不要下载。如果证据不支持"下载不全"，停下来报告，不要自行改变结论。

# 第 3 步：确定下载来源（不要假设）

不要凭记忆假设下载方式。先用证据确定：

  - 检查 $OPENSCENE_DATA_ROOT 及其上层有没有残留的下载脚本、.tgz/.tar.gz、
    wget/curl 日志、HuggingFace 缓存、aria2 控制文件
  - 检查已存在的 sensor_blobs 目录的 mtime 分布，判断是分批下载还是一次性
  - 确认官方分发是"按 log"还是"按 chunk 归档"。如果是 chunk 归档，
    需要建立 log_name -> 归档文件 的映射，然后按归档下载，而不是按 log

把确定的来源、命令和 log->归档映射先写出来给我确认，再执行。

# 第 4 步：容量检查（下载前必做）

  - 用 missing_files.txt 的条数 × 已有文件的平均字节数，估算需要的空间
  - 查 /gs/bs/tga-RLA 的配额与剩余空间
  - 需要空间 > 剩余空间的 80% 就停下来报告，不要开始下载

# 第 5 步：补下载

  - 只下载 redownload_logs.txt 里的 log（或其所属归档），不要整包重下
  - 断点续传，失败重试，逐文件校验（有官方 checksum 就用官方的，没有就至少校验字节数非零且可解码）
  - 全程写日志到 $RAP_ROOT/outputs/data_audit/download.log
  - 可中断可重入：重跑不应重复下载已完成的部分

# 第 6 步：复验

重跑第 1 步的两条命令，输出到 .../trainval_after 和 .../test_after，
并给出 before/after 对照表：coverage_pct、logs_absent、logs_partial、files_absent。

# 硬约束（违反即失败）

  - 不要修改 $OPENSCENE_DATA_ROOT/navsim_logs/ 下的任何 .pkl。元数据是帧清单的唯一来源。
  - 不要写入、移动、删除 $RAP_ROOT/outputs/alignment_stage_a/ 下的任何内容。那是冻结产物。
  - 不要碰 navhard_two_stage 与 warmup_two_stage。它们是评测专用，任何进入训练路径的动作都违反 split integrity。
  - 补完 blob 会改变候选池，但**不要**重新生成 Stage-A 的 54-log paired 子集、
    stage_a_token_manifest.json 或任何 init checkpoint。已冻结的 Stage-A run 必须保持原样可复现；
    新 blob 只对将来的 run 生效。
  - 不要重跑训练或评测。
  - 不要改动 navsim/ 下的代码。审计脚本本身如果需要修，先说明原因再改。
  - 每一步先报告再执行；任何需要删除或覆盖的操作，先列出目标再等确认。
```

---

## 本地就能先跑的一条（如果你手上有集群 shell）

```bash
python scripts/data_audit/sensor_blob_coverage.py --data-split trainval --out-dir /tmp/blob_audit
```

不加 `--dump-missing-files` / `--decode-sample` 时只做目录扫描，很快，先看四个数字就够判断了。
