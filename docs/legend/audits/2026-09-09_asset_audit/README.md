# 2026-09-09 Asset Audit

- 审计时间：2026-09-09 16:52–17:54 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex
- 工作树：`DIRTY`；审计期间存在 Human 修改的 Sprint Plan、暂存的 `tools/render_nuscenes_camera_cross.py` 和既有 untracked assets。本审计未覆盖、清理或提交这些内容。

## 结论

**Environment Check 未通过。禁止启动 Smoke Test、训练或评测。**

| 项目 | 状态 | 关键结论 |
|---|---|---|
| EF-1 | PASS | 仅发现 UGE `qsub`/`qstat`；未发现 Slurm `sbatch`/`squeue`，调度器为 qsub / UGE 2023.1.1 (8.8.1)。 |
| A1 | **FAIL / CONTAMINATED** | `dataset_norm` 与 `navtest` 重叠 10 logs / 1,365 tokens；与 `warmup_test_e2e` 重叠 62 logs / 563 tokens；`navhard_two_stage` split 定义缺失，无法核验。 |
| A2 | PASS_WITH_LIMITATION | nuScenes trainval/test metadata、RGB samples、calibration 与 sweeps 存在，核心数量一致；递归 total bytes/mtime 扫描超过 10 分钟后终止，未取得全目录字节数。 |
| A3 | PASS | 3 个 checkpoint 均完成内部结构与 SHA-256 审计；全部 `ALLOWED_AS_INIT: no`。 |
| A4 | **BLOCKED** | 当前 staged renderer 读取 source RGB 与 source calibration，属于 P2；`tools/canonical_bev/` 与既有输出目录缺失，不能作为 P1 F1 输入。 |
| A5 | **BLOCKED** | manifest、registry、F0/F1 frozen config、完整解析命令和执行契约均缺失；A1 污染门已失败。 |

## 阻断项

1. `dataset_norm` 命中 NAVSIM 评测侧数据。按 §3.1 / CP-2b，这是直接否决训练的硬门。
2. 仓库没有 `navhard_two_stage` split 定义，不能证明对应交集为 0。
3. `dataset_norm` metadata 的 414,936 个 camera 路径中，主 raster 根缺失 259,287 个，缺失率 62.4884%。
4. A4 renderer 越过 P1 source-pixel/calibration 边界，只能标 P2。
5. 10% DEV manifest、manifest SHA、F0/F1 `config.frozen.yaml`、registry 计划行、`run_id` 与 scheduler wrapper 均不存在。
6. 默认训练入口固定 `trainer.fit(..., ckpt_path='last')`，不能在未隔离输出目录和未冻结配置时证明 from-scratch。

## 报告索引

- `A1_rap_datasets.md`
- `A1_split_membership.md`
- `A2_nuscenes.md`
- `A3_ckpts.md`
- `A4_source_pixel_boundary.md`
- `A5_training_readiness.md`
- `raw/`：对应 JSONL 记录与命令日志

本轮未运行 H0、Smoke Test、训练或评测，也未补全任何 `SD-*` 或未知参数。
