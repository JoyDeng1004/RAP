# Command log

本文件记录本轮实际执行的只读命令族。长输出已结构化到同目录 JSONL；未完成的扫描明确标注，不把 partial output 当结果。

```bash
which qsub sbatch squeue qstat 2>/dev/null
qstat --version 2>/dev/null
sinfo --version 2>/dev/null

find dataset_* ...                  # 浅层结构、目录项、扩展名与计数
python -c 'pickle.load(...)'        # dataset_norm 64 个 metadata pickle
python -c 'yaml.safe_load(...)'     # navtrain/navtest/navmini/warmup split
python -c 'os.walk/os.scandir(...)' # raster 与 nuScenes sensor 文件计数

python -c 'torch.load(..., map_location="cpu", weights_only=False)' ckpts/<file>
sha256 streamed in the same Python process

rg / sed / git show / git diff      # A4 access boundary 与 A5 配置静态审计
```

未执行：H0、Smoke Test、训练、评测、renderer、数据生成、manifest 生成。

终止/中断记录：

- `dataset_aug` 首次 per-file size/mtime scan：交互中断，无结果；后以 no-stat 全量目录项计数完成。
- nuScenes per-file size/mtime scan：超过 10 分钟后主动终止，exit 130；后以 `os.scandir` 目录项计数完成核心完整度检查。
- `du -sh dataset_aug`：长时间无输出后主动终止，exit 130；未将 partial/未知值写入结论。
