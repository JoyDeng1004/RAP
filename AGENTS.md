# RAP Agent 规则

## 输出原则
- 大目录列表、大日志、进度条默认不进入上下文。
- 禁止 `ls -R`、无界 `tree`/`find`、`cat` 大文件、`tail -f`、带进度条的 `wget`/`curl`。
- 不要反复读取训练日志陪跑。
- 预计输出超过 100 行或 8 KiB 的命令，必须使用 `safe_run.py`。

## 存储规则
- 大型存储是 Lustre，海量小文件递归 metadata 操作很慢。
- 检查数据集顺序：已有 manifest/index → 浅层结构 → 有限样本 → 定向搜索 → 万不得已才扩大扫描。
- 禁止仅为了解数据集递归统计 `/gs/bs`、`/gs/fs`、`/home`。

## 工具映射
| 需求 | 用 |
|---|---|
| 看目录结构 | `safe_find.py` |
| 读源码文件 | `safe_read.py` |
| 搜代码 | `safe_grep.py` |
| 跑高输出命令 | `safe_run.py` |
| 查任务状态 | `job_status.py` |
| 排查日志 | `log_summary.py` |

## 长任务
- 长任务一律 `qsub` 提交，不在 login node 上运行。
- 任务开始后只用 `job_status.py` 查状态，不反复读日志。
- 每个工具 JSON 都有 `next_action`；需要更多信息时按它给的参数逐级放大。
