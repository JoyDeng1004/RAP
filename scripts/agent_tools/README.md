# RAP Agent 工具

## 工具用途
- `safe_find.py`：限深度、限目录条目地查看结构，避免扫爆数据集。
- `safe_read.py`：按行或正则上下文读取源码，避免整篇载入。
- `safe_grep.py`：限匹配数的代码搜索，自动跳过权重和缓存。
- `safe_run.py`：把高输出命令写入日志并限制磁盘占用。
- `job_status.py`：查询本地/PBS 状态并清理旧任务。
- `log_summary.py`：只读日志末尾并分类错误、警告、指标。

## 常用命令
```bash
python scripts/agent_tools/safe_find.py navsim --max-depth 2
python scripts/agent_tools/safe_read.py train.py --start 1 --lines 80
python scripts/agent_tools/safe_grep.py 'def train' navsim
python scripts/agent_tools/safe_run.py -- python -c 'print("hello")'
python scripts/agent_tools/job_status.py JOB_ID
python scripts/agent_tools/log_summary.py .agent_jobs/JOB_ID/stderr.log
```

## 背景与策略
Lustre 适合大文件，不适合海量小文件的递归 metadata 操作；先查 manifest/index，再做浅层结构和有限样本。长任务一律通过 `qsub` 放到计算节点，开始后只查状态。

Policy:      AGENTS.md      → 告诉 Agent 该怎么做
Enforcement: agent_tools/*.py → 真正限制输出

注意：AGENTS.md 是约定不是强制。Codex 仍能敲原始 shell 命令，
这套工具只在被调用时生效。

## 临时目录
在 compute job 中若存在 `T4TMPDIR`，可显式设置：
```bash
export AGENT_JOBS_DIR="$T4TMPDIR/rap_agent_jobs"
```
T4TMPDIR 是临时 scratch，作业结束后内容可能消失；需要保留的日志必须另存。默认不会自动切换。
