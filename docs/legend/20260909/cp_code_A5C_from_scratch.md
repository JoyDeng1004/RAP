# `CP-CODE-A5C`｜让 `from scratch` 可自证

> **状态：🔴 WAITING_HUMAN_PASS。**
> 按 Sprint Plan §9.0：Human 明确 `PASS` 前，**不得 commit 本修改、不得开始下一次关键代码修改、不得启动 Smoke Test 或 Full Experiment**。
> 本卡对应 `rap_paper_baseline.md` §2 注 C。目前**尚未做出任何代码改动**，工作树中该文件未被触碰。

---

## 修改目的（仅一个）

**让「F0/F1 从零初始化」这一冻结约束在代码层面可被证明，而不是依赖运行时的目录卫生。**

对应条目：

| 类型 | ID | 原文 |
|---|---|---|
| `SD-*` | `SD-0` | 「F0/F1 必须 from scratch；只允许**同一 run** 在 code/config/data hash 完全一致时续训」 |
| 审计门 | `A5` · `Init mode` 行 | 「agent `checkpoint_path: ''` 表面为 scratch，但 trainer 固定 `ckpt_path='last'`……**不能证明 from scratch**」→ `BLOCKED` |
| 审计门 | `README.md` 阻断项 #6 | 「默认训练入口固定 `trainer.fit(..., ckpt_path='last')`，不能在未隔离输出目录和未冻结配置时证明 from-scratch」 |
| 实验臂 | F0 / F1 | 二者都必须 from scratch |

---

## 关键文件

| 相对路径 | 改动性质 |
|---|---|
| `navsim/planning/script/run_training.py` | 修改 1 处（1 行 → 1 行） |
| `navsim/planning/script/config/training/default_training.yaml` | 新增 1 个配置键 |

无其他文件改动。不触碰 `rap_agent.py`、`navsim_config.py`、任何 agent/dataset 配置。

---

## 具体代码段

### (1) `navsim/planning/script/run_training.py` · `main()` · L186–191

**当前**（已核实）：

```python
186    trainer.fit(
187        model=lightning_module,
188        train_dataloaders=train_dataloader,
189        val_dataloaders=val_dataloader,
190        ckpt_path='last'
191    )
```

**拟改为**：

```python
186    trainer.fit(
187        model=lightning_module,
188        train_dataloaders=train_dataloader,
189        val_dataloaders=val_dataloader,
190        ckpt_path=cfg.resume_ckpt_path,
191    )
```

起止语义：`trainer.fit` 的调用点，即整个训练入口的唯一 resume 决策处。`cfg.resume_ckpt_path` 为 `None` 时 Lightning 从零开始；为路径字符串时从该 checkpoint 续训。

### (2) `navsim/planning/script/config/training/default_training.yaml` · 顶层键区（L23–25 邻近）

**当前上下文**（已核实）：

```yaml
23  use_cache_without_dataset: false
24  force_cache_computation: true
25  seed: 0
```

**拟新增**（紧随 `seed` 之后）：

```yaml
# Resume contract (SD-0): null = from scratch. Set to an explicit checkpoint
# path ONLY to resume the SAME run_id with identical code/config/data hashes.
# 'last' is deliberately NOT the default: it makes a re-used output_dir
# silently resume, which cannot be distinguished from from-scratch.
resume_ckpt_path: null
```

---

## 行为变化与风险

### 会变化的

| 场景 | 改前 | 改后 |
|---|---|---|
| 全新 `output_dir` | 从零训练 | 从零训练（**不变**） |
| **复用已有 `output_dir`** | **静默从 `last.ckpt` 续训，日志无告警** | **从零训练**；除非显式传 `resume_ckpt_path=<path>` |
| 需要续训 | 无法关闭 | `+resume_ckpt_path=/abs/path/last.ckpt` 显式开启，且该值会进 `config.frozen.yaml` 与 `hashes.json`，可审计 |

**这正是本次修改的全部收益**：resume 从"环境的副作用"变成"配置里的显式声明"，因而可被 `A5` 记录、可被 `CP-2b` 核验。

### 不应变化的（须 Human 一并确认）

- 模型结构、loss、optimizer、scheduler、batch size、precision、分布式策略：**零改动**。
- `ModelCheckpoint(save_last=True, ...)`（`rap_agent.py:596–602`）：**不动**。`last.ckpt` 照常保存，只是不再被自动读取。
- 数据加载路径、cache 逻辑、`use_cache_without_dataset` 分支：**零改动**。
- 默认行为的方向：由「隐式 resume」改为「显式 from scratch」。**这是行为变更，不是纯重构** —— 任何依赖旧默认值的既有脚本会改变行为（见下）。

### 风险

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **既有 `scripts/*.qsub` 若依赖自动 resume，改后会从零重跑** | `scripts/` 下 20 余个既有脚本仍被 gitignore（`SD-8` 只放行三个目录），本冲刺不使用它们。**但 Human 须确认没有正在跑的作业依赖旧行为。** |
| 2 | 长作业被队列杀死后，重新提交不再自动接续 | 这是**有意的**。§9 把「作业被队列杀」归类为 Technical Failure，处置是"修复后重跑"。若确需接续，走 `resume_ckpt_path` 显式声明，并按 `SD-0` 核对 hash 一致 |
| 3 | `cfg.resume_ckpt_path` 键缺失导致 `AttributeError` | 已在 (2) 中给出默认值。若 Human 担心其他 config 组未继承该键，可改用 `cfg.get('resume_ckpt_path', None)` —— **请在批注中指定用哪种** |

---

## 验证证据

**已执行的静态检查：**

| 检查 | 结果 |
|---|---|
| `sed -n '186,191p' run_training.py` | 与上文「当前」段逐字一致 |
| `grep -n "ckpt_path" run_training.py` | **全文件仅 1 处命中**（L190），无其他 resume 入口 |
| `grep -n "resume_ckpt_path" navsim/` | **0 命中** —— 新键不与任何现有键冲突 |
| `grep -n "^seed:" default_training.yaml` | L25，插入点上下文已核实 |
| `grep -n "save_last" rap_agent.py` | L597，确认 `last.ckpt` 仍会生成 |

**未执行、也不得以之代替本卡的：** Smoke Test、训练、评测（§9.0 明令「不得以 Smoke Test 代替 Human 检查」）。

**尚未改动任何文件。** 本卡获 `PASS` 后才落笔，落笔后立即提交与之一一对应的原子 commit（§8.0 第 3 步）。

---

## 待 Human 判断

请给出 **`PASS` / 需要修改 / `BLOCKED`** 之一。若 `PASS`，请一并明确两点：

1. **风险 #3 的写法**：`cfg.resume_ckpt_path` 还是 `cfg.get('resume_ckpt_path', None)`？
2. **是否接受"默认行为变更"**：改后复用 `output_dir` 将从零开始而非续训。

> 若你倾向**不改代码**，替代方案是「每个 `run_id` 强制使用全新 `output_dir`」，写死进 job wrapper。
> ⚠️ 但请注意：`scripts/jobs/` **当前是空目录**，该 wrapper 尚不存在。选这条路等于把阻断从"改一行代码"移到"先写出并审完一个 wrapper"，且 `A5` 的 `Init mode` 行在 wrapper 落地前仍为 `BLOCKED`。

---

## 附：同文件内另一处发现（**不属于本卡，需单独处置**）

核实本卡时在同一文件的 `use_cache_without_dataset` 分支（L103–169）发现一处与 F0 定义相关的问题，按 §8.0「一个目的 = 一个 commit」**不并入本卡**：

```python
149    train_data_perturbed = CacheOnlyDataset(cache_path=cfg.cache_path_perturbed, ...)
154    indices = random.sample(range(N), int(0.1*N))
158    train_data_others   = CacheOnlyDataset(cache_path=cfg.cache_path_others, ...)
165    indices = random.sample(range(N), int(0.05*N))
169    train_data = ConcatDataset([train_data, train_data_perturbed, train_data_others])
```

三点：

1. **该分支无条件混入 `cache_path_perturbed` 与 `cache_path_others`**，比例 10% / 5%，没有开关。而 §5.2 定义 F0 = 「无 external 输入，仅 10% DEV target manifest」。**若 F0 走 cache 路径，它就不是 F0。**
2. §3.7 已实测 `dataset_perturbed` **命中 navtrain 5,095 / navmini 320 tokens** —— 混入的正是一个已证实污染的根。
3. 抽样用 `random.sample`。`pl.seed_everything(cfg.seed, workers=True)`（L91）先于它执行，故**给定 seed 与相同 cache 目录内容时可复现**；但这仍是**每次运行重新抽样**，不满足 `SD-2` 的「manifest 生成一次后保存 sha256，后续只读取该文件，**不得重新抽样**」。

> 该分支是否被 F0/F1 走到，取决于 `use_cache_without_dataset`（`default_training.yaml:23` 默认 `false`）。**请 Human 裁决 F0/F1 用哪条数据路径**；若走 cache 路径，本条须升级为独立的 `CP-CODE` 并先于 Smoke Test 处理。
