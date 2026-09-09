# RAP Stage-A — Step 5 增补指令（v1-only）

Target reader: AI coding agent。Spec of record: `docs/rap-alignment-experiment-plan.md`。Runbook: `docs/rap-alignment-stage-a-runbook.md`。冻结协议: `outputs/alignment_stage_a/summary/summary_protocol.json`。On conflict, spec wins。

**前置**：runbook Step 4.1 已完成并验收通过（6 个 `last.ckpt`，epoch=19，global_step=400，`validate_stage_a_seed.py` 三 seed 全部 passed）。

**人工决策**：本轮只做 NAVSIM v1.1 评测，v2.2 延期。该决策构成对 runbook Step 2.4「No downgrade to a v1-only official experiment」的显式偏差，按 Step S.7 记录。

**本文件替代 runbook Step 5.1 的执行方式**：runbook Step 5.1 只有文字 Expected、无可执行断言，且未继承 Step 2.3 的 scorer provenance 要求。本文件补齐这两项。Step 5.2 本轮不执行。Step 5.3 按 Step S.8 以 v1-only 模式执行。

**执行顺序**（与章节顺序不同，以本表为准）：

```text
S.1 → S.1b → S.2 → S.3 → S.3b → [HALT: 路线 A/B 由人决定]
    → S.4b → S.3c → S.4 → S.5 → S.6 → S.7 → S.8
```

`S.4b`（代码修改）必须先于 `S.3c`（代码快照），`S.3c` 必须先于任何产生评测结果的 Step（S.4 起）。这样全部 8 个 CSV（2 baseline + 1 repeat + 1 smoke + ... 与 6 个正式结果）出自同一个已快照的代码状态。

## 全局禁令

1. 禁止修改 `summary_protocol.json`。需要修订时新建 `summary_protocol_v2.json` 并按其 `amendment_rule` 记录。
2. 禁止追加 epoch、重新训练、或在 checkpoint 之间做选择。primary checkpoint 固定为 `checkpoints/last.ckpt`。
3. 禁止用本地 scorer 替代官方 scorer，禁止在 Step S.3 失败后「先跑起来再说」。
4. 禁止在读取任何 PDMS 数值之后修改本文件或协议中的任何统计规则。
5. 任何 HALT 一律停止并报告，不得自行放宽条件。

## Session prelude

```bash
set -euo pipefail
source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env
cd "$RAP_ROOT"
```

---

## Step S.1 冻结协议与 cluster key

**Context**

- `summary_protocol.json` 已由人工写入 `$STAGE_A_OUT/summary/`。本 Step 只做完整性校验并生成 cluster bootstrap 所需的 `token -> log_name` 映射。
- v1 输出 CSV 的 9 列不含 `log_name`，而 spec §6 要求以 log 为 cluster bootstrap 单位。映射必须在读取任何分数之前建好并冻结。
- 映射来源是 navtest metric cache 的 metadata CSV（Step 2.2 产物，12,146 token）。

**Commands**

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import pandas as pd

S = Path(os.environ["STAGE_A_OUT"])
proto_path = S / "summary/summary_protocol.json"
proto = json.loads(proto_path.read_text())
for key in ("cluster_bootstrap", "hypothesis_family", "convergence_gate",
            "continuous_metrics", "discrete_metrics", "noise_floor",
            "practical_regression_threshold", "report_status", "run_scope"):
    assert key in proto, f"summary_protocol.json 缺少 {key}"
assert proto["cluster_bootstrap"]["n_resamples"] >= 10000
assert proto["hypothesis_family"]["status_this_run"].startswith("INCOMPLETE")
sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()

cache = Path(os.environ["NAVSIM_EXP_ROOT"]) / "metric_cache" / "metadata"
csvs = sorted(p for p in cache.iterdir() if p.suffix == ".csv")
assert len(csvs) == 1, f"metadata/ 有 {len(csvs)} 个 CSV: {[p.name for p in csvs]}"
df = pd.read_csv(csvs[0])
cols = {c.lower(): c for c in df.columns}
tok_col = cols.get("token")
log_col = cols.get("log_name") or cols.get("log_file") or cols.get("log")
assert tok_col and log_col, f"metadata CSV 缺少 token/log 列: {list(df.columns)}"

mapping = {}
for tok, log in zip(df[tok_col], df[log_col]):
    log = Path(str(log)).stem
    if tok in mapping:
        assert mapping[tok] == log, f"token {tok} 映射到多个 log"
    mapping[tok] = log
assert len(mapping) == 12146, f"token->log 映射有 {len(mapping)} 条，应为 12,146"

out = {"source_csv": str(csvs[0]), "n_tokens": len(mapping),
       "n_logs": len(set(mapping.values())),
       "protocol_sha256": sha, "mapping": mapping}
(S / "summary/token_to_log.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: out[k] for k in ("n_tokens", "n_logs", "protocol_sha256")}, indent=2))
PY
```

**Expected**

- `token_to_log.json` 写出，12,146 个 token，`n_logs` > 1。
- 打印 `protocol_sha256`，该值必须写入最终报告。
- 映射一对多 → HALT。

---

## Step S.1b convergence gate 曲线可用性审计

**Context**

- spec §4 的 convergence gate 要求记录**三条**逐 epoch 曲线：`train loss`、`val loss`、`val/score`。Step 4.1 的验收只报告了前两条，`val/score` 是否落盘从未确认。
- `run_training.py:465` 配置了 `CSVLogger(save_dir=cfg.output_dir, name="csv_logs")`，因此逐 epoch 指标应在 `$output_dir/csv_logs/version_*/metrics.csv`。`val/score` 的来源是 `agent_lightning_module.py:145` 对 loss dict 逐 key 的 `self.log`。
- `summary_protocol.json` 的 `convergence_gate.val_score_availability` 已规定：若缺失，必须在报告中记录该 artifact 缺失，**不得**以 val_loss 替代后不作说明。本 Step 把「缺没缺」变成一个落盘事实。
- 纯只读。无论结果如何都不阻塞后续 Step —— 它决定的是报告里写什么，不是能不能继续。

**Commands**

```bash
python - <<'PY'
import json, os
from pathlib import Path
import pandas as pd

S = Path(os.environ["STAGE_A_OUT"])
runs = [(c, s) for c in ("r100-noalign", "r100-fullalign") for s in (0, 1, 2)]
out, missing = {}, []
for cond, seed in runs:
    run_dir = S / "training" / cond / f"seed_{seed}"
    key = f"{cond}/seed_{seed}"
    csvs = sorted(run_dir.glob("csv_logs/version_*/metrics.csv"))
    if not csvs:
        out[key] = {"metrics_csv": None, "note": "未找到 csv_logs/version_*/metrics.csv"}
        missing.append(key)
        continue
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    cols = list(df.columns)
    score_cols = [c for c in cols if c.endswith("/score") or c.endswith("/best_score")]
    entry = {"metrics_csv": [str(p) for p in csvs], "columns": cols,
             "score_like_columns": score_cols}
    for c in ("train/loss", "val/loss", "val/score"):
        entry[c] = {"present": c in cols,
                    "n_non_null": int(df[c].notna().sum()) if c in cols else 0}
    entry["val_score_ok"] = entry["val/score"]["present"] and entry["val/score"]["n_non_null"] >= 20
    if not entry["val_score_ok"]:
        missing.append(key)
    out[key] = entry

result = {"runs": out, "runs_missing_val_score": missing,
          "status": "complete" if not missing else "val_score_missing",
          "spec_clause": "spec §4 — convergence gate 需 train loss / val loss / val/score 三条逐 epoch 曲线",
          "consequence": ("三条曲线齐全" if not missing else
                          "val/score 缺失，必须按 summary_protocol.json 的 "
                          "convergence_gate.val_score_availability 在报告中记录该 artifact 缺失")}
(S / "summary/convergence_curve_audit.json").write_text(
    json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
print(json.dumps({"status": result["status"],
                  "runs_missing_val_score": missing,
                  "score_like_columns_example": next(iter(out.values())).get("score_like_columns")},
                 indent=2, ensure_ascii=False))
PY
```

**Expected**

- `convergence_curve_audit.json` 写出。
- `status == "complete"` → 三条曲线齐全，报告正常引用。
- `status == "val_score_missing"` → **不 HALT**，继续执行；但该文件必须被 Step S.8 的报告引用，并在 convergence gate 章节显式声明缺哪条曲线、影响什么。禁止事后补跑训练来生成该曲线（会破坏预注册的 20 epochs 与两条件对照）。

---

## Step S.2 Step 2.3 人工交付物 gate

**Context**

- runbook Step 2.3 的**代码改动已全部完成**，本 Step 不需要再改代码。已核实：`default_run_pdm_score.yaml` 顶层有 `output_csv: null`；`run_pdm_score.py` 有 field mapping、9 列 reindex、`sort_values("token")`、`to_csv(index=False)`、token 集合全部走 `sorted(set(...))`。
- 缺的是人工交付物，全部由人提供，agent 不得代为编造：
  1. `$NAVSIM_V1_ROOT` 指向 `$RAP_ROOT` **之外**的 official navsim v1.1 checkout；
  2. `$NAVSIM_V1_COMMIT` 为完整 40 位 SHA（原 `env/stage_a.env` 中是 7 位短 SHA `0811876`）。
- **原第 3 项 `v1_reference.json` 已作废。** 本 Step 的核查确认 pinned worktree 内**不存在**任何 human / constant-velocity 的数值 PDMS——官方代码库只有运行命令、agent 描述与 PDMS 公式，README 仅链接外部论文与 leaderboard。因此绝对值门禁无法建立：
  - **硬门禁**改为 Step S.4 的**关系式门禁**，条件冻结在 `v1_relational_gate.json`，不依赖任何外部数值；
  - **软交叉核对**使用 `v1_paper_reference.json`（论文 Table 1 navtest 表，已提取落盘），**不作 HALT 条件**。
- 禁止用本次实验的任何运行结果反填任何参考文件。

**Commands**

```bash
test -n "$NAVSIM_V1_ROOT"
# worktree 中 .git 是文件而非目录，故用 -e 并由 git 自身确认
test -e "$NAVSIM_V1_ROOT/.git"
git -C "$NAVSIM_V1_ROOT" rev-parse --is-inside-work-tree >/dev/null
test "${NAVSIM_V1_ROOT#$RAP_ROOT}" = "$NAVSIM_V1_ROOT"   # 必须在 RAP_ROOT 之外
test ${#NAVSIM_V1_COMMIT} -eq 40
test "$(git -C "$NAVSIM_V1_ROOT" rev-parse HEAD)" \
   = "$(git -C "$NAVSIM_V1_ROOT" rev-parse "${NAVSIM_V1_COMMIT}^{commit}")"
test -z "$(git -C "$NAVSIM_V1_ROOT" status --porcelain)"   # 官方 checkout 必须干净
git -C "$NAVSIM_V1_ROOT" rev-parse HEAD > "$STAGE_A_OUT/evaluator_validation/v1_commit.txt"

# 上游归属证明：commit 必须可达自 autonomousvision/navsim 的远程 ref 或 tag，
# 「本地存在一个前缀相同的 commit 对象」不构成官方来源证明。
git -C "$NAVSIM_V1_ROOT" remote -v | grep -q "autonomousvision/navsim"
git -C "$NAVSIM_V1_ROOT" describe --tags --exact-match "$NAVSIM_V1_COMMIT" \
  > "$STAGE_A_OUT/evaluator_validation/v1_tag.txt"
test -n "$(git -C "$NAVSIM_V1_ROOT" for-each-ref --contains "$NAVSIM_V1_COMMIT" \
  --format='%(refname)' refs/remotes refs/tags)"

# 软交叉核对用的论文参考表必须已落盘（来源 papers/navsimv1.pdf Table 1，advisory 非门禁）
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["STAGE_A_OUT"]) / "evaluator_validation/v1_paper_reference.json"
d = json.loads(p.read_text())
rows = d["table1_navtest"]["rows"]
for name in ("Constant Velocity", "Human"):
    assert name in rows, f"v1_paper_reference.json 缺少 {name}"
    assert isinstance(rows[name]["PDMS"], (int, float))
for key in ("constant_velocity_agent", "human_agent"):
    assert key in d["advisory_bands"], f"advisory_bands 缺少 {key}"
assert d["advisory_bands"]["constant_velocity_agent"]["reference"] == rows["Constant Velocity"]["PDMS"]
assert d["advisory_bands"]["human_agent"]["reference"] == rows["Human"]["PDMS"]
# 该文件不得被当成门禁
assert d["cross_check"]["table3_leaderboard_1_1"]["Constant Velocity"] == 20.6
print("v1_paper_reference.json OK (advisory only)")
PY

# 确认 v1_reference.json 未被误创建 —— 绝对值门禁已作废，见 Context
test ! -e "$STAGE_A_OUT/evaluator_validation/v1_reference.json"

# output_csv 已进入 config，否则后续所有 run_pdm_score 调用会被 Hydra 拒绝
python navsim/planning/script/run_pdm_score.py --cfg job agent=constant_velocity_agent \
  experiment_name=probe 2>/dev/null | grep -q "^output_csv:"
```

**Expected**

- 全部 exit 0。checkout / 完整 SHA / clean / upstream remote / exact tag / ref 可达 六项全过。
- `v1_commit.txt`（40 位 SHA）与 `v1_tag.txt`（应为 `v1.1`）写出。
- `v1_paper_reference.json` 存在且自洽；`v1_reference.json` **不存在**。
- 任一不满足 → HALT for human。

---

## Step S.3 Scorer 同源性验证（决定性 gate）

**Context**

- runbook Step 2.3 原文：「local wrapper 必须记录实际 scorer 源文件与 SHA256，并证明它来自 `$NAVSIM_V1_ROOT@$NAVSIM_V1_COMMIT`；检查一个未被调用的 checkout 不算数。」
- `run_pdm_score.py` 从**本地** `navsim` 包 import `PDMScorer` / `PDMSimulator`，不是从 `$NAVSIM_V1_ROOT` import。因此该要求的唯一可验证形式是：**本地 pdm_planner 子树与官方 checkout 逐字节一致**。
- RAP 是 navsim 的 fork，本 Step **很可能失败**。这正是它排在评测之前的原因：先知道 scorer 是否漂移，比跑完 8 次全量评测再发现便宜得多。
- 失败时禁止自行同步、覆盖或 patch 任何一侧。列出差异文件与 diff 摘要，HALT for human。

**Commands**

```bash
python - <<'PY'
import hashlib, inspect, json, os
from pathlib import Path

import navsim  # compatibility shim
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

RAP = Path(os.environ["RAP_ROOT"]).resolve()
OFF = Path(os.environ["NAVSIM_V1_ROOT"]).resolve()
S   = Path(os.environ["STAGE_A_OUT"])

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

invoked = {}
for name, obj in (("PDMScorer", PDMScorer), ("PDMSimulator", PDMSimulator)):
    f = Path(inspect.getfile(obj)).resolve()
    rel = f.relative_to(RAP)
    off = OFF / rel
    invoked[name] = {"local_path": str(f), "relative_path": str(rel),
                     "local_sha256": sha(f),
                     "official_path": str(off),
                     "official_exists": off.is_file(),
                     "official_sha256": sha(off) if off.is_file() else None}
    invoked[name]["identical"] = (
        invoked[name]["official_exists"]
        and invoked[name]["local_sha256"] == invoked[name]["official_sha256"])

subtree = Path("navsim/planning/simulation/planner/pdm_planner")
diff, missing, extra, same = [], [], [], 0
for p in sorted((RAP / subtree).rglob("*.py")):
    rel = p.relative_to(RAP)
    off = OFF / rel
    if not off.is_file():
        missing.append(str(rel))
    elif sha(p) != sha(off):
        diff.append(str(rel))
    else:
        same += 1
for p in sorted((OFF / subtree).rglob("*.py")) if (OFF / subtree).is_dir() else []:
    if not (RAP / p.relative_to(OFF)).is_file():
        extra.append(str(p.relative_to(OFF)))

out = {"navsim_v1_root": str(OFF),
       "navsim_v1_commit": os.environ["NAVSIM_V1_COMMIT"],
       "invoked_classes": invoked,
       "subtree": str(subtree),
       "files_identical": same,
       "files_differing": diff,
       "files_missing_in_official": missing,
       "files_only_in_official": extra,
       "status": "passed" if not (diff or missing or extra)
                 and all(v["identical"] for v in invoked.values()) else "failed"}
(S / "evaluator_validation/v1_scorer_provenance.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: out[k] for k in (
    "files_identical", "files_differing", "files_missing_in_official",
    "files_only_in_official", "status")}, indent=2, sort_keys=True))
assert out["status"] == "passed", "本地 scorer 与官方 v1.1 不一致 → HALT，禁止自行同步任何一侧"
PY
```

**Expected**

- `v1_scorer_provenance.json` 写出，`status == "passed"`，`files_differing`、`files_missing_in_official`、`files_only_in_official` 全部为空，两个被实际调用的类都 `identical == true`。
- `status == "failed"` → 执行 Step S.3b 后 **HALT for human**。不得自行 patch、copy 或 symlink。

**已知**：2026-08-20 的预检已确认本地 `pdm_planner` 核心评分文件与 official v1.1 全部不一致。因此 S.3 预期为 `failed`，S.3b 是本轮的实际交付物。

---

## Step S.3b 差异内容分类（A/B 路线的决策输入）

**Context**

- S.3 只回答「是否一致」，S.3b 回答「差在哪、差得重不重」。**这是选择评测路线的唯一决策输入，在它产出之前不得设计 submission 流程，也不得启动任何评测。**
- 判定方法：对每个差异文件，剥除注释、空行、行尾空白后再比较。规范化后仍有差异 → `substantive`；规范化后相同 → `cosmetic`。
- 本 Step 只读、只分类，不作结论、不改任何一侧代码。

**Commands**

```bash
python - <<'PY'
import ast, difflib, json, os, re
from pathlib import Path

RAP = Path(os.environ["RAP_ROOT"]).resolve()
OFF = Path(os.environ["NAVSIM_V1_ROOT"]).resolve()
S   = Path(os.environ["STAGE_A_OUT"])
SUB = Path("navsim/planning/simulation/planner/pdm_planner")

def normalize(text: str) -> list[str]:
    """剥除注释、docstring、空行与行尾空白后的可比较代码行。"""
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and \
                   isinstance(getattr(body[0], "value", None), ast.Constant) and \
                   isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
        text = ast.unparse(ast.fix_missing_locations(tree))
    except SyntaxError:
        text = re.sub(r"#.*", "", text)
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]

results, substantive, cosmetic = {}, [], []
for p in sorted((RAP / SUB).rglob("*.py")):
    rel = p.relative_to(RAP)
    off = OFF / rel
    if not off.is_file():
        results[str(rel)] = {"kind": "missing_in_official"}
        substantive.append(str(rel))
        continue
    a, b = off.read_text(), p.read_text()
    if a == b:
        continue
    na, nb = normalize(a), normalize(b)
    if na == nb:
        results[str(rel)] = {"kind": "cosmetic"}
        cosmetic.append(str(rel))
        continue
    diff = list(difflib.unified_diff(na, nb, "official", "rap", lineterm="", n=1))
    results[str(rel)] = {
        "kind": "substantive",
        "normalized_added": sum(d.startswith("+") and not d.startswith("+++") for d in diff),
        "normalized_removed": sum(d.startswith("-") and not d.startswith("---") for d in diff),
        "hunks": sum(d.startswith("@@") for d in diff),
        "diff_head": diff[:40],
    }
    substantive.append(str(rel))

SCORE_CRITICAL = ("scoring/", "simulation/", "observation/")
critical = [f for f in substantive if any(k in f for k in SCORE_CRITICAL)]
out = {"official_root": str(OFF), "official_commit": os.environ["NAVSIM_V1_COMMIT"],
       "subtree": str(SUB), "cosmetic_files": cosmetic,
       "substantive_files": substantive,
       "substantive_in_score_critical_paths": critical,
       "per_file": results,
       "verdict": "cosmetic_only" if not substantive else
                  "substantive_outside_scoring" if not critical else
                  "substantive_in_scoring"}
(S / "evaluator_validation/pdm_planner_diff_analysis.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: out[k] for k in (
    "cosmetic_files", "substantive_files",
    "substantive_in_score_critical_paths", "verdict")}, indent=2, sort_keys=True))
PY

diff -ru --exclude='__pycache__' \
  "$NAVSIM_V1_ROOT/navsim/planning/simulation/planner/pdm_planner" \
  "$RAP_ROOT/navsim/planning/simulation/planner/pdm_planner" \
  > "$STAGE_A_OUT/evaluator_validation/pdm_planner_full.diff" || true
wc -l "$STAGE_A_OUT/evaluator_validation/pdm_planner_full.diff"
```

**Expected**

- `pdm_planner_diff_analysis.json` 与 `pdm_planner_full.diff` 写出，打印 `verdict`。
- **HALT for human。** 由人依据 `verdict` 选择路线，agent 不得代选：
  - `cosmetic_only` → 记录为已接受偏差，按原 Step S.4/S.6 用本地 scorer 继续，可称官方等价。
  - `substantive_outside_scoring` → 人工审阅差异后决定。
  - `substantive_in_scoring` → 本地 scorer **不可**称官方 v1.1 结果。二选一：路线 A（本地 scorer + 显式非官方标注）或路线 B（submission 隔离流程）。选定后按新增 Step 执行。

---

## Step S.3c 评测代码基线快照

**Context**

- **前置：Step S.4b 已完成且测试通过。** 见本文件头部的执行顺序表。
- spec §7 要求保存「Git commit（dirty worktree 时保存完整 patch 与 untracked source manifest）」。现有的 `$STAGE_A_OUT/baseline.patch` 是 runbook Step -1.3 在 Phase -1 冻结的，此后工作树变化很大，而且 **Step S.4b 直接修改了评测路径的代码**。没有新快照，将来无法证明 6 个 CSV 出自哪个代码状态。
- **训练与评测的代码状态本来就不同**，这一点必须显式记录而不是回避：6 个 checkpoint 训练于 S.4b 之前的代码，评测运行于 S.4b 之后。其合法性由 S.4b 的「逐位相同」测试保证——该修改只在 rendered 分支加零占位，模型输入不变。本 Step 把这个事实落盘。
- **本 Step 之后直到 Step S.6 结束，禁止修改 `navsim/` 下任何代码。** 确需修改 → 作废已产生的全部评测结果，回到本 Step 重新快照后从 S.4 重跑。

**Commands**

```bash
SNAP="$STAGE_A_OUT/evaluation_v1/code_baseline"
mkdir -p "$SNAP"
git rev-parse HEAD                        > "$SNAP/commit.txt"
git status --porcelain                    > "$SNAP/dirty.txt"
git diff --binary                         > "$SNAP/evaluation_baseline.patch"
git ls-files --others --exclude-standard  > "$SNAP/untracked.txt"

python - <<'PY'
import hashlib, json, os, platform, subprocess
from pathlib import Path

RAP  = Path(os.environ["RAP_ROOT"]).resolve()
S    = Path(os.environ["STAGE_A_OUT"])
SNAP = S / "evaluation_v1/code_baseline"

EVAL_CRITICAL = [
    "navsim/planning/script/run_pdm_score.py",
    "navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml",
    "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
    "navsim/planning/script/config/common/default_evaluation.yaml",
    "navsim/planning/script/config/common/agent/rap_agent.yaml",
    "navsim/planning/script/config/common/train_test_split/navtest.yaml",
    "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
    "navsim/evaluate/pdm_score.py",
    "navsim/agents/abstract_agent.py",
    "navsim/agents/rap_dino/rap_agent.py",
    "navsim/agents/rap_dino/rap_model.py",
    "navsim/agents/rap_dino/rap_features.py",
    "navsim/agents/rap_dino/navsim_config.py",
    "navsim/agents/rap_dino/bevformer/bev_feature_build.py",
    "navsim/agents/rap_dino/bevformer/image_encoder.py",
    "navsim/common/dataclasses.py",
    "navsim/common/dataloader.py",
]

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

files = {}
for rel in EVAL_CRITICAL:
    p = RAP / rel
    assert p.is_file(), f"评测关键文件缺失: {rel}"
    files[rel] = sha(p)

# pdm_planner 整棵子树的滚动摘要
sub = RAP / "navsim/planning/simulation/planner/pdm_planner"
roll = hashlib.sha256()
for p in sorted(sub.rglob("*.py")):
    rel = str(p.relative_to(RAP)).encode()
    roll.update(len(rel).to_bytes(4, "big")); roll.update(rel); roll.update(p.read_bytes())
subtree_digest = roll.hexdigest()

import torch
versions = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda}
import navsim  # compatibility shim
for name in ("pytorch_lightning", "hydra", "omegaconf", "transformers", "mmcv", "mmdet", "timm", "ray"):
    versions[name] = getattr(__import__(name), "__version__", "unknown")

phase_minus1 = (S / "baseline_commit.txt")
out = {
    "purpose": "evaluation code baseline — 6 个 v1 CSV 的代码来源证明（spec §7）",
    "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "phase_minus1_baseline_commit": phase_minus1.read_text().strip() if phase_minus1.is_file() else None,
    "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "eval_critical_file_sha256": files,
    "pdm_planner_subtree_digest": subtree_digest,
    "env_versions": versions,
    "training_vs_evaluation_code_state": (
        "6 个 checkpoint 训练于 Step S.4b 之前的代码状态；本快照为 S.4b 之后的状态。"
        "差异仅为 rendered 分支的零占位，其数值惰性由 S.4b 的逐位相同测试证明，"
        "模型输入与训练时一致。"),
    "freeze_rule": "本快照之后至 Step S.6 结束，navsim/ 下任何代码修改都会作废已产生的评测结果",
}
(SNAP / "code_baseline.json").write_text(
    json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
print(json.dumps({"head_commit": out["head_commit"],
                  "worktree_dirty": out["worktree_dirty"],
                  "n_eval_critical_files": len(files),
                  "pdm_planner_subtree_digest": subtree_digest[:16] + "..."},
                 indent=2, ensure_ascii=False))
PY
```

**Expected**

- `$STAGE_A_OUT/evaluation_v1/code_baseline/` 下写出 `commit.txt`、`dirty.txt`、`evaluation_baseline.patch`、`untracked.txt`、`code_baseline.json`。
- 17 个评测关键文件全部存在并记录 SHA256；`pdm_planner_subtree_digest` 写出。
- **Step S.6 的六次评测结束后必须重跑本 Step 的 python 块**，把结果写到 `code_baseline_recheck.json`，并断言 `eval_critical_file_sha256` 与 `pdm_planner_subtree_digest` 与本次完全一致。不一致 → 评测期间代码被改动，全部 CSV 作废。

---

## Step S.4 v1 evaluator 回归（关系式门禁）

**Context**

- **门禁设计已变更。** runbook Step 2.3 原本要求人工提供 human / constant-velocity 的官方参考 PDMS，冻结进 `v1_reference.json` 后做绝对值比对。但 Step S.2 的核查确认：pinned v1.1 worktree 里**没有**任何 baseline 数值，官方代码库只有运行命令与公式。因此绝对值门禁**无法建立**，改为**关系式门禁**——全部判定条件不依赖任何外部数值，只依赖 agent 定义本身与打分公式。
- **关系式门禁能抓什么**：非确定性、scorer 接反、metric cache 与场景错位、归一化错误、全 NaN / 全 0 / 常数输出、聚合公式实现错误。
- **抓不到什么**：让两个 baseline **同比例**偏移的系统性错误（例如地图版本用错）。这一块由 `v1_paper_reference.json` 的论文数值做**软交叉核对**补上——记录并说明，**不作 HALT 条件**（论文数值未必对应 pin 的 commit `0811876c`）。
- **门禁阈值必须在跑之前冻结**到 `$STAGE_A_OUT/evaluator_validation/v1_relational_gate.json`。先跑再调阈值等于没有回归测试。
- **运行复用**：若已执行 `batch_lqr` A/B 实验，其中「RAP 版 × 2」两次 constant-velocity 全量 navtest 跑可直接作为本 Step 的 G2 确定性证据，不必重复。本 Step 的边际成本因此只有 `human_agent` 一次。
- 三次跑均为全量 navtest 12,146 scenes。**必须提交到有 GPU 的计算节点**：`run_pdm_score.py` 硬编码 `Task(num_gpus=1)` 且忽略 `worker=` 覆写，登录节点或无 GPU 节点会永久挂起，即便这两个 baseline 本身不吃图像也不吃 GPU。

**Commands**

先冻结门禁：

```bash
python - <<'PY'
import json, os
from pathlib import Path
gate = {
  "_frozen_before_any_baseline_run": True,
  "_design": "关系式门禁：不依赖外部参考数值，只依赖 agent 定义与 PDMS 聚合公式",
  "G1_schema": {"columns": ["token","valid","NC","DAC","EP","TTC","C","DDC","PDMS"],
                "n_rows": 12146, "all_valid": True, "sorted_by_token": True,
                "forbidden_columns": ["TLC","LK","HC","EC","EPDMS"]},
  "G2_determinism": "两次 constant_velocity 的 DataFrame 完全相同",
  "G3_range": {"min": 0.0, "max": 1.0, "columns": ["NC","DAC","EP","TTC","C","DDC","PDMS"]},
  "G4_non_degenerate": {"pdms_std_gt": 0.0, "pdms_distinct_values_gt": 10,
                        "cv_pdms_mean_open_interval": [0.05, 0.60],
                        "human_pdms_mean_interval": [0.80, 1.00]},
  "G5_ordering": {"human_minus_cv_pdms_mean_gt": 0.30},
  "G6_human_consistency": {"_basis": "human_agent 重放日志记录的专家轨迹，按构造不应碰撞、不应驶出可行驶区域",
                           "nc_mean_ge": 0.99, "dac_mean_ge": 0.99, "ttc_mean_ge": 0.99},
  "G7_formula": {"_formula": "PDMS = NC*DAC*(5*EP + 5*TTC + 2*C + 0*DDC)/12",
                 "_basis": "docs/rap-navsim-v1-evaluation-pipeline.md §4.C；driving_direction_weight=0",
                 "_must_be_row_wise": "PDMS 是逐场景乘性量，禁止在均值上做重构校验",
                 "row_tolerance": 1e-6,
                 "zero_rule": "NC*DAC == 0 的行，PDMS 必须为 0"},
  "advisory_only": {"source": "v1_paper_reference.json", "gating": False}
}
p = Path(os.environ["STAGE_A_OUT"]) / "evaluator_validation/v1_relational_gate.json"
assert not p.exists(), f"门禁已冻结，禁止覆盖: {p}"
p.write_text(json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
print(f"gate frozen: {p}")
PY
```

再跑三次 baseline（`constant_velocity_repeat` 可复用 A/B 实验的第二次 RAP 版跑）：

```bash
for A in constant_velocity_agent human_agent; do
  python navsim/planning/script/run_pdm_score.py agent="$A" train_test_split=navtest \
    experiment_name="stage_a_evalcheck_$A" output_dir="$STAGE_A_OUT/evaluator_validation/$A" \
    output_csv="$STAGE_A_OUT/evaluator_validation/${A}.csv"
done
python navsim/planning/script/run_pdm_score.py agent=constant_velocity_agent train_test_split=navtest \
  experiment_name=stage_a_evalcheck_repeat \
  output_dir="$STAGE_A_OUT/evaluator_validation/constant_velocity_repeat" \
  output_csv="$STAGE_A_OUT/evaluator_validation/constant_velocity_repeat.csv"
```

判定：

```bash
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

S = Path(os.environ["STAGE_A_OUT"]) / "evaluator_validation"
gate = json.loads((S / "v1_relational_gate.json").read_text())
COLS = gate["G1_schema"]["columns"]
SUB = ["NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS"]

cv  = pd.read_csv(S / "constant_velocity_agent.csv")
cv2 = pd.read_csv(S / "constant_velocity_repeat.csv")
hu  = pd.read_csv(S / "human_agent.csv")
mapping = json.loads((Path(os.environ["STAGE_A_OUT"]) / "summary/token_to_log.json").read_text())["mapping"]

fail, rec = [], {}
def check(cond, msg):
    if not cond: fail.append(msg)

# G1 schema
for name, df in (("cv", cv), ("cv_repeat", cv2), ("human", hu)):
    check(list(df.columns) == COLS, f"G1 {name} 列不符: {list(df.columns)}")
    check(len(df) == 12146 and df["token"].nunique() == 12146, f"G1 {name} 行数/唯一 token: {len(df)}")
    check(bool(df["valid"].all()), f"G1 {name} 存在 valid=False")
    check(df[SUB].notna().all().all(), f"G1 {name} 存在 NaN")
    check(list(df["token"]) == sorted(df["token"]), f"G1 {name} 未按 token 排序")
    check(not (set(gate["G1_schema"]["forbidden_columns"]) & set(df.columns)), f"G1 {name} 含 v2-only 列")
    check(set(df["token"]) == set(mapping), f"G1 {name} token 集与 token_to_log 不一致")

# G2 determinism
check(cv.equals(cv2), "G2 两次 constant_velocity 不是完全相同 → evaluator 非确定性")

# G3 range
lo, hi = gate["G3_range"]["min"], gate["G3_range"]["max"]
for name, df in (("cv", cv), ("human", hu)):
    bad = {c: [float(df[c].min()), float(df[c].max())] for c in SUB
           if df[c].min() < lo - 1e-9 or df[c].max() > hi + 1e-9}
    check(not bad, f"G3 {name} 越界: {bad}")

# G4 non-degenerate
g4 = gate["G4_non_degenerate"]
for name, df in (("cv", cv), ("human", hu)):
    check(float(df["PDMS"].std()) > g4["pdms_std_gt"], f"G4 {name} PDMS 无方差（常数输出）")
    check(int(df["PDMS"].nunique()) > g4["pdms_distinct_values_gt"], f"G4 {name} PDMS 取值过少")
cv_m, hu_m = float(cv["PDMS"].mean()), float(hu["PDMS"].mean())
a, b = g4["cv_pdms_mean_open_interval"]
check(a < cv_m < b, f"G4 cv PDMS 均值 {cv_m:.4f} 不在 ({a},{b})")
a, b = g4["human_pdms_mean_interval"]
check(a <= hu_m <= b, f"G4 human PDMS 均值 {hu_m:.4f} 不在 [{a},{b}]")

# G5 ordering
check(hu_m - cv_m > gate["G5_ordering"]["human_minus_cv_pdms_mean_gt"],
      f"G5 human({hu_m:.4f}) 未显著高于 cv({cv_m:.4f})")

# G6 human consistency
g6 = gate["G6_human_consistency"]
for col, key in (("NC", "nc_mean_ge"), ("DAC", "dac_mean_ge"), ("TTC", "ttc_mean_ge")):
    m = float(hu[col].mean())
    check(m >= g6[key], f"G6 human {col} 均值 {m:.4f} < {g6[key]}")

# G7 逐行公式重构
tol = gate["G7_formula"]["row_tolerance"]
for name, df in (("cv", cv), ("human", hu)):
    mult = df["NC"].to_numpy() * df["DAC"].to_numpy()
    weighted = (5 * df["EP"].to_numpy() + 5 * df["TTC"].to_numpy() + 2 * df["C"].to_numpy()) / 12.0
    resid = np.abs(mult * weighted - df["PDMS"].to_numpy())
    rec[f"{name}_formula_max_residual"] = float(resid.max())
    rec[f"{name}_formula_rows_over_tol"] = int((resid > tol).sum())
    check(resid.max() <= tol,
          f"G7 {name} 逐行重构失败: max|resid|={resid.max():.3e}, "
          f"越界行={int((resid > tol).sum())}。若残差与「EP 未归一化」一致需人工确认")
    zero = mult == 0
    check(bool(np.all(df["PDMS"].to_numpy()[zero] == 0)),
          f"G7 {name} 存在 NC*DAC==0 但 PDMS!=0 的行")

rec.update({"cv_pdms_mean": cv_m, "human_pdms_mean": hu_m,
            "cv_pdms_mean_pct": cv_m * 100, "human_pdms_mean_pct": hu_m * 100,
            "human_minus_cv": hu_m - cv_m,
            "cv_subscore_means": {c: float(cv[c].mean()) for c in SUB},
            "human_subscore_means": {c: float(hu[c].mean()) for c in SUB}})

# 软交叉核对（非门禁）
paper = json.loads((S / "v1_paper_reference.json").read_text())
adv = {}
for key, measured in (("constant_velocity_agent", cv_m), ("human_agent", hu_m)):
    band = paper["advisory_bands"][key]
    d = abs(measured * 100 - band["reference"])
    adv[key] = {"measured_pct": measured * 100, "paper_pct": band["reference"],
                "abs_diff_pct": d,
                "band": "green" if d <= band["green"] else "yellow" if d <= band["yellow"] else "red"}
rec["advisory_vs_paper"] = adv

out = {"status": "passed" if not fail else "failed", "failures": fail,
       "gate": "v1_relational_gate.json", "measurements": rec}
(S / "v1_regression.json").write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
print(json.dumps({"status": out["status"], "failures": fail,
                  "cv_pdms_pct": rec["cv_pdms_mean_pct"], "human_pdms_pct": rec["human_pdms_mean_pct"],
                  "advisory_vs_paper": adv}, indent=2, ensure_ascii=False))
assert not fail, f"关系式门禁未通过 → HALT: {fail}"
PY
```

**Expected**

- `v1_relational_gate.json` 在任何 baseline 运行**之前**写出，且不覆盖已有文件。
- `v1_regression.json` 的 `status == "passed"`，G1–G7 全部满足。任一失败 → HALT。
- **G7 是本 Step 最强的一条**：逐行验证 `PDMS = NC·DAC·(5·EP+5·TTC+2·C)/12`，一次性覆盖乘性惩罚、加权聚合、DDC 权重为 0 三件事，且不需要任何外部参考值。失败时先看残差是否与「EP 列存的是未归一化的原始进展」一致，这是诊断信息而非直接判死。
- **软交叉核对（不 HALT）**：`advisory_vs_paper` 落 `green` 正常；`yellow` 在报告中记录并说明；`red` 停下排查后由人决定是否继续。参考值来自论文 Table 1（navtest），Table 3 的 v1.1 leaderboard 给出同一个 Constant Velocity = 20.6，说明该锚点跨版本稳定。
- ⚠️ 禁止在看到 measured 值之后修改 `v1_relational_gate.json` 或 `advisory_bands`。

---

## Step S.4b rendered 占位修复（RAP 评测的硬阻塞）

**Context**

- 依据 `docs/rap-navsim-v1-evaluation-pipeline.md` §6.4，并已逐行复核：
  - `run_pdm_score.py:59-64` 构造 `SceneLoader` 时**不传** `rendered_sensor_blobs_path`，于是 `Cameras.from_camera_dict` 得到 `rendered_image=None`；
  - `bev_feature_build.py:76` 的 `_get_bev_feature` **无条件**先执行 `LoadMultiViewImageFromFiles(agent_input, synthetic=True)`，四路都拿不到图 → `validity_list` 为空 → `bev_feature_build.py:69-70` 抛 `ValueError("Expected four camera views, got 0")`；
  - 该异常被 `run_pdm_score.py:94-98` 逐 token 捕获 → **每个 token 都被标 `valid=False`**，job 正常退出，CSV 全空、PDMS 为 NaN。
- **这是静默全失败**：不崩、不报错、退出码 0。Step S.5/S.6 的 `valid` 断言会拦住它，但那是在浪费 6×12,146 场景之后。必须先修。
- **数值惰性已验证**：`_get_bev_feature` 返回的 `camera_feature`、`camera_valid`、`img_shape`、`lidar2img` 全部取自 real 分支（`bev_feature_build.py:95-100`），`RAPModel.forward` 只消费 `features["camera_feature"]`（`rap_model.py:116`）。synthetic 分支的产物 `rendered_camera_feature` 在推理路径上无消费者。因此用零张量占位**不改变任何模型输出**。
- **不采用「接线真实 rendered 图」方案**：`$RASTER_4CAM_ROOT` 按 spec §3 只含 54 个 train/val log，10 个 navtest log 被显式排除；源 raster root 虽有 test log，但只有 F0/L0/R0 三路，仍会因 `got 3` 抛错。为一批推理时即被丢弃的图像去渲染 8,328 帧 navtest B0，代价与收益完全不成比例。
- **修改必须严格受限**，否则会掩盖训练期真实的数据缺失：
  1. 新增开关默认 **false**，训练路径行为逐位不变；
  2. 只在 `pdm_scoring` 配置链上显式置 true；
  3. 仅当四路 rendered **全部**缺失时才占位；**部分**缺失仍须抛错（部分缺失是数据损坏，不是评测场景）；
  4. 每个 worker 只 log 一次，落盘到评测输出目录；
  5. 禁止改动 real 分支的任何一行，禁止改动 `camera_valid` 的来源。

**Commands**

实现后运行：

```bash
python -m pytest tests/training/test_alignment_experiment.py -q
python -m pytest tests/ -q -k "rendered or placeholder or bev_feature"
```

**Expected**

- 既有测试全部通过，数量不低于 Step 3.4 记录的 39 passed。
- 新增测试至少覆盖三项：
  - **开关关闭时**，四路 rendered 缺失仍抛 `ValueError`（训练期严格性未被削弱）；
  - **开关开启时**，对一个 rendered 图像**存在**的 train token，占位路径与正常路径产出的 `camera_feature`、`camera_valid`、`img_shape`、`lidar2img` **逐位相同**——这是「不改变模型输入」的直接证据，不可用「代码看起来没动 real 分支」替代；
  - 开关开启且四路缺失时，`rendered_camera_feature` 全为零，且 `camera_valid` 仍取自 real 分支。
- 部分缺失（如三路有、一路无）在开关开启时**仍然抛错**。

---

## Step S.5 Checkpoint 加载 smoke

**Context**

- `RAPAgent` 有两条 checkpoint 加载路径，行为不同，且 `checkpoint_path` 非空时**两条都会执行**：
  - `init_from_pretrained()`（`rap_agent.py:82-101`）在 `__init__` 中调用，`strict=False`，key 变换是 `removeprefix('agent.')`，missing/unexpected keys 只 `print` 不抛错。
  - `initialize()`（`rap_agent.py:110-122`）是 inference 路径，`strict=True`（默认），key 变换是 `replace("agent._rap_model", "_rap_model")`。
- 训练 checkpoint 由 `AgentLightningModule` 保存，两种变换对同一份 state_dict 的处理不同。`git log` 中 `5fd8630 #10 fix checkpoint bug` 表明此处有历史问题。
- 已排除的风险：`use_spatial_align` / `use_global_align` 只在 `agent_lightning_module.py:45-46` 的训练 loss 路径被消费，不影响模型结构，两条件 checkpoint 的 key 集合一致（各 932 tensors）。**评测时不需要设置 align 开关。**
- 训练日志出现 72 条 `ms_deformable_*: invalid configuration argument` 与 5 条 `invalid value encountered in arccos`。训练时被 batch 平均掩盖，但评测是逐场景出分，单场景的 deformable attention 失败会直接污染该场景的轨迹。本 Step 顺带确认它在 eval 路径是否出现。
- 只用 **1 个** checkpoint、**8 个** scene。产物写入独立目录，不得污染 `evaluation_v1/`。

**Commands**

```bash
mkdir -p "$STAGE_A_OUT/eval_smoke"
python navsim/planning/script/run_pdm_score.py agent=rap_agent \
  agent.checkpoint_path="$STAGE_A_OUT/training/r100-fullalign/seed_0/checkpoints/last.ckpt" \
  agent.config.dino_init_from_pretrained=false \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.trajectory_sampling.interval_length=0.5 \
  agent.config.train_metric_cache_path="$NAVSIM_EXP_ROOT/train_metric_cache" \
  train_test_split=navtest train_test_split.scene_filter.max_scenes=8 \
  experiment_name=stage_a_eval_smoke \
  output_dir="$STAGE_A_OUT/eval_smoke" \
  output_csv="$STAGE_A_OUT/eval_smoke/smoke.csv" \
  2>&1 | tee "$STAGE_A_OUT/eval_smoke/smoke.log"

python - <<'PY'
import json, os, re
from pathlib import Path
import pandas as pd

S = Path(os.environ["STAGE_A_OUT"]) / "eval_smoke"
log = (S / "smoke.log").read_text()
missing   = re.findall(r"Missing keys when loading pretrained weights: (.*)", log)
unexpected= re.findall(r"Unexpected keys when loading pretrained weights: (.*)", log)
deform    = len(re.findall(r"ms_deformable_\w*: invalid configuration argument", log))
arccos    = len(re.findall(r"invalid value encountered in arccos", log))

df = pd.read_csv(S / "smoke.csv")
COLS = ["token", "valid", "NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS"]
out = {"missing_keys": missing, "unexpected_keys": unexpected,
       "ms_deformable_errors": deform, "arccos_warnings": arccos,
       "rows": len(df), "columns_ok": list(df.columns) == COLS,
       "all_valid": bool(df["valid"].all()),
       "pdms_finite": bool(df["PDMS"].notna().all()),
       "pdms_mean": float(df["PDMS"].mean())}
(S / "smoke_report.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))

assert not [m for m in missing if m.strip() not in ("[]", "")], f"加载缺 key: {missing}"
assert not [u for u in unexpected if u.strip() not in ("[]", "")], f"加载多 key: {unexpected}"
assert out["columns_ok"] and out["all_valid"] and out["pdms_finite"]
PY
```

**Expected**

- 无 missing / unexpected keys；8 行、9 列、全部 `valid`、PDMS 有限。
- `ms_deformable_errors == 0` 时可继续。**> 0 时 HALT for human**：需先确认该 CUDA kernel 报错是否影响输出轨迹，不得直接跑 6×12,146。
- `arccos_warnings > 0` 记录即可，不阻塞。

---

## Step S.6 Step 5.1 — 6 次 v1 评测

**Context**

- Step S.3、S.4、S.5 全部 passed。
- 严格使用 runbook Step 5.1 的命令，不增删任何 override。**不设 align 开关**（见 Step S.5 Context）。
- 本 Step 补上 runbook 缺失的可执行断言块。

**Commands**

```bash
for COND in r100-noalign r100-fullalign; do for SEED in 0 1 2; do
  python navsim/planning/script/run_pdm_score.py agent=rap_agent \
    agent.checkpoint_path="$STAGE_A_OUT/training/$COND/seed_$SEED/checkpoints/last.ckpt" \
    agent.config.dino_init_from_pretrained=false \
    agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.trajectory_sampling.interval_length=0.5 \
    agent.config.train_metric_cache_path="$NAVSIM_EXP_ROOT/train_metric_cache" \
    train_test_split=navtest experiment_name="stage_a_v1_${COND}_seed${SEED}" \
    output_dir="$STAGE_A_OUT/evaluation_v1/${COND}_seed${SEED}" \
    output_csv="$STAGE_A_OUT/evaluation_v1/${COND}_seed${SEED}.csv"
done; done
```

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import pandas as pd

S = Path(os.environ["STAGE_A_OUT"])
E = S / "evaluation_v1"
COLS = ["token", "valid", "NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS"]
names = [f"{c}_seed{s}.csv" for c in ("r100-noalign", "r100-fullalign") for s in (0, 1, 2)]

paths = [E / n for n in names]
assert all(p.is_file() for p in paths), f"缺少 CSV: {[p.name for p in paths if not p.is_file()]}"
assert len(sorted(E.glob("*.csv"))) == 6, f"evaluation_v1 下有多余 CSV: {[p.name for p in sorted(E.glob('*.csv'))]}"

frames, token_sets, summary = {}, {}, {}
for p in paths:
    df = pd.read_csv(p)
    assert list(df.columns) == COLS, f"{p.name} 列不符: {list(df.columns)}"
    assert len(df) == 12146, f"{p.name} 行数 {len(df)}"
    assert df["token"].nunique() == 12146, f"{p.name} token 不唯一"
    assert bool(df["valid"].all()), f"{p.name} 存在 valid=False"
    assert df[COLS[2:]].notna().all().all(), f"{p.name} 存在 NaN 分数"
    assert list(df["token"]) == sorted(df["token"]), f"{p.name} 未按 token 排序"
    frames[p.name] = df
    token_sets[p.name] = set(df["token"])
    summary[p.name] = {"pdms_mean": float(df["PDMS"].mean()),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

base = token_sets[names[0]]
for n, s in token_sets.items():
    assert s == base, f"{n} 的 token 集合与 {names[0]} 不一致"

# token 集合必须与 cluster key 映射完全对齐
mapping = json.loads((S / "summary/token_to_log.json").read_text())["mapping"]
assert base == set(mapping), "评测 token 集合与 token_to_log.json 不一致"

# 6 个 run 不得出现两两完全相同的分数矩阵（说明 checkpoint 未真正生效）
sigs = {n: hashlib.sha256(df[COLS[2:]].to_numpy().tobytes()).hexdigest() for n, df in frames.items()}
dup = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if sigs[a] == sigs[b]]
assert not dup, f"以下 run 的分数完全相同，checkpoint 可能未生效: {dup}"

(E / "step_5_1_audit.json").write_text(json.dumps(
    {"n_csv": 6, "tokens": len(base), "logs": len(set(mapping.values())),
     "per_run": summary, "score_signatures": sigs, "status": "passed"},
    indent=2, sort_keys=True) + "\n")
print(json.dumps({n: v["pdms_mean"] for n, v in summary.items()}, indent=2, sort_keys=True))
print("Step 5.1 audit passed")
PY
```

**Expected**

- 6 个 CSV，各 12,146 唯一 token、token 集合完全相同且与 `token_to_log.json` 对齐、全部 `valid`、严格 9 列且按 token 排序、无 NaN。
- 6 个 run 的分数矩阵两两不同。
- `step_5_1_audit.json` 写出，`status == "passed"`。
- 六次评测结束后立即重跑 Step S.3c 的 python 块，输出写到 `code_baseline_recheck.json`，并断言 `eval_critical_file_sha256` 与 `pdm_planner_subtree_digest` 与 S.3c 完全一致。不一致 → 评测期间代码被改动，6 个 CSV 全部作废。
- **本 Step 只做验收，不做任何统计解读。** 打印 PDMS 均值仅用于确认数值合理（非 0、非 1、非常数）；禁止在此对两条件差异做任何评论或据此调整后续步骤。

---

## Step S.7 v2 延期记录

**Context**

- runbook Step 2.4 原文：「No local scorer substitution. No downgrade to a v1-only official experiment.」本轮的 v1-only 决定构成对该条的显式偏差，必须留痕。
- 实现-only Step，只写记录文件。

**Commands**

```bash
python - <<'PY'
import json, os, subprocess
from pathlib import Path
S = Path(os.environ["STAGE_A_OUT"])
rec = {
  "deviation": "runbook Step 2.4 — 'No downgrade to a v1-only official experiment'",
  "decision": "本轮只执行 NAVSIM v1.1 评测，v2.2 EPDMS 延期",
  "decided_by": "human",
  "date": "2026-08-20",
  "rationale": "本次 run 已按 summary_protocol.json 的 run_scope 记录为 pipeline_rehearsal_v1_only："
               "400 optimizer steps、under-trained regime、候选池受 real 图像可用性非随机筛选。"
               "spec §5 与 runbook Step 5.3 在任何情况下都禁止单边官方结论，因此补齐 v2 不会使本轮"
               "获得可下结论的地位；v2 evaluator 的建设成本转移到补全数据后的正式 Stage-A run。",
  "consequences": [
    "co-primary family {PDMS, EPDMS} 不完整；EPDMS 记为 not_tested，不得记为 pass 或 fail",
    "禁止对 PDMS 施加 m=1 的 Holm 校正；p 值按未校正值报告并标注 descriptive / non-confirmatory",
    "stage_a_report.json 的 status 恒为 incomplete",
    "本轮结果不得作为 Stage-B 的效应量先导估计"
  ],
  "deferred_deliverables": [
    "NAVSIM_V2_ROOT 与完整 NAVSIM_V2_COMMIT",
    "official v2.2 evaluator environment lock",
    "RAP last.ckpt -> submission adapter 及 schema test",
    "v2_adapter_command.txt",
    "official human/baseline regression 命令与参考值",
    "navhard_two_stage 数据集"
  ],
  "future_rule": "补做 v2 时必须沿用 summary_protocol.json 已冻结的 co-primary family 与 Holm 规则；"
                 "EPDMS 届时不得被视为新增假设。",
  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
p = S / "evaluator_validation/v2_deferral_record.json"
p.write_text(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
print(f"written: {p}")
PY
```

**Expected**

- `v2_deferral_record.json` 写出。`SKIPPED.md` 保留不动。

---

## Step S.8 Step 5.3 — v1-only 汇总

**Context**

- Step S.6 passed。实现 `navsim/planning/script/summarize_stage_a.py`。
- **操作协议为 `summary_protocol_v2.json`**（路线 A 决定后按 v1 的 `amendment_rule` 新建）。v1 `summary_protocol.json` 原样保留、禁止修改，仅作为原始预注册凭据。
- **脚本的全部统计行为由 v2 驱动，不得内联任何未在协议中冻结的规则。** 脚本启动时必须做两项校验，任一失败 → HALT：
  1. v1 文件的 SHA256 == `252994acb610806b59ee7653f6b7f27a1e3ab470d1efba7e9de045dba900945d`，且等于 Step S.1 的 `token_to_log.json` 中记录的 `protocol_sha256` —— 证明 v1 未被追溯性篡改；
  2. v2 的 `supersedes_sha256` 字段等于上述值。
- v2 相对 v1 只改动了 scorer 相关条目与报告章节要求；`pairing` / `cluster_bootstrap` / `metric_classification` / `continuous_metrics` / `discrete_metrics` / `noise_floor` / `practical_regression_threshold` / `hypothesis_family` 全部逐字未变，见 v2 的 `amendment.unchanged_from_v1`。
- 必须实现的行为，逐条对应协议字段：
  - `pairing`：先 seed 内配对，再跨 seed 取均值得到 `delta_t`；额外输出 per-seed 结果。
  - `cluster_bootstrap`：以 `token_to_log.json` 的 `log_name` 为簇，整簇有放回重采样，`n_resamples=10000`，`default_rng(20260820)`，percentile 95% CI，双侧 bootstrap p。
  - `metric_classification`：按观测取值个数自动判定 discrete / continuous，分类结果写入报告。
  - `continuous_metrics`：mean / median / 5 个分位数 / CI / p / `cohens_dz` / `paired_dominance`。禁止对连续指标做二值化。
  - `discrete_metrics`：完整转移表（per-seed + 合并），regression / rescue 计数，禁止只报 net。
  - `noise_floor`：6 个 same-condition seed 对序列，走完全相同的流程。
  - `practical_regression_threshold`：由 noise_floor 导出，仅用于标注，不用于二值化。
  - `hypothesis_family`：EPDMS 记为 `not_tested`；**禁止**在 m=1 下施加 Holm；PDMS 的 p 标注为 `descriptive_non_confirmatory`。
  - `report_status`：`status` 恒为 `incomplete`，首页声明原因。
  - `required_sections`：全部章节必须存在，缺一 → 脚本自身报错。
- `evaluation_v2` 目录为空是**预期状态**，脚本必须正常产出 `status=incomplete` 的报告，不得崩溃、也不得静默把 family 缩成 m=1。
- 报告必须原样嵌入 `data_provenance_caveats_required_in_report` 的全部条目、`distribution_audit.json` 的完整 summary、以及 `convergence_gate` 的 under-trained regime 判定。
- 增加测试：协议 SHA 不匹配时 HALT；`evaluation_v2` 缺失时 `status == "incomplete"` 且 EPDMS 为 `not_tested`；bootstrap 在固定 seed 下可复现；转移表单元格之和等于场景数。

**Commands**

```bash
python navsim/planning/script/summarize_stage_a.py \
  --evaluation-v1 "$STAGE_A_OUT/evaluation_v1" \
  --evaluation-v2 "$STAGE_A_OUT/evaluation_v2" \
  --token-to-log "$STAGE_A_OUT/summary/token_to_log.json" \
  --distribution-audit "$STAGE_A_OUT/input_data/distribution_audit.json" \
  --protocol "$STAGE_A_OUT/summary/summary_protocol_v2.json" \
  --protocol-v1 "$STAGE_A_OUT/summary/summary_protocol.json" \
  --scorer-equivalence "$STAGE_A_OUT/evaluator_validation/scorer_equivalence_record.json" \
  --convergence-curve-audit "$STAGE_A_OUT/summary/convergence_curve_audit.json" \
  --out-dir "$STAGE_A_OUT/summary"

python -c 'import json,os;d=json.load(open(os.environ["STAGE_A_OUT"]+"/summary/stage_a_report.json"));assert d["status"]=="incomplete";assert d["hypothesis_family"]["EPDMS_v2_navhard_two_stage"]=="not_tested";print("report status OK")'
```

**Expected**

- `$STAGE_A_OUT/summary/stage_a_report.json` 与 `paired_deltas.csv` 写出。
- `status == "incomplete"`，EPDMS `not_tested`，PDMS 的 p 标注为 descriptive。
- 报告首页含 `run_scope` 的 `permitted_use` / `forbidden_use` 原文。
