# RAP Stage-A — Phase 4 Preflight (Step P.1–P.4)

Target reader: AI coding agent. Spec of record: `docs/rap-alignment-experiment-plan.md`。Runbook: `docs/rap-alignment-stage-a-runbook.md`。On conflict, spec wins。

前置：Phase 3 已完成（Step 3.4 PASS，39 tests passed）。本文件的全部 Step 在 runbook Step 4.1 **之前**执行。Step P.1 的 verdict 决定后续分支。

**触发原因**：Phase 1 实测 qualified train/val = 7,495/1,394，对应 spec §2 的 34,468/7,396，通过率约 21%；target cache 41,864 正常，说明损耗全部发生在候选资格判定层。该差异使 Stage-A 的 optimizer steps 从 spec §4 冻结的 ≈1,800 降到 400，改变了 spec §4 预注册解释规则所依据的数字。Phase 4 启动前必须归因并修正。

## 全局禁令

贯穿本文件全部 Step，违反任一条 → HALT：

1. **只读诊断。** 禁止运行 `build_alignment_small_data.py`、`run_dataset_caching.py`、`build_4cam_root.py`、`run_train_metric_caching.py`，或任何写入 `$STAGE_A_OUT/paired_cache`、`$STAGE_A_OUT/target_cache`、`$RASTER_4CAM_ROOT`、`b0_generation_manifest.jsonl`、`stage_a_token_manifest.json`、`dataset_manifest.json` 的命令。
2. **禁止重抽样。** spec §3：审计结果只作记录，不得据此更换抽样算法、参数或重抽。禁止修改 `select_hash_proportional`、`distribution_audit_thresholds.json` 或任何已冻结阈值。
3. **遇到 `VersionPollutionError` → HALT 报告。** 禁止删除、清空或绕过任何已冻结的 manifest / cache / raster root。
4. 禁止修改 `navsim/` 下任何训练相关代码。本文件只允许修改 `docs/rap-alignment-experiment-plan.md`，且仅限 Step P.3 列出的位置。
5. 新增产物一律写入 `$STAGE_A_OUT/input_data/`，禁止覆盖已存在的 Phase 0–3 产物。

## Session prelude

```bash
set -euo pipefail
source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env
cd "$RAP_ROOT"
```

---

## Step P.1 候选池损耗归因

**Context**

- `_audit_candidate`（`navsim/planning/script/build_alignment_small_data.py:268-334`）是唯一的资格判定入口，可抛出七类失败：4-history/10-future window 不全、`is_valid=False` 或 `roadblock_ids` 为空、target token/shape 非法、相机路径非规范、**real 或 raster 图像读失败**、real/raster shape 不一致、三路 raster 全黑。
- `candidate_pool.json` 的 `excluded_candidates` 为每个被排除 token 记录了 `repr(error)`，`FileNotFoundError` 的 repr 内含完整路径，可据此区分 real 与 raster。
- 目的：判定 21% 通过率是**可修复的数据缺失**还是**真实数据限制**。两者对 Phase 4 是否值得启动给出相反答案。
- 本 Step 纯只读。

**Commands**

```bash
python - <<'PY'
import collections, json, os, re
from pathlib import Path

S = Path(os.environ["STAGE_A_OUT"])
pool = json.loads((S / "input_data/candidate_pool.json").read_text())
excluded = pool["excluded_candidates"]
real_root = pool["sources"]["real_root"]
raster_root = pool["sources"]["raster_root"]

def classify(reason: str) -> str:
    if reason.startswith("FileNotFoundError"):
        if real_root in reason:   return "missing_real_image"
        if raster_root in reason: return "missing_raster_image"
        return "missing_image_other"
    for needle, label in (
        ("incomplete 4-history/10-future window", "window"),
        ("invalid frame or missing route",        "route_or_invalid"),
        ("shape mismatch",                        "real_raster_shape_mismatch"),
        ("all checked raster cameras are black",  "black_raster"),
        ("noncanonical camera path",              "noncanonical_camera_path"),
        ("target token mismatch",                 "target_invalid"),
        ("invalid target trajectory",             "target_invalid"),
        ("expected RGB image",                    "decode_invalid"),
        ("non-finite decoded image",              "decode_invalid"),
        ("raster too short for fixed crop",       "raster_too_short"),
        ("metadata match count",                  "metadata_match_count"),
        ("outside official train/val",            "outside_split"),
    ):
        if needle in reason:
            return label
    return "other"

def camera_of(reason: str) -> str:
    m = re.search(r"/(CAM_[A-Z0-9]+)/", reason)
    return m.group(1) if m else "n/a"

cats     = collections.Counter(classify(x["reason"]) for x in excluded)
img_cams = collections.Counter(
    camera_of(x["reason"]) for x in excluded if classify(x["reason"]).startswith("missing_")
)
by_log   = collections.defaultdict(collections.Counter)
samples  = {}
for x in excluded:
    c = classify(x["reason"])
    by_log[x["log_name"]][c] += 1
    samples.setdefault(c, x["reason"][:300])

total     = len(excluded)
qualified = pool["qualified_train"] + pool["qualified_val"]
frac_real = cats["missing_real_image"] / total if total else 0.0
frac_hard = sum(cats[k] for k in ("window", "route_or_invalid", "black_raster")) / total if total else 0.0

if frac_real >= 0.20:
    verdict, action = "data_bug", "HALT for human — real 图像缺失是可修复问题，不得启动 Phase 4"
elif frac_hard >= 0.80:
    verdict, action = "data_limit", "真实数据限制 — 继续 Step P.2"
else:
    verdict, action = "unclear", "HALT for human — 主导原因不属于已知两类"

out = {
    "source_target_count": pool["source_target_count"],
    "qualified_total": qualified,
    "qualified_train": pool["qualified_train"],
    "qualified_val": pool["qualified_val"],
    "excluded_total": total,
    "pass_rate": qualified / (qualified + total) if (qualified + total) else 0.0,
    "categories": dict(cats.most_common()),
    "missing_image_by_camera": dict(img_cams.most_common()),
    "fraction_missing_real_image": frac_real,
    "fraction_hard_data_limits": frac_hard,
    "reason_samples": samples,
    "top_logs_by_exclusions": {
        log: dict(c.most_common(3))
        for log, c in sorted(by_log.items(), key=lambda kv: -sum(kv[1].values()))[:10]
    },
    "verdict": verdict,
    "action": action,
}
(S / "input_data/exclusion_reason_histogram.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({k: out[k] for k in (
    "qualified_total", "excluded_total", "pass_rate", "categories",
    "missing_image_by_camera", "fraction_missing_real_image", "verdict", "action")},
    indent=2, sort_keys=True))
PY
```

独立交叉验证——不依赖 `repr` 分类，直接抽样 stat 磁盘上的真实文件：

```bash
python - <<'PY'
import json, os, pickle
from pathlib import Path
import yaml

S = Path(os.environ["STAGE_A_OUT"])
pool = json.loads((S / "input_data/candidate_pool.json").read_text())
real   = Path(pool["sources"]["real_root"])
raster = Path(pool["sources"]["raster_root"])
logs   = Path(pool["sources"]["log_root"])
sf = yaml.safe_load(open(
    "navsim/planning/script/config/common/train_test_split/scene_filter/paired_54log.yaml"))
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
PROBE = 200

rows, tot_probed, tot_real, tot_raster = {}, 0, 0, 0
for log_name in sf["log_names"]:
    frames = pickle.load(open(logs / f"{log_name}.pkl", "rb"))
    step   = max(1, len(frames) // PROBE)
    probe  = frames[::step][:PROBE]
    r = ra = 0
    for fr in probe:
        paths = [Path(fr["cams"][c]["data_path"]) for c in CAMS]
        r  += all((real / p).is_file() for p in paths)
        ra += all((raster / p).is_file() for p in paths)
    rows[log_name] = {"probed": len(probe), "real_all3": r, "raster_all3": ra,
                      "real_rate": r / len(probe), "raster_rate": ra / len(probe)}
    tot_probed += len(probe); tot_real += r; tot_raster += ra

out = {"cameras": list(CAMS), "probe_per_log": PROBE, "logs": len(rows),
       "probed_frames": tot_probed,
       "real_coverage": tot_real / tot_probed,
       "raster_coverage": tot_raster / tot_probed,
       "logs_with_real_coverage_below_0_9": sorted(
           k for k, v in rows.items() if v["real_rate"] < 0.9),
       "per_log": rows}
(S / "input_data/image_coverage_probe.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: out[k] for k in (
    "probed_frames", "real_coverage", "raster_coverage",
    "logs_with_real_coverage_below_0_9")}, indent=2, sort_keys=True))
PY
```

**Expected**

- `exclusion_reason_histogram.json` 与 `image_coverage_probe.json` 写出。
- 分支规则：
  - `verdict == "data_limit"`（window / route / black raster 合计 ≥ 80%）→ 继续 Step P.2。
  - `verdict == "data_bug"`（`missing_real_image` ≥ 20%）→ 执行 Step P.2 的诊断命令收集完整信息后 **HALT for human**，**不执行 Step P.3，不启动 Phase 4**。修数据属人工决策，禁止自行下载、补齐或重建任何数据版本。
  - `verdict == "unclear"` → 同上 HALT。
- 交叉验证一致性：`real_coverage` 显著低于 1.0 时必须与 `fraction_missing_real_image > 0` 相互印证；两者矛盾（如 repr 分类显示大量 real 缺失但 probe 覆盖率接近 1.0）→ HALT 并报告矛盾，不得自行取信其一。

---

## Step P.2 log 覆盖与 map_location 决策记录

**Context**

- spec §7 验收条款：「每个 official train log 均有 token 被抽中且**四个 `map_location` 全部出现**」。实测 selected train logs = 43/44、val logs = 9/10，该条款当前**未通过**。
- `input_audit.json` 的 `checks`（`build_alignment_small_data.py:676-686`）不含该项，因此它是**静默通过**的——不同于 Step 1.3 的 TVD 被显式拦下。本 Step 补一份等价的人工决策记录。
- 已有说明只解释了 1 个 log（无满足 route 条件的 frame），但 train / val 各缺 1 个共 2 个，第二个缺失原因未交代。本 Step 必须给出两个 log 各自的确切原因。
- `map_location` 覆盖不全比缺 1 个 log 严重得多（直接破坏 geography 覆盖），是硬 HALT。
- 本 Step 纯只读 + 写一份记录 JSON。禁止重抽。

**Commands**

```bash
python - <<'PY'
import collections, json, os
from pathlib import Path

S = Path(os.environ["STAGE_A_OUT"])
man  = json.loads((S / "input_data/stage_a_token_manifest.json").read_text())
pool = json.loads((S / "input_data/candidate_pool.json").read_text())

missing = man["missing_qualified_logs"]
selected = set(man["train_tokens"]) | set(man["val_tokens"])
by_token = {c["token"]: c for c in man["qualified_candidates"]}
locs = collections.Counter(by_token[t]["map_location"] for t in selected if t in by_token)

EXPECTED_LOCS = {"us-ma-boston", "us-nv-las-vegas-strip",
                 "us-pa-pittsburgh-hazelwood", "sg-one-north"}
assert set(locs) == EXPECTED_LOCS, \
    f"map_location 覆盖不全 → HALT（spec §7）: got={sorted(locs)} expected={sorted(EXPECTED_LOCS)}"

n_by_log = collections.Counter((c["split"], c["log_name"]) for c in man["qualified_candidates"])
rates = {}
for split in ("train", "val"):
    for log_name, quota in man["quotas"][split].items():
        n = n_by_log[(split, log_name)]
        rates[f"{split}/{log_name}"] = {"quota": quota, "n_l": n,
                                        "rate": quota / n if n else None}
finite = [v["rate"] for v in rates.values() if v["rate"]]

exc_by_log = collections.defaultdict(collections.Counter)
for x in pool["excluded_candidates"]:
    exc_by_log[x["log_name"]][x["reason"][:160]] += 1
qual_logs = {c["log_name"] for c in man["qualified_candidates"]}

detail = {}
for split, names in missing.items():
    for log_name in names:
        reasons = exc_by_log.get(log_name)
        detail[log_name] = {
            "split": split,
            "in_qualified_pool": log_name in qual_logs,
            "excluded_token_count": sum(reasons.values()) if reasons else 0,
            "top_exclusion_reasons": dict(reasons.most_common(5)) if reasons else {},
            "note": ("no targets in target cache — log 未出现在 candidate_pool 的任何记录中"
                     if not reasons and log_name not in qual_logs else ""),
        }

out = {
    "spec_clause": "spec §7 — 每个 official train log 均有 token 被抽中且四个 map_location 全部出现",
    "clause_status": "FAILED",
    "selected_train_logs": len(man["train_logs"]),
    "expected_train_logs": 44,
    "selected_val_logs": len(man["val_logs"]),
    "expected_val_logs": 10,
    "missing_qualified_logs": missing,
    "missing_log_detail": detail,
    "map_location_coverage": dict(locs),
    "map_location_complete": True,
    "per_log_sampling_rate": {
        "max": max(finite), "min": min(finite), "max_over_min": max(finite) / min(finite),
        "threshold": 1.5, "within_threshold": (max(finite) / min(finite)) <= 1.5,
    },
    "per_log_detail": rates,
    "resolution": "",
    "status": "pending_human",
}
(S / "input_data/log_coverage_resolution.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: out[k] for k in (
    "clause_status", "selected_train_logs", "selected_val_logs",
    "missing_qualified_logs", "missing_log_detail",
    "map_location_coverage", "per_log_sampling_rate", "status")},
    indent=2, sort_keys=True))
PY
```

**Expected**

- 四个 `map_location` 全部出现，否则断言失败 → HALT。
- `per_log_sampling_rate.max_over_min ≤ 1.5`，否则 → HALT（spec §7）。
- `log_coverage_resolution.json` 写出，`status == "pending_human"`，且 `missing_log_detail` 对两个缺失 log **各自**给出确切原因（`no targets in target cache` 或具体的 exclusion reason 分布）。
- **HALT for human ack**：把 `missing_log_detail` 与 `map_location_coverage` 呈给人，等待一句明确的接受或拒绝。人接受后由**人**把 `resolution` 填写为决策理由、`status` 改为 `accepted`；**codex 不得自行修改这两个字段**。未收到 ack 前不得进入 Step P.3。

---

## Step P.3 修正 spec §4 的 Stage-A 冻结数字

**Context**

- **前置：Step P.1 `verdict == "data_limit"` 且 Step P.2 已获人工 ack。** 否则本 Step 不执行——数据若要修复，样本数还会变，此时改 spec 是无效劳动。
- spec §2 / §4 中的 Stage-A 数字（train ≈11,489、val ≈2,465、optimizer steps ≈1,800）与实测不符。这是**事实性修正**，不是 selection-on-result，但必须发生在任何训练启动**之前**：训练后再改等于看过结果修改预注册。
- **允许修改的范围，仅限以下四处的 Stage-A 数值及直接引用它们的句子**：
  1. §2 表格 `Stage-A 1/3 子集` 列的 `≈11,489` / `≈2,465`；
  2. §4「冻结的训练预算」表 Stage-A 列的 `train / val samples` 与 `optimizer steps`；
  3. §4「Stage-A 解释规则」段中的两处 `≈1,800 steps`（一处为「Stage-A ≈1,800 steps 是 Stage-B 的约 1/3、官方配置的约 1/20」，一处为「未能在 ≈1,800 steps 下检出效应」），比例描述须按实测重算；
  4. §4「与 RAP-DINO 官方实现对照」表中「训练数据量」与「总 optimizer steps」两行的 Stage-A 数值。
- **禁止修改**：Stage-B 的任何数字、`epochs`、`effective global batch`、`seeds`、`runs`、λ、lr、weight decay、precision、评测协议、任何阈值。spec §1 明令 Stage-A 不得修改 Stage-B。
- §2 表格「通过检查」列的 34,468 / 7,396 与 §4 Stage-B 列同样与实测不符，但 Stage-B 的合格池要到 Stage-B 全量 B0 补齐后才能测定。**不得**用 Stage-A 的实测值改写 Stage-B 列；改为在 §4 预算表下方加一行脚注，说明 Stage-B 数值为待测定的估算值，以 Stage-B 数据构建时的实测 manifest 为准。
- `optimizer steps` 必须按 `⌈train_samples / 128⌉ × 20` 由实测样本数计算，不得沿用估算值。

**Commands**

```bash
python - <<'PY'
import json, math, os
from pathlib import Path

S = Path(os.environ["STAGE_A_OUT"])
man  = json.loads((S / "input_data/stage_a_token_manifest.json").read_text())
pool = json.loads((S / "input_data/candidate_pool.json").read_text())
res  = json.loads((S / "input_data/log_coverage_resolution.json").read_text())
assert res["status"] == "accepted", f"Step P.2 未获人工 ack（status={res['status']}）→ HALT"

train, val = len(man["train_tokens"]), len(man["val_tokens"])
GLOBAL_BATCH, EPOCHS = 128, 20
per_epoch = math.ceil(train / GLOBAL_BATCH)
out = {
    "qualified_train": pool["qualified_train"], "qualified_val": pool["qualified_val"],
    "stage_a_train_samples": train, "stage_a_val_samples": val,
    "selected_train_logs": len(man["train_logs"]), "selected_val_logs": len(man["val_logs"]),
    "effective_global_batch": GLOBAL_BATCH, "epochs": EPOCHS,
    "optimizer_steps_per_epoch": per_epoch,
    "optimizer_steps_total": per_epoch * EPOCHS,
    "superseded_spec_values": {"train": 11489, "val": 2465, "optimizer_steps": 1800},
    "formula": "ceil(train_samples / effective_global_batch) * epochs",
}
(S / "input_data/measured_training_budget.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY

# 列出全部待改位置，确保无遗漏
grep -n "11,489\|2,465\|1,800\|34,468\|7,396\|≈5,400" docs/rap-alignment-experiment-plan.md
```

按上述范围编辑 `docs/rap-alignment-experiment-plan.md`，然后验证：

```bash
# Stage-A 的三个旧值必须全部消失；Stage-B 的 34,468 / 7,396 / 5,400 必须保留
test "$(grep -c "11,489\|2,465\|1,800" docs/rap-alignment-experiment-plan.md)" -eq 0
grep -q "34,468" docs/rap-alignment-experiment-plan.md
grep -q "7,396"  docs/rap-alignment-experiment-plan.md
git diff docs/rap-alignment-experiment-plan.md > "$STAGE_A_OUT/spec_budget_correction.patch"
git diff --stat docs/rap-alignment-experiment-plan.md
```

**Expected**

- `measured_training_budget.json` 写出，`optimizer_steps_total == ceil(train/128) * 20`。
- 三条 grep 断言全部通过：Stage-A 旧值 0 处残留，Stage-B 数值原样保留。
- `spec_budget_correction.patch` 写出，且 diff **只**触及上述四处及新增的 Stage-B 脚注。diff 中若出现 `epochs`、`batch`、`seed`、`lambda`、`lr`、`weight decay`、`precision`、评测协议或任何阈值的改动 → 立即 `git checkout docs/rap-alignment-experiment-plan.md` 回滚并 HALT。
- 只改 spec，**不**改 runbook；runbook 的数字来自 spec，由 spec 单一来源。

---

## Step P.4 Phase 4 启动门槛

**Context**

- 本 Step 无新命令，只做放行判定。

**Expected** — 以下全部成立才允许执行 runbook Step 4.1：

1. `exclusion_reason_histogram.json` 的 `verdict == "data_limit"`；
2. `log_coverage_resolution.json` 的 `status == "accepted"`，且 `map_location_complete == true`；
3. `measured_training_budget.json` 已写出，spec §4 的 Stage-A 数字已按 Step P.3 修正且 grep 断言通过；
4. `spec_budget_correction.patch` 已保存。

任一不成立 → HALT，不启动 Phase 4。

**并行提醒**：Step 4.1 是 6 个 run × 4×H100 的长时 GPU 作业，期间 CPU 相对空闲。runbook Step 2.3 / 2.4 当前为 SKIPPED，其后果是 6 个 `last.ckpt` **没有通向结论的路径**——spec §5 为 v1+v2 co-primary，Step 5.3 明写缺 official evaluator → HALT 并标记实验不完整。Phase 4 训练窗口应同时推进 Step 2.3 / 2.4 的人工交付物（v1/v2 官方 checkout 与完整 SHA、`v1_reference.json`、v2 adapter 与 schema test、`navhard_two_stage` 数据集）。这些属 CPU/IO 工作，与训练不争 GPU。该项不阻塞 Phase 4 启动。
