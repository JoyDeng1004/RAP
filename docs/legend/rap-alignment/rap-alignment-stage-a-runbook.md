# RAP Paired Alignment — Stage-A Machine Instruction List

Target reader: AI coding agent. Spec of record: `docs/rap-alignment-experiment-plan.md`. On conflict, spec wins. On spec self-conflict or missing external evaluator: HALT.

Rules: execute Steps in order. No parallel, no skip, no continue-after-failure. Every Step's `Expected` must pass before the next Step.

Session prelude — run at the start of every independent Bash session:

```bash
set -euo pipefail
source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env
cd "$RAP_ROOT"
```

## Phase -1

### Step -1.1 Create and verify env file

**Context**
- Create `env/stage_a.env` with an editor. Never commit `HF_TOKEN`.

**Commands**

```bash
mkdir -p /gs/bs/tga-RLA/qdeng/RAP/env
# 用编辑器创建 env/stage_a.env，内容如下；禁止把 HF_TOKEN 写进仓库。
```

```bash
export PATH="/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin:${PATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"
export NAVSIM_EXP_ROOT="/gs/bs/tga-RLA/qdeng/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/gs/bs/tga-RLA/qdeng/RAP"
export RAP_ROOT="/gs/bs/tga-RLA/qdeng/RAP"
export RASTER_SRC_ROOT="${RAP_ROOT}/dataset_norm/rendered_sensor_blobs"
export RASTER_4CAM_ROOT="${RAP_ROOT}/dataset_norm/rendered_sensor_blobs_4cam_v1"
export STAGE_A_OUT="${RAP_ROOT}/outputs/alignment_stage_a"
export HF_HOME="/gs/bs/tga-RLA/qdeng/.cache/huggingface"
export MPLCONFIGDIR="${STAGE_A_OUT}/.matplotlib"

# DINO 架构参考。只读取 img_backbone 的参数名与 shape 用于比对配置；
# 规格书 §4 禁止把它作为 Stage-A 的 initialization 权重，任何步骤都不得 load 它的 tensor。
export REF_DINO_CKPT="${RAP_ROOT}/ckpts/RAP_DINO_navsimv2.ckpt"

# 外部 official evaluator；未填写真实路径/commit 时 Phase 2 不得通过。
export NAVSIM_V1_ROOT=""
export NAVSIM_V1_COMMIT="0811876"       # 必须替换为完整 SHA
export NAVSIM_V2_ROOT=""
export NAVSIM_V2_COMMIT=""              # official v2.2 完整 SHA
export NAVHARD_TWO_STAGE_ROOT="${OPENSCENE_DATA_ROOT}/navhard_two_stage"
```

```bash
set -euo pipefail
source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env
cd "$RAP_ROOT"
mkdir -p "$STAGE_A_OUT"/{input_data,evaluation_v1,evaluation_v2,evaluator_validation,summary} "$MPLCONFIGDIR"
test "$(command -v python)" = "/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python"
for p in "$NUPLAN_MAPS_ROOT" "$OPENSCENE_DATA_ROOT" "$RASTER_SRC_ROOT"; do test -d "$p"; done
test -f "$REF_DINO_CKPT"
# v2 数据集只告警：Phase 0/1 的数据构建完全不依赖它，硬 gate 在 Step 2.4。
# 放进上面的循环会让一个纯 v2 的缺失在第一步就阻塞掉两个 Phase 的可执行工作。
test -d "$NAVHARD_TWO_STAGE_ROOT" || echo "WARN: NAVHARD_TWO_STAGE_ROOT 不存在，Step 2.4 之前必须解决"
python - <<'PY'
import json, os, platform, subprocess, torch
mods = ["pytorch_lightning", "hydra", "omegaconf", "transformers", "mmcv", "mmdet", "timm", "ray"]
versions = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
import navsim  # 先应用 torch/transformers compatibility shim
for name in mods:
    module = __import__(name); versions[name] = getattr(module, "__version__", "unknown")
try: versions["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0]
except Exception as e: versions["driver_error"] = repr(e)
with open(os.environ["STAGE_A_OUT"] + "/env_versions.json", "w") as f: json.dump(versions, f, indent=2, sort_keys=True)
PY
```

**Expected**
- Exit 0. `env_versions.json` written. `NAVHARD_TWO_STAGE_ROOT` missing → WARN only.

---

### Step -1.2 Storage, GPU, DINO architecture gate

**Context**
- Step -1.1 passed. Requires ≥200 GiB free on `$STAGE_A_OUT` filesystem, a GPU allocation, and HF access. Do not train on the login node.
- **DINO 架构约束**：Stage-A 实例化的 DINO backbone，其配置必须与 `$REF_DINO_CKPT` 内 `img_backbone` 的实际形状**逐字段一致**。不一致 → HALT，禁止改 `tf_d_model` 或换 DINO 变体来"凑合跑通"。
- 该约束与规格书 §4「`RAP_DINO_navsimv2.ckpt` Stage-A 禁止使用」并不冲突，二者边界是：**读架构 shape 允许，读权重禁止**。下面的比对通过 zip + pickle metadata 解析参数名与 shape，`persistent_load` 直接返回 `None`，全过程不反序列化任何 storage，也不调用 `torch.load`。任何步骤都不得从该文件加载 tensor。
- 参考值（2026-08-18 实测，仅供人工核对，脚本以运行时解析为准）：`hidden_size=1280`、`num_hidden_layers=32`、`patch_size=16`、`intermediate_size=5120`、`num_register_tokens=4`、gated MLP。即 `facebook/dinov3-vith16plus-pretrain-lvd1689m`。`dinov3-vitl16`（`hidden_size=1024`）不满足本约束。
- `num_attention_heads` 无法从 shape 反推（q/k/v/o 均为 `hidden_size × hidden_size`），故只校验其整除 `hidden_size`，并落盘记录实际值。

**Commands**

```bash
set -euo pipefail; source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env; cd "$RAP_ROOT"
python - <<'PY'
import os, shutil
free = shutil.disk_usage(os.environ["STAGE_A_OUT"]).free
need = 200 * 1024**3
assert free >= need, f"insufficient storage: free={free/1024**3:.1f} GiB, require >=200 GiB"
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import io, json, os, pickle, re, zipfile

# ---- 1) 从参考 ckpt 解析 DINO 架构：只读 shape，不读权重 ----
class _Shape:
    __slots__ = ("size",)
    def __init__(self, size): self.size = size

def _rebuild(*a, **k): return _Shape(tuple(a[2]))

class _MetaOnly(pickle.Unpickler):
    def persistent_load(self, pid): return None          # 不触碰任何 storage
    def find_class(self, mod, name):
        if "rebuild_tensor" in name: return _rebuild
        if (mod, name) == ("collections", "OrderedDict"):
            import collections; return collections.OrderedDict
        return lambda *a, **k: None

ref_path = os.environ["REF_DINO_CKPT"]
zf = zipfile.ZipFile(ref_path)
blob = zf.read([n for n in zf.namelist() if n.endswith("data.pkl")][0])
obj = _MetaOnly(io.BytesIO(blob)).load()
sd = obj.get("state_dict", obj)
bk = {k: v.size for k, v in sd.items() if isinstance(v, _Shape) and "img_backbone" in k}
assert bk, f"{ref_path} 中找不到 img_backbone 参数"

def one(suffix):
    hits = [v for k, v in bk.items() if k.endswith(suffix)]
    assert len(hits) == 1, f"{suffix}: 匹配 {len(hits)} 个，期望 1"
    return hits[0]

ref = {
    "hidden_size":         one("embeddings.cls_token")[-1],
    "num_hidden_layers":   max(int(m.group(1)) for k in bk
                               if (m := re.search(r"\.layer\.(\d+)\.", k))) + 1,
    "patch_size":          one("patch_embeddings.weight")[-1],
    "num_register_tokens": one("embeddings.register_tokens")[1],
    "intermediate_size":   one("layer.0.mlp.up_proj.weight")[0],
    "use_gated_mlp":       any(k.endswith("layer.0.mlp.gate_proj.weight") for k in bk),
}

# ---- 2) 与将要实例化的 HF 配置比对 ----
import navsim  # 先应用 torch/transformers compatibility shim
from transformers import AutoConfig
from navsim.agents.rap_dino.navsim_config import RAPConfig

rap = RAPConfig()
cfg = AutoConfig.from_pretrained(rap.dino_model_name, cache_dir=os.environ["HF_HOME"])
got = {k: getattr(cfg, k) for k in ref}
bad = {k: {"reference_ckpt": ref[k], "hf_config": got[k]} for k in ref if ref[k] != got[k]}
assert not bad, f"DINO 配置与参考 ckpt 不一致 → HALT: {json.dumps(bad, default=str)}"

heads = cfg.num_attention_heads
assert ref["hidden_size"] % heads == 0, f"num_attention_heads={heads} 不整除 hidden_size={ref['hidden_size']}"
assert rap.tf_d_model == ref["hidden_size"], \
    f"tf_d_model={rap.tf_d_model} 与 DINO hidden_size={ref['hidden_size']} 不一致"

out = {"reference_ckpt": ref_path, "dino_model_name": rap.dino_model_name,
       "tf_d_model": rap.tf_d_model, "num_attention_heads": heads,
       "backbone_param_count": len(bk), **{k: got[k] for k in ref}}
with open(os.environ["STAGE_A_OUT"] + "/dino_reference_config.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, default=str)
print("DINO config == reference ckpt OK:", json.dumps(out, sort_keys=True, default=str))
PY
```

**Expected**
- Exit 0；打印 `DINO config == reference ckpt OK: …`；`$STAGE_A_OUT/dino_reference_config.json` 写出。
- 六个字段全部相等，且 `tf_d_model == hidden_size`。任一不等 → HALT for human，不得自行调整任何一侧。

---

### Step -1.3 Code baseline

**Context**
- Step -1.2 passed.

**Commands**

```bash
set -euo pipefail; source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env; cd "$RAP_ROOT"
git rev-parse HEAD > "$STAGE_A_OUT/baseline_commit.txt"
git status --porcelain > "$STAGE_A_OUT/baseline_dirty.txt"
git diff --binary > "$STAGE_A_OUT/baseline.patch"
git ls-files --others --exclude-standard > "$STAGE_A_OUT/baseline_untracked.txt"
python -m pytest tests/training/test_alignment_experiment.py -q | tee "$STAGE_A_OUT/baseline_tests.txt"
grep -n "tf_dropout\|distill_feature_weight\|domain_align_weight" navsim/agents/rap_dino/navsim_config.py
grep -n "WarmupCosLR" navsim/agents/rap_dino/rap_agent.py
```

**Expected**
- Tests pass. `tf_dropout=0`, `distill_feature_weight=0.002`, `domain_align_weight=0.1`, scheduler = 20 epochs / 1 warmup epoch / `min_lr=1e-5`.

---

## Phase 0

### Step 0.1 Freeze source checksums + emit frozen Hydra config

**Context**
- Step -1.3 passed.
- Implement `navsim/planning/script/freeze_source_checksums.py`:
  - Args: `--raster-src-root --log-root --split-config --out --emit-scene-filter`.
  - Emit sorted-key JSON: split YAML SHA256; entry counts 13,180/1,381/1,349; 64 raster logs; split intersection 44/10/10; F0/L0/R0/B0 file counts 51,898/51,898/51,900/48.
  - 44/10/10 = `raster dir ∩ official split`, NOT the split YAML entry counts.
  - Emit the two YAML files below; write their SHA256 and `log_names` length into `source_checksums.json`. Frozen artifacts: regenerate via this script only, never hand-edit.

```yaml
# navsim/planning/script/config/common/train_test_split/scene_filter/paired_54log.yaml
_target_: navsim.common.dataclasses.SceneFilter
_convert_: 'all'
num_history_frames: 4
num_future_frames: 10
frame_interval: 1          # 对齐规格书 §2 的统计条件；null 会退化为 stride 14
has_route: true
max_scenes: null
tokens: null
log_names:                 # 由 raster ∩ official {train,val}_logs 生成，恰好 54 条
  - ...
```

```yaml
# navsim/planning/script/config/common/train_test_split/paired_trainval.yaml
defaults:
  - scene_filter: paired_54log
data_split: trainval
```

**Commands**

```bash
python navsim/planning/script/freeze_source_checksums.py \
  --raster-src-root "$RASTER_SRC_ROOT" --log-root "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  --split-config navsim/planning/script/config/training/default_train_val_test_log_split.yaml \
  --emit-scene-filter navsim/planning/script/config/common/train_test_split \
  --out "$STAGE_A_OUT/input_data/source_checksums.json"
python - <<'PY'
import yaml
sf = yaml.safe_load(open("navsim/planning/script/config/common/train_test_split/scene_filter/paired_54log.yaml"))
split = yaml.safe_load(open("navsim/planning/script/config/common/train_test_split/paired_trainval.yaml"))
assert sf["frame_interval"] == 1, sf["frame_interval"]
assert sf["num_history_frames"] == 4 and sf["num_future_frames"] == 10
assert sf["has_route"] is True and sf["tokens"] is None and sf["max_scenes"] is None
assert len(sf["log_names"]) == 54 and len(set(sf["log_names"])) == 54
assert split["data_split"] == "trainval"
print("paired_54log OK", len(sf["log_names"]))
PY
```

**Expected**
- Prints `paired_54log OK 54`; all JSON counts match the numbers above; `log_names ∩ test_logs = ∅`.

---

### Step 0.2 Train metric cache

**Context**
- Step 0.1 passed. Must use `train_test_split=paired_trainval`. `rap_agent.yaml` default `./train_metric_cache` is forbidden; training and evaluation always override to `${NAVSIM_EXP_ROOT}/train_metric_cache`.

**Commands**

```bash
set -euo pipefail; source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env; cd "$RAP_ROOT"
python navsim/planning/script/run_train_metric_caching.py train_test_split=paired_trainval
python - <<'PY'
import os
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
cache = Path(os.environ["NAVSIM_EXP_ROOT"]) / "train_metric_cache"
csvs = sorted(p for p in (cache / "metadata").iterdir() if p.suffix == ".csv")
assert len(csvs) == 1, f"metadata/ 有 {len(csvs)} 个 CSV，loader 只读其中一个：{[p.name for p in csvs]}"
n = len(MetricCacheLoader(cache).tokens)
Path(os.environ["STAGE_A_OUT"], "metric_cache_count.txt").write_text(f"{n}\n")
assert n >= 41864, f"train metric cache 仅 {n} 个 token；落在约 3,100 说明 stride 仍是 14"
print("train metric cache tokens:", n)
PY
```

**Expected**
- Exactly 1 CSV in `metadata/`; token count ≥ 41,864. Count ≈3,100 → HALT.

---

### Step 0.3 Build 4cam root hard links

**Context**
- Step 0.2 passed.
- Implement `navsim/planning/script/build_4cam_root.py`:
  - Log list source is `paired_54log.yaml` only. Do NOT recompute the intersection from `--raster-src-root + --split-config`.
  - On start, assert the YAML SHA256 equals the value recorded in `source_checksums.json`; mismatch → HALT.
  - F0/L0/R0: source SHA256 → `os.link` → destination SHA256/inode verification. Forbidden: copy, transcode, symlink, test logs.
  - Non-empty destination → audit and HALT; never auto-overwrite.
  - Manifest line fields: `log_name,frame_token,camera,rel_path,src_sha256,linked_sha256,inode_src,inode_dst`. Summary records `expected_frame_total`.

**Commands**

```bash
python navsim/planning/script/build_4cam_root.py \
  --raster-src-root "$RASTER_SRC_ROOT" --dest-root "$RASTER_4CAM_ROOT" \
  --log-root "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  --log-list navsim/planning/script/config/common/train_test_split/scene_filter/paired_54log.yaml \
  --source-checksums "$STAGE_A_OUT/input_data/source_checksums.json" \
  --manifest "$STAGE_A_OUT/input_data/hardlink_manifest.jsonl"
```

**Expected**
- Each of F0/L0/R0 count == `expected_frame_total`; 54 dirs; no test log; dest-root log name set == `paired_54log.yaml` `log_names`.

---

### Step 0.4 Rebuild target-only cache

**Context**
- Step 0.3 passed. Must use `train_test_split=paired_trainval`. `trainval` and `navtrain` are both forbidden. Old `cache/rap_ego` is forbidden.
- Modify `run_dataset_caching.py` first:
  1. 在导入 `pytorch_lightning` 前 `import navsim`，确保 pytree compatibility shim 生效。
  2. 新增配置键 `target_only: false`；命令设为 `true`。
  3. `target_only=true` 时只取 `agent.get_target_builders()`，`feature_builders=[]`；主进程和 `cache_features()` 内的 SceneLoader 都使用 `SensorConfig.build_no_sensors()`。
  4. target-only 分支不得实例化 DINO/RAPModel；可实例化 `RAPTargetBuilder(RAPConfig(trajectory_sampling=...))`，或仅实例化轻量 target provider。
  5. 增加测试：没有 raster root/B0 时可生成 `rap_target.gz`，且目录中不存在 `rap_feature.gz`；另加一个断言 `paired_trainval` resolved 后 `frame_interval == 1` 的配置测试。
- Never pipe `--cfg job` output into a YAML parser; write to disk first.

**Commands**

```bash
# 起跑前先确认 resolved config 的 stride，避免跑几小时后才发现切分错误
python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset train_test_split=paired_trainval \
  experiment_name=stage_a_target_cache --cfg job --resolve \
  > "$STAGE_A_OUT/input_data/resolved_target_cache.yaml"
python - <<'PY'
import os, yaml
p = os.environ["STAGE_A_OUT"] + "/input_data/resolved_target_cache.yaml"
c = yaml.safe_load(open(p))["train_test_split"]["scene_filter"]
assert c["frame_interval"] == 1, f'frame_interval={c["frame_interval"]}，应为 1'
assert len(c["log_names"]) == 54, f'log_names={len(c["log_names"])}，应为 54'
print("stride/log_names OK")
PY

python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset train_test_split=paired_trainval experiment_name=stage_a_target_cache \
  cache_path="$STAGE_A_OUT/target_cache" force_cache_computation=true target_only=true \
  agent.config.trajectory_sampling.time_horizon=5 agent.config.trajectory_sampling.interval_length=0.5 \
  agent.config.train_metric_cache_path="$NAVSIM_EXP_ROOT/train_metric_cache"
find "$STAGE_A_OUT/target_cache" -name rap_target.gz | wc -l | tee "$STAGE_A_OUT/input_data/target_count.txt"
test "$(find "$STAGE_A_OUT/target_cache" -name rap_feature.gz | wc -l)" -eq 0
test "$(ls "$STAGE_A_OUT/target_cache" | wc -l)" -eq 54
```

**Expected**
- `target_count.txt` ≥ 41,864; 0 `rap_feature.gz`; 54 log dirs, no test log; sampled targets have valid trajectory shape/dtype/finite. Count ≈3,100 → HALT.

---

## Phase 1

### Step 1.1 Modify `build_alignment_small_data.py`

**Context**
- Step 0.4 passed. Implementation-only Step; no commands.
- Add `--dry-run`, `--dry-run-out PATH`, deterministic top-up, `--topup-manifest`, 1% circuit breaker, four-camera recheck.
- Artifacts: `stage_a_token_manifest.json`, `dataset_manifest.json`, `b0_generation_manifest.jsonl`, `topup_replacements.jsonl`, `input_audit.json`. B0 manifest uses the full field list from spec §3.
- `--dry-run-out` writes via a file handle. Never rely on stdout redirection.
- B0 write order: temp file in the same directory → F0 MAE / decode / shape / finite / black-frame / SHA256 checks → atomic rename → append manifest. On failure delete the temp file; never leave an unregistered B0.
- Replacement rate > 1% for any split → abort the whole data version.
- Add tests: `test_topup_replaces_failed_token_in_hash_order`, `test_topup_aborts_above_one_percent`, interrupt-recovery.
- Replace `select_hash_round_robin` with `select_hash_proportional(records, count)`:

  1. 按 log 分组，组内按 `(selection_hash, token)` 升序排序；
  2. 精确配额 `q_l = n_l / N * count`，`base_l = max(1, floor(q_l))`；
  3. `sum(base) > count` 时抛异常（目标过小，无法同时满足每 log ≥1 与比例性）；否则按余数 `q_l - floor(q_l)` 降序分配剩余名额，余数相同时以 `sha256(log_name)` 升序决胜；
  4. 每 log 取排序后前 `quota_l` 条；
  5. 断言 `sum(quota) == count`、`min(quota) >= 1`、`quota_l <= n_l`。

- No RNG, no seed, order-independent.
- Top-up draws the next unselected candidate **within the same log** by hash order. Only when that log is exhausted, fall back to global hash order and set `fallback_cross_log=true` in `topup_replacements.jsonl`.
- Add tests for quota proportionality, per-log ≥1, exact total, same-log top-up.
- Candidate records gain fields: `map_location, driving_command, actor_count_30m, endpoint_dx, endpoint_dy, endpoint_dyaw`. Computed from metadata and rebuilt targets only; never read checkpoints or predictions.

**Commands**
- None.

**Expected**
- New and existing tests in `tests/training/test_alignment_experiment.py` pass; count not below `$STAGE_A_OUT/baseline_tests.txt`.

---

### Step 1.2 Determine N, build, verify

**Context**
- Step 1.1 passed.

**Commands**

```bash
ARGS=(--real-root "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" --raster-root "$RASTER_4CAM_ROOT" \
 --target-root "$STAGE_A_OUT/target_cache" --log-root "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
 --split-config navsim/planning/script/config/training/default_train_val_test_log_split.yaml)
python navsim/planning/script/build_alignment_small_data.py --dry-run "${ARGS[@]}" \
  --dry-run-out "$STAGE_A_OUT/input_data/candidate_pool.json"
read -r TRAIN_N VAL_N < <(python -c 'import json,os; d=json.load(open(os.environ["STAGE_A_OUT"]+"/input_data/candidate_pool.json")); print(d["qualified_train"]//3,d["qualified_val"]//3)')
export TRAIN_N VAL_N
python navsim/planning/script/build_alignment_small_data.py "${ARGS[@]}" \
  --input-root "$STAGE_A_OUT/input_data" --cache-root "$STAGE_A_OUT/paired_cache" \
  --train-count "$TRAIN_N" --val-count "$VAL_N" \
  --topup-manifest "$STAGE_A_OUT/input_data/topup_replacements.jsonl" --generate-missing-back-raster
```

```bash
python - <<'PY'
import json, os
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader

STAGE = Path(os.environ["STAGE_A_OUT"])
man = json.load(open(STAGE / "input_data/stage_a_token_manifest.json"))
want = set(man["train_tokens"]) | set(man["val_tokens"])

# 1) metric cache 单 CSV + 全覆盖
cache = Path(os.environ["NAVSIM_EXP_ROOT"]) / "train_metric_cache"
csvs = sorted(p for p in (cache / "metadata").iterdir() if p.suffix == ".csv")
assert len(csvs) == 1, f"metadata/ 有 {len(csvs)} 个 CSV，loader 只会读其中一个: {[p.name for p in csvs]}"
have = set(MetricCacheLoader(cache).tokens)
miss = want - have
assert not miss, f"{len(miss)}/{len(want)} 个选中 token 不在 metric cache 中，例如 {sorted(miss)[:5]}"
print(f"metric cache 覆盖全部 {len(want)} 个选中 token（cache 共 {len(have)} 个）")

# 2) paired_cache 根目录只能是 manifest 里的 log 目录
paired = STAGE / "paired_cache"
entries = sorted(paired.iterdir())
non_dir = [p.name for p in entries if not p.is_dir()]
assert not non_dir, f"paired_cache 根目录存在非目录条目，会被当成 log: {non_dir}"
man_logs = set(man["train_logs"]) | set(man["val_logs"])
stray = {p.name for p in entries} - man_logs
assert not stray, f"paired_cache 存在 manifest 之外的 log 目录: {sorted(stray)}"
val_logs = set(man["val_logs"])
assert not (set(man["train_logs"]) & val_logs), "train/val log 集合相交，run_training 的切分会错"

# 3) token 子目录总数与 manifest 精确一致
n_tok = sum(1 for log in entries for _ in log.iterdir())
assert n_tok == len(want), f"paired_cache token 目录数 {n_tok} != manifest {len(want)}"
print(f"paired_cache OK: {len(entries)} logs / {n_tok} tokens")
PY
```

```bash
printf 'export TRAIN_N=%q\nexport VAL_N=%q\n' "$TRAIN_N" "$VAL_N" > "$STAGE_A_OUT/input_data/frozen_counts.env"
```

**Expected**
- Exact sample counts; train covers 44 logs / 4 map_locations; no test log; no duplicate B0 and all F0 MAE ≤ 1; replacement rate ≤ 1%; `input_audit.status=passed`; reshuffled candidate pool reproduces identical token order; per-log sampling rate `quota_l / n_l` max/min ≤ 1.5.
- Assertion block prints both `metric cache 覆盖全部 …` and `paired_cache OK: …`.
- `frozen_counts.env` written.

---

### Step 1.3 Subset distribution audit

**Context**
- Step 1.2 passed.
- Implement `navsim/planning/script/audit_subset_distribution.py`. Per split, compare the full candidate pool (`stage_a_token_manifest.json` → `qualified_candidates`) against the selected subset (`dataset_manifest.json`). Emit full/subset proportion tables plus `TVD = 0.5 · Σ|p_full − p_subset|` for:
  - `route_command`: metadata one-hot `driving_command`, 4 values, read directly, no reinterpretation.
  - `map_location`: 4 values.
  - `actor_density_bin`: `vehicle/pedestrian/bicycle` count within 30m, bins `0–5 / 6–10 / 11–20 / >20`.
  - `endpoint_dy_bin`: GT future trajectory lateral endpoint, bins `<−2 / −2~−0.5 / −0.5~0.5 / 0.5~2 / >2` m. `endpoint_dx_bin`, `endpoint_dyaw_bin` binned the same way.
  - `per_log_sampling_rate`: min/max/median of `quota_l / n_l`.
- Freeze thresholds into `$STAGE_A_OUT/input_data/distribution_audit_thresholds.json` BEFORE running: categorical TVD ≤ 0.02, binned continuous TVD ≤ 0.05, per-log sampling rate max/min ≤ 1.5.
- The script must not read checkpoints, predictions, PDMS or EPDMS.
- Audit output is record-only. Never re-sample or change the sampling algorithm/parameters based on it. Threshold violation → HALT for human decision, resolved before any training starts.

**Commands**

```bash
python navsim/planning/script/audit_subset_distribution.py \
  --token-manifest "$STAGE_A_OUT/input_data/stage_a_token_manifest.json" \
  --dataset-manifest "$STAGE_A_OUT/input_data/dataset_manifest.json" \
  --thresholds "$STAGE_A_OUT/input_data/distribution_audit_thresholds.json" \
  --out "$STAGE_A_OUT/input_data/distribution_audit.json"
python -c 'import json,os;d=json.load(open(os.environ["STAGE_A_OUT"]+"/input_data/distribution_audit.json"));assert d["status"]=="passed",d["violations"];print(json.dumps(d["summary"],indent=2))'
```

**Expected**
- `distribution_audit.json` has `status == "passed"`; summary printed.

---

## Phase 2

### Step 2.1 Data, loss and integration tests

**Context**
- Phase 1 passed.
- Implement `tests/training/test_stage_a_data_integrity.py` covering token/log/drive leakage, manifest alignment, four cameras, targets, raw-to-cache numerics, train/val/test isolation.

**Commands**

```bash
python -m pytest tests/training/test_stage_a_data_integrity.py tests/training/test_alignment_experiment.py -q
```

**Expected**
- All tests pass.

---

### Step 2.2 navtest metric cache

**Context**
- Step 2.1 passed. Required by the evaluator regression below.

**Commands**

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtest experiment_name=stage_a_navtest_metric_cache
python - <<'PY'
import os
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
cache = Path(os.environ["NAVSIM_EXP_ROOT"]) / "metric_cache"      # 与 train_metric_cache 是不同目录
csvs = sorted(p for p in (cache / "metadata").iterdir() if p.suffix == ".csv")
assert len(csvs) == 1, f"metadata/ 有 {len(csvs)} 个 CSV，loader 只读其中一个：{[p.name for p in csvs]}"
n = len(MetricCacheLoader(cache).tokens)
assert n == 12146, f"navtest metric cache {n} 个 token，应为 12,146；检查 train_test_split 是否为 navtest、有无 caching 失败"
print("navtest metric cache tokens:", n)
PY
```

**Expected**
- Exactly 1 CSV; token count exactly 12,146.

---

### Step 2.3 Pin v1 evaluator, schema, determinism

**Context**
- Step 2.2 passed.
- Preconditions, all mandatory: `$NAVSIM_V1_ROOT` is an official v1.1 checkout; HEAD equals `$NAVSIM_V1_COMMIT`; human-supplied human/constant-velocity reference PDMS and tolerance frozen into `$STAGE_A_OUT/evaluator_validation/v1_reference.json`. Any missing → HALT.
- **⚠️ 上面这条 precondition 的第三项已于 2026-08-20 被 `docs/rap-alignment-stage-a-step5-v1-only.md` Step S.2 取代，原文保留仅作预注册追溯凭据，不得再作为执行依据。** 取代理由：pinned v1.1 worktree 内**不存在**任何 human / constant-velocity 的数值 PDMS（官方代码库只有运行命令、agent 描述与 PDMS 公式），绝对值门禁无法建立。现行口径：
  - `v1_reference.json` **作废**，Step S.2 显式断言该文件**不得存在**；
  - 硬门禁改为 Step S.4 的**关系式门禁** G1–G7，条件运行前冻结进 `$STAGE_A_OUT/evaluator_validation/v1_relational_gate.json`，不依赖任何外部数值；
  - 论文 Table 1 数值（`v1_paper_reference.json`）仅作 **advisory 软交叉核对**，不作 HALT 条件。
  - 前两项 precondition（official checkout、HEAD == 完整 40 位 `$NAVSIM_V1_COMMIT`）仍然有效，且被 Step S.2 加强为六项检查（另加 clean worktree、位于 `$RAP_ROOT` 之外、upstream remote 归属、exact tag 可达）。
- Modify `run_pdm_score.py`: all token sets via `sorted(set(...))`; final DataFrame sorted by token; `to_csv(index=False)`; fixed output field mapping; keep `token,valid`.

```text
no_at_fault_collisions→NC; drivable_area_compliance→DAC; ego_progress→EP;
time_to_collision_within_bound→TTC; comfort→C;
driving_direction_compliance→DDC; score→PDMS
```

- Add `output_csv: null` at the **top level** of `navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`; script writes to that path when non-empty. `output_dir` already exists in `default_evaluation.yaml`; do not add it.
- The local wrapper must record the actual scorer source file and its SHA256 and prove it originates from `$NAVSIM_V1_ROOT@$NAVSIM_V1_COMMIT`. Checking an uninvoked checkout does not count.

**Commands**

```bash
test -n "$NAVSIM_V1_ROOT"; test -d "$NAVSIM_V1_ROOT/.git"
test "$(git -C "$NAVSIM_V1_ROOT" rev-parse HEAD)" \
   = "$(git -C "$NAVSIM_V1_ROOT" rev-parse "${NAVSIM_V1_COMMIT}^{commit}")"
git -C "$NAVSIM_V1_ROOT" rev-parse HEAD > "$STAGE_A_OUT/evaluator_validation/v1_commit.txt"
# output_csv 已进入 config，否则下面 9 条命令会全部被 Hydra 拒绝
python navsim/planning/script/run_pdm_score.py --cfg job agent=constant_velocity_agent \
  experiment_name=probe 2>/dev/null | grep -q "^output_csv:"
for A in constant_velocity_agent human_agent; do
  python navsim/planning/script/run_pdm_score.py agent="$A" train_test_split=navtest \
    experiment_name="stage_a_evalcheck_$A" output_dir="$STAGE_A_OUT/evaluator_validation/$A" \
    output_csv="$STAGE_A_OUT/evaluator_validation/${A}.csv"
done
python navsim/planning/script/run_pdm_score.py agent=constant_velocity_agent train_test_split=navtest \
  experiment_name=stage_a_evalcheck_repeat output_dir="$STAGE_A_OUT/evaluator_validation/constant_velocity_repeat" \
  output_csv="$STAGE_A_OUT/evaluator_validation/constant_velocity_repeat.csv"
```

**Expected**
- Two constant-velocity runs are byte-identical across the 9 columns after sorting by token (filenames/timestamps excluded); 12,146 scenes; all `valid`; no v2-only columns; both baselines match the frozen reference within tolerance.

---

### Step 2.4 v2 evaluator hard gate

**Context**
- Step 2.3 passed. Human must supply:
  1. `$NAVSIM_V2_ROOT` 与完整 `$NAVSIM_V2_COMMIT`；
  2. 官方 evaluator 的 environment lock；
  3. RAP `last.ckpt → submission pickle/agent API` adapter 及 schema test；
  4. official human/baseline regression 命令和参考值；
  5. 输出 root-scene、Stage 1/2 rollout、Gaussian weight、全部 sub-score 和 EPDMS 的固定 schema。
- No local scorer substitution. No downgrade to a v1-only official experiment.

**Commands**

```bash
test -n "$NAVSIM_V2_ROOT"; test -d "$NAVSIM_V2_ROOT/.git"
test "$(git -C "$NAVSIM_V2_ROOT" rev-parse HEAD)" \
   = "$(git -C "$NAVSIM_V2_ROOT" rev-parse "${NAVSIM_V2_COMMIT}^{commit}")"
test -d "$NAVHARD_TWO_STAGE_ROOT"
git -C "$NAVSIM_V2_ROOT" rev-parse HEAD > "$STAGE_A_OUT/evaluator_validation/v2_commit.txt"
```

**Expected**
- Exit 0 and all 5 human deliverables present. Otherwise HALT — Phase 4 must not start.

---

## Phase 3

### Step 3.1 Per-seed initialization checkpoints

**Context**
- Phase 2 passed.
- Implement `make_init_checkpoints.py --seeds 0 1 2 --out-dir "$STAGE_A_OUT/init_checkpoints"`:
  - Per seed: `pl.seed_everything(seed, workers=True)`. This is the only Step that initializes pretrained DINO from HF.
  - Save `{"state_dict": agent.state_dict()}`.
  - **`dino_init_from_pretrained` 的两个取值不是矛盾，分属两个阶段**，规格书 §4「Stage-A 采用 ①（`dino_init_from_pretrained: true`）」描述的是本 Step 的初始化来源，不是训练配置：
    - **本 Step 内** `dino_init_from_pretrained=true`：从 HF 拉 pretrained DINO backbone + 随机初始化 projector/planner，落盘成 `seed_{0,1,2}.ckpt`。
    - **Step 3.2 / 4.1 的训练配置** `dino_init_from_pretrained=false`：从上面那个文件加载，禁止在进程内重新随机初始化。
    - 这正是规格书 §4 同句后半段「该 seed 的两个 run 通过 `agent.checkpoint_path` 加载**同一文件**，禁止两条件各自在进程内随机初始化」的唯一可行实现。若训练配置误设为 `true`，两条件会各自随机初始化，实验作废。
  - 实例化前先断言 DINO 架构与 `$REF_DINO_CKPT` 一致（复用 Step -1.2 的比对逻辑，或直接读取 `$STAGE_A_OUT/dino_reference_config.json` 并逐字段核对当前 `RAPConfig`）。不一致 → HALT。
  - **禁止**从 `$REF_DINO_CKPT`（或任何 `ckpts/RAP_DINO_*.ckpt`）加载权重：规格书 §4 判定其训练来源未确认，Stage-A 禁止使用。该文件在本流程中只作架构参考。
  - 记录 provenance 到 `sha256.json` 的每个 seed 条目：`source`（`hf:{dino_model_name}@{revision}` + 随机 head）、`training_data`（`none — freshly initialized`）、`training_objective`（`none`）、`created_at`、`git_commit`。规格书 §7 要求 initialization checkpoint 记录来源/训练数据/训练目标/SHA256，缺一不可。
  - `sha256.json` per seed records three fields:
    - `file_sha256`: whole `.ckpt` file.
    - `backbone_sha256`: parameters whose name starts with `_rap_model._backbone.img_backbone.`.
    - `head_sha256`: all remaining parameters.
  - Group hash algorithm: sort by parameter name; per parameter feed `name`, `str(dtype)`, `str(tuple(shape))`, then `tensor.detach().cpu().contiguous().numpy().tobytes()`. `contiguous()` is mandatory.

**Commands**

```bash
set -euo pipefail; source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env; cd "$RAP_ROOT"
python navsim/planning/script/make_init_checkpoints.py \
  --seeds 0 1 2 --out-dir "$STAGE_A_OUT/init_checkpoints"
test "$(ls "$STAGE_A_OUT"/init_checkpoints/seed_*.ckpt | wc -l)" -eq 3
python - <<'PY'
import json, os
d = json.load(open(os.environ["STAGE_A_OUT"] + "/init_checkpoints/sha256.json"))
assert set(d) == {"seed_0", "seed_1", "seed_2"}, sorted(d)
files = {k: v["file_sha256"] for k, v in d.items()}
backs = {k: v["backbone_sha256"] for k, v in d.items()}
heads = {k: v["head_sha256"] for k, v in d.items()}
assert len(set(files.values())) == 3, f"三个 init ckpt 文件不应相同: {files}"
assert len(set(backs.values())) == 1, f"DINO backbone 在三 seed 间必须完全一致: {backs}"
assert len(set(heads.values())) == 3, f"随机 head 在三 seed 间必须互不相同: {heads}"

# provenance（规格书 §7 必须保存的 artifacts）
ref = json.load(open(os.environ["STAGE_A_OUT"] + "/dino_reference_config.json"))
for seed, v in d.items():
    for field in ("source", "training_data", "training_objective", "created_at", "git_commit"):
        assert v.get(field), f"{seed} 缺少 provenance 字段 {field}"
    assert ref["dino_model_name"] in v["source"], (seed, v["source"], ref["dino_model_name"])
    assert "RAP_DINO_navsimv" not in v["source"], \
        f"{seed} 的 initialization 来自被规格书 §4 禁止的 ckpt: {v['source']}"
print("init ckpt 分组哈希 + provenance OK")
PY
```

**Expected**
- 3 个 `seed_*.ckpt` 写出；打印 `init ckpt 分组哈希 + provenance OK`。
- 每个 seed 的 `source` 指向 Step -1.2 记录的 `dino_model_name`，且不含任何 `RAP_DINO_navsimv*`。

---

### Step 3.2 `alignment_stage_a.yaml`

**Context**
- Step 3.1 passed.
- Base on `alignment_small.yaml`. Required values: token manifest / audit / paired cache paths via `${oc.env:STAGE_A_OUT}`; `max_*_samples=null`; `limit_*_batches=1.0`; `include_auxiliary_datasets=false`; `cache_path_perturbed=null`; `cache_path_others=null`; `shuffle_train=true`; `shuffle_val=false`; 20 epochs; global batch 128 (`devices=4 × batch_size=32 × accumulate_grad_batches=1`); precision `16-mixed`; `num_sanity_val_steps=0`; alignment weights 0.002/0.1; trajectory 5/0.5.
- **`agent.checkpoint_path` 必须覆盖为 `${oc.env:STAGE_A_OUT}/init_checkpoints/seed_${seed}.ckpt`。** `alignment_small.yaml` 的默认值是 `${hydra:runtime.cwd}/ckpts/RAP_DINO_navsimv2.ckpt`——规格书 §4 明令 **Stage-A 禁止使用**该文件。忘记覆盖会静默继承它，故下面的校验块必须同时断言"两条件相同"与"指向正确的 init 文件"，只断言前者不足以拦截。
- `agent.config.dino_init_from_pretrained` 必须为 `false`（理由见 Step 3.1）。
- Add:

```yaml
exact_distributed_coverage: true
final_step_checkpoint_only: true
trainer:
  params:
    use_distributed_sampler: false
agent:
  config:
    dino_init_from_pretrained: false
```

- Use `4×32×1` only after the memory smoke test and the Step 3.3 sampler gate pass.
- `--cfg` short-circuits before `main()`, so this Step is not blocked by the `input_audit_path` gate. Both resolved configs are required artifacts.

**Commands**

```bash
for C in r100_noalign r100_fullalign; do
  python navsim/planning/script/run_training.py --config-name alignment_stage_a \
    alignment_condition="$C" seed=0 --cfg job --resolve > "$STAGE_A_OUT/resolved_config_${C}.yaml"
done
python - <<'PY'
import os, yaml
from pathlib import PurePath
S = os.environ["STAGE_A_OUT"]
cfgs = {c: yaml.safe_load(open(f"{S}/resolved_config_{c}.yaml")) for c in ("r100_noalign", "r100_fullalign")}
for c, cfg in cfgs.items():
    name = cfg["alignment_condition_name"]
    assert name == c.replace("_", "-"), f"{c} → alignment_condition_name={name}，与 Phase 4 的目录名推导不符"
    out = PurePath(cfg["output_dir"])
    assert out.name == "seed_0" and out.parent.name == name, cfg["output_dir"]
    assert str(out.parent.parent) == f"{S}/training", cfg["output_dir"]
a, b = cfgs["r100_noalign"], cfgs["r100_fullalign"]
ac, bc = a["agent"]["config"], b["agent"]["config"]
assert (ac["use_spatial_align"], ac["use_global_align"]) == (False, False), ac
assert (bc["use_spatial_align"], bc["use_global_align"]) == (True, True), bc

# init checkpoint：既要两条件相同，也要指向正确的文件。
# 只断言"相同"会让 alignment_small.yaml 的默认 RAP_DINO_navsimv2.ckpt 静默通过（规格书 §4 禁止）。
ckpt = a["agent"]["checkpoint_path"]
assert ckpt == b["agent"]["checkpoint_path"], "两条件必须加载同一 init 文件"
expected_ckpt = f"{S}/init_checkpoints/seed_0.ckpt"
assert PurePath(ckpt) == PurePath(expected_ckpt), \
    f"checkpoint_path={ckpt}，应为 {expected_ckpt}"
assert "RAP_DINO_navsimv" not in ckpt, \
    f"Stage-A 禁止使用 {ckpt}（规格书 §4：训练来源未确认）"
assert ac["dino_init_from_pretrained"] is False and bc["dino_init_from_pretrained"] is False, \
    "训练必须从 init 文件加载，不得在进程内重新随机初始化"

assert ac["distill_feature_weight"] == bc["distill_feature_weight"] == 0.002
assert ac["domain_align_weight"] == bc["domain_align_weight"] == 0.1

# 规格书 §4 冻结的训练预算与优化器超参
for cfg in (a, b):
    tp = cfg["trainer"]["params"]
    assert tp["max_epochs"] == 20, tp["max_epochs"]
    assert tp["precision"] == "16-mixed", tp["precision"]
    assert tp["gradient_clip_val"] == 0.0, tp["gradient_clip_val"]
    assert (tp["devices"], cfg["dataloader"]["params"]["batch_size"],
            tp["accumulate_grad_batches"]) == (4, 32, 1), "effective global batch 必须为 128"
    assert cfg["max_train_samples"] is None and cfg["max_val_samples"] is None
    assert tp["limit_train_batches"] == 1.0 and tp["limit_val_batches"] == 1.0
    assert cfg["shuffle_train"] is True and cfg.get("shuffle_val", False) is False
    assert cfg["include_auxiliary_datasets"] is False
    assert cfg["cache_path_perturbed"] is None and cfg["cache_path_others"] is None
    assert float(cfg["agent"]["lr"]) == 1e-4, cfg["agent"]["lr"]   # 注意在 agent.lr，不在 agent.config
    assert cfg["agent"]["config"]["tf_dropout"] == 0, cfg["agent"]["config"]["tf_dropout"]

top_diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
assert top_diff <= {"alignment_condition_name", "experiment_name", "output_dir", "agent"}, top_diff
print("条件名/目录名映射与受控性 OK")
PY
```

**Expected**
- Prints `条件名/目录名映射与受控性 OK`.
- `checkpoint_path` 指向 `$STAGE_A_OUT/init_checkpoints/seed_0.ckpt`，不含 `RAP_DINO_navsimv*`。
- `devices × batch_size × accumulate = 4 × 32 × 1 = 128`；`agent.lr=1e-4`、`gradient_clip_val=0.0`、`tf_dropout=0`。
- `weight_decay=1e-4` 与 `min_lr=1e-5 / epochs=20 / warmup_epochs=1` 硬编码在 `rap_agent.py:configure_optimizers`，不在 resolved config 中，由 Step -1.3 的 grep 覆盖。**注意**：同函数中 **DINO backbone 在训练全程被冻结** —— `rap_agent.py:571-572` 对 `_backbone.img_backbone` 的全部参数设 `requires_grad = False`，且 `rap_agent.py:576` 的 vit 参数组已被注释掉；`vit_lr = self._lr * 0.2` 在 :567 只计算不使用。因此 optimizer 只有一个参数组（`other_params`，lr `1e-4`），**backbone 不接收任何梯度**。规格书 §4 只写了 `initial learning rate = 1e-4`，未记录 backbone 冻结这一事实。该偏离对两条件完全相同，不影响内部效度，但须在结果报告中声明——且其含义比 param-group 分层更重：alignment 梯度只能作用到 `ImgEncoder` 中 DINO 之后的可训练部分（`img_neck` FPN + `cams_embeds` + `level_embeds`）与下游 planner，作用面窄于规格书 §4「pretrained frozen DINO backbone + 随机初始化 projector/planner」字面给人的印象。
  - **勘误（2026-08-24）**：本条原文写作「同函数中 `vit_lr = self._lr * 0.2`，DINO backbone 实际 lr 为 `2e-5`」，与代码不符（该参数组被注释掉）。同一错误已进入 `summary_protocol_v2.json` 的 `data_provenance_caveats_required_in_report` 第 8 条，按该文件的 `amendment_rule` 只能通过新建 `summary_protocol_v3.json` 更正，不得原地修改。

---

### Step 3.3 No-padding DDP sampler

**Context**
- Step 3.2 passed. Implementation-only Step; no commands.
- Implement `ExactDistributedSampler` in `navsim/planning/training/samplers.py`: unique global permutation from `seed+epoch`, sharded by rank stride; no padding, no drop; expose `global_token_order(epoch)`.
- `run_training.py` installs it explicitly when `exact_distributed_coverage=true`, sets DataLoader `shuffle=false`, and relies on `trainer.params.use_distributed_sampler=false` to stop Lightning from replacing it.
- Assert equal batch count per rank before launch. On mismatch adjust devices / per-device batch / accumulation. Never duplicate or drop tokens.
- Tests must cover at least the frozen Stage-A sizes **2,498 train / 464 val** samples × 4 ranks: union equals full set, pairwise intersections empty, equal batch count per rank, same seed reproduces order, different epoch changes order.（数字来自规格书 §4 的冻结训练预算；原文的 11,489/2,465 是 Phase 1 实测前的估算值，已由 `docs/rap-alignment-stage-a-phase4-preflight.md` Step P.3 修正，此处同步。）

**Commands**
- None.

**Expected**
- Sampler tests pass.

---

### Step 3.4 Callbacks

**Context**
- Step 3.3 passed. Implementation-only Step; no commands.
- Do not reference a non-existent `output_dir` inside `RAPAgent.get_training_callbacks()`. Read `cfg.output_dir` in `run_training.py`; when `final_step_checkpoint_only=true`, drop every `ModelCheckpoint` returned by the Agent and create:

```python
ModelCheckpoint(dirpath=Path(cfg.output_dir) / "checkpoints", save_last=True, save_top_k=0)
```

- `run_training.py` also creates `EpochTokenChecksumCallback`, reading `ExactDistributedSampler.global_token_order(epoch)` directly. Never iterate the live sampler. Only rank 0 appends `$output_dir/epoch_token_checksums.jsonl` with fields `epoch,stage,num_tokens,sequence_sha256`.
- `run_training.py` also creates `LossComponentCallback`。规格书 §7 的必保 artifacts 含「**每个 loss 分量、实际 weight、GRL coefficient**」，当前流程无任何步骤记录它们；缺此文件则实验结束后无法证明"唯一变量是 alignment loss"，也无法核对 §4 的 GRL schedule。
  - 只 rank 0 追加写 `$output_dir/loss_components.jsonl`，字段：

    ```text
    epoch, global_step, progress_p, task_loss, spatial_loss_raw, global_loss_raw,
    spatial_weight_effective, global_weight_effective, grl_lambda, total_loss
    ```

  - `progress_p = completed_steps / total_steps`，与规格书 §4 的 $\lambda_{GRL}(p)=0.1\left(\frac{2}{1+e^{-10p}}-1\right)$ 同源；`grl_lambda` 必须取训练时实际生效的值，不得事后按公式重算。
  - `*_raw` 是加权**前**的分量，`*_weight_effective` 是当步实际乘数（NoAlign 恒为 `0.0`）。`total_loss` 必须等于 `task_loss + spatial_weight_effective * spatial_loss_raw + global_weight_effective * global_loss_raw`，容差 `1e-6`。
  - 采样频率：每个 optimizer step 写一行（Stage-A 全程 400 行/run = `⌈2,498/128⌉ × 20`，开销可忽略；原文的约 1,800 行同为 Step P.3 修正前的估算值）。
  - `λ_global = 0.1` 与 `λ_GRL,max = 0.1` 是不同参数（规格书 §4 明令禁止混写），必须落在 `global_weight_effective` 与 `grl_lambda` 两个独立字段上，不得共用一列。
- Add tests for callback path, rank safety, 20-epoch train/val row counts, `total_loss` 重构等式, 以及 `grl_lambda` 在 `p=0 / 0.5 / 1` 三点与 §4 公式一致（`0`、`≈0.098661`、`≈0.099991`）。既有的 `test_rap_grl_schedule_has_paper_scale_and_endpoints` 已覆盖 `LambdaScheduler(gamma=10.0, scale=0.1)` 的端点，本 Step 追加的是**落盘值**与该 schedule 一致。

**Commands**
- None.

**Expected**
- Callback tests pass.

---

## Phase 4

### Step 4.1 Stage-A training, per-seed validation

**Context**
- Phase 3 passed, including the Step 2.4 v2 gate.
- Implement `validate_stage_a_seed.py`: verify init SHA, train/val tokens, epoch checksum, global steps, NoAlign `loss==task_loss`, only `last.ckpt` present, and that the resolved configs differ solely in alignment switches / name / output.
- 追加校验 `loss_components.jsonl`（Step 3.4）：
  - 两条件行数相同，且等于 optimizer step 数；
  - **NoAlign**：`spatial_weight_effective == global_weight_effective == 0.0` 且 `total_loss == task_loss` 逐行成立（容差 `1e-6`）——这是"NoAlign alignment gradient 严格为零"的运行时证据，静态测试不能替代；
  - **FullAlign**：`spatial_weight_effective == 0.002`、`global_weight_effective == 0.1` 逐行成立；
  - `grl_lambda` 随 `progress_p` 单调不减，首行 `≈0`、末行 `≈0.0999`，且逐行等于 `0.1·(2/(1+exp(-10·progress_p))−1)`（容差 `1e-6`）；
  - `total_loss` 重构等式逐行成立；
  - 两条件的 `task_loss` 序列**不要求**逐行相等（FullAlign 的梯度会改变后续参数），但第一个 optimizer step 的 `task_loss` 必须相等（容差 `1e-6`）——对应规格书 §7「两条件首个 optimizer step 前参数完全一致」。
- 断言两条件的 `agent.checkpoint_path` 均指向 `$STAGE_A_OUT/init_checkpoints/seed_$SEED.ckpt`，且不含 `RAP_DINO_navsimv*`（规格书 §4）。
- Order is fixed: `seed 0 → 1 → 2`, and within each seed `noalign → fullalign`.

**Commands**

```bash
source "$STAGE_A_OUT/input_data/frozen_counts.env"
for SEED in 0 1 2; do
  export SEED
  for C in r100_noalign r100_fullalign; do
    python navsim/planning/script/run_training.py --config-name alignment_stage_a alignment_condition="$C" seed="$SEED"
  done
  export NO="$STAGE_A_OUT/training/r100-noalign/seed_$SEED"
  export FU="$STAGE_A_OUT/training/r100-fullalign/seed_$SEED"
  python navsim/planning/script/validate_stage_a_seed.py --noalign "$NO" --fullalign "$FU" --train-count "$TRAIN_N" --epochs 20
done
test "$(find "$STAGE_A_OUT/training" -name last.ckpt | wc -l)" -eq 6
test "$(find "$STAGE_A_OUT/training" -name loss_components.jsonl | wc -l)" -eq 6
test "$(find "$STAGE_A_OUT/training" -name epoch_token_checksums.jsonl | wc -l)" -eq 6
```

**Expected**
- `validate_stage_a_seed.py` exits 0 for every seed; exactly 6 `last.ckpt`、6 `loss_components.jsonl`、6 `epoch_token_checksums.jsonl`. Any seed failure → `set -e` aborts the loop.

---

## Phase 5

### Step 5.1 v1 evaluation ×6

**Context**
- Phase 4 passed. Uses the official v1 commit/schema frozen in Step 2.3. Copy each output CSV into `$STAGE_A_OUT/evaluation_v1/` as `{condition}_seed{seed}.csv`.

**Commands**

```bash
for COND in r100-noalign r100-fullalign; do for SEED in 0 1 2; do
  python navsim/planning/script/run_pdm_score.py agent=rap_agent \
    agent.checkpoint_path="$STAGE_A_OUT/training/$COND/seed_$SEED/checkpoints/last.ckpt" \
    agent.config.dino_init_from_pretrained=false agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.trajectory_sampling.interval_length=0.5 \
    agent.config.train_metric_cache_path="$NAVSIM_EXP_ROOT/train_metric_cache" \
    train_test_split=navtest experiment_name="stage_a_v1_${COND}_seed${SEED}" \
    output_dir="$STAGE_A_OUT/evaluation_v1/${COND}_seed${SEED}" \
    output_csv="$STAGE_A_OUT/evaluation_v1/${COND}_seed${SEED}.csv"
done; done
```

**Expected**
- 6 CSVs, each 12,146 identical tokens, all `valid`, fixed 9 columns `token,valid,NC,DAC,EP,TTC,C,DDC,PDMS`.

---

### Step 5.2 v2.2 EPDMS evaluation ×6

**Context**
- Step 5.1 passed. Use only the adapter and `$NAVSIM_V2_ROOT` validated in Step 2.4. Freeze the exact adapter command into `$STAGE_A_OUT/evaluator_validation/v2_adapter_command.txt` together with the evaluator; never modify it after seeing results.
- Output to `$STAGE_A_OUT/evaluation_v2/{condition}_seed{seed}/`. Pair by `root_scene_id`. Stage 2 follow-up is not an independent test scene.

**Commands**
- Frozen adapter command from `$STAGE_A_OUT/evaluator_validation/v2_adapter_command.txt`.

**Expected**
- 6 output dirs with the Step 2.4 schema; identical `root_scene_id` set across all 6.

---

### Step 5.3 Summary

**Context**
- Steps 5.1 and 5.2 passed.
- Freeze `summary_protocol.json` before writing `summarize_stage_a.py`: bootstrap seed, ≥10,000 log-cluster resamples, percentile 95% CI, reported quantiles, paired effect-size definition, Holm correction, convergence gate numeric thresholds.
- The script pairs by seed/scene or root scene and outputs the full transition table, continuous deltas, and both co-primary metrics PDMS/EPDMS.
- The report must embed the full summary of `distribution_audit.json` from Step 1.3 and state that the Stage-A subset is a deterministic 1/3 sample stratified proportionally by log size, with the measured TVD per variable. If any variable is near its threshold, the extrapolation section must state the representativeness risk in that direction.
- Missing official evaluator, mismatched scene sets, or schema mismatch → HALT and mark the experiment incomplete. Never produce a one-sided official conclusion.

**Commands**

```bash
python navsim/planning/script/summarize_stage_a.py \
  --evaluation-v1 "$STAGE_A_OUT/evaluation_v1" \
  --evaluation-v2 "$STAGE_A_OUT/evaluation_v2" \
  --distribution-audit "$STAGE_A_OUT/input_data/distribution_audit.json" \
  --protocol "$STAGE_A_OUT/summary/summary_protocol.json" \
  --out-dir "$STAGE_A_OUT/summary"
```

**Expected**
- `$STAGE_A_OUT/summary/stage_a_report.json` and `$STAGE_A_OUT/summary/paired_deltas.csv` written.

---

## Phase 6

### Step 6.1 Stage-B boundary

**Context**
- Stage-B requires a separate runbook: complete B0 for all 54 logs, verify unique renderer/map version, verify Stage-A is a proper subset, generate labels and thresholds, human audit, full-scale training, v1/v2 stratified statistics.

**Commands**
- None.

**Expected**
- Do not start until Stage-A is complete and both evaluators pass.
