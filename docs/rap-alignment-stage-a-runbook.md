# RAP Paired Alignment — Stage-A 执行手册（Runbook）

本文件是 `docs/rap-alignment-experiment-plan.md`（下称**规格书**）的施工图纸。规格书冻结"实验事实"（数字、阈值、禁令），本手册规定"怎么做"。**任何冲突以规格书为准**；发现规格书有误则停止上报，不得自行改数。

**执行约束**：严格按 Step 顺序，禁止跳步/并行/换序；验收全绿才进下一步；任一失败**立即停止**（§3：停止冻结数据版本）；`⚠️` 处必须等人工答复。每个 bash 块首行执行 `source /gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env && cd $RAP_ROOT`（下称 **PRELUDE**）。本手册产出的 `source_checksums.json`、`hardlink_manifest.jsonl`、`baseline_*.txt`、`env_versions.json`、`epoch_token_checksums.jsonl` 是 Runbook 追加产物，不在规格书 §7 目录树内，不要据此认为规格书有遗漏。

---

## Phase -1：环境固化与前置校验

### Step -1.1 环境变量

新建 `/gs/bs/tga-RLA/qdeng/RAP/env/stage_a.env`：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"
export NAVSIM_EXP_ROOT="/gs/bs/tga-RLA/qdeng/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/gs/bs/tga-RLA/qdeng/RAP"
export RAP_ROOT="/gs/bs/tga-RLA/qdeng/RAP"
export RASTER_SRC_ROOT="${RAP_ROOT}/dataset_norm/rendered_sensor_blobs"
export RASTER_4CAM_ROOT="${RAP_ROOT}/dataset_norm/rendered_sensor_blobs_4cam_v1"
export STAGE_A_OUT="${RAP_ROOT}/outputs/alignment_stage_a"
```

**验收**：`mkdir -p $STAGE_A_OUT/{input_data,evaluation_v1,evaluator_validation}`；`NUPLAN_MAPS_ROOT / OPENSCENE_DATA_ROOT / RASTER_SRC_ROOT` 三个目录都存在；用 python 把 `sys.version / torch.__version__ / torch.version.cuda / nvidia-smi driver` 写入 `$STAGE_A_OUT/env_versions.json`（§7 artifacts 要求）。

> ⚠️ `NUPLAN_MAPS_ROOT` 与 `NAVSIM_EXP_ROOT` 的实际值规格书未记录，上面按 README 惯例推断，执行前须人工核实。

### Step -1.2 代码基线

```bash
PRELUDE
git rev-parse HEAD > $STAGE_A_OUT/baseline_commit.txt
git status --porcelain > $STAGE_A_OUT/baseline_dirty.txt
grep -n "tf_dropout\|distill_feature_weight\|domain_align_weight" navsim/agents/rap_dino/navsim_config.py
grep -n "distill_feature\|pdm_scorer" navsim/planning/script/config/common/agent/rap_agent.yaml
grep -n "WarmupCosLR" navsim/agents/rap_dino/rap_agent.py
python -m pytest tests/training/test_alignment_experiment.py -q 2>&1 | tail -3 | tee $STAGE_A_OUT/baseline_test_count.txt
```

**验收**（任一不符即停止）：`tf_dropout = 0`（§4）；`distill_feature_weight = 0.002`（λ_spatial）；`domain_align_weight = 0.1`（λ_global）；`rap_agent.yaml` 中 `distill_feature: True` 且 `pdm_scorer: True`；`WarmupCosLR(..., epochs=20, warmup_epochs=1, min_lr=1e-5)` 为硬编码，`max_epochs` 必须与其一致；`baseline_test_count.txt` 记录基线通过数（当前 16 个 `def test_`，参数化后收集数更多），Step 1.1 改造后**不得减少**。

### Step -1.3 train metric cache

`pdm_scorer: True` 使 `RAPAgent.__init__`（[rap_agent.py:73](navsim/agents/rap_dino/rap_agent.py:73)）构造 `MetricCacheLoader`。Step 0.3 的 dataset caching 也会 `instantiate(cfg.agent)`（[run_dataset_caching.py:39](navsim/planning/script/run_dataset_caching.py:39)），故本步必须在 Phase 0 之前。

```bash
PRELUDE
python navsim/planning/script/run_train_metric_caching.py train_test_split=trainval
find $NAVSIM_EXP_ROOT/train_metric_cache -type f | wc -l > $STAGE_A_OUT/metric_cache_count.txt
```

**验收**：`$NAVSIM_EXP_ROOT/train_metric_cache` 存在且非空。
> 生成位置由 `train_metric_caching.yaml` 的 `cache.cache_path=${oc.env:NAVSIM_EXP_ROOT}/train_metric_cache` 决定，而 `rap_agent.yaml` 默认读 `./train_metric_cache` —— **两者不是同一目录**，Step 3.2 与 5.2 必须显式覆写 `agent.config.train_metric_cache_path`。

---

## Phase 0：共用数据准备

> 顺序不可颠倒：Step 0.2 的 4cam root 是 Step 1.2 读 F0 的来源，B0 会被写回该 root，故须先自洽。

### Step 0.1 冻结源数据 checksum

新建 `navsim/planning/script/freeze_source_checksums.py`，接口：`--raster-src-root $RASTER_SRC_ROOT --log-root $OPENSCENE_DATA_ROOT/navsim_logs/trainval --split-config navsim/planning/script/config/training/default_train_val_test_log_split.yaml --out $STAGE_A_OUT/input_data/source_checksums.json`。

脚本行为：① 对 split config 算 SHA256，记录 `train/val/test_logs` 的**条目总数**；② 统计各相机文件数；③ **关键**——规格书 §2 的 44/10/10 是 *raster 目录 ∩ official split*，**不是** yaml 长度（yaml 实为 13,180/1,381/1,349 条），必须写成 `len(({p.name for p in raster_src_root.iterdir() if p.is_dir()} - {"missing_camera"}) & set(cfg["train_logs"]))`。

**验收**（任一不符即停止上报，说明源数据已变动）：

| 字段 | 期望 | | 字段 | 期望 |
|---|---:|---|---|---:|
| `split_config_{train,val,test}_log_count` | 13,180 / 1,381 / 1,349 | | `raster_{train,val,test}_log_count` | **44 / 10 / 10** |
| `raster_log_dirs` | 64 | | `cam_f0/l0/r0/b0_count` | 51,898 / 51,898 / 51,900 / 48 |

### Step 0.2 建立 4cam root 的 F0/L0/R0 hard link

新建 `navsim/planning/script/build_4cam_root.py`，接口：`--raster-src-root --dest-root $RASTER_4CAM_ROOT --log-root --split-config --manifest $STAGE_A_OUT/input_data/hardlink_manifest.jsonl`。

脚本行为（严格按 §3）：① 只处理 `raster_logs ∩ train_logs`(44) + `∩ val_logs`(10) = 54 个 log，显式断言 dest-root 无任何 test log 名；② 对 54 log 的**全部** metadata frame，从 `frame["cams"][cam]["data_path"]` 取相对路径；③ F0/L0/R0 三路：源 SHA256 → `os.link` → 校验 link 后 SHA256 一致；④ **禁止 copy/转码/symlink**，`os.link` 跨文件系统失败则停止上报，不得退化为 copy；⑤ manifest 每行 `log_name, frame_token, camera, rel_path, src_sha256, linked_sha256, inode_src, inode_dst`；⑥ 另写 `hardlink_manifest.summary.json`，含 `expected_frame_total`（54 log 的 metadata frame 总数）。

**验收**：

```bash
PRELUDE
ls $RASTER_4CAM_ROOT | wc -l   # 54
python - <<'PY'
import yaml, os, json, collections
cfg = yaml.safe_load(open("navsim/planning/script/config/training/default_train_val_test_log_split.yaml"))
IN = os.environ["STAGE_A_OUT"] + "/input_data"
rows = [json.loads(l) for l in open(f"{IN}/hardlink_manifest.jsonl")]
exp = json.load(open(f"{IN}/hardlink_manifest.summary.json"))["expected_frame_total"]
per = collections.Counter(r["camera"] for r in rows)
assert not set(os.listdir(os.environ["RASTER_4CAM_ROOT"])) & set(cfg["test_logs"])  # 无 test 泄漏
assert not [r for r in rows if r["src_sha256"] != r["linked_sha256"]]               # 内容一致
assert not [r for r in rows if r["inode_src"] != r["inode_dst"]]                    # 确为 hard link
assert all(per[c] == exp for c in ("CAM_F0","CAM_L0","CAM_R0")), (dict(per), exp)   # §7 一一对应
print("Step 0.2 OK", len(rows), dict(per), exp)
PY
```

### Step 0.3 重建 target cache（§2 已判废 `cache/rap_ego`）

```bash
PRELUDE
python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset train_test_split=trainval \
  experiment_name=stage_a_target_cache \
  cache_path=$STAGE_A_OUT/target_cache force_cache_computation=true \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.trajectory_sampling.interval_length=0.5 \
  agent.config.train_metric_cache_path=$NAVSIM_EXP_ROOT/train_metric_cache
find $STAGE_A_OUT/target_cache -name "rap_target.gz" | wc -l | tee $STAGE_A_OUT/input_data/target_count.txt
```

> **必须带 `experiment_name=`**：`default_training.yaml` 继承 `default_evaluation` 的 `experiment_name: ???`，且 `hydra.run.dir=${output_dir}` 引用它，缺失会在启动时抛 `MissingMandatoryValue`。
> **必须带 `dataset=navsim_dataset`**：默认是 `waymo_dataset`，而 [run_dataset_caching.py:55](navsim/planning/script/run_dataset_caching.py:55) 直接 `instantiate(cfg.dataset)`。
> **不要传 `rendered_sensor_blobs_path=` / `strict_camera_loading=`**：该脚本建 SceneLoader 时不接收这两个参数（只有 [run_training.py:169](navsim/planning/script/run_training.py:169) 才传），且用 `SensorConfig.build_no_sensors()`。target 不读 raster，**不存在缺 B0 风险**。

**验收**：`target_count.txt` ≥ **41,864**（= 34,468 + 7,396）；`ls $STAGE_A_OUT/target_cache | wc -l` = 54（若为子集须记录缺失 log）。

---

## Phase 1：Stage-A 数据构建

### Step 1.1 改造 `build_alignment_small_data.py`

**现状（已核实）**：`_render_missing_back_camera` 在 F0 复现 MAE > 1.0 时**直接 `raise RuntimeError`**（[:248](navsim/planning/script/build_alignment_small_data.py:248)），整个 build 崩溃 —— **不是**静默塞进 `failures`（`failures` 收的是候选审计异常，见 [:397](navsim/planning/script/build_alignment_small_data.py:397)）。改造目标是把"崩溃"换成"按 §3 规则补抽"。

1. 新增 `--dry-run`：算出 `qualified_train / qualified_val` 后打印 JSON 并 return，不生成 B0、不写 cache/manifest。
2. **top-up 循环**：`_render_missing_back_camera` 改为返回 `(record, ok, reason)` 而非 raise；`build()` 把 `select_hash_round_robin` 结果作初始选择，逐个做 B0 生成 + F0 校验 + 四相机复检，失败者移出、按**同一 hash 顺序**补入下一个未选过的候选，循环到集合大小 == 目标值。
3. 新增 `--topup-manifest`，每行 `split, removed_token, removed_log, reason, replacement_token, replacement_log`。
4. 熔断：单 split 累计替换数 / 目标数 > 1% 抛异常终止（§3）。
5. 复检 camera_order 用四相机 `("CAM_B0","CAM_F0","CAM_L0","CAM_R0")`；B0 生成**前**初筛仍用三相机。
6. **产物改名/新增**（当前只写 `token_manifest.json`/`samples.csv`/`input_audit.json`，与 §7 目录树不符）：`token_manifest.json` → **`stage_a_token_manifest.json`**；新增 **`dataset_manifest.json`**（`{"train":[...],"val":[...]}`，每条含 `token, log_name, split, selection_hash, map_location`）；`back_camera_generation` 数组 → 逐行追加写 **`b0_generation_manifest.jsonl`**（append-only）。
7. **B0 manifest 字段补全**（§3 点名要求，当前 [:260](navsim/planning/script/build_alignment_small_data.py:260) 全缺）：`log_name, frame_token, camera, relative_path, source_metadata_sha256, renderer_git_commit, map_version, existing_f0_sha256, regenerated_f0_sha256, f0_mae_0_to_255, generated_b0_sha256, decoded_shape, dtype, min, max, mean, std, nonzero_fraction, finite_fraction`。
8. **禁止重复生成**：当前 `if not back_path.exists()` 会静默复用已有 B0；改为已在 `b0_generation_manifest.jsonl` 中出现则抛异常（§3"同一 frame 不得生成两次，视为版本污染"）。
9. `_audit_candidate` 记录 `frame["map_location"]`，否则 §7 的"四 map_location 全覆盖"无从校验。
10. `--raster-root` 指向 **4cam root**（B0 写回该 root）；若另加 `--dest-raster-root` 则 `--raster-root` 只读源。

**验收**：`python -m pytest tests/training/test_alignment_experiment.py -q` 通过数不低于 `baseline_test_count.txt`，且新增 `test_topup_replaces_failed_token_in_hash_order`、`test_topup_aborts_above_one_percent`。

### Step 1.2 确定 N 并抽样生成 B0

`⌊N/3⌋` 的 N 来自实际候选池，**不得硬编码**：

```bash
PRELUDE
ARGS="--real-root $OPENSCENE_DATA_ROOT/sensor_blobs/trainval --raster-root $RASTER_4CAM_ROOT \
 --target-root $STAGE_A_OUT/target_cache --log-root $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
 --split-config navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
python navsim/planning/script/build_alignment_small_data.py --dry-run $ARGS \
  > $STAGE_A_OUT/input_data/candidate_pool.json
python -c "
import json;d=json.load(open('$STAGE_A_OUT/input_data/candidate_pool.json'))
print('TRAIN_N',d['qualified_train']//3);print('VAL_N',d['qualified_val']//3)"
```

> §2 期望 N = 34,468 / 7,396 → 11,489 / 2,465。**实际不符时必须改用 `⌊实际N/3⌋` 并记录。** 下文 `$TRAIN_N`/`$VAL_N` 为实际值（先 `export`）。

```bash
python navsim/planning/script/build_alignment_small_data.py $ARGS \
  --input-root $STAGE_A_OUT/input_data --cache-root $STAGE_A_OUT/paired_cache \
  --train-count $TRAIN_N --val-count $VAL_N \
  --topup-manifest $STAGE_A_OUT/input_data/topup_replacements.jsonl \
  --generate-missing-back-raster
```

**验收**（第 1、4、5、7 项任一失败即停止）：

```bash
python - <<'PY'
import json, os, collections
IN = os.environ["STAGE_A_OUT"] + "/input_data"
T, V = int(os.environ["TRAIN_N"]), int(os.environ["VAL_N"])
tok = json.load(open(f"{IN}/stage_a_token_manifest.json"))
ds  = json.load(open(f"{IN}/dataset_manifest.json"))
b0  = [json.loads(l) for l in open(f"{IN}/b0_generation_manifest.jsonl")]
top = [json.loads(l) for l in open(f"{IN}/topup_replacements.jsonl")]
assert (len(tok["train_tokens"]), len(tok["val_tokens"])) == (T, V)      # 1 样本数精确(§3)
assert len({r["log_name"] for r in ds["train"]}) == 44                   # 2 覆盖全部 train log
locs = {r["map_location"] for r in ds["train"]}; assert len(locs) == 4   # 3 四 map_location
rep = collections.Counter(r["split"] for r in top)
assert rep["train"] <= T*0.01 and rep["val"] <= V*0.01, dict(rep)        # 4 替换率 ≤1%
assert not [r for r in b0 if r["f0_mae_0_to_255"] > 1.0]                 # 5 F0 MAE ≤1.0
assert len(b0) == T + V                                                  # 6 B0 数量
assert len({(r["log_name"], r["frame_token"]) for r in b0}) == len(b0)   # 6 无重复生成
assert json.load(open(f"{IN}/input_audit.json"))["status"] == "passed"   # 7 训练入口 gate
print("Step 1.2 OK", T, V, len(b0), dict(rep), locs)
PY
```

### Step 1.3 抽样确定性复验

§7 要求"打乱输入顺序后重跑抽样结果完全一致"——**必须在候选池上验，不能在已选集合上验**（后者是恒等式，永远通过）：

```bash
PRELUDE
python - <<'PY'
import json, os, random
from navsim.planning.script.build_alignment_small_data import select_hash_round_robin
pool = json.load(open(os.environ["STAGE_A_OUT"]+"/input_data/stage_a_token_manifest.json"))["qualified_candidates"]
for split, count in [("train", int(os.environ["TRAIN_N"])), ("val", int(os.environ["VAL_N"]))]:
    recs = [r for r in pool if r["split"] == split]
    a = [r["token"] for r in select_hash_round_robin(list(recs), count)]
    random.shuffle(recs)
    b = [r["token"] for r in select_hash_round_robin(list(recs), count)]
    assert a == b, f"{split} 抽样不确定"
    print(split, "deterministic OK", len(a), "from pool", len(recs))
PY
```

---

## Phase 2：测试门槛（§7）

**Step 2.1 泄漏检查** — 新建 `tests/training/test_stage_a_data_integrity.py`，断言：① Stage-A token 所属 log 与 `test_logs` 交集为空；② train ∩ val token = ∅；③ train ∩ val log = ∅；④ `$RASTER_4CAM_ROOT` 不含 test log 目录；⑤ `stage_a_token_manifest.json` 的 token 全在 `dataset_manifest.json` 中；⑥ train/val/test 的原始 drive（log 名日期+车号前缀）无交集。验收：`pytest tests/training/test_stage_a_data_integrity.py -q` 全绿。

**Step 2.2 Loss 与集成测试** — `pytest tests/training/test_alignment_experiment.py -q -v`，重点确认（均已核实存在）：`test_four_alignment_switch_combinations`、`test_noalign_and_fullalign_use_identical_r100_task_supervision`、`test_spatial_alignment_detaches_raster_branch_only`、`test_global_alignment_grl_reverses_encoder_gradient`、`test_rap_grl_schedule_has_paper_scale_and_endpoints`、`test_predict_step_uses_real_input_and_filters_frame_names`。最后一个失败说明 GRL schedule 与 §4 公式不符，**停止上报**。

**Step 2.3 Evaluator 回归测试**（§7 门槛，Stage-A 启动前必须通过；原 Runbook 缺失）——产物落 `$STAGE_A_OUT/evaluator_validation/`：

```bash
PRELUDE
for A in constant_velocity_agent human_agent; do
  python navsim/planning/script/run_pdm_score.py agent=$A train_test_split=navtest \
    experiment_name=stage_a_evalcheck_$A
done
python navsim/planning/script/run_pdm_score.py agent=constant_velocity_agent \
  train_test_split=navtest experiment_name=stage_a_evalcheck_repeat
```

**验收**：① 两个官方 baseline 的 PDMS 复现 v1.1 参考值；② 两次 constant-velocity 的逐 scene CSV 完全相同（确定性）；③ 输出列只含 `NC, DAC, EP, TTC, C, DDC, PDMS`，**不含任何 v2-only 指标**（TLC/LK/HC/EC/EPDMS）；④ evaluator resolved commit（预期前缀 `0811876`）记入 `evaluator_validation/v1_commit.txt`。

---

## Phase 3：训练准备

### Step 3.1 per-seed initialization checkpoint

新建 `navsim/planning/script/make_init_checkpoints.py`，接口 `--seeds 0 1 2 --out-dir $STAGE_A_OUT/init_checkpoints`。行为：① 每 seed 做 `pl.seed_everything(seed, workers=True)` → `instantiate(rap_agent config)`（`dino_init_from_pretrained: true`、`checkpoint_path: ''`）；② **必须 `torch.save({"state_dict": agent.state_dict()}, path)`** —— [rap_agent.py:93](navsim/agents/rap_dino/rap_agent.py:93) 的 `init_from_pretrained` 取的是 `checkpoint['state_dict']`，直接存裸 state_dict 会 KeyError；③ 写 `sha256.json`；④ 断言 DINO backbone 权重非随机、projector/planner 为随机初始化。

**验收**：三个 `seed_*.ckpt` + `sha256.json` 存在；三个 SHA256 互不相同；`torch.load(seed_0.ckpt)` 含 `state_dict` 键。

### Step 3.2 创建 `alignment_stage_a.yaml`

以 `alignment_small.yaml` 为模板。**路径一律用 `${oc.env:...}` —— OmegaConf 不展开 shell 变量，写 `$STAGE_A_OUT` 会变成字面量路径。**

| 配置项（含完整层级） | 值 | 依据 / 陷阱 |
|---|---|---|
| `experiment_name` / `output_dir` | `stage-a-${alignment_condition_name}` / `${oc.env:STAGE_A_OUT}/training/${alignment_condition_name}/seed_${seed}` | §7 目录树；模板原值指向 `outputs/alignment_small/`，会覆盖旧 smoke run |
| `max_train_samples` / `max_val_samples` | `null` | §4 缩减走 token manifest，**禁止**截断 |
| `trainer.params.limit_train_batches` / `.limit_val_batches` | `1.0` | §4+§7 coverage；**层级在 `trainer.params` 下** |
| `subset_token_manifest_path` / `input_audit_path` | `${oc.env:STAGE_A_OUT}/input_data/{stage_a_token_manifest,input_audit}.json` | §3；后者是训练入口 gate |
| `cache_path` / `rendered_sensor_blobs_path` | `${oc.env:STAGE_A_OUT}/paired_cache` / `${oc.env:RASTER_4CAM_ROOT}` | §3 |
| `use_cache_without_dataset` / `force_cache_computation` / `include_auxiliary_datasets` / `require_camera_valid_subset` / `use_wandb_logger` | `true` / `false` / `false` / `true` / `false` | §3 资格条件 ④；§4 禁止 perturbed/others 混合 |
| `shuffle_train` | `true` | §4。**无 `shuffle_val` 这个 key**，val dataloader 在 [run_training.py:330](navsim/planning/script/run_training.py:330) 硬编码 `shuffle=False` |
| `trainer.params.max_epochs` | `20` | §4，须与 `WarmupCosLR(epochs=20)` 一致 |
| `trainer.params.strategy` | `ddp_find_unused_parameters_true` | 多卡必需：DINO backbone 冻结、NoAlign 下 domain 分支无梯度，默认 DDP 会因 unused parameters 崩 |
| `trainer.params.devices` × `dataloader.params.batch_size` × `trainer.params.accumulate_grad_batches` | 乘积 = **128**，推荐 `4 × 32 × 1` | §4，两 condition 必须完全相同 |
| `trainer.params.gradient_clip_val` / `.precision` / `.num_sanity_val_steps` | `0.0` / `16-mixed` / `0` | §4 |
| `agent.config.distill_feature_weight` / `.domain_align_weight` | `0.002` / `0.1` | §4 λ_spatial / λ_global，**显式钉死不依赖默认值** |
| `agent.config.distill_feature` / `.task_real_ratio` / `.eval_input_modality` / `.dino_init_from_pretrained` | `true` / `1.0` / `real` / `true` | §4 |
| `agent.config.trajectory_sampling.time_horizon` / `.interval_length` | `5` / `0.5` | §3「10 future」 |
| `agent.config.train_metric_cache_path` | `${oc.env:NAVSIM_EXP_ROOT}/train_metric_cache` | Step -1.3 实际生成位置 |
| `agent.checkpoint_path` | `${oc.env:STAGE_A_OUT}/init_checkpoints/seed_${seed}.ckpt` | §4；`defaults: - alignment_condition:` 由命令行覆盖 |

**同时修改** [rap_agent.py:594](navsim/agents/rap_dino/rap_agent.py:594) 的 `get_training_callbacks()`：
- `ModelCheckpoint(save_last=True, save_top_k=3, monitor='val/score')` 引入基于 val 的隐式选择，与 §4「primary checkpoint = final optimizer step / early stopping = false」冲突 → 改 `save_top_k=0, save_last=True`。
- **显式设 `dirpath=os.path.join(output_dir, "checkpoints")`** —— 否则 Lightning 会落到 `output_dir/csv_logs/version_0/checkpoints/`，后续所有 ckpt 路径都对不上。
- 新增 `EpochTokenChecksumCallback`（§7「记录每 epoch 的 token sequence checksum」）：在 `on_train_epoch_start` / `on_validation_epoch_start` 取 dataloader sampler 的实际顺序，把 token 序列 SHA256 逐行写 `output_dir/epoch_token_checksums.jsonl`（`epoch, stage, num_tokens, sequence_sha256`）。

**验收**：

```bash
PRELUDE
for C in noalign fullalign; do
  python navsim/planning/script/run_training.py --config-name alignment_stage_a \
    alignment_condition=r100_$C seed=0 --cfg job --resolve > /tmp/resolved_$C.yaml
done
grep -E "max_train_samples|max_val_samples|limit_train_batches|limit_val_batches|include_auxiliary|distill_feature|domain_align_weight|use_spatial_align|use_global_align|max_epochs|batch_size|devices|accumulate_grad|strategy|train_metric_cache_path|time_horizon|checkpoint_path|output_dir" /tmp/resolved_noalign.yaml
diff /tmp/resolved_noalign.yaml /tmp/resolved_fullalign.yaml
```

逐项人工核对上表。确认 NoAlign 下 `use_spatial_align: false` 且 `use_global_align: false`；确认所有路径已展开为绝对路径而非字面 `$VAR`。**`diff` 输出必须只有 `alignment_condition_name / use_spatial_align / use_global_align / experiment_name / output_dir` 这几行——这是"唯一变量是 alignment loss"的直接证据。**

---

## Phase 4：Stage-A 训练（6 run）

> 顺序约束：`seed 0 → 1 → 2`，每 seed 内 `noalign → fullalign`。禁止先跑完 3 个 noalign——那样无法早期发现配对不一致。

```bash
PRELUDE
for SEED in 0 1 2; do for C in r100_noalign r100_fullalign; do
  python navsim/planning/script/run_training.py --config-name alignment_stage_a \
    alignment_condition=$C seed=$SEED
done; done   # 注意：每完成一个 seed 就跑一次下面的验收，不通过不得继续下一个 seed
```

**每 seed 验收**（`NO=$STAGE_A_OUT/training/r100-noalign/seed_$SEED`，`FU=.../r100-fullalign/seed_$SEED`）：

```bash
python - <<'PY'
import json, csv, glob, math, os, torch
NO, FU, T = os.environ["NO"], os.environ["FU"], int(os.environ["TRAIN_N"])
a = json.load(open(f"{NO}/reproducibility_manifest.json"))
b = json.load(open(f"{FU}/reproducibility_manifest.json"))
assert len(a["train_tokens"]) == T                                        # 1 样本数
assert a["train_tokens"] == b["train_tokens"] and a["val_tokens"] == b["val_tokens"]
assert a["initialization_checkpoint_sha256"] == b["initialization_checkpoint_sha256"]  # 2 同一 init 文件
ca = [json.loads(l) for l in open(f"{NO}/epoch_token_checksums.jsonl")]
cb = [json.loads(l) for l in open(f"{FU}/epoch_token_checksums.jsonl")]
assert ca == cb, "两条件的 epoch batch token 顺序不一致"                   # 3 §7 核心
assert len([r for r in ca if r["stage"] == "train"]) == 20
exp = math.ceil(T/128)*20
gs = torch.load(f"{FU}/checkpoints/last.ckpt", map_location="cpu")["global_step"]
assert abs(gs - exp) <= 20, (gs, exp)                                     # 4 ≈1800 steps
rows = list(csv.DictReader(open(glob.glob(f"{NO}/csv_logs/*/metrics.csv")[0])))
last = [x for x in rows if x.get("train/task_loss")][-1]
assert abs(float(last["train/loss"]) - float(last["train/task_loss"])) < 1e-6  # 5 NoAlign 无 alignment
print("seed OK  steps", gs, "expected", exp)
PY
ls $NO/checkpoints $FU/checkpoints   # 应只有 last.ckpt
```

第 2、3、5 项是**受控性的核心验收**，任一失败立即停止。全部完成后：`find $STAGE_A_OUT/training -name "last.ckpt" | wc -l` 必须是 **6**。

---

## Phase 5：Stage-A 评测

**Step 5.1 metric cache**：`python navsim/planning/script/run_metric_caching.py train_test_split=navtest experiment_name=stage_a_navtest_metric_cache`。验收：`$NAVSIM_EXP_ROOT/metric_cache/` 非空，覆盖 navtest 的 12,146 scenes。

**Step 5.2 v1.1 PDMS 评测（6 次）**：

```bash
PRELUDE
for COND in r100-noalign r100-fullalign; do for SEED in 0 1 2; do
  python navsim/planning/script/run_pdm_score.py agent=rap_agent \
    agent.checkpoint_path=$STAGE_A_OUT/training/$COND/seed_$SEED/checkpoints/last.ckpt \
    agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.trajectory_sampling.interval_length=0.5 \
    agent.config.train_metric_cache_path=$NAVSIM_EXP_ROOT/train_metric_cache \
    train_test_split=navtest experiment_name=stage_a_v1_${COND}_seed${SEED} \
    2>&1 | tee $STAGE_A_OUT/evaluation_v1/${COND}_seed${SEED}.log
done; done
```

> **必须带 `trajectory_sampling` 覆写**：`rap_agent.yaml` 默认 `time_horizon: 4`（8 poses），训练用 5（10 poses）；[rap_agent.py:121](navsim/agents/rap_dino/rap_agent.py:121) 的 `initialize()` 用 **strict=True** 的 `load_state_dict`，维度不符会直接抛，或更糟——静默走通但轨迹长度错。
> **必须带 `train_metric_cache_path`**：`pdm_scorer: True` 使 eval 期也构造 `MetricCacheLoader`。

**验收**：`grep -h "Missing metric cache" $STAGE_A_OUT/evaluation_v1/*.log` 无输出，且

```bash
python - <<'PY'
import pandas as pd, glob, os
fs = sorted(glob.glob(os.environ["NAVSIM_EXP_ROOT"] + "/stage_a_v1_*/*/*.csv"))
assert len(fs) == 6, fs
sets = []
for f in fs:
    df = pd.read_csv(f)
    df = df[df["token"] != "average"]      # run_pdm_score 会额外追加一行 token="average"
    assert df["valid"].all(), f"{f} 含 valid=False 的 token"
    assert all(c in df.columns for c in ["NC","DAC","EP","TTC","C","DDC","PDMS"])
    assert not [c for c in df.columns if c in ("TLC","LK","HC","EC","EPDMS")]
    sets.append(set(df["token"]))
assert all(s == sets[0] for s in sets), "scene token 集合不一致"
print("6 runs share", len(sets[0]), "scenes")   # 与 12,146 比对，缺失须说明
PY
```

**Step 5.3 v2.2 EPDMS 评测 — ⛔ 阻塞**：本仓库不存在 v2.2 evaluator（全仓库搜索 `navhard`/`two_stage` 只命中 bevformer 的 transformer 模块）。规格书 §5 要求的 `navhard_two_stage` + EPDMS 零支持。执行前人工必须提供：① v2.2 官方仓库 clone 路径与 resolved commit；② `navhard_two_stage` 数据集物理路径；③ 该 evaluator 接受 RAP checkpoint 的方式（submission pickle 还是 agent 接口）。**拿到前不得自行实现**（§5：本地 scorer 不可代替官方 evaluator）。

**Step 5.4 汇总与预注册解释**：新建 `navsim/planning/script/summarize_stage_a.py`。**另起新脚本，不要改动已有的 `summarize_alignment_small.py` / `evaluate_alignment_small.py`**（那是 32/16 smoke run 的产物，§4 已声明与 Stage-A 无继承关系）。按 §6：每 seed 内按 scene token 配对算 `delta = FullAlign − NoAlign`；**以 log 为 cluster** bootstrap 报 95% CI；离散指标输出完整 transition table（不只 net count）；连续指标输出均值/中位数/分位数/effect size/95% CI；两项 co-primary 用 Holm correction。**不做**分层切片（§5 标注 Stage-B only）。

验收：产出 `$STAGE_A_OUT/summary/{stage_a_report.json, paired_deltas.csv}`；含 convergence gate 字段（读 6 run 的 val 曲线，最后一个 epoch 仍在明显趋势中则 `under_trained_regime: true`）；PDMS 的 delta CI 跨 0 时结论字段必须写 `"未能在 ≈1,800 steps 下检出效应"`，**禁止**写"alignment 无效"（§4 解释禁令）；因 5.3 阻塞须显式声明 `"仅有 v1 单侧证据，co-primary 不完整"`。

---

## Phase 6：Stage-B（本手册不展开）

Stage-B 增量：补齐 54 log 全部 43,432 frame 的 B0（复用 Step 1.2 脚本去掉 count 限制）；raster root 同质性校验 `renderer_git_commit`/`map_version` 唯一（新脚本，依赖 Step 1.1 改造点 7 已写入这两个字段）；Stage-A ⊂ Stage-B 子集校验（新测试）；场景标签生成 9 标签 + 阈值冻结 + 200 条人工审计（完整新模块，尚不存在）；分层切片统计 + BH FDR（新脚本，尚不存在）。

**Stage-B runbook 必须在 Stage-A 完成、且 Step 5.3 阻塞解除后另行编写。**

---

## 附录：阻塞项汇总

| # | 阻塞项 | 影响 Step | 需要谁 |
|---|---|---|---|
| B1 | `NUPLAN_MAPS_ROOT` / `NAVSIM_EXP_ROOT` 实际值 | -1.1 | 人工 |
| B2 | train metric cache 生成耗时与资源 | -1.3 | 人工 |
| B3 | 实际候选池 N 是否等于 34,468 / 7,396 | 1.2 | 由 Step 1.2 `--dry-run` 输出决定 |
| B4 | GPU 资源与 `devices × batch_size × accum = 128` 的拆分 | 3.2 | 人工 |
| B5 | **NAVSIM v2.2 evaluator 完全缺失** | 5.3 | 人工 |

B5 最严重：它使 §5 的 co-primary 只剩一半。若无法解决，Stage-A 结论必须按 §5「方向不一致须记为 mixed result」的精神，明确声明"仅有 v1 单侧证据"。

> 原附录的「Step 0.3 在无 B0 时能否用 `strict_camera_loading=true`」已删除：`run_dataset_caching.py` 建 SceneLoader 时用 `SensorConfig.build_no_sensors()`，既不接收也不使用该参数，target 缓存不读 raster，不存在此风险。
