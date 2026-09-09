# navtest 场景元数据画像 — Codex 任务书

Target reader: AI coding agent。工作目录 `$RAP_ROOT`。Spec of record: `docs/rap-alignment-experiment-plan.md`。冻结协议: `outputs/alignment_stage_a/summary/summary_protocol_v2.json`。On conflict, spec wins。

全程只读现有产物：不训练、不推理、不加载 checkpoint。

**前置**：先跑一次 `scripts/alignment/scene_vs_run_decomposition.py`，否则下面的 `*_tokens.txt` 不存在。

## 目标

6 次独立运行（2 条件 × 3 seed）在同一批 12,146 个 navtest 场景上的逐场景 v1 分数已经证明：失败高度场景固有（242 个场景六次全败，独立性预期 0.0045）。

本任务回答一个纯描述性问题：**这些被稳定做坏的场景，长什么样？**

为 navtest 全部场景生成元数据标签，再把若干**条件无关**的子集与全体分布对比。

## 全局禁令（违反任一条 → HALT 并报告）

1. **禁止按训练条件切分。** 所有子集定义在全部 6 次运行之上，绝不出现「FullAlign vs NoAlign」「哪个条件赢」。协议 `run_scope.forbidden_use` 禁止对 alignment 是否改善 planning performance 作任何方向性结论；本任务合法的唯一原因就是它从不比较两个条件。
2. **禁止使用 HD map。** 不得调用 `get_maps_api`、`extract_map_features` 或任何 nuplan-devkit 地图接口。依赖地图的标签属 spec §5 的 Stage-B 范围。
3. **禁止修改**：`summary_protocol*.json`、`evaluation_v1/`、`input_data/`、`training/`、`evaluator_validation/`，以及 `summary/` 下已有的 `scene_run_matrix.csv`、`*_tokens.txt`、`scene_vs_run_decomposition.json`。只新增文件。
4. 本任务的标签**不是** spec §5 的 Stage-B scene label manifest（那份要求阈值从 train split 冻结、在看到任何预测之前生成、人工核验 ≥200 条）。本任务的标签生成于分数已被观测之后，故结论只能是 **descriptive / exploratory**，不得表述为 confirmatory。必须写进输出 JSON 的 `_scope`。
5. **图像标签的表述禁令**（spec §5 原文，逐条继承）：
   - `illumination` 必须写作 illumination / luminance quantile，**未经真值验证不得称 day / night**。
   - `blur_like` 只是图像统计标签，**不等价于运动模糊、天气或 occlusion**。
   - 阈值取自 navtest 自身分位数（见 Step 1c），与 spec §5 的「只从 official train split 计算一次」不同，属**有意偏离**，必须在 `_scope` 中声明。
6. **禁止写入 `sensor_blobs`**。图像一律只读打开，不转码、不落盘副本（contact sheet 除外）。

## 输入

```
summary/token_to_log.json                        12,146 token -> log_name，136 logs
summary/hard_core_tokens.txt                     242
summary/ep_bottom_decile_all_six_tokens.txt      207
summary/safe_but_stalled_all_six_tokens.txt      31
summary/failed_exactly_once_tokens.txt           947    （对照）
summary/never_failed_tokens.txt                  9,683  （对照）
input_data/candidate_pool.json                   train/val record，含 endpoint 真值
```
（相对 `outputs/alignment_stage_a/`）

真实图像：`$NAVSIM_SENSOR_ROOT` 下与 navtest 对应的 split。相对路径由 `frame["cams"][camera]["data_path"]` 给出，与 log pickle 同源。

**优先 import 而非重写**：
- `build_alignment_small_data.py` → `_actor_count_30m`、`_read_rgb`、`_sensor_config`；字段取法 `frame["driving_command"]`、`frame["map_location"]`、`frame["anns"]["gt_boxes"|"gt_names"]`、`frame["cams"][camera]["data_path"]`
- `audit_subset_distribution.py` → `_route_command`、`_actor_bin`、`_continuous_bin`、`_proportions`、`_distribution`（含 TVD）
- 相机集合与固定 crop：从 `outputs/alignment_stage_a/resolved_config_r100_noalign.yaml` 与 `navsim/agents/rap_dino/{rap_features.py,bevformer/image_encoder.py}` 读取实际使用的相机顺序与 crop，**不要凭猜**。查不到 crop 时退回整帧，并在产物中记录 `crop: "full_frame"`。

## Step 1 — `scripts/alignment/navtest_scene_labels.py`

1. 从 `token_to_log.json` 取 token → log_name 与 136 个 log 名。
2. **定位 log pickle 目录**：在 `$NAVSIM_LOGS_ROOT` 的 `trainval` / `test` / `mini` 等候选中，找 136 个 `<log_name>.pkl` 齐全的那个。**一个都不齐 → HALT 并打印缺失清单**，不得用部分数据继续。
3. 逐 log `pickle.load`，建 `token -> (log_name, frame_index)`。同一 token 映射到多处 → HALT。
4. 逐 token 抽取标签：

| 字段 | 取法 |
|---|---|
| `route_command` | `frame["driving_command"]` → `_route_command` |
| `map_location` | `frame["map_location"]` |
| `actor_count_30m` / `actor_density_bin` | `_actor_count_30m(frame)` → `_actor_bin` |
| `endpoint_dx/dy/dyaw` + bins | 见 Step 1b |

### Step 1b — endpoint 推导与强制自校验

navtest 无 target cache，endpoint 须从 log 的 future ego pose 推。先读 `build_alignment_small_data.py` 中 target `trajectory`（shape `(10,3)`）的定义——相对哪一帧、什么坐标系、取第几个未来帧——照同一定义实现。

**自校验（必须通过才能输出 endpoint 标签）**：`candidate_pool.json` 的 train/val record 带有由 target cache 算出的 `endpoint_dx/dy/dyaw` 真值。用新实现对**至少 1,000 个**这类 token 重算，断言 `max |Δ| <= 1e-3`（米 / 弧度）。

- 通过 → 输出 endpoint 标签，并记录校验统计。
- **不通过 → 不输出 endpoint 标签**，记 `endpoint_labels: "dropped"` 与原因，其余三组照常输出。**绝不输出未经校验的 endpoint。**

### Step 1c — 真实图像统计标签

6 次评测本身就是吃真实相机图像跑出来的（12,146 个场景全部 `valid=True`），所以 navtest 的真实图像必然可读。**但先验证再用**：

1. 从 resolved config 读实际相机集合（预期 `CAM_F0 / CAM_L0 / CAM_R0 / CAM_B0`）与 current frame 索引。
2. **覆盖率探针**：抽 500 个 token，检查全部相机文件存在且可解码。覆盖率 < 100% → 记录缺失清单与逐 log 覆盖率，**并在产物中打 `image_coverage_partial: true`**。
   缺失在本项目中是 **log/chunk 块状、非随机**的（Stage-A 的 real 覆盖率只有 20.887%），因此覆盖不全时图像标签的子集画像可能有偏，必须在结论字段显式警告，不得当作随机缺失处理。
3. 逐 token 按 §5 定义计算（先按固定 crop 取有效区域，可先降采样到长边 ≤512 以控时间）：

| 标签 | 定义 |
|---|---|
| `illumination_score` | `Y = 0.2126R + 0.7152G + 0.0722B`；每相机取有效区域 median；scene score = 四路 median 的中位数 |
| `illumination_bin` | `low / typical / high`，阈值 = **navtest 自身**的 P20 / P80 |
| `contrast_score` / `low_contrast` | 四路 camera luminance IQR 的中位数；`<= P10 → true` |
| `sharpness_score` / `blur_like` | 四路 `Laplacian(Y)` variance 的中位数；`<= P10 → true` |

阈值用 navtest 自身分位数而非 train split，原因有二：train 侧真实图像只有 21% 可用；且 Step 2 做的是**同一 split 内**的「子集 vs 全体」对比，用 split 内分位数才让零假设「子集看起来像 navtest」成立。这是对 spec §5 的有意偏离，写进 `_scope`。

**断言**：阈值只计算一次并落盘；同一 token 在任何后续调用中标签一致。

### 输出

`summary/navtest_scene_labels.csv`，12,146 行：

```
token, log_name, route_command, map_location, actor_count_30m, actor_density_bin,
endpoint_dx, endpoint_dy, endpoint_dyaw, endpoint_dx_bin, endpoint_dy_bin, endpoint_dyaw_bin,
illumination_score, illumination_bin, contrast_score, low_contrast,
sharpness_score, blur_like, image_ok
```

`summary/navtest_scene_labels_meta.json`：log 目录、log 数、覆盖数、endpoint 校验结果、相机集合与 crop 来源、图像覆盖率与缺失清单、四个图像阈值的数值、每个标签的取值计数。

**断言**：恰好 12,146 行；token 集合与 `token_to_log.json` 完全一致；无缺失值；`route_command` 全部通过 one-hot 检查。任一不满足 → HALT。

## Step 2 — `scripts/alignment/subset_metadata_profile.py`

5 个子集分别与全体 12,146 对比：`hard_core`、`ep_bottom_decile_all_six`、`safe_but_stalled_all_six`、`failed_exactly_once`（对照）、`never_failed`（对照）。

对每个 `子集 × 标签` 报告：

1. **逐取值比例** + 全体比例 + **比值**（subset / full）
2. **TVD**（用 `_distribution`）
3. **cluster bootstrap 95% CI**，**cluster 单位 = `log_name`**，10,000 次重采样（与 `summary_protocol_v2.json` 的 `cluster_bootstrap` 一致）
4. **随机子集零分布**：抽 2,000 个同样大小的随机子集，算 TVD 分布，报告 p95。**观测 TVD 必须与该 p95 并列呈现**——否则 n=31 的子集会因纯抽样噪声看起来「分布很不一样」。

### 必做的混杂检查：log 集中度

242 个场景若有 200 个来自 3 个 log，那「富集」是 log 效应而非场景类型效应。逐子集报告 distinct log 数、前 5 个 log 占比、`子集每 log 场景数 / 全体每 log 场景数` 的分布。

**任一子集 >30% 来自 ≤3 个 log → 打 `log_confounded: true`**，并在结论字段注明该子集的画像不可解释为场景类型效应。

### 图像标签的处理

图像标签（`illumination_bin` / `low_contrast` / `blur_like`）与元数据标签走**完全相同**的四步流程，不另设规则。`image_ok == false` 的 token 在图像标签上按缺失处理并单列计数，不得静默丢弃。

### 输出

- `summary/subset_metadata_profile.json`
- `summary/figs/fig_d_subset_profiles.png`：每标签一组分组柱状图（full + 各子集比例），n<100 的子集柱子用不同边框标出。`matplotlib.use("Agg")`。

## Step 3 — `scripts/alignment/subset_contact_sheets.py`

给人眼看的定性材料。**这是本任务对汇报最直接的产出。**

1. **硬核片**：`hard_core` 按 `mean_pdms` 升序取前 24 个场景，每场景一张四相机拼图（按 resolved config 的相机顺序），标注 `token`、`log_name`、`route_command`、`actor_count_30m`、以及六次运行中归零的是 `NC` 还是 `DAC`。
2. **对照片（必做）**：从 `never_failed` 随机抽 24 个（固定 seed），用**完全相同**的排版渲染。
   没有对照组，人眼会在 24 张难场景里认出并不存在的规律。
3. **盲评片（可选，有时间才做）**：把上面 48 个场景打乱混排输出一张，答案另存 `blind_key.json`。先看盲评片记下判断，再对答案。

输出 `summary/figs/contact_hard_core.png`、`contact_never_failed.png`、（可选）`contact_blind.png` + `blind_key.json`。单张过大时按 8 场景一页分页。

**禁令**：只读打开图像；不得把 `sensor_blobs` 里的原图复制到别处；contact sheet 是唯一允许落盘的图像衍生物。

## 验收

1. `navtest_scene_labels.csv` 12,146 行，无缺失，断言全通过
2. endpoint 通过 1e-3 校验，或被显式 drop 并记录原因
3. 图像覆盖率已探测并落盘；不足 100% 时打 `image_coverage_partial` 并附非随机缺失警告
4. 四个图像阈值只计算一次并落盘；相机集合与 crop 的来源已记录
5. 5 个子集 × 全部标签（含 3 个图像标签）都有：逐取值比例、TVD、cluster bootstrap CI、随机零分布 p95
6. 5 个子集全部报告 log 集中度，超阈值的打 `log_confounded`
7. `fig_d_subset_profiles.png`、`contact_hard_core.png`、`contact_never_failed.png` 生成
8. 输出 JSON 含 `_scope`：子集条件无关；标签生成于分数被观测之后，故为 descriptive / exploratory；非 spec §5 的 Stage-B label manifest；图像阈值取自 navtest 自身（对 spec §5 的有意偏离）
9. 代码中不存在比较两个训练条件的路径
10. `git status` 不出现对禁令清单中任何文件的修改，`sensor_blobs` 未被写入

## 完成后报告

- 每个子集在各标签（含图像标签）上的 TVD，与对应随机零分布 p95 并列
- 哪些 `子集 × 标签` 的 CI 不跨过 full 的比例（即真正有区分度的）
- 每个子集的 log 集中度与 `log_confounded` 标记
- endpoint 自校验结果
- 图像覆盖率；若 <100%，逐 log 覆盖率与受影响的子集
- 三张 contact sheet 的路径
