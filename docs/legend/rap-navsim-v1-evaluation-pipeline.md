# RAP NAVSIM v1 评测流程分析报告

> 分析对象：本仓库 `RAP`（RAP fork），分支 `exp/rap-alignment-regression`。
> 分析方式：全局搜索 + 逐文件读码追踪执行流，未实际运行评测（本机无数据集/GPU）。
> 所有结论均标注了对应文件与行号，便于核对。

---

## 0. 结论速览（TL;DR）

1. **NAVSIM v1 的官方评测入口只有一个 Python 程序**：`navsim/planning/script/run_pdm_score.py`，由 Hydra 驱动，主配置为 `navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`。
2. **仓库里没有官方的 `.sh` 评测入口脚本**。README 的第 4 步「test navsim model」直接把读者指向上游 navsim 仓库（`README.md:157-159`）。仓库内唯一一个名字像评测的 shell 脚本 `scripts/legacy/metabev_eval.sh` 指向 `tools/eval_meta_bev_batch.py`，**该文件当前已不存在**，属于失效的历史脚本。
3. **评测是两阶段的**：
   - 阶段一（离线、CPU 密集）：`run_metric_caching.py` 跑 PDM-Closed 规划器，把「参考轨迹 + 插值后的真值障碍物观测 + 中心线 + 可行驶区域地图」冻结成 metric cache。
   - 阶段二（GPU）：`run_pdm_score.py` 加载 agent 权重 → 对每个 token 推理出 ego 系轨迹 → LQR + 运动学自行车模型仿真 → PDMScorer 算 6 个子指标 → 聚合成 PDMS → 落 CSV。
4. **PDMS 公式**（`pdm_scorer.py:156-183`）：
   `PDMS = (NC × DAC) × [ (5·EP + 5·TTC + 2·C + 0·DDC) / 12 ]`
   注意 `driving_direction_weight = 0.0`（`default_scoring_parameters.yaml:23`），即 v1 口径下 DDC 只记录不计入分数。
5. **默认评测集**：`train_test_split=navtest`，即 OpenScene/NAVSIM 的 `test` split，本仓库自有的作业脚本断言其 token 数为 **12,146**（`scripts/stage_a_step_2_2_navtest_metric_cache.qsub:37`）。
6. **有 4 个必须知道的坑**（详见第 6 节），其中最关键的是：当前 HEAD 上直接用 `agent=rap_agent` 跑 `run_pdm_score.py`，会因为 rendered（栅格化）图像未接线而**让所有 token 被标记为 invalid**，PDMS 变成 NaN。

---

## 1. 核心文件清单

### 1.1 入口程序（全部是 Hydra Python 入口，无 shell）

| 路径 | 角色 |
| --- | --- |
| `navsim/planning/script/run_pdm_score.py` | **主评测入口**。端到端：加载 agent → 推理 → 仿真 → 打分 → 写 CSV |
| `navsim/planning/script/run_metric_caching.py` | **评测前置**。为 navtest 生成 metric cache（PDM-Closed 参考轨迹等） |
| `navsim/planning/script/run_train_metric_caching.py` | 训练用 metric cache（训练时 PDM 打分头需要）。⚠️ README 第 115 行写的是 `run_training_metric_caching.py`，**该文件名不存在** |
| `navsim/planning/script/run_create_submission_pickle.py` | 生成 leaderboard 提交用的 `submission.pkl`（只推理不打分） |
| `navsim/planning/script/run_merge_submission_pickles.py` | 合并多个提交 pkl（多 seed / 集成） |
| `navsim/planning/script/run_pdm_score_from_submission.py` | 从提交 pkl 离线复现打分（不重新推理） |

**仓库内 shell / 作业脚本（均非官方评测入口）**

| 路径 | 说明 |
| --- | --- |
| `scripts/stage_a_step_2_2_navtest_metric_cache.qsub` | 本 fork 在 TSUBAME 上跑 navtest metric caching 的 qsub 作业，含 12,146 token 校验，是最接近「可直接抄」的真实命令 |
| `scripts/legacy/metabev_eval.sh` | 失效：所引用的 `tools/eval_meta_bev_batch.py` 已被删除 |
| `navsim/planning/script/evaluate_alignment_small.py` | 本 fork 自研的开环小样本评测（走 CacheOnlyDataset，不是 PDMS 官方路径） |

### 1.2 配置文件

| 路径 | 关键内容 |
| --- | --- |
| `config/pdm_scoring/default_run_pdm_score.yaml` | 主评测配置。`defaults` 组合 `default_common` + `default_evaluation` + `default_scoring_parameters` + `agent`；`override train_test_split: navtest`；`metric_cache_path: ${NAVSIM_EXP_ROOT}/metric_cache`；本分支新增 `output_csv: null` |
| `config/common/default_evaluation.yaml` | 数据路径：`navsim_log_path = ${OPENSCENE_DATA_ROOT}/navsim_logs/${train_test_split.data_split}`，`sensor_blobs_path = ${OPENSCENE_DATA_ROOT}/sensor_blobs/...`；输出目录 `output_dir = ${NAVSIM_EXP_ROOT}/${experiment_name}/${experiment_uid}` |
| `config/common/default_common.yaml` | worker 默认 `ray_distributed_no_torch`、`gpu: true`、日志等级 |
| `config/pdm_scoring/default_scoring_parameters.yaml` | **评测口径的唯一真源**：`proposal_sampling = 40 poses × 0.1s`（4 秒 @10Hz）、`PDMSimulator`、`PDMScorer` 及其权重/阈值 |
| `config/common/train_test_split/navtest.yaml` | `data_split: test` + `scene_filter: navtest` |
| `config/common/train_test_split/scene_filter/navtest.yaml` | 12,294 行的 log 白名单；`num_history_frames: 4`、`num_future_frames: 10`、`frame_interval: 1`、`has_route: true` |
| `config/common/train_test_split/scene_filter/navmini.yaml` | 小规模冒烟用（配 `train_test_split=navmini`，`data_split: mini`） |
| `config/common/agent/rap_agent.yaml` | RAP agent 实例化配置：`RAPAgent` + `RAPConfig`，`trajectory_sampling: 4s / 0.5s`，`pdm_scorer: True`，`train_metric_cache_path: './train_metric_cache'`，`checkpoint_path: ''` |
| `config/common/agent/constant_velocity_agent.yaml` / `human_agent.yaml` | 两个 sanity baseline，**不需要 GPU、不需要图像**，是验证评测链路是否正确的最佳手段 |
| `config/metric_caching/default_metric_caching.yaml` | metric cache 输出路径 `${NAVSIM_EXP_ROOT}/metric_cache` |
| `config/common/worker/*.yaml` | `ray_distributed_no_torch` / `single_machine_thread_pool` / `sequential` 三种 worker |

### 1.3 评测核心逻辑模块

| 路径 | 职责 |
| --- | --- |
| `navsim/evaluate/pdm_score.py` | **打分总装**：ego 系轨迹 → 全局系 `InterpolatedTrajectory` → 10Hz 重采样 → 调 simulator/scorer → 抽出 6 个子指标封装成 `PDMResults` |
| `navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py` | LQR 跟踪 + 运动学自行车模型，把「几何轨迹」变成「可执行状态序列」 |
| `.../simulation/batch_lqr.py`、`batch_kinematic_bicycle.py` | 上面那步的具体实现 |
| `.../scoring/pdm_scorer.py` | **6 个指标 + PDMS 聚合**（NC/DAC/DDC 乘性，EP/TTC/C 加权） |
| `.../scoring/pdm_comfort_metrics.py`、`pdm_scorer_utils.py` | 舒适度阈值、碰撞类型判定 |
| `navsim/planning/metric_caching/metric_cache_processor.py` | 生成 metric cache：跑 `PDMClosedPlanner`，插值真值障碍物到 10Hz |
| `navsim/planning/metric_caching/metric_cache.py` | `MetricCache` dataclass（lzma 压缩 pickle） |
| `navsim/planning/metric_caching/caching.py` | metric caching 的 worker 并行调度 |
| `navsim/common/dataloader.py` | `SceneLoader`（按 scene_filter 切帧、拿 `AgentInput`）、`MetricCacheLoader`（按 metadata csv 建 token→路径映射） |
| `navsim/common/dataclasses.py` | `AgentInput` / `Scene` / `Camera` / `Trajectory` / `SceneFilter` / `SensorConfig` / `PDMResults` |
| `navsim/agents/abstract_agent.py` | `compute_trajectory()` 通用推理封装（feature builder → forward → 取前 8 个 pose） |
| `navsim/planning/scenario_builder/navsim_scenario.py` | 把 NAVSIM `Scene` 适配成 nuPlan `AbstractScenario`，供 metric caching 用 |

### 1.4 RAP agent 侧（被评测的模型）

| 路径 | 职责 |
| --- | --- |
| `navsim/agents/rap_dino/rap_agent.py` | `RAPAgent`：权重加载（`init_from_pretrained:82` / `initialize:108`）、传感器配置（`get_sensor_config:124`，只用 F0/L0/R0/B0 四目当前帧）、feature/target builder |
| `navsim/agents/rap_dino/rap_model.py` | `RAPModel.forward`（`:114`）：DINOv3 backbone → BEV → 多轮 refine 出 64 条 proposal → scorer 头出 pdm_score → **argmax 选一条**（`:172-174`） |
| `navsim/agents/rap_dino/rap_features.py` | `RAPFeatureBuilder.compute_features`（`:38`）：图像特征 + `ego_status`（pose/velocity/acceleration/driving_command） |
| `navsim/agents/rap_dino/bevformer/bev_feature_build.py` | `_get_bev_feature`（`:76`）：**同时**构造 real 与 rendered 两路四目图像张量 —— 第 6 节坑点的根源 |
| `navsim/agents/rap_dino/score_module/compute_navsim_score.py` | 训练期用的可微 PDM 打分（不参与官方评测） |

---

## 2. 前置条件：环境变量与数据布局

评测前必须导出（`README.md:55-64`，本仓库真实取值见 `env/stage_a.env`）：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/rap_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/rap_workspace/dataset"
export NAVSIM_EXP_ROOT="$HOME/rap_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/rap_workspace/navsim"
```

配置解析出的目录要求（`default_evaluation.yaml:3-7`）：

```
$OPENSCENE_DATA_ROOT/
  navsim_logs/test/*.pkl          # navtest 的 log 标注（scene_filter 按文件名白名单过滤）
  sensor_blobs/test/...           # navtest 的真实相机图像
$NUPLAN_MAPS_ROOT/                # nuplan-maps-v1.0，metric caching 必需
$NAVSIM_EXP_ROOT/
  metric_cache/                   # 阶段一产物；metadata/*.csv + 每 token 一个 metric_cache.pkl
  <experiment_name>/<timestamp>/  # 阶段二输出目录（含 CSV 与 hydra 配置快照）
```

---

## 3. 评测执行指令

### 3.1 阶段一：生成 navtest 的 metric cache（必须先做，一次性）

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtest experiment_name=navtest_metric_cache worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=64
```

产物：`$NAVSIM_EXP_ROOT/metric_cache/`，其中 `metadata/*.csv` 是 `MetricCacheLoader` 唯一的索引来源（`dataloader.py:289-300`）。校验方式（本仓库 qsub 里的做法）：token 数应为 **12,146**，且 `metadata` 目录下**只能有一个 csv**（多个 csv 时 `MetricCacheLoader` 只取第一个，会静默丢 token）。

### 3.2 阶段二 · 先跑 baseline 验证链路（强烈建议）

```bash
python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=constant_velocity_agent experiment_name=cv_navtest
```

```bash
python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=human_agent experiment_name=human_navtest
```

`human_agent` 用真值未来轨迹（`human_agent.py:35-41`，`requires_scene=True`，由 `run_pdm_score.py:80-84` 分支处理），PDMS 应接近上限；`constant_velocity_agent` 应显著更低。这两个数能确认「metric cache + scorer」这半条链路是对的，再去查模型侧的问题会省很多时间。

两个 baseline 本身都是 `build_no_sensors()`、不吃图像也不吃 GPU，但注意 **`run_pdm_score.py` 硬编码了 `Task(num_gpus=1)`**（见 6.1），所以即使跑 baseline 也需要机器上有可见 GPU，否则 Ray 无法调度、任务会一直挂起。

### 3.3 阶段二 · 评测 RAP 模型

```bash
python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=rap_agent agent.checkpoint_path=/abs/path/to/RAP_DINO_navsimv1.ckpt agent.config.train_metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache experiment_name=rap_navtest output_csv=$NAVSIM_EXP_ROOT/rap_navtest.csv
```

这条命令里有三个**不能省**的覆写，原因见第 6 节：

- `agent.checkpoint_path=...`：不给的话 `initialize()` 直接跳过（`rap_agent.py:111`），`self.device` 从未被赋值，`compute_trajectory` 里 `.to(self.device)` 会 AttributeError。
- `agent.config.train_metric_cache_path=...`：`RAPAgent.__init__` 无条件构造 `MetricCacheLoader(config.train_metric_cache_path)`（`rap_agent.py:73`），默认值 `./train_metric_cache` 是相对路径，不存在就在实例化阶段崩。指向阶段一的 navtest cache 即可（评测路径不会用到它）。
- `output_csv=...`：本分支新增的字段，指定后 CSV 落到固定路径而不是时间戳目录，便于脚本化对比。

⚠️ **这条命令在当前 HEAD 上会让所有 token 失败**（PDMS=NaN），必须先处理第 6.4 节的 rendered 图像问题。

### 3.4 小规模冒烟（改数据集为 mini split）

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navmini experiment_name=navmini_metric_cache worker=sequential
```

```bash
python navsim/planning/script/run_pdm_score.py train_test_split=navmini agent=constant_velocity_agent experiment_name=cv_navmini
```

### 3.5 Leaderboard 提交路径（HuggingFace `AGC2024-P/e2e-driving-navtest`）

```bash
python navsim/planning/script/run_create_submission_pickle.py train_test_split=private_test_e2e agent=rap_agent agent.checkpoint_path=/abs/path/to/ckpt experiment_name=rap_submission team_name=YOUR_TEAM authors=YOUR_NAME email=you@example.com institution=YOUR_LAB country=YOUR_COUNTRY
```

产出 `$NAVSIM_EXP_ROOT/<experiment_name>/<timestamp>/submission.pkl`（`run_create_submission_pickle.py:91-93`），结构为 `{team_name, authors, email, institution, country / region, predictions: [ {token: Trajectory} ]}`。本地想复现 leaderboard 打分（需要对应 split 的 metric cache）：

```bash
python navsim/planning/script/run_pdm_score_from_submission.py submission_file_path=/abs/path/to/submission.pkl metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache output_dir=$NAVSIM_EXP_ROOT/submission_rescore train_test_split=navtest
```

注意 `run_create_submission_pickle.py:33-39` 显式禁止 `agent.requires_scene=True` 的 agent —— 提交路径下不允许看到标注场景。

---

## 4. 代码级执行流程（Pipeline）

### 4.A 阶段一：metric caching

1. `run_metric_caching.py:17` → `build_logger` → `build_worker(cfg)`（`builders/worker_pool_builder.py`）→ `cache_data(cfg, worker)`。
2. `caching.py:235` 用 `SceneLoader(sensor_blobs_path=None, sensor_config=build_no_sensors())` 只读标注、不读图像，按 `scene_filter` 把每个 log 切成 14 帧（4 历史 + 10 未来）的 scene，key 是第 4 帧（当前帧）的 token（`dataloader.py:112-130`）。
3. 按 log 打包成 `data_points`，`worker_map` 分发到 `cache_scenarios`（`caching.py:28`）。
4. 每个 scene → `Scene.from_scene_dict_list` → `NavSimScenario(scene, map_root=$NUPLAN_MAPS_ROOT)`（`caching.py:48-55`），把 NAVSIM 数据适配成 nuPlan scenario 接口。
5. `MetricCacheProcessor.compute_metric_cache`（`metric_cache_processor.py:209`）：
   - 跑 `PDMClosedPlanner`（IDM 5 档速度 × 3 条横向偏移，`:50-61`）得到 **PDM-Closed 参考轨迹**（5 秒 @10Hz）；
   - `_interpolate_gt_observation`（`:90`）把 2Hz 的真值障碍物插值到 10Hz，得到 `PDMObservation`；
   - 连同 `initial_ego_state`、中心线 `PDMPath`、`route_lane_ids`、`PDMDrivableMap` 一起 dump 成 lzma 压缩 pickle。
6. `save_cache_metadata` 写出 `metadata/*.csv`，即后续 `MetricCacheLoader` 的索引。

> 这一阶段决定了评测的「世界模型」：**其他交通参与者是真值回放（非交互式）**，ego 的动作不会改变他们的轨迹。这正是 NAVSIM v1「非反应式 pseudo-closed-loop」的定义。

### 4.B 阶段二：`run_pdm_score.py` 主流程

1. **主进程建 token 列表**（`run_pdm_score.py:115-129`）：`SceneLoader`（无传感器）取 navtest 全部 token，`MetricCacheLoader` 取 cache 里全部 token，两者**取交集**；缺失/多余各自 warning 后跳过。
2. **按 log 分片**（`:131-138`）：每个 log 一个 `data_point`，携带完整 `cfg`。
3. **Ray 并行**（`:139-155`）：硬编码 `RayDistributed`（**忽略 `worker=` 配置**），按 `torch.cuda.device_count()` 把分片均分，`Task(fn=run_pdm_score, num_cpus=16, num_gpus=1)`，即**一个 GPU 跑一个 worker 进程**。
4. **worker 内初始化**（`:48-65`）：`instantiate(cfg.simulator)` / `instantiate(cfg.scorer)`（断言两者 `proposal_sampling` 一致）→ `instantiate(cfg.agent)` → `agent.initialize()`（加载 ckpt、`.to(device)`）→ 建 `MetricCacheLoader` 与带传感器的 `SceneLoader`（`sensor_config=agent.get_sensor_config()`，RAP 只加载 4 目当前帧）。
5. **逐 token 推理**（`:69-92`）：
   - lzma 解压读出 `MetricCache`；
   - `scene_loader.get_agent_input_from_token(token)` 构造 `AgentInput`（4 帧历史 ego 状态 + 当前帧图像）；
   - `agent.compute_trajectory(agent_input)`：`AbstractAgent.compute_trajectory`（`abstract_agent.py:62`）跑 feature builder → `forward` → **取前 8 个 pose**（`:79`）→ 包成 `Trajectory`（默认采样 4s / 0.5s，`dataclasses.py:282`）。RAP 内部是先出 64 条 proposal，再用打分头 argmax 选一条（`rap_model.py:164-174`）。
   - 单 token 抛异常会被捕获（`:94-98`），只把该行标 `valid=False`，**不中断整个 job**。
6. **打分**（`evaluate/pdm_score.py:83`）：
   - `transform_trajectory`：ego 系相对位姿 → 以 `initial_ego_state.rear_axle` 为原点转到全局系 → `InterpolatedTrajectory`（`:24-52`，速度/加速度置零，反正 LQR 不用）；
   - `get_trajectory_as_array`：把 PDM-Closed 参考轨迹与模型轨迹都按 `proposal_sampling`（40×0.1s）重采样成状态数组（`:55-80`）；
   - 两条轨迹拼成 `(2, 41, state_dim)`，`simulator.simulate_proposals` 用 LQR + 自行车模型仿真出实际可执行状态（`:107-109`）；
   - `scorer.score_proposals(...)` 输入仿真状态 + cache 里的观测/中心线/route/可行驶区域（`:111-117`）；
   - 取 `pred_idx=1`（索引 0 是 PDM-Closed 参考，只用来做 progress 归一化的分母）抽出 6 个子指标，返回 `PDMResults`（`:120-140`）。
7. **聚合与落盘**（`run_pdm_score.py:158-187`）：列名映射成 `NC/DAC/EP/TTC/C/DDC/PDMS`，按 token 排序，写 CSV，日志打印成功/失败数与 `PDMS` 均值。

### 4.C `PDMScorer` 内部（`pdm_scorer.py:114-183`）

先算 `_calculate_ego_area`（车身角点落在哪些语义区域），再算 6 个指标：

| 指标 | 类型 | 含义 | 代码位置 |
| --- | --- | --- | --- |
| `NC` no_at_fault_collisions | 乘性 | 是否发生「己方过错」碰撞（后方来车撞我不算） | `:293-349` |
| `DAC` drivable_area_compliance | 乘性 | 是否驶出可行驶区域 | `:351-358` |
| `DDC` driving_direction_compliance | 加权（**权重 0**） | 1 秒窗口内逆行位移，2m/6m 两档阈值 | `:360-396` |
| `EP` ego_progress | 加权，权重 5 | 沿中心线的进展，除以 PDM-Closed 参考的进展做归一化 | `:398-412` + `:167-173` |
| `TTC` time_to_collision | 加权，权重 5 | 按当前速度外推 0.5s/1.0s 是否会撞 | `:414-498` |
| `C` comfort | 加权，权重 2 | 加速度/加加速度/横摆率是否全部在舒适阈值内 | `:500-509` |

聚合（`_aggregate_scores`，`:156-183`）：

```
mult      = NC × DAC                  # _multi_metrics 只有这两项，见 pdm_enums.py:156-160
EP_norm   = raw_progress × mult / max(raw_progress × mult)   # 分母含 PDM-Closed 参考；若最大进展 < 5m 则整体置 1（mult=0 的置 0）
weighted  = (5·EP_norm + 5·TTC + 2·C + 0·DDC) / (5+5+2+0)
PDMS      = mult × weighted
```

> 注：`DDC` 被写进 `_weighted_metrics`（`pdm_scorer.py:396`）而非 `_multi_metrics`——`MultiMetricIndex` 只有 `NO_COLLISION` 与 `DRIVABLE_AREA` 两项（`pdm_enums.py:156-160`）。权重为 0 时 DDC 对 PDMS 无影响，但 CSV 里仍会输出该列。这与 NAVSIM v1 的口径一致。

---

## 5. 关键参数速查

| 参数 | 值 | 出处 |
| --- | --- | --- |
| 评测 split | `test`（navtest），12,146 token | `train_test_split/navtest.yaml`、`stage_a_step_2_2_navtest_metric_cache.qsub:37` |
| 场景切分 | 4 历史帧 + 10 未来帧 @2Hz，`frame_interval=1`，`has_route=true` | `scene_filter/navtest.yaml:4-7` |
| 模型输出轨迹 | 8 poses × 0.5s = **4 秒** | `rap_agent.yaml:11-12`、`abstract_agent.py:79` |
| 仿真/打分采样 | 40 poses × 0.1s = **4 秒 @10Hz** | `default_scoring_parameters.yaml:4-5` |
| metric cache 参考轨迹 | PDM-Closed，50 poses × 0.1s = 5 秒 | `metric_cache_processor.py:44` |
| 真值障碍物 | 2Hz 采样后插值到 10Hz，**非反应式** | `metric_cache_processor.py:90-190` |
| RAP 输入相机 | CAM_F0 / L0 / R0 / B0，仅当前帧（history index 3） | `rap_agent.py:124-136`、`bev_feature_build.py:29` |
| RAP proposal 数 | 64，用打分头 argmax 选一条 | `navsim_config.py:43`、`rap_model.py:172` |
| 权重 | EP 5 / TTC 5 / C 2 / DDC 0 | `default_scoring_parameters.yaml:20-23` |

---

## 6. 已知差异与坑（跑之前务必看）

### 6.1 `run_pdm_score.py` 硬编码 Ray + 每任务 1 GPU

`:139-155` 里 `build_worker(cfg)` 被注释掉（`:111`），改成硬编码 `RayDistributed` 和 `Task(num_cpus=16, num_gpus=1)`。这段是 **RAP 官方 release 就带的改动**（`git show 20cddfd` 确认），不是本分支引入。后果：

- `worker=sequential` 之类的覆写在**评测阶段完全无效**（metric caching 阶段仍然有效）；
- 无 GPU 机器上 `num_gpus=1` 的任务无法被 Ray 调度，会一直挂起；
- 每个 worker 要 16 CPU，核数不足时并行度会被 Ray 限死。
- README 里给的 `ray start --head ...` 那段（`README.md:126-137`）是**训练**时加速 PDM 打分用的，评测这里 `RayDistributed` 会自行起本地 Ray。

### 6.2 本分支（未提交）对评测入口的修改

`git diff` 显示 `run_pdm_score.py` 与 `default_run_pdm_score.yaml` 有未提交改动，全部是为了可复现性：

- 所有 token 集合改走 `sorted(set(...))`，消除 Python set 迭代顺序带来的分片不确定性；
- 输出列固定为 `token,valid,NC,DAC,EP,TTC,C,DDC,PDMS` 并按 token 排序，`to_csv(index=False)`；
- 新增顶层 `output_csv`（默认 `null`），支持指定固定输出路径；
- **移除了原版末尾追加的 `average` 行**——现在 CSV 里没有 average 行，均值只出现在日志里。对比历史 CSV 时注意这一点。

### 6.3 `RAPAgent` 实例化就会去读 train metric cache

`rap_agent.py:39-78`：只要 `config.cache_data == False`（评测就是这个值），构造函数就会执行 `MetricCacheLoader(Path(config.train_metric_cache_path))`。该分支**不受 `pdm_scorer` 开关保护**。默认值 `'./train_metric_cache'` 是相对当前工作目录的路径，不存在则直接抛异常，且发生在 Ray worker 内部，报错信息容易被淹没。解法：评测时显式覆写 `agent.config.train_metric_cache_path` 指向任一有效 metric cache 目录。

### 6.4 rendered（栅格化）图像未接线 —— 当前 HEAD 下 RAP 评测必然全失败

- `run_pdm_score.py:60-65` 构造 `SceneLoader` 时**没有传** `rendered_sensor_blobs_path`；
- 于是 `Cameras.from_camera_dict`（`dataclasses.py:80-96`）走 `rendered_image = None, rendered_valid = False`；
- 但 `_get_bev_feature`（`bev_feature_build.py:77`）**无条件**先构造 synthetic 一路：`LoadMultiViewImageFromFiles(agent_input, synthetic=True)` 里 `selected_image = cam.rendered_image` 为 `None` → `continue` → 收集到 0 张图 → `raise ValueError("Expected four camera views, got 0")`（`:69-70`）；
- 该异常被 `run_pdm_score.py:94-98` 捕获，**每个 token 都被标成 `valid=False`**，最终 CSV 全是空值、`PDMS` 均值为 NaN，而 job 会「正常」结束。这种失败模式最阴险：不崩，只是全 NaN。

对照上游 RAP release（`git show 1d9a886:navsim/common/dataclasses.py`）：那时 rendered 路径是由 `str(image_path).replace('sensor_blobs', 'rendered_sensor_blobs')` 推出来的，且读不到时 **fallback 成全零图**，所以官方版本能跑通（rendered 那路在推理时其实不进 `RAPModel.forward`，它只取 `features["camera_feature"]`，即 real 那路）。本 fork 为了对齐实验的严格性去掉了静默 fallback，代价就是评测路径断了。

两种修法（择一）：

1. **接线**：给 `run_pdm_score.py` 的 `SceneLoader` 加 `rendered_sensor_blobs_path=`（可复用 `run_training.py:282-296` 里已有的 cfg 读法），并在评测配置里加同名字段；
2. **恢复 fallback**：让 `LoadMultiViewImageFromFiles` 在 `synthetic=True` 且无 rendered 图时用零图占位（等价于上游行为）。

推荐 (1)：显式接线不会掩盖数据缺失，而且和本仓库 Stage-A 的 `strict_camera_loading` 设计一致。

### 6.5 其他小坑

- **必须给 `agent.checkpoint_path`**：`rap_agent.py:111` 判断的是 `!= ""`，而 yaml 默认就是 `''`；不给则 `initialize()` 空转，`self.device` 未定义 → `compute_trajectory` 里 `.to(self.device)` AttributeError。
- **权重会被加载两次**：`__init__` 里 `init_from_pretrained`（strict=False，剥 `agent.` 前缀）和 `initialize()`（strict=True，把 `agent._rap_model` 换成 `_rap_model`）各加载一次，key 规则还不一样。ckpt 格式不匹配时，第二次会以 strict 报错的形式暴露。
- **轨迹长度不一致**：训练命令用 `agent.config.trajectory_sampling.time_horizon=5`（10 poses），而评测按 `Trajectory` 默认 4s/8 poses，`compute_trajectory` 用 `[:8]` 硬截断。用 5s 训练的 ckpt 评测时无需改动，但如果你改了 `interval_length`，这里会静默错位。
- **`MetricCacheLoader` 只取 metadata 目录下的第一个 csv**（`dataloader.py:296`）。多次 metric caching 会留下多个 csv，导致 token 集合悄悄变小 —— 交集逻辑会把它当成「missing metric cache」warning 掉。
- **README 的 `run_training_metric_caching.py` 不存在**，实际文件名是 `run_train_metric_caching.py`。

---

## 7. 附：训练期的 PDMS 代理评估

训练时并不跑上面这套完整流程，而是走 `navsim/agents/rap_dino/score_module/compute_navsim_score.py`：`RAPAgent.compute_score`（`rap_agent.py:248`）拿 64 条 proposal 直接对着 **train metric cache** 批量算可微的 PDM 子分数，作为 `score` / `best_score` 记到日志，并由 `ModelCheckpoint(monitor='val/score', mode='max')`（`rap_agent.py:596-605`）挑选 checkpoint。它与官方 PDMS 口径接近但不等价（没有 LQR 仿真那一步），**不能替代 `run_pdm_score.py` 的结果**用于论文/榜单汇报。

---

## 8. 复现检查清单

- [ ] 5 个环境变量已导出，`$OPENSCENE_DATA_ROOT/navsim_logs/test` 与 `sensor_blobs/test` 存在
- [ ] `$NUPLAN_MAPS_ROOT` 下有 `nuplan-maps-v1.0`
- [ ] 阶段一完成，`$NAVSIM_EXP_ROOT/metric_cache/metadata/` 下**只有一个** csv，token 数 = 12,146
- [ ] `agent=constant_velocity_agent` 与 `agent=human_agent` 两个 baseline 跑通，PDMS 数值合理（human ≫ cv）
- [ ] 已处理 6.4 的 rendered 图像问题
- [ ] RAP 评测命令带齐 `agent.checkpoint_path`、`agent.config.train_metric_cache_path`
- [ ] 机器上有 GPU 且 Ray 能起来（`ulimit -u` 足够大）
- [ ] 结果 CSV 的 `valid` 列全为 True；若有 False，去 worker 日志找 traceback
