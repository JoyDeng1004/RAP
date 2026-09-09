# RAP Paired Alignment 受控实验计划

## 1. 结论与边界

**唯一研究问题**：在相同 real planning supervision 下，paired real-raster feature alignment 是否改善整体 planning performance，或在特定场景中引入被平均分掩盖的 regression？

| RAP 组件 | raster 的作用 | 本实验 |
|---|---|---|
| Paired real-raster alignment | 同场景干净结构视图，作 feature-space teacher / regularizer | **覆盖** |
| Raster-only augmentation | recovery perturbation、cross-agent view、无配对真实照片的额外样本 | **不覆盖** |

结论只能落在 paired alignment 上；正结果不得声称验证了 recovery 或 cross-agent augmentation。术语统一：模块名 Raster-to-Real alignment，spatial gradient direction 为 `real → detached raster`。非 RAP Figure 5 严格复现（Figure 5 用 RAP-ResNet、85k paired + 100k raster-only + 12k real val）。

**两阶段结构**：模型、超参、条件定义、泄漏边界、评测协议完全相同，仅训练集规模不同。Stage-A 用 official train 的确定性 1/3 子集、不做场景标签与分层切片；Stage-B 用全部合格 official train、执行分层切片。两阶段评测均为 v1+v2 co-primary。Stage-A 不替代 Stage-B，其结果**不得**用于修改 Stage-B 的任何超参、样本量、标签规则、阈值或评测协议（否则 Stage-B 变为 selection-on-result）。

## 2. 数据与隔离

```text
real images:      /gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/trainval
source raster:    /gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs
4cam raster root: /gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs_4cam_v1
NAVSIM logs:      /gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/navsim_logs/trainval
official split:   /gs/bs/tga-RLA/qdeng/RAP/navsim/planning/script/config/training/default_train_val_test_log_split.yaml
```

`cache/rap_ego` 判废：只覆盖少量已有 target，旧 `rap_feature.gz` 中 real/raster feature 已发现全黑与 `camera_valid=False`。target 与 paired cache 必须从 NAVSIM metadata、real 原图、raster 原图重新生成。

**盘点结果（2026-08-16，只读）**：原始 raster root 有 64 个有效 log 目录 + 1 个空 `missing_camera` 目录。

| 项目 | 数量 | | Split | Logs | 完整 window | 通过检查 | Stage-A 1/3 子集 |
|---|---:|---|---|---:|---:|---:|---:|
| NAVSIM metadata frames | 51,890 | | train | 44 | 35,334 | **34,468** | **2,498**（实测，≈57/log） |
| `CAM_F0` files | 51,898 | | val | 10 | 7,396 | **7,396** | **464**（实测，≈46/log） |
| `CAM_L0` files | 51,898 | | test/navtest | 10 | 8,328 | **8,328** | 仅评测，禁止训练 |
| `CAM_R0` files | 51,900 | | | | | | |
| `CAM_B0` files | 48 | | | | | | |

统计条件：`4 history + 10 future`、`frame_interval=1`、`is_valid=True`、`has_route=True`、三路现有 raster 可解码。文件数超出 metadata frame 数的部分为 orphan files，不得自动进入数据集；NAVSIM metadata 是 frame 枚举和相机相对路径的唯一来源。以上为补全四相机前的盘点值，正式数量以冻结 manifest 为准。

**Stage-A 实测修正**：表中通过检查的 34,468 / 7,396 不含 real 图像可用性检查；冻结候选池实测合格 train / val 为 7,495 / 1,394。差异源于 real `sensor_blobs` 三相机覆盖率仅 20.887%。Stage-B 数量仍须在全量数据构建时由实测 manifest 决定。

**Test 泄漏边界**：10 个 test log 全部出现在 NAVSIM v1 `navtest`，**禁止进入** train/val dataset、normalization/quantile/label threshold 拟合、checkpoint selection/early stopping/超参选择、alignment loss 的 paired raster cache。"使用全部数据训练和验证"严格表示：使用全部合格 official train 和 val，同时永久隔离 official test/navtest/navhard。禁止把 64 个 raster log 重新随机切成 train/val——官方 split 已在原始 drive 级别无交叉，按 token 随机切分会引入同一 drive 的时序泄漏。

## 3. 四相机协议

```text
camera order = [CAM_B0, CAM_F0, CAM_L0, CAM_R0]
ImgEncoder.num_cams = 4
```

以官方发布的模型代码为准补全 `CAM_B0`，不改三相机版本。新建版本化 root `rendered_sensor_blobs_4cam_v1`，只含 44 个 official train log + 10 个 official val log：

- `CAM_F0/L0/R0`：SHA256 核验的原文件，可用只读 hard link，禁止转码。必须**一次性**对 54 个 log 的全部 metadata frame 建立（B0 生成时的 F0 校验从本 root 读 F0）。
- `CAM_B0`：对 54 个 log 的全部 43,432 个 metadata frame 重新生成。原始目录已有的 48 张 smoke-test B0 不复用。
- 10 个 test/navtest log 不复制到本 root，不生成训练 cache。不得向原始三相机 root 混写 B0。
- **分批生成**：Stage-A 只为被抽中的 ≈13,954 个 train+val token 生成 B0，Stage-B 启动前补齐其余。root 为 append-only，`b0_generation_manifest.jsonl` 追加写，禁止重写已冻结条目；同一 frame 不得生成两次（视为版本污染，回滚整个 root）。**Stage-B 启动前必须校验整个 root 的 `renderer_git_commit` 与 `map_version` 全条目唯一**，不通过则不得启动。

**B0 与 F0 同时重渲染**：每个 metadata frame 调用同一版本 `process_data.helpers.renderer.ScenarioRenderer`，`camera channels = [CAM_F0, CAM_B0]`；生成 B0 的同时重渲染 F0，用原 F0 校验 renderer 可复现性：

```text
mean(abs(F0_existing_uint8 - F0_regenerated_uint8)) <= 1.0
```

阈值 `1.0` 位于 `0–255` 像素空间，相当于归一化空间约 `1/255`。任一 frame 超阈值时：① 不接受该 frame 的 B0；② 记录 log、token、renderer 输入和误差；③ **停止冻结数据版本**；④ 排查 renderer commit / map version / metadata 不一致，禁止静默跳过。manifest 至少记录：

```text
log_name, frame_token, camera, relative_path, source_metadata_sha256,
renderer_git_commit, map_version, existing_f0_sha256, regenerated_f0_sha256,
f0_mae_0_to_255, generated_b0_sha256, decoded_shape, dtype,
min, max, mean, std, nonzero_fraction, finite_fraction
```

**样本资格条件**（须同时满足）：① token 唯一映射到 log 和 frame；② 有 4 帧 history 和 10 帧 future；③ `is_valid=True` 且 `has_route=True`；④ 四路 real 与四路 raster 文件存在且可解码；⑤ real/raster 的 log、frame、camera calibration 和 camera order 一致；⑥ target 能由原始 Scene 重新生成且 shape/dtype/数值合法；⑦ 所有输入 tensor 为 finite；⑧ 四路 raster 不得全部纯黑（单路纯黑须显式记录）；⑨ train/val/test 的 token、log 和原始 drive 均无交集。只有 `input_audit.json` 中 `status=passed`，训练入口才允许启动。

**Stage-A 抽样**：用 `navsim/planning/script/build_alignment_small_data.py` 的 `select_hash_proportional`——**按 log 规模比例分层的确定性哈希抽样**。**抽样单位是 frame token，不是 log**（44 个 train log 跨 4 个 `map_location` 且光照分布不均，整 log 抽样会丢失城市或时段覆盖）。算法：① 按 log 分组，组内按 `(sha256(split, log_name, token), token)` 升序排序；② 精确配额 `q_l = n_l / N × count`，`base_l = max(1, ⌊q_l⌋)`；③ `Σbase > count` 时抛异常（目标过小，无法同时满足每 log ≥1 与比例性），否则按余数 `q_l − ⌊q_l⌋` 降序分配剩余名额，余数相同时以 `sha256(log_name)` 升序决胜；④ 每 log 取排序后前 `quota_l` 条；⑤ 断言 `Σquota = count`、`min(quota) ≥ 1`、`quota_l ≤ n_l`。全程无随机数、无 seed，结果与输入顺序无关。流程：在通过全部资格条件的候选池上抽样（B0 相关条件 ④⑧ 在 B0 生成后复检，生成前只用 `CAM_F0/L0/R0` 判定）→ 取 `⌊N_train/3⌋` 与 `⌊N_val/3⌋` → 为选中 token 生成 B0 并执行 F0 校验。

**已弃用**：早期版本的 `select_hash_round_robin`（每轮从每个 log 各取 1 个）等价于**等配额**抽样。在 44 个 train log、均值约 783 候选、目标约 261/log 的实际分布下，小 log 采样率可达约 87%、大 log 仅约 17%，per-log 采样率极差约 5 倍，使 Stage-A 训练分布系统性偏离 Stage-B 全量分布，削弱 Stage-A 作为 Stage-B 先导估计的外推力。比例分层在保留"每 log 至少 1 个"与确定性的同时消除该畸变，**不得回退**。

**top-up 规则**：选中 token 在 B0 生成或 F0 校验（MAE ≤ 1.0）失败时，**在同一 log 内**按 hash 序补抽下一个未选候选，以保持 per-log 配额不变；该 log 候选耗尽时才回退到全局 hash 序，并在 manifest 标记 `fallback_cross_log=true`。manifest 记录被替换 token、替换者 token、失败原因；**禁止静默减少样本数**，最终数必须精确等于目标值；单 split 替换率 **> 1%** 时停止冻结并排查 renderer 一致性。Stage-A token 集合必须是 Stage-B 的真子集，冻结为 `outputs/alignment_stage_a/input_data/stage_a_token_manifest.json`，Stage-B manifest 生成后必须校验该关系。

**抽样分布审计**（抽样后、训练前执行）：比较全候选池与选中子集在 `route_command`、`map_location`、`actor_density`（30m 内 vehicle/pedestrian/bicycle 计数分箱 `0–5/6–10/11–20/>20`）、GT future 轨迹端点 `Δx/Δy/Δψ` 分箱（`Δy` 分箱 `<−2/−2~−0.5/−0.5~0.5/0.5~2/>2` m）上的比例，报告 `TVD = 0.5·Σ|p_full − p_subset|`，并报告 per-log 采样率 `quota_l/n_l` 的 min/max/中位数。阈值运行前冻结：categorical TVD ≤ 0.02、binned continuous TVD ≤ 0.05、per-log 采样率 max/min ≤ 1.5。审计器不读取 checkpoint、prediction、PDMS/EPDMS。**审计结果只作记录与报告，不得据此更换抽样算法、参数或重抽**——抽样确定性且无 seed 可换，"审计不过就重抽"会退化为 selection-on-data；超阈值时停止并由人工决定是否修订本节，修订必须发生在任何训练启动之前。结果冻结为 `distribution_audit.json`，并在 Stage-A 报告中引用。

## 4. 受控训练条件

| 条件 | Task input | Task supervision | Spatial align | Global align | Eval input |
|---|---|---|---|---|---|
| `R100-NoAlign` | real | 100% real | Off | Off | real |
| `R100-FullAlign` | real | 100% real | On | On | real |

**唯一变量是 alignment loss 项。** FullAlign 可读 paired raster feature 计算 alignment loss，但 **raster sample 不参与 planning task loss**；两条件的 planning supervision、样本暴露次数、模型结构必须完全相同。不包含：raster-only planning task loss、recovery-oriented perturbation dataset、cross-agent dataset、额外 100k raster-only reproduction、perturbed 或 auxiliary cache 混合训练。

$$\mathcal{L}_{spatial}=\operatorname{MSE}(F_{real},\operatorname{stopgrad}(F_{raster}))\qquad \mathcal{L}_{NoAlign}=\mathcal{L}_{task,real}$$
$$\mathcal{L}_{FullAlign}=\mathcal{L}_{task,real}+0.002\,\mathcal{L}_{spatial}+0.1\,\mathcal{L}_{global}\qquad \lambda_{GRL}(p)=0.1\left(\frac{2}{1+\exp(-10p)}-1\right),\ p=\frac{completed\ steps}{total\ steps}$$

Global alignment：paired real/raster feature 用相同 global pooling；domain classifier 同时读两个 domain 且数量平衡；raster encoder/projector branch detach；real branch 经 GRL 接收反向梯度；domain classifier 正常最小化 BCE。`λ_global=0.1` 与 `λ_GRL,max=0.1` 是不同参数，禁止混写。

**冻结的训练预算**：

| | Stage-A | Stage-B | | Stage-A | Stage-B |
|---|---:|---:|---|---:|---:|
| train / val samples | **2,498 / 464** | ≈34,468 / ≈7,396 | seeds | [0, 1, 2] | [0, 1, 2] |
| epochs | **20** | **20** | runs | 6 | 6 |
| effective global batch | **128** | **128** | final-step ckpt | 6 | 6 |
| optimizer steps | **400** | ≈5,400 | 每样本暴露次数 | 20 | 20 |

optimizer steps 按 `⌈samples / 128⌉ × 20` 估算（`drop_last=False`），随冻结 manifest 微调。

注：Stage-B 的 ≈34,468 / ≈7,396 samples 与 ≈5,400 optimizer steps 均为待测定的估算值，以 Stage-B 数据构建时的实测 manifest 为准。

```text
optimizer = AdamW              gradient clipping = 0.0     lambda_spatial = 0.002
initial learning rate = 1e-4   dropout = 0                 lambda_global = 0.1
weight decay = 1e-4            precision = 16-mixed        early stopping = false
schedule = cosine decay, 1 warmup epoch, min_lr = 1e-5     shuffle_train = true
max_train_samples = null       limit_train_batches = 1.0   shuffle_val = false
max_val_samples = null         limit_val_batches = 1.0
primary checkpoint = final optimizer step
```

Stage-A 的缩减通过冻结 token manifest 实现，**不通过** `max_train_samples` 或 `limit_train_batches` 截断（后者破坏 DataLoader coverage 要求，也是旧 smoke run 失效的直接原因）。显存不足时只能用 per-device batch + gradient accumulation 保持相同 effective batch，两条件必须使用完全相同的设备数、batch 划分和 accumulation。**同一 seed 下必须共享**：initialization checkpoint 与 SHA256、train/val manifest、每 epoch 的 batch token 顺序、augmentation 随机数流、optimizer/scheduler/总 optimizer step 数、task target/task mask/task sample count、validation 与 inference 输入、checkpoint 保存时点。

**初始化 checkpoint**：共同 initialization **不得包含 R2R alignment 训练历史**。允许 ① 同一 pretrained frozen DINO backbone + 相同随机初始化 projector/planner；② 同一个明确以 100% real、NoAlign 训练得到的 checkpoint。每次运行记录来源、训练数据、训练目标、SHA256。Stage-A 采用 ①（`dino_init_from_pretrained: true`）：每 seed 先生成一份 initialization state_dict 落盘并记 SHA256，该 seed 的两个 run 通过 `agent.checkpoint_path` 加载**同一文件**，禁止两条件各自在进程内随机初始化。`ckpts/RAP_DINO_navsimv2.ckpt` 训练来源未确认，**Stage-A 禁止使用**；若已接受过 alignment，只能用于 pipeline test 或 continued-alignment diagnostic。

**与 RAP-DINO 官方实现对照**——已对齐：optimizer AdamW、lr 1e-4、scheduler cosine、weight decay 1e-4、gradient clipping 0、total batch size (NAVSIM) 128、epochs 20（单阶段）、λ_spatial 0.002、λ_global 0.1、camera 数 4。已知偏离（有意选择，须在结果报告中声明）：

| 项 | 官方论文 | 本实验 | 原因 |
|---|---|---|---|
| dropout | 0.1 | **0** | released code 中 `RAPConfig.tf_dropout = 0`，按"以官方发布代码为准"立场取 0；两阶段必须同值 |
| 训练阶段结构 | 20 pretraining + 20 finetuning | 单阶段 20 epochs | pretraining 依赖已被排除的 raster-only augmentation |
| 训练数据量 | 85k paired + 100k raster-only | ≈34,468（B）/ 2,498（A） | 可与 metadata 对齐的 paired 数据上限 |
| 总 optimizer steps | ≈4 万+ | ≈5,400（B）/ 400（A） | 数据量与单阶段结构的必然结果 |
| GPU 配置 | 4 × H100 | 不约束 | 只约束 effective global batch = 128 与两条件设备一致 |

`dropout = 0` 会系统性放大 alignment 的观测效应（两者均为正则化）；该偏差同时作用于两阶段，不影响两条件比较的内部效度，但影响与论文数值的可比性。

**Stage-A 解释规则（预注册，运行前冻结）**：记录两条件 train loss、val loss、`val/score` 逐 epoch 曲线作 **convergence gate**，若最后一个 epoch 仍在明显趋势中则全部结论标注 **under-trained regime**。Stage-A 400 steps 是 Stage-B 估算值的约 1/14、官方配置的至多约 1/100，为"保持每样本暴露 20 次"的直接结果。**Δ 的 CI 跨 0** 时只能结论为「未能在 400 steps 下检出效应」，**不得**结论为「alignment 无效」，Stage-B 照常执行；**Δ 显著为负**视为强信号，Stage-B 仍需执行以确认是否为小数据特有现象。**外推方向警告**：正则化收益随数据量减少而系统性放大 ⇒ Stage-A 的 Δ 为 Stage-B 的**乐观侧估计**，Stage-B 效应量预期更小，Stage-A 的 Δ 接近 0 时 Stage-B 出现更小或反向效应可预期，不构成矛盾。

**旧 32/16 smoke checkpoint 勘误**：`outputs/alignment_small/alignment-small-r100-{noalign,fullalign}` 不能描述为"由全部 32 个 train samples 训练"。两次运行虽构建 32 train + 16 val，但配置为 `batch_size=1`、`devices=1`、`limit_train_batches=8`、`shuffle_train=false`、`max_epochs=2`，故每 epoch 只有排序靠前的相同 8 条参与更新，`last.ckpt` 共 16 个 optimizer steps，只能证明有限的 pipeline 可运行性。与 Stage-A **无继承关系**（从 `RAP_DINO_navsimv2.ckpt` 出发且 `dino_init_from_pretrained: false`），不得沿用其任何 checkpoint 或 cache。

## 5. 评测与分层

| Benchmark | Official split | Primary metric | 统计单位 |
|---|---|---|---|
| NAVSIM v1.1 | `navtest`（完整 12,146 scenes） | PDMS | scene token，按 log cluster |
| NAVSIM v2.2 | `navhard_two_stage` | EPDMS | root scene |

PDMS 与 EPDMS **分别报告，禁止合并成自定义平均分**；方向不同时结论必须写成 mixed result，不得择优。评测集不随训练集缩减，两阶段均在完整评测集上运行。Sub-scores：v1 报告 `NC`（No at-fault Collision）、`DAC`（Drivable Area Compliance）、`EP`（Ego Progress）、`TTC`（Time to Collision）、`C`（Comfort）、`DDC`（Driving Direction Compliance）、`PDMS`；v2 报告 `NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC, EPDMS`，并同时保留 `seed, condition, log_id, root_scene_id, stage, rollout_token, followup_id, start_position, stage1_endpoint, raw_kernel_weight, normalized_kernel_weight, all sub-scores, EPDMS`。Stage 2 follow-up rollout **不是独立 test scene**，paired comparison 以 `root_scene_id` 为单位，follow-up transition 仅作诊断输出。正式 scorer 使用隔离的官方版本，本地 scorer 不可代替：

```text
NAVSIM v1: official v1.1 tag，保存 resolved commit（预期前缀 0811876）
NAVSIM v2: official v2.2 tag，保存 resolved commit
```

**场景标签（Stage-B only）**：`NavSimScenario.scenario_type` 固定返回 `unknown`，不可用。在查看任何预测前生成 condition-independent scene label manifest；同一 scene 可有多个标签，主分析不做 `city × illumination × intersection × density` 交叉切片。

| 标签 | 定义 | 阈值 | 表述禁令 |
|---|---|---|---|
| `route_command` | 直读 metadata one-hot `driving_command`：`TURN_LEFT, GO_STRAIGHT, TURN_RIGHT, UNKNOWN` | — | 禁止按预测轨迹或人工角度阈值重新解释 |
| `intersection_context` | 官方 HD map `INTERSECTION` polygon + logged expert trajectory。`inside`：ego center 在 polygon 内；`approach`：不在 polygon 内但当前至未来 **5 秒** expert trajectory LineString 与 polygon 相交；`non_intersection`：以上均不满足。优先级 `inside > approach > non_intersection` | — | — |
| `traffic_light_context` | `none, green_only, any_red` | — | 不得写成"自车受红灯控制"（附近 lane connector 的灯不一定属自车 route） |
| `actor_density` | 当前 ego frame 中心距离 ≤ **30m** 的 `vehicle, pedestrian, bicycle` 计数 → `low/medium/high` | train `P33/P67` | — |
| `VRU_present` | 30m 内存在 `pedestrian` 或 `bicycle` → `true` | — | — |
| `map_location` | 直读 metadata：`us-ma-boston`、`us-nv-las-vegas-strip`、`us-pa-pittsburgh-hazelwood`、`sg-one-north` | — | 表示 geography/domain bundle，不单独解释为天气或道路风格 |
| `illumination` | 四路 real current-frame 按模型一致的固定 crop 计算 $Y=0.2126R+0.7152G+0.0722B$；每 camera 取有效区域 median，scene score 为四路 median 的中位数 → `low/typical/high` | train `P20/P80` | 必须写作 illumination/luminance quantile，未经真值验证不得称 day/night |
| `image_quality` | `contrast` = 四路 camera luminance IQR 的中位数；`sharpness` = 四路 camera Laplacian(Y) variance 的中位数。`contrast <= P10 → low_contrast=true`；`sharpness <= P10 → blur_like=true` | train `P10` | `blur_like` 只是图像统计标签，不等价于运动模糊、天气或 occlusion |
| `paired_feature_gap` | alignment 训练前用同一 frozen DINO backbone：四路 paired real/raster 相同预处理 → 去 special/padding token → 对应 camera/token 算 `1 - cosine_similarity` → 对有效 token 和四路 camera 求平均 → `low/medium/high` | train `P33/P67` | 衡量 initialization 下的 representation gap，不得解释为天气或 causal difficulty |

navtest/navhard 若需 feature-gap 标签，可只为 label generation 生成 evaluation-only paired raster，其 root、manifest、cache 必须与训练数据物理隔离，不得进入训练、validation 或 checkpoint selection。**冻结要求**：所有 threshold 只从 official train split 计算一次（val、navtest、navhard 复用同一数值阈值）；label generator 不读取 checkpoint prediction、PDMS、EPDMS 或 sub-score；同一 scene 在所有 condition 和 seed 下标签完全相同；v2 follow-up rollout 继承 root scene 标签；从 v1/v2 分层抽取**至少 200 条 scene** 人工核验 metadata mapping、intersection、traffic light、illumination、image-quality；规则修订必须发生在正式模型评分之前。manifest 至少包含：

```text
benchmark, split, log_name, scene_token, root_scene_id, route_command,
intersection_context, traffic_light_context, actor_count_30m, actor_density,
vru_present, map_location, luminance_score, illumination, contrast_score,
low_contrast, sharpness_score, blur_like, paired_feature_gap_score,
paired_feature_gap, label_algorithm_version, threshold_manifest_sha256
```

## 6. 统计分析

本节集中全部统计规则，其他章节不重复。

- **配对与重采样**：每 seed 内按相同 scene token（v1）或 `root_scene_id`（v2）对 NoAlign/FullAlign 做 paired comparison；**以 log 为 cluster bootstrap 单位**，报告 95% cluster-bootstrap CI。
- **离散指标**：Align 低于 NoAlign 记 regression，高于记 rescue；报告**完整 transition table**，不得只报 net count。
- **连续指标 / aggregate sub-score / PDMS / EPDMS**：报告 paired delta `FullAlign − NoAlign` 的均值、中位数、分位数、effect size、95% CI；**不把任意微小负数二值化成 regression**；若要设 practical regression threshold，必须在查看结果前依据 metric resolution 或独立 null run 冻结。
- **两项 co-primary hypothesis 使用 Holm correction。**
- **分层切片（Stage-B only）**：每个标签值报告 `n, condition mean, paired delta mean/median, 95% cluster-bootstrap CI, regression count, rescue count, transition table`。最低样本量 **v1 slice n ≥ 200 scenes、v2 slice n ≥ 100 root scenes**；低于阈值仍可列描述性结果，但不得作显著性或机制结论。
- **多重比较**：family 分开定义——① v1 几何/语义标签 × v1 sub-scores；② v1 真实外观标签 × v1 sub-scores；③ v2 几何/语义标签 × v2 sub-scores；④ v2 真实外观标签 × v2 sub-scores。每个 family 内使用 **Benjamini-Hochberg FDR**。**禁止根据结果临时添加、合并或删除标签值。**

## 7. 验收与复现

**数据测试**：54 个 train/val log 与 official split 完全一致；10 个 raster test/navtest log 不出现在任何训练 artifact；train/val/test 的 token、log、原始 drive 交集均为空；metadata frame 与 F0/L0/R0/B0 相对路径一一对应；B0 数量与冻结四相机 root 的 metadata frame 数一致；所有 F0 reproduction MAE 在 0–255 空间 ≤ 1.0；缺失/损坏/全黑/NaN/Inf/shape/camera order 检查；source target 与重建 target 逐字段检查；raw raster 到 cached tensor 的有效像素一致性检查。**Stage-A 追加**：token 集合是 Stage-B 的真子集；每个 official train log 均有 token 被抽中且四个 `map_location` 全部出现；per-log 配额满足 `Σquota = ⌊N/3⌋`、`min(quota) ≥ 1`、`quota_l ≤ n_l`，且 per-log 采样率 `quota_l/n_l` 的 max/min ≤ 1.5；top-up 替换记录完整、优先同 log 补抽、跨 log 回退已标记，且单 split 替换率 ≤ 1%；最终样本数精确等于 `⌊N/3⌋`；打乱输入顺序后重跑抽样结果完全一致；`distribution_audit.json` 的全部 TVD 与采样率指标在冻结阈值内（超阈值时停止，不得重抽）。

**DataLoader coverage**：`limit_train_batches=1.0`、`limit_val_batches=1.0`；每个 train token 每 epoch 恰好出现一次；每次 validation 覆盖全部 val token；同一 seed 下两条件的 batch token 顺序逐 step 相同；多 GPU sampler 不重复或遗漏 token；记录每 epoch 的 token sequence checksum。

**Loss 与集成**：spatial/global 四种开关组合；spatial loss 只更新 real branch；raster branch detach；global domain batch real/raster 数量平衡；GRL 梯度方向与 schedule 起点/中点/终点；**NoAlign alignment gradient 严格为零**；两条件首个 optimizer step 前参数完全一致；**FullAlign 只比 NoAlign 多 alignment gradient**；validation/predict 只读 real input；fixed final-step checkpoint 输出格式一致。

**标签测试**（Stage-B 前置）：command one-hot 映射；intersection inside/approach/non-intersection 边界；traffic-light 三值；30m actor 边界与 VRU 类型；luminance/contrast/sharpness 数值回归；paired feature gap 对相同输入接近零；threshold manifest 只引用 train split；label manifest 不含 prediction 或 score 字段；v2 follow-up 正确继承 root label。

**Evaluator 回归**：官方 human/constant-velocity baseline 能复现对应版本参考结果；固定轨迹重复运行结果确定；v1 不输出 v2-only 指标；v2 Stage 1、Stage 2 rollout、Gaussian weight、root aggregation 与最终 EPDMS 全部验证；同一 prediction 在两条件的输出 schema 完全一致。

**门槛**：Stage-A 启动前需通过数据（含追加项）、DataLoader、loss、evaluator 测试；标签测试为 Stage-B 前置，Stage-A 不要求。全部通过后三 seed 实验才允许启动。

**必须保存的 artifacts**：Git commit（dirty worktree 时保存完整 patch 与 untracked source manifest）；四相机 raster root 版本及完整 manifest checksum；official split checksum；重新生成的 target 与 paired cache checksum；initialization checkpoint 来源/训练历史/SHA256；resolved Hydra config；Python / PyTorch / CUDA / driver 及关键依赖版本；seed、sampler 顺序、epoch、optimizer step；每个 loss 分量、实际 weight、GRL coefficient；scene label algorithm、threshold manifest、人工审计结果；NAVSIM v1/v2 evaluator commit、数据版本、metric cache checksum；逐 scene/rollout CSV、root aggregate CSV、summary；所有测试结果。目录：

```text
outputs/alignment_stage_a/{input_data/{stage_a_token_manifest.json, dataset_manifest.json,
    b0_generation_manifest.jsonl, topup_replacements.jsonl, input_audit.json,
    distribution_audit_thresholds.json, distribution_audit.json},
    init_checkpoints/{seed_0,seed_1,seed_2}.ckpt, paired_cache/,
    training/{r100-noalign,r100-fullalign}/, evaluation_v1/, evaluation_v2/, evaluator_validation/}

outputs/alignment_full/{input_data/{dataset_manifest.json, b0_generation_manifest.jsonl,
    input_audit.json, threshold_manifest.json}, init_checkpoints/, paired_cache/,
    scene_labels/{navtest_scene_labels.parquet, navhard_root_scene_labels.parquet},
    training/{r100-noalign,r100-fullalign}/, evaluation_v1/, evaluation_v2/, evaluator_validation/}
```

Stage-A 无 `scene_labels/`。两阶段的 `b0_generation_manifest.jsonl` 指向同一版本化 raster root，Stage-B 的 manifest 是 Stage-A 的超集。

## 8. 执行顺序

**阶段 0 — 共用数据准备**
1. 冻结 source raster、NAVSIM metadata、official split checksum。
2. 枚举 54 个 official train/val log 的全部 metadata frame，审计 F0/L0/R0。
3. 在 `rendered_sensor_blobs_4cam_v1` 中为 54 个 log 的全部 metadata frame 建立 F0/L0/R0 的 SHA256 核验 hard link。
4. 重建 54 个 log 的 target cache（`cache/rap_ego` 已判废，不得作 token whitelist 或 target 来源）。
5. 以 §3 资格规则生成候选池（B0 相关条件暂缓判定）。

**阶段 A — 先导实验**
6. 抽取 `⌊N/3⌋` train/val token，冻结 `stage_a_token_manifest.json`。
7. 仅为选中 token 生成 B0，逐 frame 验证 F0 reproduction，按 top-up 规则补抽。
8. 生成 Stage-A 的 paired cache、`dataset_manifest.json`、`input_audit.json`。
9. 运行泄漏检查、DataLoader coverage test、Stage-A 追加项。
10. 完成 alignment loss、GRL、paired-condition 集成测试。
11. 为每 seed 生成并冻结 initialization checkpoint，记 SHA256。
12. 按 20 epochs、global batch 128 运行 3 个 paired seeds 的 NoAlign/FullAlign，共 6 run。
13. 用 final-step checkpoint 分别运行官方 v1.1 与 v2.2 evaluator（完整 navtest 与 navhard_two_stage）。
14. 按 §5、§6 生成 co-primary 与 sub-score 报告，**严格按 §4 预注册规则**撰写结论；不做分层切片。

**阶段 B — 全量正式实验**
15. 补齐 54 个 log 剩余全部 metadata frame 的 B0，逐 frame 验证 F0 reproduction。
16. 校验整个 raster root 的 `renderer_git_commit` 与 `map_version` 唯一；不通过不得继续。
17. 以全量资格规则生成 target、paired cache、冻结 manifest，并校验 Stage-A token 为其真子集。
18. 重跑全部数据、DataLoader、loss 测试。
19. 用 official train split 冻结 actor/illumination/image-quality/feature-gap threshold。
20. 在模型评分前生成并人工审计 NAVSIM v1/v2 scene labels。
21. 生成 Stage-B 的 per-seed initialization checkpoint（不复用 Stage-A 的）。
22. 按 20 epochs、global batch 128 运行 3 个 paired seeds 的 NoAlign/FullAlign。
23. 用 final-step checkpoint 分别运行官方 v1.1 与 v2.2 evaluator。
24. 按 co-primary、sub-score、标签 slice 与多重比较协议生成最终报告。

**待补齐的工程缺口**（执行前置条件）

| 缺口 | 步骤 | 说明 |
|---|---|---|
| `alignment_stage_a.yaml` | 12 | 参照 `alignment_small.yaml`，去掉 `max_train_samples` / `limit_train_batches` 截断，改用 token manifest |
| F0/L0/R0 hard link 脚本 | 3 | 当前无此步骤；`build_alignment_small_data.py` 的 `_render_missing_back_camera` 把 B0 写回 `--raster-root` 本身，故该 root 必须先自洽 |
| top-up 补抽逻辑 | 7 | 当前 F0 校验失败时只记录 failure，会静默少样本 |
| 全量 target 重建 | 4 | 需用 `run_dataset_caching.py` 覆盖 54 个 log |
| per-seed init checkpoint 生成 | 11, 21 | 当前无脚本，两条件需加载同一文件 |
| checkpoint 回调调整 | 12, 22 | 现有 `ModelCheckpoint(save_last=True, save_top_k=3, monitor='val/score')` 引入基于 val 的隐式选择，与 final-step 协议冲突 |
