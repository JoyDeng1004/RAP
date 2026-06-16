# Codex 执行 Prompt：SCA-Only Recovery Observation Fine-Tune 实验

> 本文件是给 Codex 的分步执行说明。请**严格按 Step 顺序**执行，每个 Step 都有「实现内容 + 必写测试 + 断言 + 完成判据」。
> 未通过当前 Step 的断言前，不要进入下一个 Step。

## 角色与约束
- 项目根目录：`/Users/joy/pythonProject/RAP`，当前分支 `sync`。
- 目标：在不修改 `image_feature` / `lidar2img`、不做物理一致扰动的前提下，用 **attention-level perturbation**（只偏移 SCA 的 `ref_2d`）+ **pure-pursuit recovery 监督目标**，做小样本（4–16）fine-tune，并与原始 GT fine-tune 对比。
- 复用优先：能复用现有工具/类就不要新写（见下方"可复用资产"）。
- 每步先写/跑测试，断言不过就停下来修，不要继续往下。

## 已核实事实（无需重新探索）
- 轨迹输出与 `targets["trajectory"]` 均为自车后轴局部系 `[x前进, y左, heading]`，T=8，dt=0.5。
- `encoder.py` 两条 ref_2d 路径：静态网格（75–78 行，归一化[0,1]）、proposal 路径（266–295 行，米制[-32,32]）。proposal 路径里 **TSA 的 `ref_pos→hybird_ref_2d`（267–269 行）与 SCA 的 `compute_corners/ref_3d`（285–289 行）共用同一 `ref_2d`**。`ref_pos=(ref_2d[...,:2]+32)/64`。
- `point_sampling`（`encoder.py:155`）用 `features['lidar2img']`，与 shift 无关。
- 调用链：`rap_model.forward`（129–130 行 refine 循环）→ `traj_refiner.forward`（调 `Bev_refiner(proposals, bev_feature, image_feature)`，约 32 行）→ `bev_refiner.forward(pose, prev_bev, image_feature)`（约 106–118 行，`ref_2d=pose.detach()`）→ `encoder.forward`。**注意：`bev_refiner.forward` 当前不接收 `features`。**
- `pdm_scorer` 标志只控制 loss（`rap_agent.py:459,479`）；scorer 在 forward 始终运行（`rap_model.py:137`），推理永远 `argmax(pdm_score)` 选轨迹（156–168 行）。`return_score=True` 时输出全部 `proposals` + `score`。
- 轨迹 loss 用 `.amin(1)`（`rap_agent.py:~521`，对 64 条 proposal 取离目标最近的一条）；`weight[~score_mask]=0.1`（:524）。`score_mask=targets['score_mask'] & metric_cache_mask`（:453,477）。
- 组合 loss（`rap_agent.py:~551`）权重：`trajectory_weight=1, sub_score_weight=0, final_score_weight=1, pred_ce_weight=1, pred_l1_weight=0.1, pred_area_weight=2, agent_class_weight=1, agent_box_weight=0.1, bev_semantic_weight=1`。其中 `pred_ce/pred_l1/pred_area` 受 `if pdm_scorer` 门控，`agent_class/agent_box/bev_semantic` **不受门控**。
- `proposal_num=64, num_poses=8` → ref_2d 形状 `[B,640,3]`。`rear_axle_to_center=1.461`。

## 可复用资产（不要重写）
- `navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py` 的 `BatchKinematicBicycleModel`（`max_steering_angle=np.pi/3`，用 `get_pacifica_parameters().wheel_base`）。
- `get_pacifica_parameters` 来自 `nuplan.common.actor_state.vehicle_parameters`。
- `navsim/planning/script/tools/validate_recovery_trajectory.py`：`_metrics`、`compute_recovery_metrics`（含 `lateral_error_reduction_ratio`、`time_to_recover`）、`_constant_velocity_baseline`（CV/不动 baseline，`beats_cv`）。
- `tools/viz_ref2d.py`：从 npz dump 渲染 `ref_2d/ref_pos/corners/reference_points_cam/bev_mask` 投影链。

---

# Step 0 — 配置开关（无行为变化）
**实现**：`navsim/agents/rap_dino/navsim_config.py` 增加字段（默认值保证旧行为不变）：
`ref2d_observation_aug: bool=False`、`ref2d_aug_scope: str="sca"`、`ref2d_aug_y_range: Tuple[float,float]=(-1.0,1.0)`、`ref2d_aug_prob: float=1.0`、`recovery_target_enabled: bool=False`。
**测试/断言**：导入 `RAPConfig()` 成功；全部默认值符合上述；旧的单 batch forward 行为不变（与改前一致）。
**完成判据**：默认配置下 forward 输出与改动前逐元素一致。

# Step 1 — recovery_target 工具 + 单测（先写测试）
**实现**：新建 `navsim/agents/rap_dino/recovery/recovery_target.py`，函数签名约：
`make_recovery_trajectory(trajectory: Tensor[B,T,3], shift_y: Tensor[B] | float, dt: float=0.5, lookahead_m=..., wheel_base=get_pacifica_parameters().wheel_base, max_steer=np.pi/3) -> Tensor[B,T,3]`。
- 复用 `BatchKinematicBicycleModel`，**不自己重写车辆动力学**。
- `v0` 由原始 `trajectory` 的有限差分推出（不要新增入参要求外部传 v0）。
- lookahead **速度自适应**：`lookahead = clip(k*v0 + min_lookahead, lo, hi)`，把 `k/min_lookahead/lo/hi` 设为带默认值的参数；默认下当 v0≈典型城市速度时 lookahead≈4m。
- 输出在**原始 ego-local 系**；`shift_y=0` 原样返回输入。
- heading wrap 到 `[-pi,pi]`。

**必写测试** `tests/recovery/test_recovery_target.py`，断言：
1. `make_recovery_trajectory(traj, shift_y=0)` 与 `traj` 逐元素近似相等（`atol=1e-5`）。
2. `shift_y=+1.0`：`|out[:,t,1] - traj[:,t,1]|` 随 t **单调不增**（横向误差收敛），且 `out[:,-1,1]` 比 `out[:,0,1]` 更接近 `traj[:,-1,1]`。
3. 输出 `shape==traj.shape`、`dtype==traj.dtype`；`out[...,2]` 全在 `[-pi,pi]`。
4. `v0` 推导：对一条已知匀速直线 traj，反推 `v0` 误差 `<5%`。
5. lookahead 自适应：低速样本与高速样本得到的 lookahead 不同，且都在 `[lo,hi]` 内。

**完成判据**：5 条断言全过。

# Step 2 — SCA-only shift + plumbing（守 R5）
**实现**：
- `encoder.py` proposal 路径：在 285 行 `compute_corners` 之前 `ref_2d_sca = ref_2d.clone()`；若 `features` 含 `ref2d_aug_shift_y`，执行 `ref_2d_sca[...,1] += shift_y[:,None]`（米制）；`compute_corners/ref_3d/reference_points_cam/bev_mask` 改用 `ref_2d_sca`。**267 行的 `ref_pos→hybird_ref_2d`（TSA）继续用原始 `ref_2d`**。静态网格路径（75–78）不动；`lidar2img` 不动。
- 透传 `ref2d_aug_shift_y`：让 `bev_refiner.forward` 接收 `features`（或至少该张量），并沿 `rap_model → traj_refiner → bev_refiner → encoder` 贯通。

**必写测试** `tests/bevformer/test_sca_only_shift.py`（构造一个最小 features，含非零 `shift_y`），断言：
1. **TSA 不变**：开启 shift 后 `hybird_ref_2d`（或其来源 `ref_pos`）与 `shift_y=0` 时逐元素相等。
2. **SCA 改变**：`reference_points_cam` 与 `shift_y=0` 时存在显著差异（`max abs diff > 1e-4`）。
3. **bev_mask 改变**（或至少可改变）：在合理 shift 下 mask 的 True 数量发生变化。
4. **lidar2img 不变**：`features['lidar2img']` 调用前后 id/值不变。
5. **R5 plumbing 硬断言**：把 `shift_y` 写进 `features`，跑完整 `rap_model.forward`，断言内部 `ref_3d`（用 hook 或返回中间量）相对 `shift_y=0` **确实变化**；若无变化测试必须失败。

**完成判据**：5 条全过（尤其第 5 条，防止静默空操作）。

# Step 3 — 符号一致性验证（守 R1，训练前必过）
**实现/脚本**：复用 `tools/viz_ref2d.py` 的 dump 逻辑，对同一样本分别用 `shift_y=0` 与 `shift_y=+1.0` 生成 npz，并写一个断言脚本/测试 `tests/bevformer/test_shift_sign.py`：
1. 计算前视相机投影点 `reference_points_cam` 在 `shift_y=+1.0` 相对 `0` 的**位移方向**（u 方向符号），记录为 `obs_dir`。
2. 计算 `make_recovery_trajectory(traj, shift_y=+1.0)` 的第一步横向符号 `sign(out[0,1]-traj[0,1])`，记为 `tgt_dir`。
3. **断言两者满足约定关系**（同号或预期固定关系，并在注释里写死该约定：`shift_y>0` 表示自车在原 ego frame 左侧）。
**完成判据**：方向关系断言通过；若不通过，说明 shift 与 recovery 目标方向相反——**必须先修正符号约定再继续**。

# Step 4 — 训练接入 + loss 配置（守 R2、R3）
**实现**：`navsim/planning/training/agent_lightning_module.py` 增加 recovery 分支：
- 采样 `shift_y ~ Uniform(ref2d_aug_y_range)`，写入 `features['ref2d_aug_shift_y']`。
- 用 `make_recovery_trajectory` 把 `targets['trajectory']` 替换成 recovery target（仅在 `recovery_target_enabled` 时）。
- **R2**：给 recovery 样本显式 `targets['score_mask']=True`（确保轨迹 loss 满权重，不被 ×0.1）。
- **R3**：在 recovery 类配置里把**所有非轨迹权重置 0**：`sub_score_weight=final_score_weight=pred_ce_weight=pred_l1_weight=pred_area_weight=agent_class_weight=agent_box_weight=bev_semantic_weight=0`，`pdm_scorer=False`，只留 `trajectory_weight=1`。
- 关掉 pdm_scorer 后确认不再走 metric cache 加载分支（`rap_agent.py:459-477`）。

**必写测试** `tests/training/test_recovery_loss.py`，断言：
1. recovery 配置下单 batch forward+`compute_loss` 不抛异常（不会因缺 agent/bev GT 崩）。
2. `loss ≈ trajectory_loss`（其余 loss 项为 0 或不参与）。
3. recovery 样本的轨迹 loss **未被 ×0.1**（对比人为设 `score_mask=False` 时数值应差约 10 倍）。
4. 同一样本下 Baseline target 与 Offset-Recovery target 不同。
**完成判据**：4 条全过。

# Step 5 — 实验矩阵（2×2）+ 统一协议
**配置 4 个 cell**（同 checkpoint / 同 token 列表 / 同 seed / 同步数；建议 8 train + 4 val）：
- `Baseline` = (shift 关, GT 目标)：保留**原始**权重与 pdm_scorer（即原始 fine-tune），不改 score_mask 逻辑。
- `Recovery-only` = (shift 关, recovery 目标)：Step4 的 recovery loss 配置。
- `Shift-only` = (shift 开, GT 目标)：开 shift，但目标用原始 GT（不变性/鲁棒性诊断）。
- `Offset-Recovery` = (shift 开, recovery 目标)：主方案。
- 训练 shift：`Uniform(-1,1)`；验证固定 grid `[-1.0,-0.5,0.5,1.0]`，recovery target 用对应 shift_y 生成。
**完成判据**：4 个 cell 各能跑通单 batch forward/backward。

# Step 6 — 评估与指标（守 R4、R6）
- **R6 统一协议**（不可比问题）：
  - (a) 标称评估：所有 cell 都用 `shift_y=0` 对**原始 GT** 算 ADE/FDE/AHE/FHE（看是否损害标称驾驶）。
  - (b) recovery 评估：给所有 cell **喂相同的偏移观测 + 相同 recovery target**，算 recovery_ADE/FDE 及 `lateral_error_reduction_ratio`（复用 `compute_recovery_metrics`）。
  - 始终附 **CV/不动 baseline**（`_constant_velocity_baseline` + `beats_cv`）与**未训练 checkpoint** 作参照。
- **R4 双轨指标**：每个 cell 同时报告
  - `selected ADE/FDE`：`prediction["trajectory"]`（scorer 选的，部署行为）；
  - `best-proposal ADE/FDE`：`forward(return_score=True)` 取全部 `proposals` 对目标取 min（训练优化上界）。
  - 报告两者差值，量化 scorer 陈旧造成的选择损失。
- 复用 `validate_recovery_trajectory.py` 的 `_metrics`，不要重造指标。

# Step 7 — 可视化与归因
- 用 `tools/viz_ref2d.py` 导出每个 cell 至少 1 个 shift 样本的投影链调试图。
- 叠加可视化：original GT / shifted initial pose / pure-pursuit recovery target / prediction。
- 归因表：对比 4 个 cell 的 (a)(b) 指标 + 双轨 ADE，判断收益来自 shift、recovery 监督，还是二者协同。

---

## 风险 → Step 对照
- R1 符号一致性 → Step 3（训练前必过）
- R2 score_mask 全 False ×0.1 → Step 4 断言 3
- R3 非轨迹 loss 崩/虚假梯度 → Step 4 断言 1–2
- R4 陈旧 scorer 选择偏差 → Step 6 双轨指标
- R5 shift_y 未到 encoder（静默空操作）→ Step 2 断言 5

## 最终验证清单
1. `pytest tests/recovery tests/bevformer tests/training` 全绿。
2. Step 3 符号断言通过。
3. 4 个 cell 各跑通单 batch forward/backward（CPU 即可）。
4. `validate_recovery_trajectory.py` 在少量样本上跑通，确认指标复用与 CV 对比。
5. 每个 cell 产出 (a) 标称指标、(b) recovery 指标、双轨 ADE，以及至少 1 张投影链调试图。