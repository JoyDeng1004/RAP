# RAP 官方 Baseline 训练超参数

## 口径与数据来源

- **唯一数据源**：`vita-epfl/RAP` 官方 GitHub 仓库；未读取或参考本地工作区中的 RAP 源码。
- **版本快照**：官方 `main`，commit [`5fd8630ae54442dd41827de4a9afe1690e3f02bb`](https://github.com/vita-epfl/RAP/commit/5fd8630ae54442dd41827de4a9afe1690e3f02bb)（2025-12-04，`#10 fix checkpoint bug`）。固定 commit 链接可避免 `main` 后续变化导致数值漂移。
- **Baseline 定义**：官方 README 中“train navsim model”的 RAP-DINO / NAVSIM 训练命令。本文不把 Waymo finetuning 的 `20 epochs / batch size 16 / lr 1e-5` 混入 NAVSIM Baseline。
- **取值优先级**：README 命令行覆盖项 > `rap_agent.yaml` > `RAPConfig` dataclass > `default_training.yaml` / 源码硬编码。
- 表中的“官方默认值”指按官方 README 命令启动后得到的**有效值**；若它覆盖了较底层声明值，会同时注明两者。

## 1. Baseline 入口与数据设置

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| Agent | `rap_agent`（`RAPAgent`） | [`README.md` L141-L143](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L141-L143); [`rap_agent.yaml` L1-L6](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L1-L6) | RAP-DINO 训练 Agent。 |
| Dataset | `navsim_dataset` | [`README.md` L153-L155](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L153-L155) | NAVSIM 数据集构建器。 |
| Train/test split config | `navtrain` | [`README.md` L145-L147](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L145-L147) | 使用官方 `navtrain` Hydra 配置组。 |
| Data split | `trainval` | [`README.md` L146-L148](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L146-L148) | Baseline 明确指定训练/验证数据分区。 |
| Ego cache | `./cache/rap_ego` | [`README.md` L147-L150](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L147-L150) | 原始 ego-view 缓存。 |
| Recovery-perturbed cache | `./cache/rap_perturbed` | [`README.md` L148-L150](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L148-L150) | recovery-oriented perturbation 缓存。 |
| Cross-agent cache | `./cache/rap_aug` | [`README.md` L149-L151](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L149-L151) | cross-agent view 缓存。 |
| `use_cache_without_dataset` | `True` | [`README.md` L150-L152](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L150-L152) | 直接从缓存加载；此模式下 scene filter 被忽略。 |
| `force_cache_computation` | `False` | [`README.md` L151-L153](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L151-L153); [`run_training.py` L104-L111](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L104-L111) | 缓存训练模式要求关闭强制重算。 |
| Perturbed-cache sampling ratio | `10%` | [`run_training.py` L149-L156](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L149-L156) | 从完整 perturbed cache 中随机抽取 `int(0.1 * N)`。这是独立抽样比例，不是最终 concat 数据集中的归一化占比。 |
| Cross-agent-cache sampling ratio | `5%` | [`run_training.py` L158-L167](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L158-L167) | 从完整 cross-agent cache 中随机抽取 `int(0.05 * N)`。 |
| Training-set composition | `ego + 10% perturbed + 5% cross-agent` | [`run_training.py` L149-L169](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L149-L169) | 三个数据子集以 `ConcatDataset` 拼接。 |
| Cross-agent score mask | `False` | [`run_training.py` L158-L165](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L158-L165) | cross-agent 样本不参与需要有效 PDM score 的分支。 |
| Training shuffle | `True` | [`run_training.py` L175-L178](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L175-L178) | 训练 DataLoader 打乱数据。 |
| Random seed | `0` | [`default_training.yaml` L23-L26](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L23-L26); [`run_training.py` L91-L92](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py#L91-L92) | `pl.seed_everything(seed, workers=True)`，包括 worker 随机性。 |

## 2. 优化器、学习率与训练器

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| Optimizer | `AdamW` | [`rap_agent.py` L566-L579](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L566-L579) | 官方代码只显式传入参数组、学习率和 weight decay；`betas`、`eps` 等沿用所装 PyTorch 的默认值。 |
| Base learning rate | `1e-4` | [`rap_agent.yaml` L20-L21](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L20-L21) | README 的 NAVSIM 命令未覆盖该值。 |
| Weight decay | `1e-4` | [`rap_agent.py` L573-L577](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L573-L577) | 应用于非 DINO-backbone 的可训练参数组。 |
| Backbone learning rate | 不适用；backbone 冻结 | [`rap_agent.py` L566-L576](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L566-L576) | 代码计算了 `vit_lr = 0.2 * lr`，但对应参数组被注释，并将 backbone 的 `requires_grad=False`；因此它不是有效 Baseline LR。 |
| LR scheduler | `WarmupCosLR` | [`rap_agent.py` L579-L580](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L579-L580) | 线性 warm-up 后进行 cosine decay。 |
| Scheduler minimum LR | `1e-5` | [`rap_agent.py` L579-L580](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L579-L580) | 余弦调度下限。 |
| Scheduler epochs | `20` | [`rap_agent.py` L579-L580](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L579-L580) | 调度器周期硬编码为 20 epochs。 |
| Warm-up epochs | `1` | [`rap_agent.py` L579-L580](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L579-L580) | 线性 warm-up 1 epoch。 |
| Max training epochs | `100` | [`default_training.yaml` L36-L40](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L36-L40) | README 的 NAVSIM 命令未覆盖此值。注意它与源码写死的 20-epoch scheduler 周期不一致，见“复现注意事项”。 |
| Batch size | `64` | [`README.md` L151-L155](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L151-L155) | 每个 DataLoader / 进程的 batch size；官方命令与 YAML 默认值一致。分布式全局 batch size 还取决于实际 GPU/进程数。 |
| Gradient accumulation | `1` | [`default_training.yaml` L50-L56](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L50-L56) | 每个 batch 更新一次。 |
| Gradient clipping value | `0.0` | [`default_training.yaml` L53-L56](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L53-L56) | 等价于未启用有效裁剪。 |
| Gradient clipping algorithm | `norm` | [`default_training.yaml` L55-L56](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L55-L56) | 梯度裁剪算法；因阈值为 0.0，Baseline 中无实际裁剪效果。 |
| Precision | `16-mixed` | [`default_training.yaml` L45-L48](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L45-L48) | Lightning 混合精度训练。 |
| Accelerator | `gpu` | [`default_training.yaml` L45-L48](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L45-L48) | GPU 训练。 |
| Distributed strategy | `ddp_find_unused_parameters_true` | [`default_training.yaml` L45-L48](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L45-L48) | DDP，并允许 unused parameters。 |
| Number of nodes | `1` | [`default_training.yaml` L45-L49](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L45-L49) | 单节点；README 的 Ray 辅助评分示例另使用 8 GPU，但 Trainer 的设备数未在该命令中显式固定。 |
| Train batch limit | `1.0` | [`default_training.yaml` L42-L43](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L42-L43) | 每 epoch 使用完整训练集。 |
| Validation batch limit | `1.0` | [`default_training.yaml` L42-L43](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L42-L43) | 使用完整验证集。 |
| Validation cadence | 每 `1` epoch；`val_check_interval=1.0` | [`default_training.yaml` L38-L40](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L38-L40) | 每个 epoch 末验证一次。 |
| Sanity validation steps | `2` | [`default_training.yaml` L50-L51](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L50-L51) | 正式训练前运行两个验证 step。 |
| Fast dev run | `False` | [`default_training.yaml` L50-L51](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L50-L51) | 不启用单 batch 调试模式。 |

## 3. DataLoader 与轨迹采样

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| `num_workers` | `4` | [`default_training.yaml` L28-L34](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L28-L34) | 每个 DataLoader 的数据加载 worker 数。 |
| `pin_memory` | `True` | [`default_training.yaml` L28-L34](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L28-L34) | 使用 pinned host memory。 |
| `prefetch_factor` | `2` | [`default_training.yaml` L28-L34](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L28-L34) | 每个 worker 预取 2 个 batch（YAML 注释写“samples”，但传给 PyTorch DataLoader 的参数语义是 batch）。 |
| `drop_last` | `False` | [`default_training.yaml` L28-L34](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml#L28-L34) | 保留最后一个不完整 batch。 |
| Trajectory time horizon | `5 s`（README 覆盖；agent YAML 原值 `4 s`） | [`README.md` L153-L155](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L153-L155); [`rap_agent.yaml` L8-L12](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L8-L12) | 官方 Baseline 命令的有效预测时域。 |
| Trajectory interval | `0.5 s` | [`rap_agent.yaml` L8-L12](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L8-L12) | 轨迹采样时间间隔。 |
| Number of trajectory proposals | `64` | [`navsim_config.py` L27-L33](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L27-L33) | 每个样本生成/评分的候选轨迹数。 |
| Refinement stages (`ref_num`) | `4` | [`navsim_config.py` L12-L20](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L12-L20); [`rap_model.py` L98-L103](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_model.py#L98-L103) | 共享 trajectory refiner 重复应用 4 次。 |
| Command classes | `4` | [`navsim_config.py` L41-L43](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L41-L43) | 路由/驾驶命令类别数。 |

## 4. RAP 模型、增强与开关

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| Image backbone | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | [`image_encoder.py` L25-L31](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py#L25-L31) | Hugging Face 预训练 DINOv3 ViT-H/16+。 |
| Backbone trainability | Frozen | [`rap_agent.py` L566-L576](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L566-L576) | DINO backbone 不参与反向更新。 |
| Number of cameras | `4` | [`image_encoder.py` L13-L21](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py#L13-L21) | 图像编码器使用 4 路相机。 |
| Camera input size | `1024 × 256` | [`navsim_config.py` L87-L95](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L87-L95) | 宽 × 高。 |
| Transformer model dimension | `1280` | [`navsim_config.py` L44-L50](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L44-L50) | `tf_d_model`。 |
| Transformer FFN dimension | `1024` | [`navsim_config.py` L44-L50](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L44-L50) | `tf_d_ffn`。 |
| Transformer layers | `3` | [`navsim_config.py` L44-L50](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L44-L50) | `tf_num_layers`。 |
| Transformer attention heads | `8` | [`navsim_config.py` L44-L50](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L44-L50) | `tf_num_head`。 |
| Transformer dropout | `0` | [`navsim_config.py` L44-L50](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L44-L50) | RAP transformer/refiner 中不使用 dropout。 |
| BEV layers | `2` | [`navsim_config.py` L49-L51](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L49-L51) | `num_bev_layers`。 |
| PDM scorer | `True` | [`rap_agent.yaml` L14-L18](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L14-L18); [`README.md` L142-L145](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L142-L145) | Baseline 使用 PDM scorer 监督/评分。 |
| Raster-to-Real feature distillation | `True` | [`rap_agent.yaml` L14-L18](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L14-L18); [`README.md` L142-L145](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L142-L145) | 进入 `_step_distill` 训练路径。dataclass 自身写 `False`，但 agent YAML 与 README 都覆盖为 `True`。 |
| `cache_data` | `False` | [`rap_agent.yaml` L14-L18](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L14-L18) | 构建完整模型而不是只做特征缓存。 |
| `latent` | `False` | [`rap_agent.yaml` L14-L16](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml#L14-L16) | 不启用 latent 模式。 |
| `traj_bev` | `True` | [`navsim_config.py` L17-L21](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L17-L21) | trajectory refiner 使用 BEV。 |
| `traj_proposal_query` | `True` | [`navsim_config.py` L17-L21](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L17-L21) | 使用 trajectory proposal query。 |
| `agent_pred` | `True` | [`navsim_config.py` L22-L26](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L22-L26) | 启用 agent-state prediction 辅助头。 |
| `area_pred` | `True` | [`navsim_config.py` L22-L26](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L22-L26) | 启用 area prediction 辅助头。 |
| `double_score` | `False` | [`navsim_config.py` L20-L24](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L20-L24) | 不启用第二套 score head。 |
| `bev_map` / `bev_agent` | `False` / `False` | [`navsim_config.py` L27-L31](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L27-L31) | scorer 内不额外启用 BEV map/agent 分支。 |
| GridMask enabled | `True` | [`image_encoder.py` L25-L29](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py#L25-L29); [`image_encoder.py` L76-L88](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py#L76-L88) | 只在训练模式中施加图像 GridMask。 |
| GridMask parameters | `rotate=1`, `offset=False`, `ratio=0.5`, `mode=1`, `prob=0.7` | [`image_encoder.py` L25-L29](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py#L25-L29) | 官方图像增强设置。 |

## 5. 损失函数权重

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| Trajectory loss weight | `1` | [`navsim_config.py` L53-L65](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L53-L65); [`rap_agent.py` L523-L533](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L523-L533) | 候选轨迹回归主损失权重。 |
| Inter-proposal diversity weight | `0` | [`navsim_config.py` L53-L57](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L53-L57); [`rap_agent.py` L490-L502](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L490-L502) | 默认关闭 diversity loss 对总损失的贡献。 |
| Previous-refinement accumulation weight | `0.1` | [`navsim_config.py` L57-L62](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L57-L62); [`rap_agent.py` L490-L502](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L490-L502) | 每个 refinement stage 以前一阶段累计损失的 0.1 倍递推。 |
| Non-score-mask trajectory sample weight | `0.1`（硬编码） | [`rap_agent.py` L490-L498](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L490-L498) | `score_mask=False` 样本的 min trajectory loss 权重；cross-agent 数据被设置为该类。 |
| Sub-score loss weight | `0` | [`navsim_config.py` L53-L58](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L53-L58) | 不计入总损失。 |
| Final-score loss weight | `1` | [`navsim_config.py` L56-L59](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L56-L59) | 最终 proposal score 监督。 |
| Prediction CE loss weight | `1` | [`navsim_config.py` L57-L60](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L57-L60) | agent prediction 分类损失。 |
| Prediction L1 loss weight | `0.1` | [`navsim_config.py` L58-L61](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L58-L61) | agent-state regression 损失。 |
| Prediction area loss weight | `2` | [`navsim_config.py` L59-L62](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L59-L62) | area prediction 损失。 |
| Agent-class loss weight | `1.0` | [`navsim_config.py` L61-L64](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L61-L64) | 辅助检测分类损失。 |
| Agent-box loss weight | `0.1` | [`navsim_config.py` L62-L65](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L62-L65) | 辅助检测框回归损失。 |
| BEV semantic loss weight | `1.0` | [`navsim_config.py` L63-L65](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L63-L65) | BEV 语义分割交叉熵权重。 |
| Feature distillation / Raster-to-Real MSE weight | `0.002` | [`navsim_config.py` L10-L16](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py#L10-L16); [`agent_lightning_module.py` L246-L250](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/training/agent_lightning_module.py#L246-L250) | 渲染 BEV（detach）与真实 BEV 间 MSE 的权重。 |
| Domain-adversarial loss weight | `0.001`（硬编码） | [`agent_lightning_module.py` L224-L245](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/training/agent_lightning_module.py#L224-L245) | 合成/真实域二分类 BCE；正类权重根据当前 batch 的合成/真实数量动态计算。 |
| Gradient-reversal schedule gamma | `10.0` | [`rap_model.py` L98-L107](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_model.py#L98-L107) | 域对抗 gradient reversal 的调度强度。 |
| Domain-adversarial progress horizon | `20 epochs`（硬编码） | [`agent_lightning_module.py` L135-L143](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/training/agent_lightning_module.py#L135-L143) | `progress=(current_epoch+1)/20`，与 scheduler 的 20-epoch 周期一致。 |

## 6. Checkpoint 默认行为

| 参数名称 (Parameter) | 官方默认值 (Official Default) | 所在文件/代码位置 (Source File) | 简要说明 (Description) |
|---|---:|---|---|
| Save last checkpoint | `True` | [`rap_agent.py` L594-L602](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L594-L602) | 始终保存最后一个 checkpoint。 |
| Save top-k | `3` | [`rap_agent.py` L594-L602](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L594-L602) | 额外保留验证指标最好的 3 个 checkpoint。 |
| Checkpoint monitor | `val/score`, mode=`max` | [`rap_agent.py` L594-L602](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L594-L602) | 按验证 score 最大化选优。 |
| Checkpoint filename | `{epoch}-{step}` | [`rap_agent.py` L594-L602](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py#L594-L602) | checkpoint 文件名模板。 |

## 复现注意事项

1. **100 vs 20 epochs**：官方 NAVSIM README 命令没有设置 `trainer.params.max_epochs`，因此按 Hydra 配置合成后的 Trainer 有效值是 **100**。但 `WarmupCosLR(..., epochs=20)` 和域对抗 `progress=(epoch+1)/20` 都在源码中硬编码为 **20**。本文忠实记录此不一致；不要仅凭 scheduler 参数把 Trainer epoch 数改写为 20。
2. **README 路径拼写**：README 写的是 `navsim/planing/script/run_training.py`，仓库实际目录是 `navsim/planning/script/run_training.py`。超参数解析按实际源码文件进行。
3. **Ray 不是 Trainer GPU 数**：README 在训练前给出 `ray start --num-gpus 8`，用于加速 PDM score 计算；这不等价于 Lightning Trainer 明确设置为 8 GPU。并且官方 `rap_agent.py` 当前仍是 `self.ray=False`，README 要求手工改为 `True`。因此本文没有把“8 GPUs”写成 Trainer 的确定默认设备数。
4. **未显式设置项**：AdamW 的 `betas`、`eps`、`amsgrad` 等没有在 RAP 源码中显式传入；为了不把依赖库默认值误写成 RAP 自身设定，表中不为它们填造数值。
5. **过期/非有效字段**：`RAPConfig` 仍含 `image_architecture="resnet34"` 等继承字段，但实际 RAP-DINO 图像骨干由 `AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")` 构造。本文以实际执行路径为准。

## 官方源码索引

- [官方仓库首页](https://github.com/vita-epfl/RAP/tree/5fd8630ae54442dd41827de4a9afe1690e3f02bb)
- [NAVSIM Baseline 启动命令](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/README.md#L139-L155)
- [训练默认配置](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/training/default_training.yaml)
- [RAP Agent Hydra 配置](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/config/common/agent/rap_agent.yaml)
- [RAPConfig 模型与损失参数](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/navsim_config.py)
- [RAP optimizer、loss 与 checkpoint](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/rap_agent.py)
- [训练数据混合逻辑](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_training.py)
- [Raster-to-Real / domain loss](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/training/agent_lightning_module.py)
- [DINOv3 image encoder 与 GridMask](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/agents/rap_dino/bevformer/image_encoder.py)
