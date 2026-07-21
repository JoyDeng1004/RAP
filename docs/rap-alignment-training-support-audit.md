# RAP alignment 训练支撑集审计

## 审计范围

- 仓库：`/gs/bs/tga-RLA/qdeng/RAP`
- commit：`918b60e54b709dfaac1328bcd582a024b2594f14`
- 发布 checkpoint：
  - `ckpts/RAP_DINO_navsimv1.ckpt`
  - `ckpts/RAP_DINO_navsimv2.ckpt`
- 审计方式：只读静态代码、config、README 和 checkpoint 元数据；未写实验代码，未运行训练。

本文严格区分：

- **当前源码行为**：由本仓库代码直接确定。
- **README 发布配方**：作者公开给出的训练命令，但不等于 checkpoint 实际命令。
- **checkpoint 可证事实**：从发布权重自身读取的 state、epoch、global step 等。
- **无法确定**：checkpoint 没有保存 Hydra config、cache manifest 或随机采样清单，无法反推。

## 当前结论

1. spatial alignment 对当前帧的 `B0/F0/L0/R0` 四个槽位整体算 MSE，不是只对前视。
2. global/domain alignment 只取相机索引 `1`，按实际相机顺序就是 `F0`。
3. RAP 只加载 history index `3`。前三个历史时刻没有图像 alignment，`L1/L2/R1/R2` 也从未进入 RAP 图像分支。
4. normal 和 perturbed renderer 默认只生成 `F0/L0/R0`。`B0` raster 缺失后被 loader 替换成全零图，但仍进入 spatial MSE。因此 `B0` 有 loss，却没有有效的 Raster-to-Real 配对。
5. real/raster 和所有相机共用同一个 DINOv3、同一个 FPN。四个相机槽位分别加 learned camera embedding；相机内外参不输入 projector，只供后面的 planner cross-attention 使用。
6. spatial MSE 在 raster 侧调用 `detach()`，所以当前 backward 中是 **real feature 被拉向 raster feature**。global loss 也切断 raster encoder 路径，只通过 GRL 更新 real 路径。
7. 两份发布 checkpoint 都没有 `domain_classifier.*` 权重，optimizer 也只保存 103 个参数状态。当前源码却注册了含 6 个参数张量的 `DomainClassifier`。因此，无法从发布 checkpoint 证明 global/GRL alignment 真参与了这两份权重的训练。
8. 两份 checkpoint 都没有 `hyper_parameters`。实际完整 config、cache 内容和随机抽中的 perturbed/cross-agent 样本无法恢复。

---

## A. alignment 的监督覆盖范围

### A1. alignment loss 在哪些相机上计算？

#### 结论

- spatial MSE：当前帧的 `B0/F0/L0/R0` 四个相机整体计算。
- global/domain BCE：只使用 `F0`。
- `L1/L2/R1/R2` 完全不参与。
- 如果任意一个所需 real camera 文件缺失，`camera_valid=False`，该样本所有相机的 real branch 都被排除，不做 alignment。

#### 证据

相机顺序写死为 `B0, F0, L0, R0`：

`navsim/agents/rap_dino/bevformer/bev_feature_build.py:27-29`

```python
for cameras in agent_input.cameras[-1:]:
    for cam in [cameras.cam_b0, cameras.cam_f0,
                cameras.cam_l0, cameras.cam_r0]:
```

四个相机的 feature 整体放进 `bev_feature`：

`navsim/agents/rap_dino/rap_model.py:145`

```python
output["bev_feature"] = image_feature[0].permute(2, 0, 1, 3)
```

spatial MSE 没有 camera slice：

`navsim/planning/training/agent_lightning_module.py:246-250`

```python
bev_feature = prediction['bev_feature']
render_bev = bev_feature[:batch_size][real_valid_mask].detach()
real_bev = bev_feature[batch_size:]
loss_render = F.mse_loss(render_bev, real_bev)
```

global loss 只取索引 `1`。结合相机顺序，它是 `F0`：

`navsim/agents/rap_dino/rap_model.py:147-152`

```python
feat = image_feature[0][[1]]
feat_grad = feat[:, :, :self.batch_size].detach()
feat_no_grad = feat[:, :, self.batch_size:]
mixed_feat = torch.cat([feat_grad, feat_no_grad], dim=2)
domain_logits = self.domain_classifier(mixed_feat, lambd=lambda_)
```

real validity 是四相机逻辑 AND：

`navsim/agents/rap_dino/bevformer/bev_feature_build.py:68`

```python
image_result["validity"] = torch.all(torch.stack(validity_list))
```

#### 置信度

- 当前源码：高。
- global 是否进入发布 checkpoint 的实际训练：低，见 E2。

### A2. projector 是否共享？有没有 conditioning？

#### 结论

所有相机和 real/raster 共用同一个 DINOv3 和同一个 FPN，不存在 per-camera projector。

conditioning 情况：

- 有：每个相机槽位一个 learned `cams_embeds`，等价于 camera ID conditioning。
- 有：learned `level_embeds`。
- 没有：projector 本身不读取 `lidar2img`、相机内参或外参。
- 内外参随后传给 planner 的 BEVFormer cross-attention，不用于 spatial MSE 前的 FPN 投影。

#### 证据

只有一套 backbone 和 FPN：

`navsim/agents/rap_dino/bevformer/image_encoder.py:30,42-49`

```python
self.img_backbone = AutoModel.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m"
)
self.img_neck = FPN(
    in_channels=[64, 128, 256, 1280][-self.num_outs:],
    out_channels=_dim_,
    ...
)
```

四个 camera embedding：

`navsim/agents/rap_dino/bevformer/image_encoder.py:50-52`

```python
self.level_embeds = nn.Parameter(
    torch.randn(self.num_feature_levels, self.embed_dims)
)
self.cams_embeds = nn.Parameter(
    torch.randn([self.num_cams, self.embed_dims])
)
```

应用 camera embedding：

`navsim/agents/rap_dino/bevformer/image_encoder.py:109-115`

```python
feat = feat.flatten(3).permute(1, 0, 3, 2)
feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
```

内外参只随 `kwargs` 返回，之后由 planner 使用：

`navsim/agents/rap_dino/bevformer/image_encoder.py:128`

```python
return feat_flatten[-1], spatial_shapes[-1], level_start_index, kwargs
```

`navsim/agents/rap_dino/bevformer/encoder.py:122-133`

```python
lidar2img = img_metas['lidar2img']
...
reference_points_cam = torch.matmul(
    lidar2img.to(torch.float32), reference_points.to(torch.float32)
)
```

checkpoint 也保存了：

```text
cams_embeds:  (4, 1280)
level_embeds: (2, 1280)
FPN conv:     1280 -> 1280
```

#### 置信度

高。

### A3. 每帧每相机都参与吗？有没有采样？

#### 结论

单个已进入训练集且 `camera_valid=True` 的样本：

- spatial alignment 使用当前帧四个相机，没有随机选相机。
- global alignment 固定只用 `F0`。
- 没有“每 N 个 training iteration 才算 alignment”的逻辑。
- validation 不算 alignment，只跑 real branch。
- 图像会用 `GridMask(prob=0.7)`，但它是图像增强，不是相机采样。

数据集级采样：

- base cache：全部有效 cache。
- perturbed cache：训练启动时随机抽 `10%`。
- cross-agent cache：随机抽 `5%`。
- 具体抽到了哪些样本没有保存在 checkpoint 中。

#### 证据

训练 batch 始终拼接 raster 和有效 real：

`navsim/planning/training/agent_lightning_module.py:149-160,211-220`

```python
real_valid_mask = features['camera_valid']
real_features = {
    k: v[real_valid_mask]
    for k, v in features.items()
    if k not in ['camera_valid', 'rendered_camera_feature']
}
...
all_features[k] = torch.cat([v, real_features[k]], dim=0)
```

validation 走 real-only 分支：

`navsim/planning/training/agent_lightning_module.py:163-166`

```python
if not self.training or real_only:
    if real_valid_mask.any():
        prediction = self.agent.forward(real_features, real_targets)
```

perturbed/cross-agent 抽样：

`navsim/planning/script/run_training.py:149-169`

```python
indices = random.sample(range(N), int(0.1 * N))
train_data_perturbed = Subset(train_data_perturbed, indices)
...
indices = random.sample(range(N), int(0.05 * N))
train_data_others = Subset(train_data_others, indices)
```

#### 置信度

- 采样代码：高。
- 具体 checkpoint 抽样结果：无法从代码确定。

### A4. split、scene、帧率和过滤

#### 结论

README 发布配方是：

- metadata 来源：nuPlan `trainval`。
- caching/training：`train_test_split=navtrain`。
- `navtrain` 实际默认继承 `scene_filter=navall`。
- `navall` 静态列表含 `14,539` 个 log、`401,300` 个 current-frame token。
- 其中 `13,158` 个 log 属于 train，`1,381` 个属于 val。
- cache-only training 从 base cache 中排除 `val_logs`。
- 数据时间间隔为 `0.5s`，即 `2Hz`。
- scene window 为 `4 history + 10 future`。
- 每相邻一帧滑动，`frame_interval=1`。
- 图像只使用第 4 个 history frame，即当前时刻。

过滤包括：

- window 不足 14 帧：跳过。
- `is_valid=False`：跳过。
- token 不在静态 token list：跳过。
- README 把 `has_route` 改成 `false`，所以不按 route 过滤。
- cache 文件不完整：training dataset 不加载。
- real camera 缺失：样本仍可能在 cache，但不做 alignment。

**实际 checkpoint 的确切 scene/token 集合无法从代码确定。** checkpoint 没保存 config 或 cache manifest。

#### 证据

README 配方：

`README.md:74-119,139-155`

```bash
train_test_split=navtrain
train_test_split.scene_filter.has_route=false
cache_path=./cache/rap_ego
...
agent.config.distill_feature=True
cache_path_perturbed=./cache/rap_perturbed
cache_path_others=./cache/rap_aug
```

`navtrain -> navall`：

`navsim/planning/script/config/common/train_test_split/navtrain.yaml:1-4`

```yaml
defaults:
  - scene_filter: navall
data_split: trainval
```

帧配置：

`navsim/planning/script/config/common/train_test_split/scene_filter/navall.yaml:4-9`

```yaml
num_history_frames: 4
num_future_frames: 10
frame_interval: 1
has_route: true
max_scenes: null
```

2Hz：

`navsim/common/dataclasses.py:26`

```python
NAVSIM_INTERVAL_LENGTH: float = 0.5
```

原始 nuPlan metadata 每 10 个 lidar frame 取一个：

`process_data/create_openscene_metadata.py:244-245,496-498`

```python
lidar_pc_list = lidar_pc_list[start_idx :: args.sample_interval]
...
parser.add_argument("--sample-interval", type=int, default=10)
```

过滤：

`navsim/common/dataloader.py:112-130`

```python
if len(frame_list) < scene_filter.num_frames:
    continue
current_frame = frame_list[scene_filter.num_history_frames - 1]
if current_frame.get("is_valid", True) == False:
    continue
if scene_filter.has_route and len(current_frame["roadblock_ids"]) == 0:
    continue
if filter_tokens and token not in tokens:
    continue
```

cache-only base train/val 划分：

`navsim/planning/script/run_training.py:113-146`

```python
train_logs = [x for x in cached_logs if x not in cfg.val_logs]
val_logs = [x for x in cached_logs if x in cfg.val_logs]
```

#### 发布配方内部不一致

perturbed 文件名是 `log_name + "_" + current_token`；cross-agent 文件名是 `log_name + "_" + vehicle_id`。但 caching 会先要求 pickle 文件名必须在原始 `navall.log_names` 中，因此按发布命令直接跑，这两类文件会被过滤：

`navsim/common/dataloader.py:107-109`

```python
if log_pickle_path.name.replace(".pkl", "") not in scene_filter.log_names:
    continue
```

`process_data/create_openscene_metadata_purturbed.py:329`

```python
log_name_this_batch = log_name + "_" + str(current_lidar_pc_token)
```

`process_data/create_openscene_metadata_aug.py:878-880`

```python
pkl_file_path_vehicle = f"{args.out_dir}/{log_name}_{vehicle_id}.pkl"
```

所以实际训练必然用了未发布的路径调整、已有 cache，或不同 scene filter；具体是哪种，无法从代码确定。

#### 置信度

中。静态配置和过滤逻辑明确；实际 checkpoint 数据清单不可恢复。

---

## B. alignment loss 的具体形式

### B1. spatial/global 定义、形状和权重

#### Spatial

定义：

$$
L_{spatial}=\operatorname{MSE}\left(\operatorname{stopgrad}(F_{raster}),F_{real}\right)
$$

权重：

$$
\lambda_{spatial}=0.002
$$

标准发布图像经过 `0.4` resize 和 32 对齐后是 `448x768`，DINO patch size 为 16，因此典型 feature 形状为：

```text
image_feature[0]: (4 cameras, 1344 patches, B, 1280)
bev_feature:      (B, 4, 1344, 1280)
MSE inputs:       (B_valid, 4, 1344, 1280)
```

权重证据：

`navsim/agents/rap_dino/navsim_config.py:12-13`

```python
distill_feature: bool = False
distill_feature_weight: float = 0.002
```

预处理证据：

`navsim/agents/rap_dino/bevformer/bev_feature_build.py:79-90`

```python
image_result = RandomScaleImageMultiViewImage(image_result)  # 432,768
image_result = PadMultiViewImage(image_result)                # 448,768
```

#### Global

当前源码定义：

1. 只取 `F0`。
2. 对 camera/patch 求 mean，得到 `(B,1280)`。
3. domain label：raster=`0`，real=`1`。
4. 使用 `BCEWithLogitsLoss`。
5. GRL 系数随 epoch 从 0 向 1 增长。
6. 总 loss 权重固定 `0.001`。

聚合：

`navsim/agents/rap_dino/rap_model.py:55-68`

```python
C, P, B, D = feat.shape
feat = feat.reshape(C * P, B, D)
feat = feat.mean(dim=0)  # (B, D)
```

loss：

`navsim/planning/training/agent_lightning_module.py:234-244`

```python
domain_labels = torch.cat([
    torch.zeros(N_synth),
    torch.ones(N_real),
], dim=0)
domain_loss = bce_logits(domain_logits, domain_labels.float())
loss_dict['loss'] += 0.001 * domain_loss
```

作用是让 classifier 区分域，同时通过 GRL 让 real feature 更难被识别为 real。

**checkpoint 限制：两份发布权重均无 `domain_classifier.*`。所以以上是当前源码定义，不能证明它实际训练了发布 checkpoint。**

#### 置信度

- spatial：高。
- global 形式：高。
- global 对发布 checkpoint 的实际应用：低。

### B2. 梯度流向

#### 结论

spatial：

- raster feature：`detach()`，当前 backward 不收梯度。
- real feature：接收 MSE 梯度，被拉向 raster feature。
- 不是 raster 被拉向 real。
- 两域共享 FPN，因此 real 路径更新 FPN 后，下一轮 raster feature 也会间接改变；raster 不是固定 teacher。

global：

- raster feature 在进入 GRL 前同样 `detach()`。
- real feature 通过 GRL 接收反向梯度。
- classifier 同时从 raster 和 real 样本获得正常 BCE 梯度。

#### 证据

Spatial：

`navsim/planning/training/agent_lightning_module.py:246-250`

```python
render_bev = bev_feature[:batch_size][real_valid_mask].detach()
real_bev = bev_feature[batch_size:]
loss_render = F.mse_loss(render_bev, real_bev)
```

Global：

`navsim/agents/rap_dino/rap_model.py:148-152`

```python
feat_grad = feat[:, :, :self.batch_size].detach()  # raster
feat_no_grad = feat[:, :, self.batch_size:]        # real
mixed_feat = torch.cat([feat_grad, feat_no_grad], dim=2)
domain_logits = self.domain_classifier(mixed_feat, lambd=lambda_)
```

GRL：

`navsim/agents/rap_dino/rap_model.py:18-25`

```python
def backward(ctx, grad_output):
    return -ctx.lambd * grad_output, None
```

#### 置信度

高（当前源码）。

### B3. image encoder 是否完全 frozen？

#### 结论

- DINOv3 所有参数都被设成 `requires_grad=False`。
- optimizer 明确排除 `_backbone.img_backbone`。
- 没有 partial unfreeze。
- FPN、camera embedding、level embedding 没冻结，会训练。
- 代码没有显式执行 `img_backbone.eval()`；Lightning 的 `train()` 会把它放在 train mode。
- 两份 checkpoint 的整个 `state_dict` 都没有 `running_mean`、`running_var` 或 `num_batches_tracked`，没有发现 BN running statistics。
- 因此：参数层面 DINO 完全 frozen；mode 层面没有被锁成 eval，但没有发现 BN 统计量更新。

#### 证据

`navsim/agents/rap_dino/rap_agent.py:566-577`

```python
other_params = [
    p for n, p in self._rap_model.named_parameters()
    if p.requires_grad and '_backbone.img_backbone' not in n
]
for p in self._rap_model._backbone.img_backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW([
    {"params": other_params, "lr": self._lr, "weight_decay": 1e-4},
])
```

当前代码没有任何针对 DINO 的 `eval()` 或 `train()` override。

#### 置信度

- 参数冻结：高。
- “完全固定 train/eval 行为”：中。

---

## C. planner 侧

### C1. planning loss 是否施加到 real 和 raster？

#### 结论

是。训练时先拼接：

```text
[raster: B 个样本, real: B_valid 个样本]
```

然后一次性调用 planner 和 `compute_loss`。因此 planning loss 同时作用于每个 raster 样本和每个有效 real 样本。raster 的 `detach()` 只针对 spatial/global alignment，不影响 planning loss。

#### 证据

`navsim/planning/training/agent_lightning_module.py:211-222`

```python
all_features[k] = torch.cat([v, real_features[k]], dim=0)
all_targets[k] = torch.cat([v, real_targets[k]], dim=0)
prediction = self.agent.forward(all_features, all_targets)
loss_dict = self.agent.compute_loss(all_features, all_targets, prediction)
```

planning loss 对整个 batch 求 mean：

`navsim/agents/rap_dino/rap_agent.py:490-501`

```python
min_loss = torch.linalg.norm(
    proposals_i - target_trajectory[:, None], ...
).mean(-1).amin(1)
...
min_loss = (min_loss * weight).mean()
```

#### 置信度

高。

### C2. 推理时 planner 的确切输入

#### 结论

推理输入路径是：

```text
real RGB
-> frozen DINOv3
-> shared FPN
-> camera embedding + level embedding
-> BEVFormer planner
```

不是 raw `f_real`，也不是 domain classifier 输出。planner 消费的是 alignment 所在的 post-FPN projected feature，以及标定 metadata。

#### 证据

推理只保留 real features：

`navsim/planning/training/agent_lightning_module.py:447-455`

```python
real_features = {
    k: v[real_valid_mask]
    for k, v in features.items()
    if k not in ['camera_valid', 'rendered_camera_feature']
}
prediction = self.agent.forward(real_features, None, return_score=True)
```

模型先过 image encoder：

`navsim/agents/rap_dino/rap_model.py:109-129`

```python
camera_feature = features["camera_feature"]
image_feature = self._backbone(camera_feature, img_metas=features)
...
bev_feature, proposal_list = refine(
    bev_feature, proposal_list, image_feature
)
```

refiner 把 post-FPN feature 当 cross-attention key/value：

`navsim/agents/rap_dino/bevformer/bev_refiner.py:122-138`

```python
feat_flatten, spatial_shapes, level_start_index, kwargs = image_feature
...
self.bev_decoder(
    ...,
    feat_flatten,
    feat_flatten,
    ...
)
```

#### 置信度

高。

---

## D. 渲染接口

### D1. 能否接受任意 SE(3) 相机位姿？

#### 结论

不能通过现有 public interface 直接传任意 SE(3) 相机位姿。

`ScenarioRenderer.observe(scenario)`：

- scenario 只提供 `ego_heading`。
- 相机相对外参从内部 `camera_params` 固定读取。
- translation 还被硬编码修改：`z += 0.8`、`x -= 2`。
- 不接收 pitch、roll、camera translation 或自定义旋转矩阵。
- 最多从 8 个固定 camera model 中选择 channel。

底层 `world_to_camera_T()` 数学上接收 translation/rotation，但 `observe()` 没暴露这些参数。

#### 证据

`process_data/helpers/renderer.py:694-715`

```python
class ScenarioRenderer:
    def __init__(
        self,
        camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0'],
        width=1920,
        height=1120,
        depth_max=120.0,
    ):
        ...

    def observe(self, scenario):
        lidar_pos = np.zeros(3)
        lidar_yaw = scenario['ego_heading']
        ...
        cam_t = cam_model["sensor2lidar_translation"].copy()
        cam_t[2] += 0.8
        cam_t[0] -= 2
        cam_R = cam_model["sensor2lidar_rotation"]
        T_w2c = world_to_camera_T(lidar_pos, lidar_yaw, cam_t, cam_R)
```

#### 置信度

高。

### D2. renderer 需要哪些输入？从哪里读？

#### 结论

`scenario` 至少需要：

```text
scenario['ego_heading']
scenario['map_features'][id]['type']
scenario['map_features'][id]['polygon'/'polyline']
scenario['traffic_lights']
scenario['anns']['gt_boxes_world']
scenario['anns']['gt_names']
```

`ego_pos` 在生成脚本里设置了，但当前 renderer 没读取。

数据来源：

- nuPlan `.db`：ego pose、lidar boxes、traffic-light status、camera records。
- nuPlan map：`get_maps_api(...)`。
- real image blob：`cams[cam_id]['data_path']`。
- 输出 metadata：每 log 一个 `.pkl`。
- 输出 raster：沿 real camera 相对路径写到 `rendered_sensor_blobs`。

#### 证据

renderer 读取字段：

`process_data/helpers/renderer.py:717-758`

```python
for feat in scenario['traffic_lights']:
    ...
for feat in scenario['map_features'].values():
    ...
anns = scenario["anns"]
bboxes = anns["gt_boxes_world"]
names = anns["gt_names"]
```

normal metadata 构造：

`process_data/create_openscene_metadata.py:297-302,438-448`

```python
scenario['map_features'] = map_features
scenario['ego_pos'] = [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y]
scenario['ego_heading'] = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]
scenario['traffic_lights'] = traffic_lights
...
scenario['anns'] = info['anns']
```

读取 nuPlan DB 和 map：

`process_data/create_openscene_metadata.py:224-230`

```python
log_db = NuPlanDB(..., log_db_name + ".db", None)
map_api = get_maps_api(
    NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_location
)
```

写 raster：

`process_data/create_openscene_metadata.py:456-460`

```python
real_path = cams[cam_id]['data_path']
full_rendered_path = os.path.join(render_sensor_path, real_path)
cv2.imwrite(full_rendered_path, rendered[:, :, ::-1])
```

#### 置信度

高。

### D3. 是否有现成 perturbation 路径？

#### 结论

有。`create_openscene_metadata_purturbed.py` 扰动的是 ego pose，不是 camera-to-lidar 外参：

- `x/y`：`±0.5m`
- yaw：`±15°`
- velocity：`±20%`
- acceleration：`±10%`

只有 window 的 current frame，即 `batch[3]` 被扰动和渲染。相机仍使用固定 relative extrinsics，所以等价于整个固定相机 rig 随 ego 改变。

另外有 cross-agent 路径，把另一个 vehicle 的 pose 当作 pseudo ego；它也不修改相机外参。

#### 证据

`process_data/create_openscene_metadata_purturbed.py:235-264`

```python
if perturb:
    x += random.uniform(-0.5, 0.5)
    y += random.uniform(-0.5, 0.5)
    yaw += math.radians(random.uniform(-15, 15))
    vx += random.uniform(-0.2, 0.2) * vx
    vy += random.uniform(-0.2, 0.2) * vy
```

只扰动 current frame：

`process_data/create_openscene_metadata_purturbed.py:320-336`

```python
batch = lidar_pc_list[lidar_i:lidar_i + 14]
current_lidar_pc_token = batch[3].token
...
perturb = lidar_pc_token == current_lidar_pc_token
```

cross-agent pose：

`process_data/create_openscene_metadata_aug.py:619-634`

```python
new_ego_pose_relative = info['anns']['gt_boxes_world'][vehicle_index]
abs_new_ego_pos = (
    info['ego2global_translation'] + new_ego_pose_relative[:3]
)
...
new_scenario['ego_heading'] = new_ego_yaw_world
```

#### 置信度

- 代码路径：高。
- 发布 checkpoint 实际用了哪些 perturbation：无法确定。

---

## E. checkpoint

### E1. 对应 config

#### 结论

不存在单独的 `navsimv1` 或 `navsimv2` config 文件。README 指向统一的：

```text
agent=rap_agent
+ README command-line overrides
```

两份 checkpoint 都没有 `hyper_parameters`，所以无法确认实际命令行 override。

相关静态字段：

| 字段 | 发布配方/默认值 | 类型 |
|---|---:|---|
| `distill_feature` | `True` | `rap_agent.yaml` 写死 |
| spatial weight | `0.002` | dataclass 默认 |
| global weight | `0.001` | loss 代码写死 |
| `pdm_scorer` | `True` | YAML / README |
| cameras | `B0/F0/L0/R0 @ history[3]` | agent 写死 |
| `trajectory_sampling.time_horizon` | YAML 默认 `4`；README override `5` | 可配 |
| `interval_length` | `0.5` | YAML 默认 |
| `lr` | `1e-4` | YAML 默认 |
| batch size | `64` | training YAML/README |
| base/perturbed/cross-agent | 全部 + `10%` + `5%` | training 代码写死 |
| optimizer | `AdamW`, wd=`1e-4` | 代码写死 |
| scheduler | 20 epochs、1 warmup、min LR `1e-5` | 代码写死 |
| trainer max epochs | 默认 `100` | config；README 未 override |

#### 证据

`navsim/planning/script/config/common/agent/rap_agent.yaml:8-21`

```yaml
trajectory_sampling:
  time_horizon: 4
  interval_length: 0.5
distill_feature: True
pdm_scorer: True
...
lr: 1e-4
```

`navsim/agents/rap_dino/rap_agent.py:573-580`

```python
optimizer = torch.optim.AdamW(
    [{"params": other_params, "lr": self._lr, "weight_decay": 1e-4}]
)
scheduler = WarmupCosLR(
    optimizer=optimizer,
    lr=self._lr,
    min_lr=1e-5,
    epochs=20,
    warmup_epochs=1,
)
```

#### 置信度

低到中。能确认 release 默认和 checkpoint scheduler，不能恢复完整实际 config。

### E2. checkpoint 变体和训练差异

#### 结论

本地有两个 NAVSIM checkpoint，网络 state schema 相同：

| checkpoint | epoch | global step | callback 原始目录 | domain head |
|---|---:|---:|---|---|
| `RAP_DINO_navsimv1.ckpt` | 17 | 50,346 | `ipad_dinov3_1280_all` | 缺失 |
| `RAP_DINO_navsimv2.ckpt` | 14 | 21,930 | `dinov3_perturbed_distill_10perc` | 缺失 |

checkpoint callback 路径说明：

- v1 目录名带 `all`。
- v2 目录名带 `perturbed_distill_10perc`。

这强烈暗示 v2 使用了 10% perturbed + distillation。但目录名不是 config，不能据此确定其余训练差异。

两份 checkpoint 均：

- `state_dict` 926 项。
- optimizer 参数状态 103 项。
- camera embedding `(4,1280)`。
- FPN `1280 -> 1280`。
- 没有 `domain_classifier.*`。
- 没有 `hyper_parameters`。

当前源码明确注册 domain classifier：

`navsim/agents/rap_dino/rap_model.py:104-106`

```python
self.scorer = Scorer(config)
self.domain_classifier = DomainClassifier(config.tf_d_model)
self.lambda_scheduler = LambdaScheduler(gamma=10.0)
```

加载时使用 `strict=False`，所以缺失的 head 会被随机初始化：

`navsim/agents/rap_dino/rap_agent.py:90-100`

```python
state_dict = checkpoint['state_dict']
missing_keys, unexpected_keys = self.load_state_dict(
    state_dict, strict=False
)
```

README 还列出 Waymo 结果，但本地 `ckpts/` 没有 Waymo checkpoint。

#### 置信度

- checkpoint 元数据：高。
- 具体训练差异：低。

---

## 发布 checkpoint 中确定存在的 alignment 盲区

### 1. 历史图像帧全部没有监督

每个 2Hz scene 有 4 个 history frame，但只加载 index `3`。因此相对当前时刻约 `-1.5s/-1.0s/-0.5s` 的三帧不进图像 encoder，也没有 alignment。

`navsim/agents/rap_dino/rap_agent.py:124-136`

```python
return SensorConfig(
    cam_f0=[3],
    cam_l0=[3],
    cam_l1=[],
    cam_l2=[],
    cam_r0=[3],
    cam_r1=[],
    cam_r2=[],
    cam_b0=[3],
    lidar_pc=[],
)
```

### 2. `L1/L2/R1/R2` 从未监督

四个相机的 `SensorConfig` 都为空列表，既不参与 spatial，也不参与 global。

### 3. global alignment 只覆盖 `F0`

即使假设当前 global 代码用于训练，`B0/L0/R0` 也从未接受 global/GRL alignment。

### 4. camera-to-lidar 外参扰动没有显式监督

renderer 使用固定外参；现成 perturbation 改的是 ego pose，不是单独的 camera translation/rotation。特别是以下区域不在已发布 renderer 的训练接口内：

- 相机高度变化。
- pitch/roll 扰动。
- 单相机 yaw/translation 偏移。
- 非固定 rig calibration。

### 5. normal/perturbed 数据里的 `B0` 没有有效 raster 配对

默认 renderer 不生成 `B0`。loader 把缺失 raster 变成全零图：

`navsim/common/dataclasses.py:76-89`

```python
try:
    rendered_image = np.array(Image.open(rendered_image_path))[20:-20]
except:
    rendered_image = np.zeros((1080, 1920, 3), dtype=np.float32)
```

因此 `B0` 确实进入 spatial loss，但被监督到“空白 raster feature”，不是有效的后视 Raster-to-Real alignment。只有 cross-agent renderer 显式请求了 `B0`；这些数据是否真正进入发布 checkpoint，无法确认。

### 6. 有限训练样本之外的连续 SE(3) 位姿未覆盖

代码只可能覆盖：

- 静态 token list 中的 logged current poses。
- 实际随机抽中的少量 ego `x/y/yaw` perturbation。
- 实际随机抽中的 cross-agent ego poses。

它没有形成连续 SE(3) 覆盖。

## 无法从发布代码/checkpoint 确定的部分

- 两份 checkpoint 实际用了哪些 token。
- perturbed/cross-agent cache 是否成功加载。
- 随机抽中的具体 10%/5% 样本。
- v1 是否也用了 perturbation。
- global loss 是否真正参与发布 checkpoint 训练。
- 每个 checkpoint 实际见过的精确全局 camera pose 集合。

## 对后续实验的直接含义

代码上能保证未被 alignment 监督的测试区域是：

1. 对 `F0/L0/R0/B0` 的 camera-to-lidar extrinsic 做非零 perturbation，尤其是 pitch、roll、高度或单相机平移。
2. 直接测试 `L1/L2/R1/R2`。
3. 测试前三个历史时刻的图像输入。

其中 `B0` 需要单独处理：normal/perturbed 训练路径虽然对它计算 spatial MSE，但 target 是缺图产生的全零 raster，不能视为正常的 Raster-to-Real 监督。
