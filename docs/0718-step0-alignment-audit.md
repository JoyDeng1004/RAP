# 步骤 0：栅格 / 对齐信息边界审计

范围：本文后续实验仅使用标准 `dataset_norm`。`dataset_aug` 与 `dataset_perturbed` 不参与 Step 0 的统计、验收或阻断判定。

## 1. 目的

在探查对齐损失如何影响图像特征（步骤 1/2）之前，先明确解读探查结果时所依据的事实：

代码级梯度推导与图中模块映射的详细背景见 [`raster-to-real-alignment-code-analysis.md`](raster-to-real-alignment-code-analysis.md)。本文在此基础上进一步给出 Step 0 的可执行实验协议。

- **Q1** 遮挡与逐层合成后，栅格中的 agent、map 和 traffic lights 分别还保留了什么？栅格是否包含相机无法观测的信息（结构私有信息），反之又是否成立？
- **Q2** 栅格编码了哪些字段（类别、速度、实例、历史、交通信号灯）？
- **Q3** 对齐梯度沿哪个方向流动——谁被拉向谁，又经过哪个模块？
- **Q4** 规划器在推理时接收什么输入？

以下静态结论均引用了 `文件:行号` 证据；需要动态确认的部分由 0-A～0-F 实验覆盖。

## 2. Q1——遮挡与视场：原有先验恰好相反

此前讨论中的工作假设是：栅格会剔除被遮挡的内容并裁剪到视场（FOV）内，因此结构私有集合几乎为空。代码表明事实正好相反。

### 2.1 不透明的画家算法，无透明度

- `draw_cuboids_with_occlusion` 收集所有智能体长方体的全部 6 个面，按从远到近的顺序排序，并使用 `cv2.fillConvexPoly` **不透明地**填充这些面（`process_data/helpers/renderer.py:360-463`，排序见 `:447`，填充见 `:463`）。
- `alpha = clip((depth_max - depth)/depth_max)` 项（`renderer.py:456`）乘到面颜色上——它表示**基于深度的亮度衰减，而不是透明度**。当较近的智能体与较远的智能体重叠时，较远智能体的像素会被完全覆盖：智能体之间的相互遮挡不会在重叠区域留下任何残余（较远智能体未重叠的部分仍然可见）。

### 2.2 栅格不包含环境几何体

- `ScenarioRenderer.observe` 恰好绘制三层：交通信号灯、地图要素，最后在最上层绘制智能体长方体（`renderer.py:704-758`）。其中没有建筑、植被、围栏或任何未标注的遮挡物。
- 智能体框直接来自标注（`anns["gt_boxes_world"]`，`renderer.py:755-758`）；代码从未检查标注的可见性。
- **结果：** 如果一个已标注智能体在真实相机图像中因建筑、树篱或模式定义之外的停放卡车遮挡而不可见，它仍会完整、清晰地渲染到栅格中。这是真正的**结构私有**信息。[0-B：在真实场景上确认]
- 无论深度如何，智能体总是绘制在地图元素之上（`renderer.py:758` 最后执行），因此地图要素绝不可能遮挡智能体。

### 2.3 地图与智能体的 FOV 处理不对称

- 地图多边形通过 Sutherland–Hodgman 算法正确裁剪（`renderer.py:240-311`，调用见 `:345`）；折线则使用 `cv2.clipLine` 并进行近裁剪面 z 裁剪（`:197-228`）。
- 智能体会被**整体丢弃**：如果长方体 8 个投影角点中有效的少于 4 个（位于相机前方且在图像范围内），就跳过该长方体——`if valid.sum() < 4: continue`（`renderer.py:419-420`）。一辆跨越 FOV 边缘、尺寸较大且距离较近的车辆，可能在真实图像中部分可见，却从栅格中完全消失。[0-C：按距离量化丢弃率]

### 2.4 渲染相机不是真实相机

- 虚拟相机相对于标定外参存在偏移：`cam_t[2] += 0.8; cam_t[0] -= 2`（`renderer.py:711-712`）——向上 0.8 m、向后 2 m。因此，即使内容在两者中都可见，栅格与真实图像也并非像素对齐。

### 2.5 修正后的信息不对称性

| 方向 | 内容 | 所支持的机制 |
|---|---|---|
| **结构私有**（仅栅格） | 被未标注环境遮挡的已标注智能体；显式地图几何先验与交通信号灯状态；因夜间、雨天、眩光或距离而在 real 中难以辨认、但仍被清晰绘制的元素 | 机制 B（后验注入）**确实有实际目标** |
| **图像私有**（仅相机） | 天气、光照、纹理、真实道路外观、建筑/植被/天空、行人姿态、模式定义之外的物体；智能体相互遮挡区域的像素；因角点少于 4 个或 renderer 不支持而在 raster 中缺失的元素 | 机制 A（语义抹除）的目标集合，已确认 |

此前“结构私有 ≈ 空集”的先验已被推翻。步骤 2 中的两种机制都有具体且可枚举的目标变量。

### 2.6 Raster target 是三类显式元素加黑色未占用区域

`ScenarioRenderer.observe` 从全零画布开始，依次绘制 `traffic_lights → map_features → agents`。因此本文中的 raster target 不能简化成“agent 图”；至少要区分：

1. **agent**：由标注 3D box 生成的长方体面；
2. **map**：当前 renderer 分支实际支持的 lane、crosswalk/speed bump、boundary/solid 等地图要素；
3. **traffic lights**：由位置和红/非红状态生成的固定尺寸长方体；
4. **blank/unoccupied**：没有任何上述元素写入的全黑区域。它不是经过验证的“背景类别”，也不代表完整的道路、建筑或天空。

训练中的 spatial MSE 作用于整张 raster 经过 DINOv3/FPN 后的 feature，而不是只作用于 agent mask。因此 0-C 既要保留原有的 agent 实例可见性统计，也必须审计 map/traffic-light 的投影、覆盖关系和各组件对 feature target/MSE 梯度的影响。由于 encoder 非线性且组件存在绘制覆盖，组件结果只能解释为受控消融，不能假定三类 feature 或梯度可线性相加。

## 3. Q2——编码字段

- **不按类别着色。** 每个智能体都使用相同的固定六面调色板绘制；颜色编码的是**面的朝向**，而非类别（`renderer.py:374-381`；按类别着色的路径已被注释掉，见 `renderer.py:766-796`）。`COLOR_TABLE` 中的类别颜色（`renderer.py:14-26`）实际上并未用于智能体。
- **不编码速度、不使用跨帧一致的实例着色，也不堆叠历史帧。** `ScenarioRenderer.observe` 只读取当前 `scenario`，并且智能体渲染只消费 `anns["gt_boxes_world"]`（`renderer.py:704-758`）。扰动数据脚本也只在当前 lidar token 上调用一次渲染（`process_data/create_openscene_metadata_purturbed.py:524-535`；注意仓库中的文件名实际拼写为 `purturbed`）。
- **显式编码交通信号灯状态：** 在信号灯位置绘制红色或绿色长方体（`renderer.py:717-732`）。
- **地图不是照片级背景。** 当前 renderer 只对 `ftype` 命中 `LANE`、`CROSSWALK/SPEED_BUMP`、`BOUNDARY/SOLID` 的要素进入绘制分支（`renderer.py:734-754`）；其余未占用像素保持全黑。0-C 必须分别统计 metadata 中的 map feature 数、renderer 支持数、实际产生像素数，不能把黑色区域解释为“无道路”标签。
- 几何设置：默认只为 `CAM_F0/L0/R0` 生成透视相机视角栅格，原始画布为 1920×1120（`renderer.py:694-702`）。加载时上下各裁掉 20 像素，得到 1920×1080（`navsim/common/dataclasses.py:77-82`），随后缩放 ×0.4 至 768×432，并填充为 768×448（`navsim/agents/rap_dino/bevformer/bev_feature_build.py:12-13,78-90`）。标准渲染路径没有生成 `CAM_B0`；其文件读取失败后会被静默替换为全零图像，而增强渲染脚本是否包含 B0 必须单独区分。
- 训练时的栅格已**预先渲染并存储在磁盘上**。本文只审计以下 norm 数据：

| 变体 | 生成脚本 | dataset root | metadata root | raster image root |
|---|---|---|---|---|
| 标准 / norm | `/gs/bs/tga-RLA/qdeng/RAP/process_data/create_openscene_metadata.py` | `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm` | `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/navsim_logs/mini` | `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs` |

标准 loader 的默认实现会将 `image_path` 中的 `sensor_blobs` 替换为 `rendered_sensor_blobs`（`navsim/common/dataclasses.py:77-82`），若文件缺失则静默回退到全零图像（`dataclasses.py:82`）。该字符串替换只直接匹配 norm 命名

## 4. Q3——对齐梯度：当前 step 内，真实图像被拉向 stop-gradient raster target

### 4.1 对齐对象与模块映射

- 前向批次顺序为 `cat([rendered, real])`（`navsim/planning/training/agent_lightning_module.py:211-220`）。
- 冻结的 Image Encoder 是 DINOv3：`facebook/dinov3-vith16plus-pretrain-lvd1689m`（`navsim/agents/rap_dino/bevformer/image_encoder.py:30,88`）。
- 图中的 Feature Projector 在当前代码中没有独立同名类；其主体是 DINOv3 后的可训练 FPN `img_neck`，输出还叠加可训练的 camera/level embedding（`image_encoder.py:42-52,95-115`）。
- 被对齐的 `prediction["bev_feature"]` 来自 `image_feature[0].permute(2,0,1,3)`（`navsim/agents/rap_dino/rap_model.py:145`）。虽然名字叫 `bev_feature`，它实际上是 post-FPN 的多相机图像 token，典型形状为 `(B, 4 cams, 1344 tokens, 1280)`。
- real 与 raster 不存在两个独立 projector，而是在 batch 维拼接后经过同一个 `ImgEncoder`、同一个 FPN 和同一组 embedding。
- **B0 风险：** spatial MSE 使用全部四个相机，但标准 renderer 默认只生成 F0/L0/R0。若训练数据确实走标准路径，B0 的 raster 文件缺失会由 loader 替换为全零图像；经过归一化、DINOv3 和 FPN 后，它不是数值全零 feature，而是“空白输入”的 feature target。这样 real B0 会被显式拉向空白 raster 表示。该问题必须在 0-A 中按相机核实，并在 0-F 中报告 per-camera MSE/gradient，不能只看总平均。

### 4.2 Spatial-level MSE 梯度

损失位置为 `agent_lightning_module.py:246-250`：

```python
render_bev = bev_feature[:batch_size][real_valid_mask].detach()
real_bev = bev_feature[batch_size:]
loss_render = F.mse_loss(render_bev, real_bev)
loss_dict["loss"] += 0.002 * loss_render
```

令 \(F^{real}\) 和 \(F^{raster}\) 分别表示配对的 real/raster post-FPN feature，则：

\[
L_{align}
=
\frac{0.002}{N}
\sum_i
\left(F_i^{real}-\operatorname{stopgrad}(F_i^{raster})\right)^2
\]

因此：

\[
\frac{\partial L_{align}}{\partial F_i^{real}}
=0.002\frac{2}{N}(F_i^{real}-F_i^{raster}),
\qquad
\frac{\partial L_{align}}{\partial F_i^{raster}}=0
\]

**就 MSE 这条路径而言，计算图方向是单向 real → raster。** 但 `image_feature` 是中间张量，AdamW 不会直接修改它；梯度继续回传并更新生成它的 FPN、camera embedding 和 level embedding。DINOv3 参数被排除在 optimizer 外并设置为 `requires_grad=False`（`navsim/agents/rap_dino/rap_agent.py:566-577`）。

### 4.3 “冻结 raster anchor”只能描述当前 iteration

原先“真实图像被拉向冻结的栅格锚点”的说法不够准确。`.detach()` 只表示当前 backward 不沿 raster activation 回传；由于两个域共享 FPN，本 step 通过 real 分支更新 FPN 后，下一 step 重新计算得到的 raster feature 也会改变。因此更准确的描述是：

> raster 是当前 iteration 的 stop-gradient target，不是跨 iteration 固定不变的 teacher。

该细节会影响后续机制解释：若要验证 target 是否漂移，应在固定输入、关闭随机增强的条件下记录一次仅由 alignment 驱动的 optimizer step 前后，raw DINO token、real post-FPN feature 和 raster post-FPN feature 的变化量，见实验 0-F。

### 4.4 Global-level GRL 梯度

第二种作用力是域对抗损失（`agent_lightning_module.py:226-244`，权重 0.001）：

- `feat = image_feature[0][[1]]` 只取相机下标 1；相机顺序为 `[B0,F0,L0,R0]`，因此它只使用 `CAM_F0`（`rap_model.py:147-152`；相机顺序见 `bev_feature_build.py:27-28`）。
- `feat[:, :, :batch_size].detach()` 切断 raster encoder 分支；后半段 real feature 没有 detach。代码变量名 `feat_grad`/`feat_no_grad` 与真实梯度状态相反，不能按变量名理解。
- GRL backward 将流向 real projector 的梯度乘以 \(-\lambda\)（`rap_model.py:17-28`），因此 DomainClassifier 正常最小化域分类 BCE，而 real projector 被训练为让域分类更困难。
- DomainClassifier 参数仍同时从 real/raster 样本得到正常 BCE 梯度；detach 只阻止梯度继续进入 raster 特征生成路径。

### 4.5 完整训练梯度不能只看 alignment

总损失可概括为：

\[
L_{total}=L_{planning}+0.002L_{MSE}+0.001L_{domain}
\]

必须区分三条路径：

| 路径 | Real encoder/projector | Raster encoder/projector | 其他直接更新模块 |
|---|---|---|---|
| Planning loss | 有梯度 | 有梯度 | trajectory refiner、scorer 等 |
| Spatial MSE | 有梯度，四个相机 | 无，detach | 无 |
| Domain + GRL | 有反向梯度，仅 F0 | 无，detach | DomainClassifier 正常 BCE 梯度 |

因此“raster 没有梯度”只适用于两条 alignment encoder 路径；raster 样本仍通过 `L_planning` 更新共享 FPN 和规划网络。任何观测到的语义抹除/后验注入也不能仅归因于 MSE，至少要同时考虑 GRL 和 planning loss。

## 5. Q4——规划器输入：严格逐帧，无历史信息

- 推理时，规划器接收：通过可变形交叉注意力得到的图像特征（`SpatialCrossAttention` → `MSDeformableAttention3D`，`navsim/agents/rap_dino/bevformer/bev_refiner.py:66-77`），以及**仅当前帧**的 11 维自车状态——`features["ego_status"][:,-1]`（`rap_model.py:110`），该状态由位姿(3)+速度(2)+加速度(2)+驾驶指令(4)组成（`navsim/agents/rap_dino/rap_features.py:44-57`）。
- **自车历史和图像历史都不会传入规划器。** 自车状态先按历史堆叠，随后由 `[:,-1]` 只取最后一帧；相机输入在 feature builder 中已经通过 `agent_input.cameras[-1:]` 只保留当前帧（`bev_feature_build.py:27`）。
- 推理时**不使用栅格**：`real_features` 在替换局部 `features["camera_feature"]` 之前已经构造，实际传入 `forward` 的是 `real_features`（`agent_lightning_module.py:442-455`）。
- 注意力是可变形注意力（采样位置 + 权重），而非稠密注意力图；后续探查应 hook sampling locations/attention weights，而不是寻找稠密注意力图。
- 骨干网络是 **DINOv3**（`facebook/dinov3-vith16plus-pretrain-lvd1689m`），而非 DINOv2（`image_encoder.py:30,88`）。

### 对时序假设（Ball T）的影响

历史帧打乱/移除实验在 RAP-DINO 中**没有可作用的目标**：该架构从设计上就是逐帧处理的。“RAP 缺少时序状态”是对架构的描述，而不是某条时序路径中可度量的故障。检验时序假设需要使用其他模型或修改架构——这超出了冻结检查点诊断的范围。

## 6. 对步骤 1 / 步骤 2 设计的影响

1. **探查层**（步骤 1）：原始 DINOv3 token（参数冻结，不会被训练更新）、FPN/embedding 后的 projected feature（受到规划损失 + MSE 对齐 + GRL 的共同塑造）、轨迹解码器之前的规划器内部状态。图中的 “Feature Projector” 在当前实现中应操作化为“FPN neck 为主体，加上 camera/level embedding”，而不是寻找一个不存在的独立 projector 类。
2. **机制 B 的目标**（步骤 2）：被环境遮挡的已标注智能体（栅格可见、图像不可见）、显式地图先验与交通信号灯状态。三类内容都可通过 metadata + 几何信息枚举，但需要不同的诊断 mask。agent 可见性标签必须区分智能体之间的相互遮挡（栅格中也会发生）和环境遮挡（只发生在 real 中）；map/traffic-light 则必须额外区分“metadata 存在但 renderer 不支持”“投影后零面积”和“被后绘制元素覆盖”。
3. **机制 A 的目标**（步骤 2）：天气/光照/纹理与真实道路外观，以及因角点少于 4 个而被丢弃的智能体、metadata 存在但 renderer 不支持或投影后为零面积的 map/light——这些是栅格侧的盲点。
4. **归因注意事项：** MSE 与 GRL 都沿真实图像→栅格的方向作用；明确归因需要进行无对齐重训练（7/23 之后）；在此之前，应将两者作为共同作用力报告。
5. **像素未对齐注意事项：** 虚拟相机存在 0.8 m/2 m 的偏移，因此逐 token 位置的 MSE 比较的是略有不同的视角；这种对齐在空间上本就具有“软”特性。值得在 7/23 的演示文稿中用一句话说明。
6. **共享 projector 注意事项：** raster 只在当前 step 是 stop-gradient target；共享 FPN 更新后，两域下一次 forward 的 projected feature 都可能移动。若 Step 1 比较 checkpoint 间的 feature 演化，应同时报告 real 与 raster 两侧的漂移，而不能默认 raster 表示恒定。

## 7. Step 0 实验协议

### 7.1 总体原则

Step 0 不是性能 benchmark，而是为 Step 1/2 固定信息边界和梯度边界。所有实验应遵守以下原则：

1. **固定数据清单。** 首次运行生成 `manifest.jsonl.gz`，记录 split、log、frame token、camera、真实图像路径、raster 路径和文件哈希；后续所有图表只使用同一清单。
2. **区分三种坐标尺度。** 同时记录 renderer 原始画布 1920×1120、训练裁剪后 1920×1080、模型输入 768×448；面积指标至少报告原始和模型输入两种尺度。
3. **确定性。** 固定随机种子；几何/渲染审计不启用 GridMask 或 photometric augmentation；保存 Python、OpenCV、NumPy 和 PyTorch 版本。
4. **分母显式。** 所有比例必须写明分母，例如“所有标注 agent”“几何上有可见面积的 agent”或“通过生产 `<4 corner` 规则的 agent”。
5. **不把自动规则当真实可见性。** ID-pass 能测 raster 内 agent-agent 遮挡，不能自动判断真实图像中的建筑/植被遮挡；后者必须人工标注或使用单独验证过的可见性模型。
6. **组件分离但不假定可加。** agent、map、traffic-light 的整数 mask 和 component-only/full-minus-component raster 必须使用同一虚拟相机、裁剪和预处理；DINO/FPN feature 与梯度按消融差异解释，不把三个组件的结果相加冒充 full raster。
7. **先 smoke 后全量。** 先在 32 帧 × 3 个相机上通过一致性检查，再扩展到主扫描集。建议几何/内容主扫描至少覆盖 10,000 帧，并按城市、昼夜/天气（若元数据可用）、相机和距离分层；需要 checkpoint 的 feature/梯度组件消融使用固定的分层子集，不要求对 10,000 帧全部 backward。

建议统一输出目录：

```text
outputs/step0_alignment_audit/
├── 0a_integrity/
│   ├── manifest.jsonl.gz
│   ├── config.json
│   ├── environment.txt
│   ├── integrity.csv
│   ├── cache_tensor_sample.csv
│   ├── pixel_delta.csv
│   └── evidence/
├── 0b_panels/
├── 0c_raster_components/
│   ├── masks/
│   ├── panels/
│   │   ├── overview/
│   │   ├── agent_fov_drop/
│   │   ├── agent_occlusion/
│   │   ├── map_unsupported/
│   │   ├── map_overwritten/
│   │   ├── traffic_light_overwritten/
│   │   └── component_ablation/
│   ├── ablations/
│   │   ├── component_only/
│   │   ├── leave_one_out/
│   │   └── token_heatmaps/
│   └── visualization_index.csv
├── 0d_field_interventions/
├── 0e_camera_offset/
└── 0f_gradient_route/
```

### 7.2 0-A：数据完整性与零扰动可复现性

#### 目的

确认训练实际读取到的是预期 raster，而不是缺失文件触发的静默零图；同时确认诊断重渲染与磁盘训练 raster 使用同一几何和绘制逻辑。

#### 本实验的绝对输入路径

0-A 只审计以下 norm 数据，不能用其他 workspace 数据根目录代替：

```text
norm metadata:      /gs/bs/tga-RLA/qdeng/RAP/dataset_norm/navsim_logs/mini
norm raster:        /gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs
```

norm metadata 的 `cams[*].data_path` 是相对于 raster image root 的路径。落盘目录按原始 log 命名；当前 norm raster 只包含 F0/L0/R0。

#### A1. Raster 路径完整性

对 manifest 中每个 frame-camera 记录：

```text
split, log, frame_token, camera,
real_path, raster_path,
real_exists, raster_exists,
raster_shape, raster_dtype,
all_zero, nonzero_fraction, sha256
```

统计要求：

- `CAM_F0/L0/R0` 分别报告缺失率和全零率；正常训练集理想目标应为 0；
- `CAM_B0` 单独报告，因为标准 renderer 默认不生成 B0，不能与前三个相机混为数据缺失；
- 同时确认训练使用的是标准 raster 还是包含 B0 的 augmented raster；如果 B0 确实为空白 target，应将其列为进入 Step 1/2 前必须解释或修正的阻断项；
- 将“文件缺失后 loader 生成零图”和“磁盘上真实存在的全零 PNG”分开计数；
- 随机抽查 loader 输出，验证 `[20:-20]` 裁剪后的 shape 确实为 1080×1920。

#### A2. 零扰动重渲染

从 manifest 固定抽取至少 100 帧，使用原始 metadata 和当前 `ScenarioRenderer` 重渲染。比较时必须走与训练文件相同的颜色通道、PNG 保存/读取和 `[20:-20]` 裁剪路径。

每个 frame-camera 输出：

```text
exact_equal,
different_pixel_count,
different_pixel_fraction,
mean_abs_delta,
max_abs_delta,
psnr,
foreground_iou
```

同时保存：磁盘 raster、重渲染 raster、绝对差分热图。若无法逐像素完全一致，优先按以下顺序排查：metadata 版本、renderer commit、RGB/BGR、裁剪范围、OpenCV 抗锯齿版本、相机列表和保存压缩，而不是直接用高 PSNR 宣称等价。

#### 验收条件

- F0/L0/R0 没有未解释的静默零图；
- 任意不一致都能归因到明确的 pipeline 差异；
- 生成 `integrity.csv`、`pixel_delta.csv` 和 `summary.md`。

#### 0-A 正式结果（2026-07-17）

审计脚本为 [`tools/audit_step0_0a.py`](../tools/audit_step0_0a.py)，norm 结果见 [`summary.md`](../outputs/step0_alignment_audit/0a_norm/summary.md) 和 [`summary.json`](../outputs/step0_alignment_audit/0a_norm/summary.json)。随机种子为 `20260717`。路径存在性覆盖全部 51,867 帧；每个 required camera 蓄水池抽样 256 张做解码、全零、shape、裁剪和 SHA256 检查；抽取 100 帧，经过与生成端一致的 OpenCV JPEG round-trip 和 loader `[20:-20]` 裁剪后逐像素比较。

| 数据 | metadata | 有效 raster 分母 | 路径缺失 | 0-byte | 100 帧重渲染 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| norm | 64/64 pkl 可读 | 51,867 × F0/L0/R0 | 0 | 0 | 300/300 camera rows exact | `PASS_WITH_WARNINGS` |

抽样中发现少量可解码的全黑侧视图（L0 1/256、R0 6/256，F0 0/256）。全量文件大小扫描确认 F0/L0/R0 均无 0-byte 文件，且零扰动重渲染逐像素完全复现，因此这些是 renderer 的合法空视野，记为 warning，而不是 loader 缺失回退。

因此在本文的 norm-only 范围内，**0-A 结论为 `PASS_WITH_WARNINGS`，可以继续执行 0-F**。warning 不来自路径损坏，而来自少量合法全黑侧视图以及 norm 不生成 B0；0-F 必须单独报告 B0 的 alignment 梯度贡献。

### 7.3 0-B：定性信息边界图组

#### 目的

用真实图像、生产 raster 和几何诊断 mask 并排证明以下边界案例：

1. raster 可见、real 不可见的环境遮挡 agent；
2. agent-agent 遮挡在 raster 中按远到近覆盖；
3. `<4 valid corners` 导致的整车 FOV drop；
4. 夜间、雨天、眩光、远距离等 real 成像退化但 raster 仍清晰的案例；
5. raster 盲点或 image-private 内容的反例。
6. map 先验在 raster 中清晰存在，但 real 中受透视、磨损、光照或遮挡影响而不清晰的案例，以及 metadata 存在但 renderer 不支持/未产生像素的反例；
7. traffic light 成功投影、被 map/agent 覆盖、或与 real 可见性/灯态难以对应的案例。

#### 候选挖掘

- 环境遮挡候选：选择几何投影面积足够大、0-C 中 raster visibility 较高、且没有被更近标注 agent 大幅覆盖的 agent；再由人工检查真实图像。
- agent-agent 遮挡候选：按 0-C 的 `1 - visible_area / isolated_area` 排序。
- FOV drop 候选：生产规则面积为 0，但严格几何 clipping 后仍有非零面积；按模型输入尺度下可见面积排序。
- map 候选：分别从 0-C 的 `unsupported_by_renderer`、`zero_area_after_rasterization`、高 `map_overwrite_fraction`、高 `agent_overwrite_fraction` 和高最终 coverage 中分层抽样。
- traffic-light 候选：分别从成功投影、被 map 完全覆盖、被 agent 完全覆盖及模型尺度小于 4/16 像素的实例中分层抽样。
- 成像退化候选：按距离和场景条件分层采样，不应只挑选最显眼的成功例。

#### Panel 规范

每个 panel 至少包含：

```text
真实整图 | raster 整图 | 真实局部 crop | raster/ID-mask 局部 crop
```

agent panel 标注 frame token、camera、track token、类别、距离、8 角点有效数、isolated/visible area、raster visibility 和人工 real-visibility 标签；map/light panel 改为标注 feature/connector ID、raw type/灯态、各绘制阶段面积和 overwrite 来源。真实图像 crop 应使用真实标定投影框；raster crop 使用虚拟渲染相机投影框，不能因为两相机存在偏移而强行共用同一个像素框。

#### 人工标签

建议标签集合：

```text
clear_visible
degraded_visible
environment_occluded
annotated_agent_occluded
outside_or_projection_mismatch
uncertain
```

上述集合用于 agent。map/traffic-light 不强行套用 agent 遮挡标签，而是在同一 `labels.csv` 中增加通用字段：

```text
element_type, element_id,
real_visibility, raster_presence_stage,
semantic_consistency, projection_quality,
reviewer_confidence
```

其中 `raster_presence_stage` 至少区分 `isolated_only`、`after_component`、`final_visible`、`not_rendered`；`semantic_consistency` 至少区分 `consistent`、`conflict`、`not_judgable`。traffic-light 灯态在 real 中不可可靠辨认时必须标为 `not_judgable`，不能按 raster 颜色反推 real 标签。

至少两名标注者独立标注 200 个分层抽样候选，报告一致率/Cohen's kappa，并对分歧样本复核。定性图组可从已确认类别中各选 5～10 个代表例，但频率统计必须来自完整人工样本，不能由精选图片推断。

#### 输出

- `labels.csv`：人工标签与几何指标；
- `panels/*.png`：固定布局的证据图；
- `panel_index.html` 或 `contact_sheet.png`：快速浏览；
- `summary.md`：每类案例数、比例及 95% bootstrap CI。

### 7.4 0-C：Raster 组件审计——agent、map 与 traffic lights

#### 目的与范围

0-C 不再只回答“哪些 agent 存在、被遮挡或被错误删除”，而是回答以下四组问题：

1. metadata 中的 agent、map feature 和 traffic light 各有多少真正进入了生产 raster；
2. 三类元素在 `traffic_lights → map → agents` 的逐层合成中被谁覆盖、最终还剩多少像素；
3. 每类元素在 renderer 原始尺度和模型输入尺度占据多少面积/token；
4. 移除或单独保留某类元素后，raster target、real/raster MSE 及其 FPN 梯度如何变化。

几何/内容主统计只使用 norm 的 `CAM_F0/L0/R0`。`CAM_B0` 没有 norm raster，不进入组件覆盖率和消融主统计；其空白 fallback 风险继续由 0-A/0-F 单独报告。

#### C0. 生产等价的分阶段 ownership pass

最终 RGB 颜色不能可靠标识实例或图层。实现只用于诊断的整型 ownership buffer，不修改磁盘训练 raster，并同时保存三个绘制阶段：

```text
after_traffic_lights
after_map
final_after_agents
```

要求：

- agent、map feature、traffic light 分别分配唯一 ID，并保留 `component_type` 与语义子类；
- RGB 路径必须保持生产 renderer 的相机偏移、几何、深度/迭代排序、线宽、颜色衰减和抗锯齿；
- 整型 mask 使用相同投影与绘制顺序，但关闭抗锯齿，避免 ID 混色；
- 对每个实例/要素另渲染 isolated mask，再与分阶段 composite ownership mask 比较；
- mask 从 1920×1120 变换到裁剪/模型尺度时只用 nearest-neighbor；RGB 仍走训练的裁剪、缩放和 padding；
- 全组件诊断 RGB 必须先在 0-A 的 100 帧样本上与磁盘 raster 做逐像素复现，ownership 统计只能在复现通过后运行。

所有元素表同时保留多列布尔 `drop/overwrite flags`；单值 `drop_reason` 仅作为便于分组的 primary reason，按“renderer 不支持 → 几何不可投影/超距 → rasterization 零面积 → 同组件覆盖 → 后续组件覆盖 → not_dropped”的固定优先级生成，避免一个元素同时满足多个条件时被重复或随意归类。

绘制覆盖不一律称为“物理遮挡”：agent 内部的远近覆盖称为 occlusion；map feature 之间以及 map 对 traffic light 的覆盖由代码迭代顺序产生，应报告为 overwrite。尤其 traffic light 先画、map 后画、agent 最后画，因此 map 和 agent 都可能覆盖信号灯，agent 也可能覆盖地图。

每个 frame-camera-scale 另写一行 `component_coverage.csv`：

```text
split, log, frame_token, camera, scale, total_pixels,
traffic_light_component_only_pixels, map_component_only_pixels,
agent_component_only_pixels,
final_traffic_light_owned_pixels, final_map_owned_pixels,
final_agent_owned_pixels, final_blank_pixels,
traffic_light_coverage, map_coverage, agent_coverage, blank_fraction
```

`final_*_owned_pixels` 来自互斥 ownership mask，所以三类 final coverage 与 blank 相加必须为 1；`component_only_pixels` 之间可以空间重叠，不能相加当作 full foreground。

#### C1. Agent 实例可见性与 FOV drop

保留原有 agent ID-pass。每个 agent 使用与生产 renderer 相同的 cuboid 面、全局面深度排序和 `<4 valid corners` 规则，并定义：

- `isolated_area_prod`：agent 单独按生产规则渲染的面积；
- `visible_area_prod`：`final_after_agents` 中最终属于该 agent 的面积；
- `isolated_area_clip`：取消 `<4 corner` 整体丢弃，对 cuboid 面进行近裁剪面和图像边界严格 clipping 后的几何可见面积；
- `valid_corner_count`：生产 renderer 中 8 个角点的有效数量。

\[
raster\_visibility
=
\frac{visible\_area\_prod}{\max(isolated\_area\_prod,1)},
\qquad
agent\_occlusion\_fraction=1-raster\_visibility
\]

FOV 整体丢弃定义为：

```text
isolated_area_prod == 0
and isolated_area_clip >= area_threshold
```

在模型输入尺度同时报告 `area_threshold ∈ {4,16,64}` 像素，不能只选择一个阈值。`agent_raster_visibility.csv` 至少包含：

```text
split, log, frame_token, camera,
agent_index, instance_token, track_token, class,
center_x, center_y, center_z, distance,
valid_corner_count, drop_reason,
isolated_area_prod_render, visible_area_prod_render,
isolated_area_clip_render,
isolated_area_prod_model, visible_area_prod_model,
isolated_area_clip_model,
raster_visibility, agent_occlusion_fraction
```

`drop_reason` 至少区分 `behind_camera`、`valid_corner_lt4`、`no_face_with_valid_vertex`、`zero_area_after_rasterization` 和 `not_dropped`。

#### C2. Map feature/layer pass

不能把 `scenario["map_features"]` 中的所有条目都当作已画入 raster。先按生产 `ftype` 分支分类：

```text
lane
crosswalk_or_speed_bump
boundary_or_solid
unsupported_by_renderer
```

每个 map feature 单独按生产逻辑渲染，并在保持原始 metadata 迭代顺序的 composite pass 中记录：

- `metadata_present`：metadata 是否存在该要素；
- `renderer_supported`：是否命中当前 renderer 的三个分支之一；
- `isolated_area`：单独绘制产生的像素面积；
- `visible_after_map`：所有 map feature 绘制完成后仍由该 feature 拥有的面积；
- `visible_final`：agent 最后绘制后仍由该 feature 拥有的面积；
- `map_overwrite_fraction = 1 - visible_after_map / max(isolated_area, 1)`；
- `agent_overwrite_fraction = (visible_after_map - visible_final) / max(visible_after_map, 1)`。

`map_element_visibility.csv` 至少包含：

```text
split, log, frame_token, camera,
map_feature_id, raw_ftype, map_category,
renderer_supported, distance_min, geometry_kind,
isolated_area_render, visible_after_map_render, visible_final_render,
isolated_area_model, visible_after_map_model, visible_final_model,
map_overwrite_fraction, agent_overwrite_fraction, drop_reason
```

map 的 `drop_reason` 至少区分 `unsupported_by_renderer`、`beyond_depth_max`、`behind_camera`、`outside_fov_after_clipping`、`zero_area_after_rasterization`、`fully_overwritten_by_map`、`fully_overwritten_by_agent` 和 `not_dropped`。按 `map_category × camera` 报告 metadata 数、支持数、非零 isolated 数、最终非零数和 union pixel/token coverage。黑色未占用区域单独报告 `blank_fraction`，但不得称为“背景准确率”。

#### C3. Traffic-light instance pass

每个 `scenario["traffic_lights"]` 条目使用生产代码中的位置、固定 `dims=(0.5,0.5,1.0)`、`z_base=5` 和红/非红颜色单独渲染。用 lane connector ID（若存在）和场景内索引标识实例，记录：

- `isolated_area`：单灯绘制面积；
- `visible_after_lights`：所有灯绘制后的剩余面积；
- `visible_after_map`：地图绘制后的剩余面积；
- `visible_final`：agent 绘制后的最终剩余面积；
- light-light、map、agent 三阶段各自的 overwrite fraction；
- 红/非红状态、距离、投影有效角点数和零面积原因。

`traffic_light_visibility.csv` 至少包含：

```text
split, log, frame_token, camera,
traffic_light_index, lane_connector_id, is_red, distance,
valid_corner_count, isolated_area_render,
visible_after_lights_render, visible_after_map_render, visible_final_render,
isolated_area_model, visible_after_lights_model,
visible_after_map_model, visible_final_model,
light_overwrite_fraction, map_overwrite_fraction,
agent_overwrite_fraction, drop_reason
```

按 camera、状态和距离分箱报告：metadata 中灯的数量、投影非零比例、最终非零比例、完全被 map/agent 覆盖比例以及模型尺度面积分布。由于 renderer 只编码红与“非红”两种颜色，统计中不能把 `is_red=False` 自动命名为经过验证的绿色真值；灯态语义正确性仍由 0-D 的红↔非红受控干预和 metadata 检查确认。

#### C4. 组件 raster 的 feature/MSE/梯度消融

几何 coverage 不能说明哪一类元素主导 feature alignment。固定 0-F 使用的 checkpoint、确定性预处理和 norm 配对样本，为每个 frame-camera 生成以下 raster 变体：

```text
full
blank
agent_only
map_only
traffic_light_only
full_without_agents
full_without_map
full_without_traffic_lights
```

`full` 必须与磁盘生产 raster exact equal；所有变体只改变绘制组件，不能改变相机、裁剪、resize、padding 或归一化。forward 分层子集建议至少 256 帧 × F0/L0/R0；梯度子集建议至少 32 帧并按 log/camera 分层。若资源不足可先做 32 帧 smoke，但不能把 smoke 的均值当作总体频率。

主 MSE 明确只在 F0/L0/R0 上求均值，B0 不参与：

\[
L^{3cam}_{MSE}=\frac{1}{3}\sum_{k\in\{F0,L0,R0\}}MSE(F^{real}_k,\operatorname{stopgrad}(F^{raster}_k))
\]

每个变体要做两种作用范围：`joint_3cam` 同时替换三个相机的 raster；`single_camera` 只替换一个相机并单独对该 camera slice 构造 MSE，用于可比较的 per-camera 梯度。不能一边把 B0 空白 fallback 混入平均，一边把结果称作 norm 三组件贡献。

对每个变体分别记录：

```text
foreground_fraction_render, foreground_fraction_model,
raw_dino_distance_to_full, post_fpn_distance_to_full,
raw_mse_real_to_variant, weighted_mse_real_to_variant,
lateral_conv_grad_norm, fpn_conv_grad_norm,
grad_cosine_vs_full_mse, grad_cosine_vs_planning
```

其中 MSE/梯度必须按 F0/L0/R0 分别报告。对每个组件 (c)，leave-one-out 结果额外计算：

\[
\Delta L_c=L_{MSE}(real,R_{-c})-L_{MSE}(real,R_{full})
\]

\[
\Delta g_c=g_{-c}-g_{full}
\]

并报告 `||Δg_c||` 及其与 full-MSE/planning 梯度的余弦。这里的正负只表示“移除该组件后，当前样本的 real/raster feature 距离变大或变小”，不能直接解释为 planner 性能改善或下降。`agent_only/map_only/traffic_light_only` 与 leave-one-out 是互补消融；由于 DINO/FPN 非线性、组件存在覆盖，严禁把三个 component-only MSE 或梯度相加。

组件级结果写入 `component_ablation.csv` 和 `component_gradient.csv`。若 planning probe 仍使用 0-F 的简化 trajectory component，必须原样继承并报告其限制，不能称为完整 PDM planning loss。

#### C5. 聚合与 bootstrap

- agent 按 camera、class、距离 `[0,10), [10,20), [20,40), [40,80), [80,120)` 米分箱；map 按 category/camera；traffic light 按状态/camera/距离分箱；
- 所有实例比例同时给出分子、分母和有效样本数；面积同时报告 renderer/model 两个尺度；
- component coverage 至少报告均值、中位数、P10/P25/P75/P90；
- 用 log 级 bootstrap 计算 95% CI，避免相邻帧被当成独立样本；

#### C6. 基于 camera image 的可视化

可视化用于把 C1～C4 的数字定位回真实 camera image，但不能把 raster ownership mask 当作真实图像语义标签。F0/L0/R0 每个 panel 必须同时标明 frame token、camera、real/raster 投影模式、元素 ID、距离、面积和 drop/overwrite 原因。

##### C6.1 双坐标系几何 overlay

同一张真实 camera image 上允许同时绘制两套轮廓，但必须使用不同线型并给出图例：

```text
实线：真实相机标定投影（real-calibrated）
虚线：带 +0.8 m / -2 m 偏移的 renderer 虚拟相机投影（raster-shifted）
```

agent 绘制 3D box 或可见面轮廓；map 绘制 polygon/polyline；traffic light 绘制投影 cuboid/中心点。不得把 raster-shifted mask 直接 alpha-blend 到 real image 后称为 pixel alignment ground truth。两套投影并列的目的是解释 FOV 进入/退出、质心偏移和局部 crop 差异；相机偏移的总体定量仍由 0-E 完成。

元素状态使用固定颜色：

```text
green   final_visible
yellow  partially_overwritten
red     metadata_present_but_final_missing
purple  unsupported_by_renderer
gray    outside_fov_or_behind_camera
```

agent overlay 显示 `class/track_token/distance/valid_corner_count/raster_visibility`；map overlay 显示 `raw_ftype/map_category/supported/isolated_area/visible_final`；traffic-light overlay 显示 `lane_connector_id/is_red/distance/visible_after_map/visible_final`。`is_red=False` 在图例中写作 `non-red`，除非人工能够从 real crop 验证真实灯态。

##### C6.2 分阶段 ownership 与覆盖来源

每个代表样本并排显示：

```text
real camera
| after_traffic_lights
| after_map
| final_after_agents
| overwrite-source heatmap
```

ownership 固定配色为 traffic light=红/绿、map=蓝、agent=橙、blank=黑；overwrite-source heatmap 至少区分：

```text
light_overwritten_by_light
light_overwritten_by_map
light_overwritten_by_agent
map_overwritten_by_map
map_overwritten_by_agent
agent_occluded_by_agent
```

颜色只编码诊断类别，不复用生产 RGB 的深度亮度，且必须附 legend。局部元素 panel 另显示 `real crop | calibrated projection | raster crop | isolated mask | final ownership`。real crop 与 raster crop 分别从各自投影框加相同相对 margin 得到，不能强行共用同一像素框。

##### C6.3 Component-only 与 leave-one-out panel

对 agent、map、traffic light 各生成固定布局：

```text
real camera
| full raster
| component_only
| full_without_component
| abs(full - full_without_component)
```

像素差分图使用统一色标，并在标题显示变更像素数、foreground fraction、raw/post-FPN distance、MSE before/after 和对应梯度范数/余弦。对于删除上层组件后重新显露的下层元素，panel 中必须明确标为 `revealed_by_removal`，不能把它误算成被删除组件自身的像素。

##### C6.4 Token/feature/gradient heatmap

在模型 token 网格上至少生成三类热图，再以 nearest 或明确标注的插值方式上采样到 768×448，并半透明叠加在 real camera image 上：

\[
H_{real-full}(p)=\left\|F^{real}_p-F^{full}_p\right\|_2
\]

\[
H_{remove-c}(p)=\left\|F^{full}_p-F^{without-c}_p\right\|_2
\]

\[
H_{grad}(p)=\left\|\frac{\partial L_{MSE}}{\partial F^{real}_p}\right\|_2
\]

分别输出 agent/map/traffic-light 的 `H_remove-c`，并为所有 panel 保存未上色的 float array（`.npy`）和统一 percentile 色标范围，避免每张图自动拉伸造成强度不可比较。heatmap 表示模型 token 的响应位置；受虚拟相机偏移、DINO 自注意力和大感受野影响，它不能解释为精确的物体像素归因，也不能替代 planner attribution。

##### C6.5 标准 panel、抽样与索引

overview panel 使用三行固定布局：

```text
row 1: real raw | real-calibrated overlay | raster-shifted overlay | full raster
row 2: after-light ownership | after-map ownership | final ownership | overwrite heatmap
row 3: agent ablation | map ablation | light ablation | MSE/gradient token heatmap
```

除按异常指标排序的 top-k 外，每个 camera/category 还必须保存同样数量的分层随机 control panels，防止只展示极端成功/失败案例。所有图片写入 `visualization_index.csv`：

```text
split, log, frame_token, camera, panel_type,
element_type, element_id, selection_rule, rank,
real_path, raster_path, panel_path, heatmap_npy_path
```

同时生成 `panel_index.html` 或 contact sheets，支持按 camera、component、drop reason、overwrite reason 和 selection rule 浏览。

#### 验收条件

- 全组件 RGB 在 0-A 固定样本上逐像素复现生产 raster；
- agent isolated/composite mask 与生产 cuboid 边界吻合，并重现远 agent 被近 agent 覆盖的排序；
- map 与 traffic-light ownership pass 重现生产绘制顺序，能够区分 metadata 未支持、几何零面积和被后续组件覆盖；
- `blank_fraction`、各 component union coverage 和最终 ownership coverage 的分母明确，且 final ownership 互斥像素加 blank 后覆盖完整画布；
- feature/梯度消融使用同一 checkpoint、real 输入与预处理，关闭随机增强，只报告 F0/L0/R0 主结果；
- 双投影 overlay 明确区分 real-calibrated 与 raster-shifted，局部 crop 不跨坐标系复用；ownership/overwrite/heatmap 使用固定 legend 与跨样本统一色标；
- top-k 异常 panel 同时配套分层随机 control，且 `visualization_index.csv` 能追溯到原始 real/raster、选择规则和数值记录；
- 生成 `agent_raster_visibility.csv`、`map_element_visibility.csv`、`traffic_light_visibility.csv`、`component_coverage.csv`、`component_ablation.csv`、`component_gradient.csv`、`visualization_index.csv`、证据 panels、token heatmap arrays 和 `summary.md`；
- 0-C 只给出 raster target 内容与梯度归因，不据此宣称 MSE 已改善或损害 planner 性能；性能因果结论仍需要后续受控训练/评估消融。

#### 实际执行结果（2026-07-19）

审计脚本为 [`tools/audit_step0_0c_geometry.py`](../tools/audit_step0_0c_geometry.py)、[`tools/audit_step0_0c_feature.py`](../tools/audit_step0_0c_feature.py) 和 [`tools/audit_step0_0c_overlays.py`](../tools/audit_step0_0c_overlays.py)。正式输出见 [`summary.md`](../outputs/step0_alignment_audit/0c_raster_components/summary.md)、[`geometry_summary.md`](../outputs/step0_alignment_audit/0c_raster_components/geometry_summary.md)、[`feature_summary.md`](../outputs/step0_alignment_audit/0c_raster_components/feature_summary.md) 和 [`panel_index.html`](../outputs/step0_alignment_audit/0c_raster_components/panel_index.html)。

几何主扫描从 0-A 固定 manifest 以种子 `20260717` 抽取 10,000 帧，覆盖全部 64 个 log 和 `CAM_F0/L0/R0`，共 30,000 个 frame-camera。16 个 shard 各完成 625/625 帧，错误数为 0。用于 production 等价检查的 336 个 camera rows 在 JPEG round-trip 与 loader crop 后全部逐像素相等；render/model 两个尺度的最终 ownership 加 blank 均完整覆盖画布，失败数为 0。因此 0-C 的几何/ownership 边界判定为 **`PASS`**。

| 对象 | metadata rows | renderer 支持 | isolated 非零 | final 非零 |
|---|---:|---:|---:|---:|
| agent-camera | 3,088,599 | 不适用 | 675,608 | 608,888 |
| map-element-camera | 12,560,592 | 8,605,128（68.51%） | 1,296,037 | 760,148 |
| traffic-light-camera | 259,686 | 259,686 | 73,398 | 70,801 |

map 中有 410,299 个要素在 map composite 后被其他 map 要素完全 overwrite，另有 125,590 个在 agent 层之后完全不可见。traffic light 中有 717 个在最终 agent 层之后完全不可见。这里的计数按 element-camera 计算，不表示独立物理实例数。

`<4 valid corners` 不是只存在于代码里的边缘条件。以严格 clipping 后模型尺度面积至少 16 px 的 agent-camera 为分母，25,413 / 700,994（3.63%）被 production 整体丢弃；近距离最严重：

| 距离 | FOV 整体丢弃 | clipping 后面积 ≥16 px | 丢弃率 |
|---|---:|---:|---:|
| `[0,10)` m | 11,498 | 40,508 | 28.38% |
| `[10,20)` m | 5,696 | 101,843 | 5.59% |
| `[20,40)` m | 4,690 | 205,339 | 2.28% |
| `[40,80)` m | 3,515 | 351,869 | 1.00% |
| `[80,120)` m | 14 | 1,435 | 0.98% |

三档面积阈值的总体结论稳定：4/16/64 px 下分别为 25,761/701,355（3.67%）、25,413/700,994（3.63%）、23,976/651,215（3.68%）。这证明 production 的角点门槛会系统性删除真实几何上仍有可见面积的 agent，且主要集中在 20 m 内的大目标/FOV 边缘案例。

最终互斥 ownership coverage 如下；区间为 log-level bootstrap 95% CI：

| 尺度 | traffic light | map | agent | blank |
|---|---:|---:|---:|---:|
| renderer 1920×1120 | 0.2088% `[0.1798%,0.2383%]` | 4.3415% `[3.9875%,4.7452%]` | 11.8507% `[10.8404%,12.7872%]` | 83.5990% `[82.7322%,84.5281%]` |
| model 768×448 | 0.2050% `[0.1768%,0.2342%]` | 4.2838% `[3.9385%,4.6746%]` | 11.6735% `[10.6762%,12.5922%]` | 83.8377% `[82.9827%,84.7393%]` |
| token 48×28 | 0.2110% `[0.1814%,0.2409%]` | 4.1859% `[3.8574%,4.5654%]` | 11.5177% `[10.5303%,12.4243%]` | 84.0854% `[83.2314%,84.9835%]` |

C4 使用与 0-F 相同的 checkpoint `ckpts/RAP_DINO_navsimv2.ckpt`（SHA256 `9accbb101f30541187c7bb061689bf88f481b2125ad9a9c563a488275ea0311d`），在 H100/PyTorch 2.1.0+cu121 上关闭 GridMask。32 帧来自 32 个不同 log；real preprocessing 与 cache 的最大差值为 0。每帧生成 8 个 raster 变体，forward 和 gradient 子集均为 32 帧 × F0/L0/R0，B0 不进入任何主损失。

| 从 full 移除 | joint-3cam Δ raw MSE（95% CI） | `||Δg||` | `cos(g_variant,g_full)` | `cos(g_variant,g_planning probe)` |
|---|---:|---:|---:|---:|
| agent | +0.09883 `[+0.07937,+0.12099]` | 9.03e-4 | 0.9250 | 0.0321 |
| map | +0.12327 `[+0.09971,+0.14783]` | 9.86e-4 | 0.9277 | 0.0122 |
| traffic light | +0.007965 `[+0.004102,+0.012196]` | 1.77e-4 | 0.9892 | 0.0252 |

这个 checkpoint/子集上，map 和 agent 都对 feature target 与 MSE gradient 产生实质影响；traffic light 的平均影响更小，但非零。移除三类组件都使 real/raster raw MSE 增大；该正号只表示 full raster 更接近当前 real feature，不能外推为 planner 性能改善。三类 variant gradient 与 planning probe 的余弦均接近 0；planning probe 仍是 0-F 的简化 trajectory component，不包含 PDM-score auxiliaries。

可视化交付共 573 条索引记录：48 个分层随机 control overview、45 个按 agent FOV/occlusion、map unsupported/overwrite 和 traffic-light overwrite 选择的 top-k 双投影 panel，以及 480 个 token heatmap panel；同时保留 480 个未着色 `.npy`。双投影 panel 在同一真实图上用实线表示 real-calibrated、虚线表示 raster-shifted。最终 0-C 综合判定为 **`PASS`**，但结论边界仍是 raster target 内容与 alignment gradient 归因，不包含 planner 性能因果。

### 7.5 0-D：字段编码的受控干预

#### 目的

通过 one-factor-at-a-time 干预验证 Q2，而不只依赖静态代码阅读。对同一个固定 scenario 分别改变一个字段，比较 raster 像素差异。

| 干预 | 代码先验 | 预期结果 |
|---|---|---|
| 只交换 `gt_names` | renderer 不读取类别进行着色 | 像素差为 0 |
| 只改变 `gt_velocity_3d` | renderer 不读取速度 | 像素差为 0 |
| 只交换 instance/track token | renderer 不读取 token | 像素差为 0 |
| 只改变非当前历史帧 | renderer 只读当前 scenario | 当前 raster 像素差为 0 |
| 红灯切换绿灯 | 显式颜色编码 | 仅信号灯区域发生变化 |
| 改变 box 位置/尺寸/yaw | 显式几何编码 | 对应 cuboid 区域发生变化 |
| 平移一个受支持的 map polygon/polyline | 显式地图几何编码 | 对应地图局部发生变化；若与其他层重叠，需报告被揭示/覆盖像素 |
| 将一个受支持 `ftype` 改为不支持类型 | renderer 分支由字符串匹配决定 | 该 map feature 消失，底层元素可能重新显露 |
| 删除所有 traffic lights | 灯先于 map/agent 绘制 | 只有最终未被 map/agent 覆盖的灯像素必然变化；完全被覆盖的灯可为零差异 |
| 分别删除全部 agent/map/traffic lights | 验证组件边界与绘制顺序 | 与 0-C leave-one-out raster exact equal |

每种干预至少覆盖 20 个包含对应目标的场景，输出 `field_interventions.csv`，记录 `different_pixel_count`、差异 bounding box、PSNR 和是否符合预期。零差异项应要求 exact equality，而不是只要求“差异很小”。

### 7.6 0-E：虚拟相机偏移造成的空间错位

#### 目的

量化 `cam_t[2] += 0.8; cam_t[0] -= 2` 对逐 token MSE 的影响，判断同位置 token 实际对应多大几何错位。

#### 对照渲染

对同一 scenario 生成两套诊断 mask：

1. `shifted`：保持生产 renderer 的 +0.8 m / -2 m 偏移；
2. `calibrated`：去掉这两个偏移，使用标定外参。

优先比较语义/ID mask，而不是 RGB，因为深度亮度会随相机位置改变。每个 frame-camera 报告：

```text
foreground_iou,
boundary_chamfer_px,
agent_centroid_shift_px,
agent_box_iou,
token_shift = centroid_shift_model_px / 16
```

分别按距离和相机聚合。`token_shift` 将模型输入上的像素偏移换算成 DINO patch/token 尺度，可直接说明逐位置 MSE 比较的是同一物体的近邻 token，还是明显不同的空间位置。

#### 解释限制

该实验只量化渲染相机偏移，不等同于完整 real/raster pixel correspondence：真实图像还包含镜头畸变、环境几何、动态时间误差和渲染简化。

### 7.7 0-F：动态梯度路径与共享 anchor 漂移 smoke test

#### 目的

用真实 autograd 验证静态 Q3 结论，并区分中间张量有梯度与 optimizer 参数被更新。

#### F1. MSE-only backward

在 2～4 个配对样本上关闭 GridMask/随机增强，单独构造代码中的 MSE loss，执行一次 backward。记录：

```text
real post-FPN feature grad norm
raster pre-detach feature grad norm
DINO parameter grad norm
FPN parameter grad norm
cam/level embedding grad norm
trajectory refiner grad norm
scorer grad norm
```

预期：real feature、FPN 和 embedding 非零；raster slice、DINO、refiner、scorer 对 MSE-only 为零或 `None`。

另外必须按 `B0/F0/L0/R0` 分别记录未加权 MSE、乘 0.002 后的 feature gradient norm 以及占总 MSE 的比例。若 B0 raster 是空白输入，该拆分用于判断 B0 是否异常地主导 alignment。

#### F2. Domain-only backward

单独对 domain BCE backward，分别记录 raster/real F0 feature、DomainClassifier 和 FPN 的梯度范数；再临时令 \(\lambda=0\) 和 \(\lambda=1\) 做对照。预期：

- raster encoder slice 为零；
- real F0 梯度在 \(\lambda=0\) 时为零，在 \(\lambda=1\) 时非零；
- DomainClassifier 在两种 \(\lambda\) 下都有正常梯度；
- 使用 hook 验证 GRL 前后梯度余弦相似度接近 -1。

#### F3. Combined loss 梯度分解

分别对 `L_planning`、`0.002L_MSE`、`0.001L_domain` 使用 `torch.autograd.grad`，输出关键 FPN 层的：

```text
grad_norm_planning
grad_norm_mse_weighted
grad_norm_domain_weighted
cos(planning, mse)
cos(planning, domain)
cos(mse, domain)
```

仅看 loss 权重 0.002/0.001 无法判断实际作用强弱；加权梯度范数和夹角才能判断 alignment 是否被 planning 梯度淹没、协同或冲突。

#### F4. Shared-projector anchor drift

固定输入并缓存一次 forward 的：

```text
raw DINO real/raster tokens
post-FPN real/raster features
MSE(real, raster)
```

只用 alignment loss 做一个实际 optimizer step，然后在完全相同输入上重新 forward。报告：

```text
delta_raw_dino_real, delta_raw_dino_raster,
delta_post_fpn_real, delta_post_fpn_raster,
mse_before, mse_after
```

预期 raw DINO 差异为 0；real post-FPN 改变；由于 FPN 共享，raster post-FPN 也可能改变。该实验直接检验“当前 step stop-gradient target，但跨 step target 会移动”的结论。为排除 AdamW weight decay 与随机增强干扰，需要同时提供：

- 实际 AdamW 设置；
- 无 weight decay 的诊断 SGD/Adam 对照；
- 参数未更新但重复 forward 的确定性基线。

#### 输出与验收

- `gradient_norms.csv`、`gradient_cosines.csv`、`anchor_drift.csv`；
- `summary.md` 中明确写出每条 loss 的计算图边界；
- 任一结果与静态预期不符时，Step 1/2 暂停，先解释 autograd 路径。

#### 0-F 正式结果（2026-07-17）

审计脚本为 [`tools/audit_step0_0f.py`](../tools/audit_step0_0f.py)，正式输出见 [`summary.md`](../outputs/step0_alignment_audit/0f_gradient_route/summary.md)、[`gradient_norms.csv`](../outputs/step0_alignment_audit/0f_gradient_route/gradient_norms.csv)、[`gradient_cosines.csv`](../outputs/step0_alignment_audit/0f_gradient_route/gradient_cosines.csv) 和 [`anchor_drift.csv`](../outputs/step0_alignment_audit/0f_gradient_route/anchor_drift.csv)。实验使用 H100、PyTorch 2.1.0+cu121、固定随机种子 `20260717`，并关闭 GridMask。checkpoint 为 `ckpts/RAP_DINO_navsimv2.ckpt`（epoch 14、global step 21,930、SHA256 `9accbb101f30541187c7bb061689bf88f481b2125ad9a9c563a488275ea0311d`）。

现有历史 cache 的 real camera tensor 与从真实图像磁盘重建的预处理结果逐元素一致，但其中 rendered tensor 是四相机空白 fallback，不能用于 0-F。正式实验因此只复用 cache 中的标定张量与轨迹 target；real 输入从 `/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/mini` 重建，raster 输入从 `/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs` 重建。B0 缺失时保留生产 loader 的全零回退行为。

F1/F2 使用以下 2 个 norm 配对样本：

```text
2021.05.12.22.00.38_veh-35_01008_01518/b214f8e744075e96
2021.05.12.22.00.38_veh-35_01008_01518/ca7be5152b3a5466
```

**F1：MSE-only 路径符合静态预期。** raster pre-detach activation 梯度为 0，real post-FPN feature 梯度范数为 `1.125e-6`；FPN、camera embedding、level embedding 的梯度范数分别为 `2.819e-3`、`2.843e-5`、`3.799e-5`。冻结 DINO、trajectory refiner、scorer 和 DomainClassifier 均为 0/`None`。

| 相机 | 未加权 MSE | 总 per-camera MSE 占比 | `0.002*MSE` feature grad norm | raster 文件 |
|---|---:|---:|---:|---|
| B0 | 1.6144 | **37.09%** | 6.850e-7 | 不存在，零图回退 |
| F0 | 0.7298 | 16.77% | 4.606e-7 | 存在 |
| L0 | 0.9398 | 21.59% | 5.226e-7 | 存在 |
| R0 | 1.0685 | 24.55% | 5.573e-7 | 存在 |

B0 是四个相机中最大的 alignment 项，证明它不是可忽略的小噪声：norm 不生成 B0，却在四相机 MSE 中把 real B0 拉向空白输入表征。后续机制实验必须至少提供“排除 B0 的 alignment mask”对照；正式训练前应生成 B0 raster 或从 alignment loss 中去掉 B0。

**F2：detach 与 GRL 路径符合静态预期。** λ=0 时 real/raster F0 encoder 梯度均为 0，但 DomainClassifier 梯度范数为 `0.6881`；λ=1 时 raster F0 仍为 0，real F0 feature 与 FPN 梯度范数分别为 `4.501e-4` 和 `0.2868`。GRL λ=+1 与 λ=-1 无反转对照的 real-F0 activation 梯度余弦为 `-1.00011`，在浮点误差内等于 -1。

需要单独记录一个 checkpoint warning：v1/v2 官方 checkpoint 都没有 `domain_classifier.*` tensor，当前源码加载 v2 时该头为固定种子下的新初始化。因此 F2 能验证 autograd/GRL 的符号与边界，但不能证明一个已训练 DomainClassifier 的决策边界；F3 中 domain 梯度的**绝对大小**也只能视为 route smoke-test。

**F3：加权梯度由 planning 主导，三路在抽查层上近似正交且略同向。** norm token 与现有 PDM metric cache 无交集，因此 planning probe 使用 `RAPAgent.rap_loss` 的精确 trajectory component，并按 `score_mask=False` 使用代码规定的 0.1 sample weight；PDM-score auxiliary terms 不在本次分解内。

| FPN 参数 | planning grad | `0.002*MSE` grad | `0.001*domain` grad | cos(plan,MSE) | cos(plan,domain) | cos(MSE,domain) |
|---|---:|---:|---:|---:|---:|---:|
| lateral conv weight | 3.390e-2 | 2.587e-3 | 2.057e-4 | 0.0364 | 0.1453 | 0.1888 |
| FPN 3×3 conv weight | 1.431e-2 | 4.188e-4 | 1.989e-4 | 0.0123 | 0.0166 | 0.0263 |

在这两个参数上 planning 梯度分别约为 MSE 的 13.1×/34.2×。夹角没有出现负余弦冲突，但多数接近 0，说明 alignment 主要提供与 planning 近似正交的小梯度，而不是简单地放大同一更新方向。domain 数值受未加载 head 的限制，不能跨 checkpoint 外推。

**F4：共享 projector 会使 raster anchor 跨 step 漂移。** 无参数更新的重复 forward 中 raw DINO 与 post-FPN 均逐元素不变。一次仅由 `0.002*MSE` 驱动的 projector 更新后结果如下：

| optimizer | raw DINO real/raster | real post-FPN relative L2 | raster post-FPN relative L2 | MSE before → after |
|---|---:|---:|---:|---:|
| AdamW, lr=1e-5, wd=1e-4 | 0 / 0 | 1.600% | 0.602% | 1.08836 → 1.07815 |
| AdamW, lr=1e-5, wd=0 | 0 / 0 | 1.600% | 0.602% | 1.08836 → 1.07815 |
| SGD, lr=1e-5, wd=0 | 0 / 0 | 4.64e-8 | 4.26e-8 | 1.08836 → 1.08836 |

AdamW 有/无 weight decay 的首步结果在 float32 精度下相同，排除了本次明显漂移由 weight decay 造成的解释；SGD 的微小但非零漂移也独立确认两域共享 FPN。raw DINO 始终严格不变，而 raster post-FPN 会随 real 分支对共享 projector 的更新而移动，因此“raster 是当前 backward 的 stop-gradient target，而不是跨 iteration 固定 teacher”得到动态验证。

综上，0-F 的**计算图边界结论为 `PASS`**；连同 checkpoint 与 B0 限制后的整体判定为 **`PASS_WITH_WARNINGS`**。MSE/共享-anchor 路径可以进入 Step 1/2，但 learned domain-gradient 的大小需要包含 DomainClassifier 权重的 checkpoint 才能解释；任何涉及四相机 alignment 的后续结果必须报告排除 B0 的对照。

## 8. 执行顺序与最小闭环

推荐顺序：

1. **0-A 数据完整性**：先排除静默零图和版本不一致；
2. **0-F 梯度 smoke test**：确认研究对象的真实优化路径；
3. **0-C raster 组件/ownership 扫描**：分别量化 agent、map、traffic lights 的进入率、覆盖关系、面积/token coverage，在固定子集上完成组件 feature/梯度消融，并生成双投影 camera overlay、ownership/overwrite panel 与 token heatmap；
4. **0-B 人工信息边界标注与 panels**：在 0-C 的 agent/map/light 定量候选上建立 real/raster 信息差证据；
5. **0-E 相机偏移量化**：为逐 token MSE 的空间解释提供尺度；
6. **0-D 字段干预**：完成 Q2 的动态验证。

最小可交付闭环不是“找到几张漂亮例图”，而是：

```text
数据路径可信（0-A）
    + 梯度路径可信（0-F）
    + raster 三类组件的进入率、覆盖关系和 feature/梯度贡献有统计（0-C）
    + real/raster 信息差有人工证据（0-B）
    + 像素错位有量化尺度（0-E）
```

完成上述闭环后，Step 1 的 layer probe 和 Step 2 的机制实验才有稳定的解释基础。

## 9. Step 0 交付清单

| 实验 | 核心问题 | 主要输出 |
|---|---|---|
| 0-A | 训练是否读取正确 raster；能否重渲染复现 | `integrity.csv`、`pixel_delta.csv`、差分图、`summary.md` |
| 0-B | agent/map/light 的 real/raster 私有信息是否真实存在 | `labels.csv`、证据 panels、人工一致性与 bootstrap CI |
| 0-C | agent、map、traffic lights 是否进入 raster、被谁覆盖，并各自怎样影响 feature/MSE 梯度 | `agent_raster_visibility.csv`、`map_element_visibility.csv`、`traffic_light_visibility.csv`、`component_coverage.csv`、`component_ablation.csv`、`component_gradient.csv`、`visualization_index.csv`、双投影/ownership/ablation panels、token heatmaps |
| 0-D | 类别/速度/实例/历史/地图几何/灯态与组件开关到底是否编码 | `field_interventions.csv`、差分图 |
| 0-E | 虚拟相机偏移造成多少 token 级错位 | `camera_offset.csv`、距离分箱图 |
| 0-F | MSE/GRL 梯度流向何处；raster anchor 是否漂移 | `gradient_norms.csv`、`gradient_cosines.csv`、`anchor_drift.csv` |
