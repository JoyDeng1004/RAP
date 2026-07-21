# Step 0 Alignment Audit 实验结果总览

本文重新梳理 [`outputs/step0_alignment_audit`](../outputs/step0_alignment_audit) 下的正式结果。它不替代实验协议 [`0718-step0-alignment-audit.md`](0718-step0-alignment-audit.md)，而是把已经确认的事实串成一条完整证据链：训练读到了什么 raster、raster 编码了什么、alignment gradient 流向哪里，以及这些事实能否解释 planner 性能。

## 1. 总结论

当前 Step 0 对 norm checkpoint 的核心机制已经解释清楚：

1. **Norm 数据路径可信。** 51,867 个有效帧的 F0/L0/R0 raster 路径完整、无 0-byte 文件；production renderer 可以逐像素复现磁盘 raster。
2. **Raster 是稀疏的当前帧几何瓶颈。** 模型尺度约 83.84% 为 blank，主要有效内容是 agent 几何和部分 map 几何；traffic light 最终只占约 0.205%。
3. **Raster 不编码完整动态场景。** 它不编码 agent 类别、速度、identity 或历史；灯态只有 `red vs non-red`；map 只支持有限字符串分支。
4. **存在系统性 target 信息损失。** `<4 valid corners` 会删除严格 clipping 后仍有明显面积的 agent；map 还存在 unsupported 和覆盖损失。
5. **Alignment 的实际更新对象是 real 侧 projected feature。** Raster activation 在当前 backward 中 detach；冻结 DINO 不更新，FPN 与 camera/level embedding 更新。
6. **Raster target 不是跨 step 固定 teacher。** FPN 共享，一次 MSE optimizer step 后 real 和 raster 的 post-FPN feature 都会移动，只有 raw DINO anchor 保持不变。
7. **Map 和 agent 主导当前 feature alignment，traffic light 平均影响较弱。** 但 alignment gradient 与简化 planning probe 近似正交，现有结果不能证明 alignment 改善或损害 planner。
8. **四相机 MSE 存在 B0 空白 target 风险。** Norm 不生成 B0 raster，生产 loader 使用零图 fallback；0-F 的两样本测试中，B0 占四相机 MSE 总和的 37.09%。
9. **Aug 数据当前不可作为完整训练输入。** 22 个 metadata pkl 损坏，且有效 metadata 引用 2,065 个 0-byte raster JPEG。

因此，Step 0 已经足以进入受控性能消融，但还没有任何 planner 性能因果结论。用户已决定不执行 0-B 和 0-E；0-D 只完成静态代码审计，没有执行正式动态 intervention protocol。

## 2. 结果状态与证据等级

| 项目 | 状态 | 证据类型 | 当前用途 |
|---|---|---|---|
| 0-A norm | `PASS_WITH_WARNINGS` | 全量路径/0-byte 扫描、抽样内容检查、300 rows exact rerender | 支撑 norm 数据路径与 renderer 等价性 |
| 0-A perturbed | `PASS_WITH_WARNINGS` | 5,095 个中心帧扫描、300 rows exact rerender | 支撑 perturbed 当前帧输入 |
| 0-A aug | `FAIL` | 全量 metadata/0-byte 扫描、400 rows exact rerender | 阻止把 aug 当作完整训练输入 |
| 0-C | `PASS` | 10,000 帧几何扫描、32 帧 checkpoint 消融 | 定量 raster 内容、ownership 和组件 feature/gradient 影响 |
| 0-D | `STATIC_AUDIT_COMPLETE` | production metadata/renderer 静态数据流 | 确定字段是否进入 raster；不是动态实验 `PASS` |
| 0-F | `PASS_WITH_WARNINGS` | 真实 autograd、梯度分解、一次 optimizer step | 确定 alignment/GRL 路径和 shared-FPN anchor 漂移 |
| 0-B | 跳过 | 无 | 不提供人工 real visibility 频率 |
| 0-E | 跳过 | 已确认代码存在偏移，但未做像素/token 量化 | 不提供 camera shift 的 token 位移分布 |

开发过程中的 `0c_smoke`、`0c_smoke32` 和 `0c_feature_smoke_v2` 不进入正式统计；正式 0-C 以 [`0c_raster_components/summary.md`](../outputs/step0_alignment_audit/0c_raster_components/summary.md) 为准。

## 3. 机制链：从 metadata 到 optimizer update

```text
当前帧 metadata
    │
    ├─ gt_boxes_world ────────────────┐
    ├─ 部分 map type + geometry ──────┤
    ├─ traffic-light xy + red/non-red ┤
    └─ ego_heading ───────────────────┘
                     │
                     ▼
        fixed shifted virtual-camera renderer
        traffic light → map → agent
                     │
                     ▼
       稀疏 raster：约 84% blank token
                     │
                     ▼
           frozen DINOv3 + shared FPN
                     │
            raster post-FPN detach
                     │
                     ▼
       MSE 只反传 real activation 一侧
                     │
                     ▼
        更新 shared FPN/camera/level embedding
                     │
                     ▼
    下一 step 的 real 和 raster post-FPN 都漂移
```

这条链条给出一个关键解释：训练不是让原始 DINO 学会 raster，而是让 DINO 后面的共享 projected feature 空间在 planning、MSE 和 GRL 的共同梯度下变化。Raster 只在当前 backward 是 stop-gradient target，不是长期固定 teacher。

## 4. 0-A：先确认训练读到的输入可信

### 4.1 Norm 可以用于当前机制分析

[`0a_norm/summary.md`](../outputs/step0_alignment_audit/0a_norm/summary.md) 的正式结果：

- 64/64 个 metadata 文件可读；
- 51,867/51,867 个有效帧进入审计；
- F0/L0/R0 路径缺失为 0；
- F0/L0/R0 全量 0-byte 文件为 0；
- 300/300 个 camera rows 经相同 JPEG round-trip 和 loader crop 后逐像素相等；
- 抽样发现 L0 1 张、R0 6 张可解码全黑图，它们是 renderer 的合法空视野，不是缺失文件 fallback。

所以后续 0-C/0-F 使用 norm 时，可以把磁盘 raster 当作当前 renderer 的真实 production 输出。

### 4.2 Norm 的 B0 不存在，但四相机 loss 会使用 fallback

Norm 的 CAM_B0 在 51,867 帧中全部没有 raster 文件。0-A 将 B0 标为非 required camera，因此 norm 三相机数据完整性仍为通过；但模型 alignment 路径使用四相机时，loader 会给 B0 零图 fallback。

这个问题不能只看 0-A 状态，必须与 0-F 联合解释：B0 空白输入经过 normalization、DINO 和 FPN 后并不是数值零 feature，而会形成稳定的“blank-raster target”。

### 4.3 Perturbed 当前帧可用

[`0a_perturbed/summary.md`](../outputs/step0_alignment_audit/0a_perturbed/summary.md) 的结果：

- 5,095/5,095 个 metadata 文件可读；
- 每个 14 帧窗口只把 index=3 的 5,095 个中心帧作为 raster 输入分母；
- F0/L0/R0 路径完整且无 0-byte 文件；
- 300/300 个 camera rows exact rerender；
- 少量侧视图全黑，同样是合法可解码输出。

### 4.4 Aug 是硬失败

[`0a_aug/summary.md`](../outputs/step0_alignment_audit/0a_aug/summary.md) 的结果：

- 17,299 个 metadata pkl 中只有 17,277 个可读；
- 21 个 `EOFError`，1 个 `UnpicklingError`；
- 有效 metadata 引用的 raster 中存在 2,065 个 0-byte JPEG：B0 524、F0 517、L0 521、R0 503；
- 这些文件会触发 loader 零图 fallback；
- 对可正常处理的样本，400/400 rows 可以 exact rerender，说明问题是数据损坏，不是 renderer 版本不一致。

结论边界：norm checkpoint 的机制诊断可以继续；任何依赖 aug 的重新训练在修复 22 个 metadata 和 2,065 个 0-byte raster 前都不可信。

## 5. 0-C：Raster 实际提供了什么 target

### 5.1 扫描规模与生产等价性

正式 0-C 使用：

- 10,000 帧；
- 64 个 log；
- F0/L0/R0 共 30,000 个 frame-camera；
- 16 个 shard，每个完成 625/625 帧；
- 336/336 个 production RGB rows 逐像素相等；
- ownership 互斥完备性失败数为 0；
- 几何扫描错误数为 0。

因此，下面统计的是 production-equivalent raster 内容，而不是另一个近似 renderer。

### 5.2 Target 约 84% 为空白

最终互斥 ownership coverage：

| 尺度 | traffic light | map | agent | blank |
|---|---:|---:|---:|---:|
| renderer 1920×1120 | 0.2088% | 4.3415% | 11.8507% | 83.5990% |
| model 768×448 | 0.2050% | 4.2838% | 11.6735% | 83.8377% |
| token 48×28 | 0.2110% | 4.1859% | 11.5177% | 84.0854% |

这些比例是最终互斥 ownership，不能把 component-only coverage 相加。它们说明 raster 的主要功能不是照片级重建，而是把少量当前帧几何和先验投影到黑底画布。

### 5.3 `<4 valid corners` 会删除仍有明显面积的 agent

全量包含 3,088,599 个 agent-camera rows。更有解释力的分母不是全部 rows，而是“严格 clipping 后在模型尺度仍达到面积阈值”的 agent-camera：

| 严格 clipping 面积阈值 | production FOV drop | 条件分母 | 丢弃率 |
|---:|---:|---:|---:|
| ≥4 px | 25,761 | 701,355 | 3.67% |
| ≥16 px | 25,413 | 700,994 | 3.63% |
| ≥64 px | 23,976 | 651,215 | 3.68% |

阈值 16 px 的距离分布：

| 距离 | FOV drop | 条件分母 | 丢弃率 |
|---|---:|---:|---:|
| `[0,10)` m | 11,498 | 40,508 | 28.38% |
| `[10,20)` m | 5,696 | 101,843 | 5.59% |
| `[20,40)` m | 4,690 | 205,339 | 2.28% |
| `[40,80)` m | 3,515 | 351,869 | 1.00% |
| `[80,120)` m | 14 | 1,435 | 0.98% |

这证明角点规则不是只影响微小远处物体。它主要删除近距离、跨越 FOV 边缘的大目标；这些目标严格 clipping 后仍可能覆盖多个 DINO token。

### 5.4 Map 同时存在 unsupported 和 overwrite 损失

Map element-camera 结果：

- metadata rows：12,560,592；
- renderer 支持：8,605,128，约 68.51%；
- isolated 非零：1,296,037；
- final 非零：760,148；
- 被其他 map 完全覆盖：410,299；
- 又被 agent 完全覆盖：125,590。

这里必须区分三件事：

1. `unsupported`：metadata 有，但 renderer 没有匹配分支；
2. geometry/rasterization zero：类型受支持，但没有形成像素；
3. overwrite：曾经形成像素，但后绘制组件拿走了最终 ownership。

### 5.5 Traffic light 面积很小，且状态表示受限

Traffic-light element-camera 结果：

- metadata rows：259,686；
- isolated 非零：73,398；
- final 非零：70,801；
- 最终被 agent 完全覆盖：717。

最终 token coverage 只有约 0.211%。这能解释它对全局平均 MSE 的贡献较小，但不能说明 traffic light 对路口规划不重要。平均面积和任务条件重要性不是同一个量。

## 6. 0-D 静态审计：Raster 编码哪些字段

完整结论见 [`0d_field_interventions/summary.md`](../outputs/step0_alignment_audit/0d_field_interventions/summary.md)。状态为 `STATIC_AUDIT_COMPLETE`，没有运行正式动态 intervention protocol。

### 6.1 明确编码

- agent `gt_boxes_world` 的位置、长宽高和 yaw；
- agent face 深度及由此产生的亮度和远近遮挡顺序；
- 受支持 map type 的 polygon/polyline 几何；
- traffic-light `[x,y]`；
- traffic-light 的 `red vs non-red` 二值状态；
- 当前 `ego_heading`；
- 固定 shifted virtual-camera 投影。

### 6.2 明确不编码

- agent `gt_names`；
- `gt_velocity_3d`；
- instance token；
- track token；
- 非当前历史帧；
- map feature ID；
- 每帧真实相机标定；
- traffic-light 的真实尺寸和高度。

### 6.3 部分编码与硬编码

- 原始灯态先被压缩为 `is_red = (status == red)`；renderer 将所有 `False` 画成绿色，因此 yellow/unknown 不再保留；
- traffic-light z=5 m、dims=(0.5,0.5,1.0) 均为硬编码；
- map z 统一为 0；
- map 只支持 `LANE`、`CROSSWALK/SPEED_BUMP`、`BOUNDARY/SOLID` 字符串分支；
- camera 使用固定 `camera_params`，并执行 `cam_t[2]+=0.8; cam_t[0]-=2`。

### 6.4 绘制顺序决定 target 语义

```text
traffic light → map → agent
```

因此：

- map 可以覆盖 traffic light；
- agent 可以覆盖 map 和 traffic light；
- 删除 agent 会显露底层 map/light；
- 删除 map 会显露底层 light；
- leave-one-out 不是简单把某种颜色设为黑色。

0-D 对 0-C 的解释约束是：`remove_agent` 反映的是 agent 几何 target，不是类别、速度、身份或历史 target。

## 7. 0-C Feature 消融：哪些组件改变 target

固定 checkpoint：

```text
ckpts/RAP_DINO_navsimv2.ckpt
SHA256 9accbb101f30541187c7bb061689bf88f481b2125ad9a9c563a488275ea0311d
epoch 14, global step 21930
```

使用 32 帧、32 个不同 log、F0/L0/R0，关闭 GridMask，B0 完全排除。共生成 1,024 条 ablation、1,024 条 gradient 和 480 个 float heatmap。

Joint-3cam leave-one-out：

| 从 full raster 删除 | Δ raw MSE mean（95% CI） | `||Δg||` | `cos(g_variant,g_full)` | `cos(g_variant,g_planning probe)` |
|---|---:|---:|---:|---:|
| map | +0.12327 `[+0.09971,+0.14783]` | 9.86e-4 | 0.9277 | 0.0122 |
| agent | +0.09883 `[+0.07937,+0.12099]` | 9.03e-4 | 0.9250 | 0.0321 |
| traffic light | +0.007965 `[+0.004102,+0.012196]` | 1.77e-4 | 0.9892 | 0.0252 |

可以确认：

- full raster 比删除 map/agent/light 后更接近当前 real feature；
- map 和 agent 对当前 feature target/MSE gradient 的影响最大；
- traffic light 的平均影响较小但非零；
- 三类 component gradient 与简化 planning probe 的余弦都接近 0。

不能确认：

- map 或 agent 会改善 planner；
- traffic light 对安全关键路口不重要；
- 三个组件的效果可以线性相加；
- 余弦接近 0 等价于 alignment 有害。

### 7.1 Heatmap 的正确解释

- `*_remove_agent/map/traffic_light.png`：full raster 与删除组件 raster 的 post-FPN token feature difference；
- `*_real_full.png`：real 与 full raster 的 post-FPN token feature difference；
- 热区经过 DINO self-attention 和 FPN 后通常比较分散，不能当作精确物体像素 attribution；
- `real_full` 图定性显示大量 token 存在差异，但没有叠加 foreground mask，因此不能严格证明高差异 token 位于 raster foreground 外。

当前 `*_grad.png` 还有一个明确的显示限制：所有 heatmap 共用 `global_p99=45.3022`，而 weighted gradient 数值约为 `1e-8`。因此 gradient PNG 近乎全黑；数值 `.npy` 和 `component_gradient.csv` 有效，但 gradient PNG 不适合判断空间强弱。

## 8. 0-F：Alignment gradient 实际流向哪里

正式结果见 [`0f_gradient_route/summary.md`](../outputs/step0_alignment_audit/0f_gradient_route/summary.md)。

### 8.1 MSE-only 路径

两个配对样本的结果：

- raw MSE：1.0881311；
- `0.002×MSE`：0.0021762622；
- real post-FPN activation gradient norm：1.12e-6；
- raster pre-detach gradient：0；
- FPN gradient norm：0.002819；
- camera embedding gradient norm：2.84e-5；
- level embedding gradient norm：3.80e-5；
- 冻结 DINO、trajectory refiner、scorer 和 domain head 不接收 MSE-only gradient。

因此当前 MSE 不是训练 DINO，也不是直接训练 planner decoder；它训练 real 分支共享的 projected feature 层。

### 8.2 B0 空白 target 是四相机 MSE 最大单项

| Camera | Raster 是否存在 | Raw MSE | 四相机 MSE 占比 |
|---|---:|---:|---:|
| B0 | 否，零图 fallback | 1.6144 | 37.09% |
| F0 | 是 | 0.7298 | 16.77% |
| L0 | 是 | 0.9398 | 21.59% |
| R0 | 是 | 1.0685 | 24.55% |

该结果只有两个样本，不能当作全数据集精确比例；但它已经证明 B0 fallback 可以成为 alignment 的主要梯度来源。0-C 的三相机组件结果不能自动外推到包含 B0 的 production MSE。

### 8.3 GRL 路径正确，但 domain head 没有已训练权重

- `lambda=0` 时 real-F0 encoder gradient 为 0，DomainClassifier 仍有梯度；
- `lambda=1` 时 real-F0 encoder gradient 非零；
- raster-F0 始终 detach；
- 正反梯度对照余弦为 -1.0001，GRL 反向路径符合预期；
- checkpoint 中 DomainClassifier tensor 数量为 0，该 head 是用固定种子随机初始化。

因此 F2 只证明 autograd/GRL wiring 正确，不能解释当前 checkpoint 是否已经学到有效 domain decision boundary，也不能把 domain gradient 大小当作训练后行为。

### 8.4 Planning 与 alignment gradient 近似正交

在两个被检查的 FPN 权重上：

| 参数 | `cos(planning,mse)` | `cos(planning,domain)` | `cos(mse,domain)` |
|---|---:|---:|---:|
| lateral conv | 0.0364 | 0.1453 | 0.1888 |
| FPN conv | 0.0123 | 0.0166 | 0.0263 |

这些数字说明瞬时梯度方向关系很弱，但样本和参数范围有限。Planning probe 只包含精确 RAP trajectory component，`score_mask=False` 使用代码定义的 0.1 权重，不包含 PDM-score auxiliary losses。

### 8.5 Raster 只在当前 backward 是固定 target

确定性重复 forward 的所有 delta 为 0。执行一次 MSE optimizer step 后：

- raw DINO real 最大差值：0；
- raw DINO raster 最大差值：0；
- post-FPN real 相对 L2 变化：0.015998；
- post-FPN raster 相对 L2 变化：0.006015；
- MSE 从 1.08836 降到 1.07815（production AdamW 设置）。

这证明：

- DINO 是固定 anchor；
- 当前 backward 的 raster activation 没有梯度；
- 共享 FPN 更新后，下一 step 的 raster target 仍会变化；
- MSE 下降不能简单解释为 real feature 靠近一个永久固定的 raster teacher。

## 9. 合并后的研究判断

### 9.1 已确认事实

1. Norm 三相机 raster 文件和 renderer 路径可信。
2. Raster target 高度稀疏，主要内容是当前帧 agent/map 几何。
3. Raster 不包含类别、速度、identity 和历史信息。
4. 部分近距离/FOV 边缘 agent 被角点规则错误删除。
5. 大量 map metadata 不进入最终 raster，原因包括 unsupported、零面积和 overwrite。
6. MSE 当前 step 把 real projected feature 拉向 detached raster projected feature。
7. FPN 共享使 raster projected target 跨 step 漂移。
8. Map 和 agent 显著改变 alignment target，traffic light 平均影响较弱。
9. B0 空白 fallback 可以主导四相机 MSE。
10. Aug 数据有明确损坏，未经修复不能用于可靠训练。

### 9.2 有证据支持但仍属于推论

1. Alignment 很可能主要塑造几何/场景布局相关表示，而不是类别、速度或 temporal identity，因为后者不在 raster target 中。
2. 大面积黑底、人工颜色、几何 cuboid 与真实照片之间的跨模态差异，会让 MSE 同时包含语义差异、appearance gap 和空间错位。
3. FOV drop、unsupported map 和 B0 blank 会向 real 分支提供不完整或错误的 target，但是否损害 planning 取决于后续训练动态。
4. Traffic light 的平均 gradient 较小可能主要来自覆盖面积小；它在条件化路口场景中的任务价值仍可能很高。

### 9.3 当前不能声称

1. Alignment 提高或降低 planner/PDM 性能；
2. Map 比 agent 对 planner 更重要；
3. Traffic light 对规划不重要；
4. 黄色 heatmap token 是精确物体 attribution；
5. Camera shift 是性能下降的主因；
6. Raster-private/real-private 样本在数据集中的人工可见性频率；
7. 当前 checkpoint 已经学习到有效 domain classifier。

## 10. 跳过 0-B/0-E 与静态 0-D 的影响

用户决定不做 0-B 和 0-E，并用代码审计替代正式 0-D。这个决定不会推翻 0-A/0-C/0-F 的事实，但会收窄最终论文表述：

| 跳过项 | 仍然知道什么 | 不再拥有的证据 |
|---|---|---|
| 0-B | 代码和 raster ownership 能区分 FOV/unsupported/overwrite | 无法报告真实图像环境遮挡、可识别性和人工一致率 |
| 0-E | 代码确认 virtual camera 有 +0.8 m/-2 m 偏移 | 无法报告不同距离/相机上的像素和 token shift 分布 |
| 动态 0-D | 静态数据流能严格判断字段是否被读取 | 没有 20 场景 OFAT pixel-difference、bbox 和 PSNR 证据 |

所以完整协议没有形式上全部完成，但 norm alignment 的数据、target 和计算图边界已经足够清楚。

## 11. 下一步最有价值的实验

下一步不应继续增加 renderer 描述性检查，而应验证性能因果。最低限度建议比较：

| 组别 | 设置 | 核心问题 |
|---|---|---|
| A | 当前 checkpoint/训练设置 | 当前基线 |
| B | 去掉 MSE 与 domain alignment | Alignment 总体是否有净收益 |
| C | 保留 alignment，但排除 B0 | B0 blank target 是否损害训练 |
| D | F0/L0/R0 component-balanced 或 foreground-aware alignment | 稀疏 coverage 不均衡是否主导梯度 |

如果算力只允许一个对照，优先做 A vs B；如果允许第二个，优先加入 C。所有组必须保持初始化、数据顺序、训练步数、augmentation 和 evaluation split 一致，并报告 planner/PDM 指标，而不是只比较 feature MSE。

## 12. 正式结果入口

- 0-A 汇总：[`summary_0a.md`](../outputs/step0_alignment_audit/summary_0a.md)
- 0-A norm：[`0a_norm/summary.md`](../outputs/step0_alignment_audit/0a_norm/summary.md)
- 0-A perturbed：[`0a_perturbed/summary.md`](../outputs/step0_alignment_audit/0a_perturbed/summary.md)
- 0-A aug：[`0a_aug/summary.md`](../outputs/step0_alignment_audit/0a_aug/summary.md)
- 0-C 综合：[`0c_raster_components/summary.md`](../outputs/step0_alignment_audit/0c_raster_components/summary.md)
- 0-C 几何：[`geometry_summary.md`](../outputs/step0_alignment_audit/0c_raster_components/geometry_summary.md)
- 0-C feature：[`feature_summary.md`](../outputs/step0_alignment_audit/0c_raster_components/feature_summary.md)
- 0-C panels：[`panel_index.html`](../outputs/step0_alignment_audit/0c_raster_components/panel_index.html)
- 0-D 静态审计：[`0d_field_interventions/summary.md`](../outputs/step0_alignment_audit/0d_field_interventions/summary.md)
- 0-F 梯度路径：[`0f_gradient_route/summary.md`](../outputs/step0_alignment_audit/0f_gradient_route/summary.md)

