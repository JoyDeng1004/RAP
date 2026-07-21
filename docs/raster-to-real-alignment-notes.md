# RAP Raster-to-Real 特征对齐：代码、梯度与模块映射

## 1. 文档目的

本文整理 RAP-DINO 中 Raster-to-Real 特征对齐相关的核心问题：

1. raster feature 和真实图像 feature 在哪里生成、如何配对；
2. spatial-level MSE alignment 涉及哪些梯度，梯度向哪个分支传播；
3. image feature 本身是否会在梯度下降中被更新；
4. global-level domain alignment 中 GRL 如何工作；
5. 论文示意图中的 Image Encoder、Feature Projector 和 Projected Features 分别对应哪些代码；
6. 为什么 real image 与 raster image 共享 Feature Projector，以及这种设计的取舍。

![RAP Pipeline](rap_pipeline.png)

本文分析的是当前仓库中的 RAP-DINO 实现。核心结论是：

> DINOv3 backbone 被冻结；real/raster 两类图像通过同一个 DINOv3 和同一个 FPN。MSE 和 GRL 都把 raster feature 作为当前 iteration 内不接收梯度的参照，alignment 梯度主要通过 real 分支更新共享 FPN 及其后的表示参数。

---

## 2. 训练数据如何进入模型

### 2.1 两类图像输入

特征构建阶段同时生成两类输入：

- `camera_feature`：真实相机图像；
- `rendered_camera_feature`：预生成的 raster 图像。

相关代码位于 `navsim/agents/rap_dino/bevformer/bev_feature_build.py:73-98`：

```python
synthetic_camera_feature = torch.tensor(
    np.ascontiguousarray(np.stack(imgs, axis=0))
)

real_camera_feature = torch.tensor(
    np.ascontiguousarray(np.stack(imgs, axis=0))
)

features = {
    "camera_feature": real_camera_feature,
    "camera_valid": real_image_result["validity"],
    "rendered_camera_feature": synthetic_camera_feature,
}
```

相机顺序为：

```text
[CAM_B0, CAM_F0, CAM_L0, CAM_R0]
```

对应代码位于 `bev_feature_build.py:27-28`。

### 2.2 Raster 与 real 在 batch 维拼接

开启 `distill_feature` 后，训练代码先筛选有效 real 样本，然后将 raster 放在 batch 前半部分、real 放在后半部分：

```python
all_features[k] = torch.cat(
    [rendered_features[k], real_features[k]], dim=0
)
prediction = self.agent.forward(all_features, all_targets)
```

对应 `navsim/planning/training/agent_lightning_module.py:211-220`。

因此 batch 排列为：

```text
[全部 raster 样本 B_s, 有效 real 样本 B_r]
```

两类输入不是分别经过两个模型，而是一次性通过同一个 `RAPModel`。

---

## 3. 图中模块与代码的对应关系

### 3.1 Image Encoder：冻结的 DINOv3

图中的 `Image Encoder` 对应：

```python
self.img_backbone = AutoModel.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m"
)
```

定义位置：`navsim/agents/rap_dino/bevformer/image_encoder.py:30`。

前向计算为：

```python
img_feats = self.img_backbone(
    pixel_values=img
)["last_hidden_state"]
```

调用位置：`image_encoder.py:88`。

图中的雪花表示该模块冻结。代码中 DINOv3 参数确实被排除在 optimizer 之外并设置为不可训练：

```python
other_params = [
    p for n, p in self._rap_model.named_parameters()
    if p.requires_grad and "_backbone.img_backbone" not in n
]

for p in self._rap_model._backbone.img_backbone.parameters():
    p.requires_grad = False
```

对应 `navsim/agents/rap_dino/rap_agent.py:566-577`。

### 3.2 Feature Projector：以 FPN neck 为主体

当前代码中不存在名为 `FeatureProjector` 的独立类。图中的 `Feature Projector` 最接近 DINOv3 后面的 FPN neck：

```python
self.img_neck = FPN(
    in_channels=[64, 128, 256, 1280][-self.num_outs:],
    out_channels=_dim_,
    start_level=0,
    add_extra_convs="on_output",
    num_outs=self.num_outs,
    relu_before_extra_convs=True,
)
```

定义位置：`image_encoder.py:42-49`。

调用位置：

```python
img_feats = self.img_neck([img_feats])
```

对应 `image_encoder.py:95-96`。

在 FPN 后，代码还会加入可训练的 camera embedding 和 level embedding：

```python
self.level_embeds = nn.Parameter(
    torch.randn(self.num_feature_levels, self.embed_dims)
)
self.cams_embeds = nn.Parameter(
    torch.randn([self.num_cams, self.embed_dims])
)
```

以及：

```python
feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
feat = feat + self.level_embeds[
    None, None, lvl:lvl + 1, :
].to(feat.dtype)
```

对应 `image_encoder.py:50-52,112-115`。

所以更完整的映射是：

```text
Image Encoder
    = frozen DINOv3

Feature Projector
    ≈ trainable FPN neck

Projected Features
    = FPN 输出 + camera embedding + level embedding
```

当前配置中，DINOv3 和 FPN 输出维度都是 1280。因此这里的 projector 主要用于任务适配和域对齐，而不是简单降维。

### 3.3 Projected Features 对应的张量

`ImgEncoder` 最终返回：

```python
return feat_flatten[-1], spatial_shapes[-1], level_start_index, kwargs
```

其中主特征张量的排列为：

```text
(num_camera, num_token, batch, feature_dim)
```

对应 `image_encoder.py:123-128`。

进入 `RAPModel` 后，该结果被命名为 `image_feature`：

```python
image_feature = self._backbone(
    camera_feature, img_metas=features
)
```

对应 `navsim/agents/rap_dino/rap_model.py:118`。

用于 MSE alignment 的张量为：

```python
output["bev_feature"] = image_feature[0].permute(2, 0, 1, 3)
```

对应 `rap_model.py:145`。虽然变量名是 `bev_feature`，这里实际上仍是 post-FPN 的多相机图像 token，典型形状为：

```text
(batch, 4 cameras, 1344 tokens, 1280 dimensions)
```

---

## 4. Spatial-level MSE alignment

### 4.1 核心代码

MSE alignment 位于 `agent_lightning_module.py:246-256`：

```python
bev_feature = prediction["bev_feature"]

render_bev = bev_feature[:batch_size][real_valid_mask].detach()
real_bev = bev_feature[batch_size:]

loss_render = F.mse_loss(render_bev, real_bev)

loss_dict["loss"] += (
    self.agent._config.distill_feature_weight * loss_render
)
```

配置权重为：

```python
distill_feature_weight: float = 0.002
```

定义于 `navsim/agents/rap_dino/navsim_config.py:13`；训练配置在 `navsim/planning/script/config/common/agent/rap_agent.yaml:15` 中将 `distill_feature` 设为 `True`。

### 4.2 数学表达

令：

- \(F^{real}\) 为 real image 的 post-FPN feature；
- \(F^{raster}\) 为配对 raster image 的 post-FPN feature；
- \(N\) 为参与平均的全部 feature 元素数。

实际加入总损失的 alignment 项是：

\[
L_{align}
=
\frac{0.002}{N}
\sum_i
\left(
F_i^{real}
-
\operatorname{stopgrad}(F_i^{raster})
\right)^2
\]

由于 raster 侧调用了 `.detach()`：

\[
\frac{\partial L_{align}}{\partial F_i^{real}}
=
0.002\frac{2}{N}
\left(F_i^{real}-F_i^{raster}\right)
\]

\[
\frac{\partial L_{align}}{\partial F_i^{raster}}
=0
\]

因此，当前 iteration 内的计算图方向是：

```text
real feature   ──接收 MSE 梯度──→ 向 raster feature 靠近
raster feature ──detach─────────→ 不接收 MSE 梯度
```

### 4.3 梯度真正更新的是网络参数

如果暂时把 feature 本身视为可优化变量，普通 SGD 的直观形式是：

\[
F^{real}
\leftarrow
F^{real}
-
\eta\,0.002\frac{2}{N}
\left(F^{real}-F^{raster}\right)
\]

但实际训练中，`real_bev` 是非叶子中间张量，不是 optimizer 中的 `nn.Parameter`。设 projector 参数为 \(\theta\)：

\[
F^{real}=P_{\theta}(E(x^{real}))
\]

真正计算的是：

\[
\nabla_{\theta}L_{align}
=
J_{F^{real}}(\theta)^T
\cdot
0.002\frac{2}{N}
\left(F^{real}-F^{raster}\right)
\]

因为 raster 侧被 detach，不存在 raster 分支的 Jacobian 项。

---

## 5. Image feature 会不会被更新

答案需要区分“当前 forward 得到的 feature 张量”和“下一次 forward 重新计算的 feature”。

### 5.1 当前 feature 张量不会被 optimizer 直接修改

`image_feature` 是由网络前向计算产生的中间张量，不是 `nn.Parameter`，也没有被放进 optimizer。因此 optimizer 不会执行：

```python
image_feature -= learning_rate * image_feature.grad
```

反向传播只是计算经过这个张量的梯度，然后将梯度继续传给生成该张量的参数。

此外，PyTorch 默认不会把非叶子张量的梯度长期保存在 `.grad` 中。如需调试查看，通常需要在 backward 前调用：

```python
image_feature[0].retain_grad()
```

这只影响梯度观察，不改变训练逻辑。

### 5.2 下一次 forward 的 post-FPN feature 会改变

当前 iteration 中：

```text
feature 差异
    ↓
计算 MSE 梯度
    ↓
梯度回传到共享 FPN/embedding
    ↓
AdamW 更新这些参数
```

下一次 iteration 会使用更新后的参数重新计算：

\[
F_{t+1}^{real}
=
P_{\theta_{t+1}}(E(x^{real}))
\]

因为 \(\theta_{t+1}\neq\theta_t\)，新的 post-FPN image feature 会发生变化。

### 5.3 Raw DINO feature 与 post-FPN feature 的区别

| Feature 层级 | 生成模块 | 是否因 alignment 更新参数而变化 |
|---|---|---|
| Raw DINO token | 冻结的 DINOv3 | 不通过参数训练而改变 |
| Post-FPN feature | 可训练 FPN | 会改变 |
| 加入 cam/level embedding 后的 feature | FPN + 可训练 embedding | 会改变 |

需要注意，训练中的 GridMask 等随机增强可能让相同输入在不同 forward 中产生不同激活；这与参数是否被 optimizer 更新是两个不同问题。

### 5.4 Raster feature 也可能跨 iteration 间接变化

`.detach()` 只阻止当前 backward 沿 raster 分支回传。因为 raster 与 real 共享 FPN：

```text
当前 iteration：
    raster activation 被 detach
    alignment 只通过 real 分支更新共享 FPN

下一 iteration：
    raster 再次通过已经更新的共享 FPN
    新的 raster feature 也会发生变化
```

因此，raster feature 是“当前 iteration 的 stop-gradient anchor”，而不是跨 iteration 永久冻结的 teacher feature。

---

## 6. Global-level domain alignment

除了逐 token 的 MSE，代码还有一条基于 Gradient Reversal Layer（GRL）的域对抗路径。

### 6.1 使用前视相机特征

`rap_model.py:147-152` 中：

```python
lambda_ = self.lambda_scheduler(self.progress)
feat = image_feature[0][[1]]

feat_grad = feat[:, :, :self.batch_size].detach()
feat_no_grad = feat[:, :, self.batch_size:]

mixed_feat = torch.cat([feat_grad, feat_no_grad], dim=2)
domain_logits = self.domain_classifier(
    mixed_feat, lambd=lambda_
)
```

由于相机顺序是 `[B0, F0, L0, R0]`，`[[1]]` 取的是前视相机 `CAM_F0`。

这里变量名具有误导性：

- `feat_grad` 实际调用了 `.detach()`，是没有 encoder 梯度的 raster feature；
- `feat_no_grad` 没有调用 `.detach()`，实际上是有 encoder 梯度的 real feature。

### 6.2 GRL 的反向传播

GRL 实现在 `rap_model.py:17-28`：

```python
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None
```

前向传播时 GRL 是恒等映射；反向传播时，它把进入 encoder/projector 的梯度乘以 \(-\lambda\)。

域标签定义为：

```text
raster = 0
real   = 1
```

域损失权重为：

```python
loss_dict["loss"] += 0.001 * domain_loss
```

对应 `agent_lightning_module.py:226-244`。

因此两个模块具有相反目标：

| 模块 | 梯度方向 | 优化目标 |
|---|---|---|
| Domain classifier | 正常 BCE 梯度 | 尽可能区分 real/raster |
| Real projector/encoder 路径 | 经过 GRL 的反向梯度 | 让 real feature 更难被判断为 real |
| Raster projector/encoder 路径 | 输入已 detach | 不接收 domain loss 的 encoder 梯度 |

Domain classifier 参数仍会同时使用 raster 和 real 样本训练。输入 detach 只阻止梯度继续传入 raster 的特征生成路径，不会阻止 classifier 自身更新。

GRL 系数随训练进度变化：

\[
\lambda(p)=\frac{2}{1+e^{-10p}}-1
\]

其中代码设置：

\[
p=\frac{epoch+1}{20}
\]

所以训练前期域对抗较弱，后期 \(\lambda\) 逐渐接近 1。

---

## 7. 哪些变量和参数会被更新

必须区分“是否具有梯度”和“是否被 optimizer 更新”：

- 中间 feature 可以具有反向梯度，但不是 optimizer 参数；
- 只有加入 optimizer 的叶子参数才会被 `optimizer.step()` 更新。

### 7.1 Alignment 路径汇总

| 对象 | MSE alignment 梯度 | Domain alignment 梯度 | 是否由 optimizer 更新 |
|---|---:|---:|---:|
| Real post-FPN feature | 有，全部相机 | 有，仅前视相机 | 否，中间张量 |
| Raster post-FPN feature | 无，已 detach | 无，已 detach | 否，中间张量 |
| DINOv3 参数 | 无，冻结 | 无，冻结 | 否 |
| FPN neck 参数 | 有，来自 real 分支 | 有，来自 real 前视分支 | 是 |
| Camera embedding | 有 | 可能有 | 是 |
| Level embedding | 有 | 有 | 是 |
| Domain classifier 参数 | 无 | 有，正常 BCE 梯度 | 是 |
| Trajectory refiner/scorer | 无直接 MSE 梯度 | 无直接 domain 梯度 | 会由 planning loss 更新 |
| Real/raster 图像输入 | 通常不保留输入梯度 | 通常不保留输入梯度 | 否 |
| Renderer、3D box、相机标定 | 无计算图连接 | 无计算图连接 | 否 |

### 7.2 不要把“alignment detach”理解为“raster 对完整训练没有梯度贡献”

总的 planning loss 是在拼接后的 raster 和 real batch 上计算的：

```python
prediction = self.agent.forward(all_features, all_targets)
loss_dict = self.agent.compute_loss(
    all_features, all_targets, prediction
)
```

因此：

- 对 MSE alignment，raster 分支没有梯度；
- 对 domain alignment，raster encoder 分支没有梯度；
- 对 planning loss，raster 样本仍可通过规划计算图更新共享 FPN、trajectory refiner 和 scorer。

---

## 8. 总损失与参数下降方式

整体损失可概括为：

\[
L_{total}
=
L_{planning}
+0.002L_{MSE}
+0.001L_{domain}
\]

对于同时连接这些损失的参数，最终梯度是各项梯度之和：

\[
g_{\theta}
=
\nabla_{\theta}L_{planning}
+0.002\nabla_{\theta}L_{MSE}
+0.001\nabla_{\theta}L_{domain}
\]

当前训练使用 AdamW：

```python
optimizer = torch.optim.AdamW(
    [{
        "params": other_params,
        "lr": self._lr,
        "weight_decay": 1e-4,
    }]
)
```

对应 `rap_agent.py:573-578`。配置中的基础学习率为 `1e-4`，并使用 20 epoch 的 warmup-cosine scheduler，最低学习率为 `1e-5`。

Lightning 的 `training_step` 返回总损失后，由自动优化流程完成大致如下的操作：

```text
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

因此实际更新不是简单的普通 SGD，而是 AdamW 根据梯度的一阶、二阶移动平均进行自适应更新，并施加 decoupled weight decay。

---

## 9. 为什么 real/raster 共享 Feature Projector

代码能够直接证明“共享”，但没有注释直接陈述设计原因。本节是根据结构和 loss 做出的设计推断。

### 9.1 共享方式

两类输入先在 batch 维拼接，随后只调用一个：

```python
self._backbone = ImgEncoder(config)
```

其中只有一套：

```python
self.img_backbone
self.img_neck
self.cams_embeds
self.level_embeds
```

不存在独立的 `real_projector` 和 `raster_projector`。

### 9.2 让两个域处于同一表示空间

共享 projector 时：

\[
F^{real}=P_{\theta}(E(x^{real}))
\]

\[
F^{raster}=P_{\theta}(E(x^{raster}))
\]

两边通过同一个函数映射，因此逐 token、逐 channel 的 MSE 比较更容易具有统一语义。

若使用两个独立 projector：

\[
F^{real}=P_{\theta_r}(E(x^{real}))
\]

\[
F^{raster}=P_{\theta_s}(E(x^{raster}))
\]

两个 projector 可能学习各自的通道尺度和语义。即使 MSE 下降，也可能只是两个独立映射相互适配，而不一定形成稳定的共享特征空间。

### 9.3 在冻结 DINO 后提供可训练的域适配能力

DINOv3 保留预训练视觉语义，FPN 则承接 planning、MSE 和 GRL 梯度：

```text
Frozen DINOv3：保持通用表示
Shared trainable FPN：学习规划任务与域对齐
```

这使 alignment 不需要破坏预训练 backbone，也仍然有可训练模块能够调整 real-image 表示。

### 9.4 参数效率和训练/推理一致性

共享 projector 只需一套 FPN。训练时使用 raster 辅助，推理时只输入 real image，但 real image 仍经过训练时的同一个 projector：

```text
训练：raster + real → shared projector
推理：real          → same shared projector
```

不需要推理阶段进行分支选择，也不需要保留仅供 raster 使用的额外 projector。

### 9.5 配合 domain classifier

共享 projector 可以减少由“不同 projector 本身”引入的域差异，使 domain classifier 更集中地检测输入域造成的特征差异，而不是检测两套网络参数产生的差异。

---

## 10. 共享设计的代价与替代方案

共享 projector 的一个重要代价是：raster target 不是跨 iteration 固定的。

虽然当前 backward 中 raster feature 被 detach，但 real 分支更新的是共享 FPN。下一次 forward 时，raster 也经过更新后的 FPN，因此 raster target 会移动。

如果需要更稳定的 teacher，可以考虑两套 projector：

```text
Real branch：trainable student projector
Raster branch：frozen 或 EMA teacher projector
```

数学上可表示为：

\[
F^{real}=P_{\theta}(E(x^{real}))
\]

\[
F^{raster}=P_{\bar{\theta}}(E(x^{raster}))
\]

其中 teacher 参数采用指数滑动平均：

\[
\bar{\theta}
\leftarrow
\mu\bar{\theta}
+(1-\mu)\theta
\]

这种方案能够提供更稳定的 target，但会引入额外参数、状态同步和超参数。当前 RAP-DINO 选择的是结构更简单、参数更少的 shared-projector 方案。

---

## 11. 常见理解误区

### 误区一：`image_feature` 会被 AdamW 直接更新

不会。它是中间张量。AdamW 更新的是生成 feature 的 FPN、embedding 等参数；下一次 forward 重新生成的 feature 才会变化。

### 误区二：所有 image feature 都被 alignment 改变

不是。冻结的 raw DINO token 不通过参数更新而改变；主要被训练塑造的是 post-FPN feature。

### 误区三：`.detach()` 表示 raster feature 永久不变

不是。`.detach()` 只在当前计算图中切断 raster 分支梯度。共享 FPN 更新后，下一次 forward 的 raster feature 仍可能改变。

### 误区四：raster 数据对训练完全没有梯度贡献

不是。raster 在 MSE/GRL 的 encoder 路径中被 detach，但仍参与 planning loss，也参与 domain classifier 参数的训练。

### 误区五：代码里存在独立的 `FeatureProjector` 模块

当前 RAP-DINO 实现没有这个命名的独立模块。图中的 Feature Projector 在代码层面主要对应 `ImgEncoder.img_neck`，即 FPN neck；其输出还叠加了 camera/level embedding。

---

## 12. 最终总结

RAP-DINO 的对齐机制可以压缩为以下计算图：

```text
Raster image ─┐
              ├─ Frozen shared DINOv3
Real image ───┘
                       ↓
                Shared trainable FPN
                       ↓
              Camera/level embeddings
                       ↓
                Projected features
                  ↙             ↘
      Spatial MSE alignment    Global GRL alignment
      raster: detach           raster: detach
      real: gradient           real: reversed gradient
                  ↘             ↙
                 更新共享 FPN/embedding
```

最关键的三点是：

1. 图中的 Feature Projector 在当前代码中主要对应 DINOv3 后面的 FPN neck；
2. MSE 和 GRL 都在 alignment 路径中切断 raster encoder 分支，只让 real 分支承担特征对齐梯度；
3. feature 张量不会被 optimizer 直接修改，但生成 post-FPN feature 的共享参数会更新，因此下一次 forward 的 real 和 raster feature 都可能变化。
