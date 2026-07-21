# RAP Planner 代码实现与监督机制总结

## 1. 文档目的

本文总结 RAP-DINO planner 的代码实现，包括：

- planner 的输入、输出和主要张量形状；
- trajectory head 如何生成并迭代细化候选轨迹；
- planner query 与候选轨迹的对应关系；
- 规则 PDM scorer 与神经 Scorer 的职责和差异；
- PDM 分数的输入、输出和计算过程；
- 训练与推理阶段如何给轨迹打分并选择最终轨迹；
- trajectory head 的监督信号来自哪里；
- 为什么当前实现动态计算 PDM 标签；
- 阅读代码时容易混淆的命名和实现细节。

本文讨论默认 NAVSIM 配置，即 `b2d=False`、`proposal_num=64`、`ref_num=4`、`pdm_scorer=True`。

## 2. 核心结论

RAP planner 不是“构建一个常规二维 BEV 网格，然后直接回归一条轨迹”，而是：

1. 使用当前帧四路相机提取图像 token；
2. 初始化 `64 × 8 = 512` 个轨迹 query；
3. 用一套共享参数的 trajectory head 执行 4 轮迭代细化；
4. 每轮生成 64 条候选轨迹，每条包含 8 个 `(x, y, heading)`；
5. 使用候选轨迹的位置从多相机图像中采样特征，并更新对应 query；
6. 使用神经 Scorer 为最后一轮的 64 条候选预测综合 PDM 分数；
7. 推理时选择预测分数最高的候选轨迹。

整体数据流为：

```text
四路当前帧图像 ─→ DINOv3 + FPN ─→ 多相机图像 token
                                           │
当前帧11维自车状态 ─→ 512个轨迹 query       │
                                           ↓
      [轨迹解码 → 以轨迹为锚点读取图像 → 更新query] × 4
                                           ↓
                         64条最终候选轨迹
                                           ↓
                          神经Scorer预测分数
                                           ↓
                             argmax选择最终轨迹
```

## 3. 关键代码位置

| 功能 | 文件与位置 |
|---|---|
| Agent 入口 | `navsim/agents/rap_dino/rap_agent.py:25-148` |
| Planner 主体 | `navsim/agents/rap_dino/rap_model.py:83-169` |
| 轨迹解码与单轮细化 | `navsim/agents/rap_dino/traj_refiner.py:6-34` |
| 基于轨迹位置更新 query | `navsim/agents/rap_dino/bevformer/bev_refiner.py:10-141` |
| 轨迹参考点构造与投影 | `navsim/agents/rap_dino/bevformer/encoder.py:159-309` |
| 多相机可变形交叉注意力 | `navsim/agents/rap_dino/bevformer/spatial_cross_attention.py:20-200` |
| 当前 query 自注意力 | `navsim/agents/rap_dino/bevformer/temporal_self_attention.py:18-267` |
| 图像编码器 | `navsim/agents/rap_dino/bevformer/image_encoder.py:13-128` |
| 神经 Scorer | `navsim/agents/rap_dino/score_module/scorer.py:8-76` |
| 规则 PDM 标签生成入口 | `navsim/agents/rap_dino/rap_agent.py:248-387` |
| 规则 PDM 指标计算 | `navsim/agents/rap_dino/score_module/train_pdm_scorer.py:113-526` |
| NAVSIM 轨迹仿真与分数打包 | `navsim/agents/rap_dino/score_module/compute_navsim_score.py:22-115` |
| trajectory/scorer 损失 | `navsim/agents/rap_dino/rap_agent.py:389-556` |
| target 构造 | `navsim/agents/rap_dino/rap_features.py:105-134` |
| NAVSIM 未来轨迹提取 | `navsim/common/dataclasses.py:307-333` |

## 4. Planner 的输入

### 4.1 相机输入

特征构造位于 `navsim/agents/rap_dino/bevformer/bev_feature_build.py:16-98`。

每个样本使用当前帧四路相机，顺序为：

```text
CAM_B0, CAM_F0, CAM_L0, CAM_R0
```

图像经过归一化、缩放和 padding 后，典型输入形状为：

```text
camera_feature: [B, 4, 3, 448, 768]
```

代码只遍历 `agent_input.cameras[-1:]`，因此 planner 不接收图像历史。

### 4.2 自车状态输入

`RAPFeatureBuilder` 将自车状态拼接为 11 维：

```text
pose(3) + velocity(2) + acceleration(2) + driving_command(4)
```

虽然 feature 中可能包含多个历史状态，但 `RAPModel.forward` 明确使用：

```python
ego_status = features["ego_status"][:, -1]
```

因此 planner 只接收当前帧自车状态，不使用自车历史。

## 5. 图像编码与张量形状

`ImgEncoder` 使用：

```text
facebook/dinov3-vith16plus-pretrain-lvd1689m
```

主要步骤为：

1. 将 batch 与 camera 维合并；
2. 使用 DINOv3 ViT-H/16+ 提取 token；
3. 去掉 CLS/register 等非 patch token；
4. 将 patch token 恢复为二维特征图；
5. 经过一个 FPN 输出层；
6. 加入相机 embedding 与 level embedding；
7. 展平为多相机 token。

对于 448×768 输入和 16×16 patch：

```text
patch grid = 28 × 48 = 1344
```

图像特征典型形状为：

```text
image_feature[0]: [4, 1344, B, 1280]
```

DINOv3 骨干在 optimizer 构建阶段被冻结；主要训练 FPN、planner query、注意力、trajectory head 和 Scorer。

## 6. Trajectory head 的结构

### 6.1 只有一套轨迹 head 参数

`RAPModel.__init__` 中的定义为：

```python
shared_refiner = Traj_refiner(config)
self._trajectory_head = nn.ModuleList(
    [shared_refiner for _ in range(ref_num)]
)
```

默认 `ref_num=4`，但 `ModuleList` 中四个位置引用的是同一个 `Traj_refiner` 对象。

因此：

| 观察角度 | 数量 |
|---|---:|
| 不同结构的 trajectory head | 1 |
| 不同参数的 trajectory head | 1 |
| trajectory head 的调用次数 | 4 |
| 产生的中间轨迹集合 | 4 |

从参数角度是一套共享 head；从执行阶段角度是四轮细化。

### 6.2 Planner query 初始化

默认配置：

```text
proposal_num = 64
poses_num = 8
tf_d_model = 1280
```

模型创建：

```python
self.init_feature = nn.Embedding(
    poses_num * proposal_num,
    tf_d_model,
)
```

因此初始 query 数量为：

```text
64 × 8 = 512
```

当前自车 11 维状态经过线性层映射到 1280 维，然后广播加到全部 query：

```python
ego_feature = self.hist_encoding(ego_status)[:, None]
bev_feature = ego_feature + self.init_feature.weight[None]
```

此处名为 `bev_feature` 的变量实际是轨迹 query：

```text
[B, 512, 1280] = [B, 64×8, 1280]
```

它不是传统规则二维 BEV 网格。

### 6.3 真正输出轨迹坐标的模块

`Traj_refiner` 内部定义：

```python
self.traj_decoder = MLP(
    config.tf_d_model,
    config.tf_d_ffn,
    3,
)
```

MLP 结构为：

```text
Linear(1280,1024)
LayerNorm
ReLU
Linear(1024,1024)
LayerNorm
ReLU
Linear(1024,3)
```

前向过程：

```python
proposals = self.traj_decoder(bev_feature).reshape(
    batch_size,
    64,
    8,
    3,
)
```

张量变化为：

```text
[B,512,1280]
      ↓ trajectory MLP
[B,512,3]
      ↓ reshape
[B,64,8,3]
```

每个轨迹点输出：

```text
(x, y, heading)
```

### 6.4 四轮细化

每轮 `Traj_refiner.forward` 执行：

```python
proposals = self.traj_decoder(bev_feature)
proposal_list.append(proposals)
bev_feature = self.Bev_refiner(
    proposals,
    bev_feature,
    image_feature,
)
```

四轮数据流为：

```text
初始query
  → proposals_0 → 根据proposals_0读取图像并更新query
  → proposals_1 → 根据proposals_1读取图像并更新query
  → proposals_2 → 根据proposals_2读取图像并更新query
  → proposals_3 → 根据proposals_3读取图像并更新query
  → 神经Scorer
```

四轮输出全部保存在：

```python
proposal_list = [
    proposals_0,
    proposals_1,
    proposals_2,
    proposals_3,
]
```

最终候选轨迹为：

```python
proposals = proposal_list[-1]
```

需要注意：最后一轮先从更新前 query 解码 `proposals_3`，然后再根据 `proposals_3` 更新 query；神经 Scorer 使用的是这次更新后的 query。

## 7. 候选轨迹如何用于读取图像

### 7.1 Query 与轨迹点一一对应

512 个 query 的布局可以理解为：

```text
query[p,t] ↔ 第p条候选轨迹的第t个未来位姿
```

其中：

```text
p = 0...63
t = 0...7
```

同一个 query 经过 `traj_decoder` 后输出对应的 `(x,y,heading)`。

### 7.2 构造车辆空间参考点

对于每个预测位姿，`BEVFormerEncoder.compute_corners` 会根据：

- 自车半长；
- 自车半宽；
- 后轴到车辆中心的距离；
- 预测 heading；

计算车辆 footprint 的四个角点。

随后在 4 个不同高度放置参考点：

```text
4个角点 × 4个高度 = 16个三维参考点/轨迹点
```

这些三维点通过 `lidar2img` 投影到四路相机。

### 7.3 多相机可变形交叉注意力

对于每个 query：

1. 判断它的参考点在哪些相机中可见；
2. 只把 query 发送给对应相机；
3. 在投影参考点附近使用 `MSDeformableAttention3D` 采样图像特征；
4. 汇总多个相机的结果；
5. 按可见相机数量归一化；
6. 通过残差连接更新 query。

因此 planner query 编码的是“候选轨迹经过位置附近的视觉上下文”，而不是整幅图像的简单全局池化。

### 7.4 轨迹参考点被 detach

`Bev_refiner.forward` 中存在：

```python
ref_2d = pose.detach()
```

含义为：

- 前向时，轨迹坐标仍决定图像采样位置；
- 反向时，损失不会通过采样坐标回传到 `traj_decoder`；
- 图像信息会更新 query，并影响下一轮轨迹；
- trajectory MLP 主要由轨迹回归损失训练。

### 7.5 `TemporalSelfAttention` 不使用历史帧

虽然模块名为 `TemporalSelfAttention`，当前调用没有传入 `prev_bev`。

当 `value is None` 时，代码复制当前 query：

```python
value = torch.stack([query, query], 1)
```

所以它实际执行的是当前 512 个轨迹 query 之间的可变形自注意力，而不是跨帧时序注意力。

## 8. Trajectory head 与 Scorer 的边界

Trajectory head 负责输出：

```text
proposals: [B,64,8,3]
```

神经 Scorer 负责输出：

```text
pred_logit: [B,64,6]
```

二者职责为：

```text
traj_decoder：生成候选轨迹坐标
Bev_refiner：根据轨迹位置读取图像并更新query
Scorer：预测候选轨迹质量
RAPModel：根据分数选择最终轨迹
```

“默认生成 64 条候选轨迹，每条 8 个未来位姿，做 4 轮共享参数细化”描述的是 trajectory head；“预测 6 个 logit，并用综合 PDM 分数选择轨迹”描述的是神经 Scorer 和 `RAPModel` 的选择逻辑。

## 9. 两种 Scorer

RAP 中存在两种功能不同的 scorer：

| 属性 | 规则 PDM scorer | 神经 Scorer |
|---|---|---|
| 输入 | 候选轨迹 + metric cache + 未来场景信息 | 与候选轨迹关联的 planner query 特征 |
| 是否使用未来真值 | 是 | 否 |
| 是否可微 | 否 | 是 |
| 计算成本 | 高，需要插值、仿真和几何检测 | 低，主要为 MLP |
| 训练阶段作用 | 生成 PDM 监督标签 | 学习逼近规则 PDM 分数 |
| 推理阶段作用 | 不运行 | 给 64 条候选排序 |

二者不是并列 ensemble，而是教师与学生的关系：

```text
规则PDM scorer：定义什么是好轨迹
神经Scorer：在没有未来真值时预测哪条轨迹更好
```

## 10. 规则 PDM scorer

### 10.1 输入

默认 NAVSIM 分支中，规则 PDM scorer 只接收最后一轮的 64 条候选：

```text
proposals_3: [B,64,8,3]
```

每条候选包含未来 4 秒、0.5 秒间隔的 8 个局部位姿。

输入在评分前被 detach：

```python
proposals = proposals.detach()
```

规则 scorer 还需要 metric cache 中的：

- 当前自车绝对状态；
- 未来动态智能体 observation；
- 道路中心线；
- route lane IDs；
- 可行驶区域地图；
- baseline PDM progress。

这些未来信息在真实在线推理中不可获得。

### 10.2 轨迹转换与仿真

每条 8 点局部轨迹首先被转换到全局坐标系，然后插值为：

```text
40个未来状态 × 0.1秒
```

接着由 `PDMSimulator` 产生约：

```text
[N,41,11]
```

的仿真状态。41 包括当前状态和 40 个未来状态；11 维包含：

- x、y、heading；
- x/y 速度；
- x/y 加速度；
- 转向角和转向率；
- 角速度和角加速度。

PDM 评分对象是仿真后的车辆状态，而不只是原始 8 个轨迹点。

### 10.3 六个输出分数

每条候选得到：

```text
[no_collision,
 drivable_area,
 progress,
 ttc,
 comfort,
 final_pdm_score]
```

输出形状为：

```text
target_scores: [B,64,6]
```

具体语义如下：

1. `no_collision`：无责任碰撞通常为 1；与动态智能体发生责任碰撞为 0；部分静态物体责任碰撞为 0.5。
2. `drivable_area`：任一时间步进入不可行驶区域则为 0，否则为 1。
3. `progress`：轨迹起终点沿中心线的正向进度，并相对 baseline progress 归一化。
4. `ttc`：根据速度外推 footprint 并检查未来碰撞风险；存在有效风险为 0，否则为 1。
5. `comfort`：检查加速度、jerk、横向动态和转向相关指标，全部满足阈值为 1。
6. `final_pdm_score`：乘法安全门控与加权指标的综合结果。

### 10.4 最终 PDM 公式

默认乘法门控为：

```text
M = NoCollision × DrivableArea
```

默认加权部分为：

```text
W = (5×Progress + 5×TTC + 2×Comfort) / 12
```

最终分数为：

```text
PDMScore = M × W
```

代码还计算 driving direction compliance，但默认权重为 0，因此不影响最终分数。

### 10.5 规则 scorer 的额外训练输出

训练模式下还会返回：

- 导致责任碰撞的关键智能体角点；
- 导致 TTC 风险的关键智能体角点；
- 对应的有效标签；
- 自车是否处于多车道、不可行驶区域和逆向区域。

这些数据用于神经 Scorer 的辅助 head。

## 11. 神经 Scorer

### 11.1 输入不是显式轨迹坐标

神经 Scorer 定义：

```python
self.pred_score = MLP(1280, 1024, 6)
```

`Scorer.forward(proposals, bev_feature)` 虽然接收 `proposals`，但轨迹坐标本身没有直接送入 `pred_score`。

代码实际执行：

```python
p_size = proposals.shape[1]
t_size = proposals.shape[2]

proposal_feature = bev_feature.reshape(
    batch_size,
    p_size,
    t_size,
    -1,
).amax(-2)

pred_logit = self.pred_score(proposal_feature)
```

张量变化为：

```text
planner query [B,512,1280]
       ↓ reshape
[B,64,8,1280]
       ↓ 时间维max pooling
[B,64,1280]
       ↓ 评分MLP
[B,64,6]
```

`proposals` 在 `Scorer.forward` 中只用于获得 `64` 和 `8` 两个 reshape 维度。

### 11.2 为什么 query 可以表示某条轨迹

因为：

```text
query[p,t] ↔ proposal[p,t]
```

并且 query 已经根据该轨迹点的 `(x,y,heading)`，从对应空间位置的多相机图像中读取过视觉信息。

因此 `proposal_feature[p]` 可以理解为：

```text
第p条候选轨迹沿途的、包含场景上下文的高维表示
```

神经 Scorer 学习的是：

```text
score_p = f(沿候选轨迹p提取到的场景特征)
```

而不是单纯：

```text
score_p = f(x1,y1,h1,...,x8,y8,h8)
```

### 11.3 轨迹坐标仍会间接影响评分

“不直接输入”不等于“完全无影响”。轨迹坐标通过以下路径影响 Scorer：

```text
(x,y,heading)
      ↓
决定车辆footprint和图像投影位置
      ↓
决定从相机图像的什么位置采样
      ↓
更新对应planner query
      ↓
输入评分MLP
```

坐标没有与 query 显式拼接，但它决定了 query 获取到的视觉上下文。

仅输入坐标无法区分“相同几何轨迹、不同场景风险”；使用图像条件 query 可以让相同轨迹在空旷道路和前方有车时得到不同分数。

### 11.4 当前评分损失的实际监督

神经 Scorer 输出 6 个 logit，并计算：

```python
sub_score_loss = BCE(pred_logit, target_scores)
final_score_loss = BCE(
    pred_logit[..., -1],
    target_scores[..., -1],
)
```

但默认配置为：

```text
sub_score_weight = 0
final_score_weight = 1
```

因此：

- 第 6 个综合 PDM 分数有直接监督；
- 前 5 个分项 logit 的联合损失虽然被计算，但不进入总损失；
- 当前 checkpoint 中前 5 个 logit 不应被视为可靠的分项指标预测；
- 碰撞智能体和危险区域由独立辅助 head 监督。

## 12. 训练与推理阶段的评分差异

### 12.1 四轮轨迹的监督范围

| 轨迹集合 | trajectory regression | 规则 PDM 评分 | 神经 Scorer | 推理候选 |
|---|---:|---:|---:|---:|
| `proposals_0` | 是 | 否 | 否 | 否 |
| `proposals_1` | 是 | 否 | 否 | 否 |
| `proposals_2` | 是 | 否 | 否 | 否 |
| `proposals_3` | 是 | 是 | 是 | 是 |

规则 PDM scorer 和神经 Scorer 都只处理最后一轮的 64 条候选。

### 12.2 训练阶段

训练流程为：

```text
trajectory head生成最后一轮64条候选
                 │
                 ├→ 规则PDM scorer生成target_scores
                 │
                 └→ 神经Scorer生成pred_logit
                                      │
                    BCE(pred_logit, target_scores)
```

只有 `score_mask=True` 且具有有效 metric cache 的样本会运行规则 PDM 评分。

### 12.3 推理阶段

推理阶段不运行：

- `PDMSimulator`；
- metric cache；
- 未来智能体 observation；
- 规则 PDM scorer。

仅执行：

```python
pdm_score = torch.sigmoid(pred_logit)[..., -1]
best_idx = torch.argmax(pdm_score, dim=1)
trajectory = proposals[batch_index, best_idx]
```

输入输出为：

```text
最终候选：[B,64,8,3]
预测分数：[B,64]
最终轨迹：[B,8,3]
```

当 `return_score=True` 时，不执行选择，而是返回全部候选和分数：

```text
trajectory: [B,64,8,3]
score:      [B,64]
```

这只是同一组候选的两种返回方式，不是两种 trajectory head。

## 13. 为什么需要两种 Scorer

### 13.1 只有规则 PDM scorer 的问题

规则 PDM scorer 能准确评估候选，但：

- 需要未来场景真值，在线推理不可获得；
- 计算成本高；
- 不可微；
- 推理时使用会造成未来信息泄漏；
- 如果推理时不使用它，64 条候选没有可靠的选择器。

best-of-64 trajectory loss 只保证候选集合中至少有一条接近 GT，并不保证固定索引对应最佳轨迹。

### 13.2 只有神经 Scorer 的问题

神经 Scorer 本身不知道什么是正确分数。没有规则 PDM 标签时：

- scorer head 缺少监督目标；
- trajectory loss 不会自动教会 scorer 排序；
- `argmax` 选择不可微；
- 输出可能保持随机或产生无意义排序。

可以用 ADE/FDE、碰撞标签或人工规则替代，但此时学习目标不再是当前定义的综合 PDM 分数。

### 13.3 当前设计的分工

```text
规则PDM scorer：使用特权未来信息定义“好轨迹”
神经Scorer：学习在没有未来信息时近似这个判断
```

这是“高成本教师 + 低成本学生”或“规则 evaluator + amortized scorer”的设计。

## 14. 为什么 PDM 标签需要动态计算

### 14.1 PDM 标签绑定具体轨迹，不绑定候选编号

PDM 分数是：

```text
PDMScore = f(场景, 具体候选轨迹)
```

不是：

```text
PDMScore = f(场景token, proposal索引)
```

同一个场景中，不同轨迹有不同 PDM 分数；同一个 `proposal_0` 在不同训练阶段也可能表示完全不同的轨迹。

### 14.2 候选轨迹随模型参数变化

设 trajectory head 为：

```text
trajectory = G_theta(scene)
```

训练更新参数后：

```text
theta_t != theta_(t+1)
```

因此同一场景生成的候选一般也会变化：

```text
trajectory_t != trajectory_(t+1)
```

旧轨迹对应的 PDM 标签不能直接监督新轨迹。

例如：

```text
Epoch 1 proposal_5：直行撞车 → PDM 0.0
Epoch 10 proposal_5：提前减速 → PDM 0.9
```

如果始终使用 Epoch 1 保存的 `proposal_5 → 0.0`，就会错误压低新的安全轨迹分数。

### 14.3 当前动态标签流程

当前训练每次使用模型最新候选：

```text
1. 当前trajectory head生成候选
2. 规则PDM scorer给当前候选打分
3. 神经Scorer预测当前候选分数
4. 计算评分损失
5. 更新模型
6. 下次对新候选重新评分
```

这样标签始终与当前候选分布匹配。

### 14.4 何时可以离线预计算标签

以下情况可以把规则 PDM 评分移到离线阶段：

1. 冻结 trajectory head，使同一场景始终生成相同候选；
2. 使用固定的轨迹模板库；
3. 定期冻结 checkpoint，重新生成候选和一版 PDM 标签。

如果冻结 trajectory head，可以采用两阶段训练：

```text
阶段1：训练并冻结轨迹生成器
阶段2：生成固定候选 → 离线PDM评分 → 单独训练神经Scorer
```

训练代码运行时可以只读取离线标签，但系统的数据准备阶段仍然需要规则 PDM scorer。

## 15. Trajectory head 的监督来源

### 15.1 标准 NAVSIM 样本

`RAPTargetBuilder` 调用：

```python
scene.get_future_trajectory(
    num_trajectory_frames=8
).poses
```

`Scene.get_future_trajectory` 从 NAVSIM Scene 中读取：

```text
当前帧ego pose + 未来8帧ego pose
```

然后将未来全局位姿转换到当前自车后轴局部坐标系：

```text
全局未来ego pose
       ↓ absolute-to-relative SE(2)
局部未来轨迹 [8,3]
```

因此标准样本的监督语义来自 NAVSIM/nuPlan 记录的人类驾驶未来轨迹，但实际 target 已经过坐标转换。

### 15.2 训练时读取缓存

target builder 的输出被保存为：

```text
<cache>/<log>/<token>/rap_target.gz
```

训练时 `CacheOnlyDataset` 直接读取缓存，而不是每次重新解析原始日志。

所以：

| 角度 | 来源 |
|---|---|
| 轨迹语义 | NAVSIM/nuPlan 未来 ego pose |
| 坐标表示 | 当前自车后轴局部坐标 |
| 训练时实际读取 | 处理并缓存的 `rap_target.gz` |
| 是否由 PDM 生成 | 否 |
| 是否为模型伪标签 | 否 |

### 15.3 扰动恢复数据

完整训练还加载 `training_cache_perturbed`。

扰动生成脚本对当前关键帧施加：

- x/y ±0.5 m；
- heading ±15°；
- 速度 ±20%；
- 加速度 ±10%。

未来帧仍来自原始记录。因此相对轨迹 target 表示：

```text
从扰动后的当前状态，回到原始未来驾驶路线的恢复轨迹
```

### 15.4 跨智能体增强数据

`training_cache_others` 来自 cross-agent synthesis：

- 选择场景中的其他车辆作为虚拟自车；
- 当前状态来自该车辆；
- 未来 trajectory target 来自该车辆后续真实 track；
- 其他场景元素被变换到虚拟自车坐标系。

该数据的 `score_mask=False`，对应 trajectory regression 样本权重降为 0.1，不运行规则 PDM 评分。

### 15.5 Trajectory regression loss

四轮候选都与同一个：

```text
targets["trajectory"]: [B,8,3]
```

比较。每轮使用 best-of-64：

```python
min_loss = norm(
    proposals_i - target_trajectory[:, None],
    p=1,
).mean(time).amin(proposal)
```

含义为：

1. 计算 64 条候选与 GT 的轨迹误差；
2. 选择误差最小的候选；
3. 使用该最小误差训练当前轮 trajectory head。

四轮损失递推：

```python
trajectory_loss = 0.1 * previous_loss + current_min_loss
```

四轮最终近似权重为：

```text
第1轮：0.001
第2轮：0.01
第3轮：0.1
第4轮：1.0
```

PDM scorer 不提供轨迹坐标监督；它只给最后一轮候选提供质量分数监督。

## 16. 训练时真实图像与渲染图像

开启 `distill_feature=True` 时，训练 wrapper 会将渲染图像 batch 与有效真实图像 batch 拼接，并复制相应 target。

因此同一个场景的渲染分支和真实图像分支使用相同的：

```text
targets["trajectory"]
```

栅格图像不会生成另一套轨迹标签。它主要通过：

- 共享图像编码器；
- 特征 MSE 对齐；
- 域对抗损失；

影响真实图像特征的学习。

推理时只输入真实相机图像，不输入渲染栅格。

## 17. 容易混淆的实现细节

### 17.1 `bev_feature` 有两个不同含义

`RAPModel.forward` 内部的局部变量 `bev_feature` 通常表示：

```text
planner trajectory query [B,512,1280]
```

但输出字典中的：

```python
output["bev_feature"] = image_feature[0].permute(2,0,1,3)
```

实际是 FPN 后的图像 token，不是 planner query。

### 17.2 `self._trajectory_head` 的名字容易让人误以为有四套 head

它包含四个列表位置，但四个位置引用同一个 `shared_refiner`，所以是共享参数的重复调用。

### 17.3 `TemporalSelfAttention` 不是时序历史模块

当前没有 `prev_bev`，只处理当前 query。

### 17.4 神经 Scorer 不直接拼接轨迹坐标

评分 MLP 的直接输入是时间池化后的 1280 维 query 特征；轨迹坐标通过图像采样位置间接影响该特征。

### 17.5 六个评分 logit 并非全部得到有效监督

默认 `sub_score_weight=0`，只有最后一个综合 PDM logit 直接进入评分总损失。

### 17.6 `return_score` 不代表另一种 trajectory head

`return_score=False` 返回最高分单条轨迹 `[B,8,3]`；`return_score=True` 返回全部候选 `[B,64,8,3]` 和分数 `[B,64]`。两者来自同一套候选。

## 18. 最终总结

RAP planner 可以分成四个职责清晰的部分：

```text
1. ImgEncoder
   当前帧四路图像 → 多相机DINOv3/FPN token

2. Shared Trajectory Refiner
   512个query → 64×8轨迹点 → 轨迹条件图像采样
   同一套参数循环执行4次

3. Rule PDM Scorer（仅训练/验证标签生成）
   最后一轮候选 + 未来场景metric cache
   → 仿真、碰撞、道路、进度、TTC、舒适度评分

4. Neural Scorer（训练和推理）
   轨迹相关planner query → 综合PDM预测分数
   → 推理时从64条候选中选择最终轨迹
```

最重要的三条边界是：

1. trajectory head 负责“生成候选”，神经 Scorer 负责“给候选排序”；
2. trajectory head 的坐标监督来自 NAVSIM 派生的局部未来轨迹，规则 PDM 只提供质量分数；
3. 规则 PDM 使用未来特权信息定义教师标签，神经 Scorer 学习在推理时没有未来信息的情况下近似这些标签。
