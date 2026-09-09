# A3｜Checkpoint 结构与加载方式

- 审计时间：2026-09-09 17:32–17:36 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex
- `torch`：2.1.0+cu121

## 判定

**PASS（格式审计完成）；全部 `ALLOWED_AS_INIT: no`。** 本轮仅 CPU load + hash，没有把任何 checkpoint 加载进实验模型。

## `RAP_DINO_navsimv1.ckpt`

- Size：3,936,976,557 bytes
- SHA-256：`4823cc6c1dcb7801138eef2dd835287380817741d88d0c9e02b2b93cc4baf4da`
- Top-level：`dict`
- Lightning 判据：含 `state_dict`、`epoch`、`global_step`
- Tensors：926；首级前缀全部为 `agent.`
- 未发现 `hyper_parameters` top-level key。
- 加载方式：本仓库 `RAPAgent.init_from_pretrained()` 取 `checkpoint["state_dict"]` 后去除 `agent.`；`initialize()` 替换 `agent._rap_model`。由于 checkpoint 不含 Lightning `hyper_parameters` 且 module 构造需要 `agent`，不能只写成无参数 `AgentLightningModule.load_from_checkpoint()`。
- **ALLOWED_AS_INIT: no**

## `RAP_DINO_navsimv2.ckpt`

- Size：3,936,971,245 bytes
- SHA-256：`9accbb101f30541187c7bb061689bf88f481b2125ad9a9c563a488275ea0311d`
- Top-level / keys / tensor count / prefix 与 v1 相同。
- 加载方式：同 v1。
- **ALLOWED_AS_INIT: no**

## `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`

- Size：3,363,232,567 bytes
- SHA-256：`7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488`
- Top-level：`OrderedDict`，全部 value 为 tensor-like。
- Tensors：552。
- Prefix：`blocks` 544；`patch_embed` 2；`norm` 2；另有 `cls_token`、`storage_tokens`、`mask_token`、`rope_embed`。
- 类别：纯 backbone state_dict / HF-timm 风格。
- 格式加载方式：目标 backbone 对齐 key 后调用 `model.load_state_dict(torch.load(...))`；本轮只做格式审计。
- **ALLOWED_AS_INIT: no**

F0/F1 必须 from scratch。只允许同一 `run_id` 且 code/config/data hash 完全一致时从该 run 自身 checkpoint 续训。
