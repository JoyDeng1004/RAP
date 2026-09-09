# RAP Paired Alignment Stage-A 最终报告

> 证据口径：本报告只读转录已落盘产物，不从 CSV、日志或 cache 重算实验统计量。冻结统计主分析来自 `outputs/alignment_stage_a/summary/stage_a_report.json`；另纳入 `per_run_subscore_means.json` 的描述性 per-run 均值、`loss_component_diagnostics.json` 的训练期机制诊断，以及 `requirement_coverage.json` 的覆盖审计。`summary_protocol_v3.json` 明确处于 `DRAFT`，本文只披露其内容和状态，不把它当作统治性协议。表内“来源”覆盖该行全部量化事实；逐字引文和 JSON 代码块由其引导句统一标注来源。章节号仅用于文档导航，不是实验数字。

## 0. 状态声明

**status = `incomplete`。** 原因逐字引用如下：

> 官方 v2.2 evaluator 缺失（spec §5 co-primary；runbook Step 5.3：Missing official evaluator -> HALT and mark the experiment incomplete）。

（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`report_status.reason`）

因此，本报告不构成正式 Stage-A 结论；它是一份 `pipeline_rehearsal_v1_only` 的不完整报告（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`run_scope.designation`、`report_status`）。

三类输入在本文中的角色是：实验计划定义“本该做什么”，runbook 定义“实际怎么做”，`outputs/alignment_stage_a/**` 定义“做出了什么”。`stage_a_report.json` 的 `operative_protocol` 仍为 `summary_protocol_v2.json`；`summary_protocol_v3.json` 的 `freeze_status` 明确要求人工审阅，在 `amended_by`、`amended_at`、`human_assessment` 未填写前不得作为分析输入，因此当前不能取代 v2。评测阶段以 `rap-alignment-stage-a-step5-v1-only.md` 的 S.1 至 S.8 和实际产物为准，不采用 runbook 被取代的 Step 2.3、Step 5.1、Step 5.2、Step 5.3 原流程（`outputs/alignment_stage_a/summary/stage_a_report.json`，`operative_protocol`；`summary_protocol_v3.json`，`freeze_status`；`docs/rap-alignment-stage-a-step5-v1-only.md`）。

## 1. 实验定位与范围

本轮不是规格书 §1 定义的 Stage-A pilot。协议给出的三项实质偏离为：候选池仅为标称值的 `21%`，real sensor blobs 覆盖率为 `20.887%` 且缺失呈非随机块状；实测预算为 `400` optimizer steps，约为官方配置的 `1/100`；官方 `v2.2` evaluator 未接入，co-primary family 不完整（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`run_scope.rationale`）。这使 null 结果成为该预算下并不意外的观察，但不授权作方向性结论。其中“`400` 相对规格书冻结值的偏离”与当前计划文档存在后续审计发现的不一致，见第 12 节；此处按统治性协议原意转录。

协议允许用途原文如下（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`run_scope.permitted_use`）：

> - 端到端 pipeline 可运行性验证
> - 两条件受控性的运行时证据（NoAlign alignment 梯度严格为零、首个 optimizer step 的 task_loss 两条件相同）
> - 训练与评测的时间/显存成本实测，用于 Stage-B 排期
> - 统计脚本与 artifact 链条的演练

协议禁止用途原文如下（同文件，`run_scope.forbidden_use`）：

> - 对 paired real-raster alignment 是否改善 planning performance 给出任何方向性结论
> - 作为 Stage-B 的效应量先导估计（spec §1 赋予 Stage-A 的唯一职能）
> - 用于修改 Stage-B 的任何超参、样本量、标签规则、阈值或评测协议（spec §1 明令禁止）
> - 对外表述为 NAVSIM v1.1 官方 Stage-A 结果

## 2. 执行概览：需求、执行与结果

### Phase -1：环境、架构与代码基线

需求是冻结环境入口、核对 DINO 架构并保存代码基线；执行结果为 Step `-1.1`、`-1.2`、`-1.3` 全部 passed，基线 commit 为 `6ed7a39bf9d2521b42b0edde0a28c49d8ca332e6`。关键产物为 `env_versions.json`、`dino_reference_config.json`、`phase_minus1_gpu_gate.log` 与 `baseline_*`（`outputs/alignment_stage_a/phase_minus1_to_phase3_summary.json`）。

### Phase 0：冻结源、构建缓存与四相机根

需求是冻结源校验、构建 train metric cache、四相机硬链接和 target-only cache；实际 Step `0.1` 至 `0.4` 均 passed。产物集中于 `input_data/source_checksums.json`、`hardlink_manifest*.json*`、`b0_generation_manifest.jsonl`、`target_cache_checksum.json` 与 `target_cache/`（`outputs/alignment_stage_a/phase_minus1_to_phase3_summary.json`）。全量硬链接存在未单独留痕的排除项，见第 12 节（`outputs/alignment_stage_a/summary/requirement_coverage.json`，`EXP-006`）。

### Phase 1：候选池、确定性抽样与分布审计

需求是按 log 比例做确定性哈希抽样、冻结 manifest 并在训练前审计分布。实际构建与数据审计通过，但 Step `1.3` 因 val `route_command` TVD 超阈值被人工以 deviation 接受；未重抽、未改算法、未改阈值（`outputs/alignment_stage_a/phase_minus1_to_phase3_summary.json`；`outputs/alignment_stage_a/input_data/distribution_audit_resolution.json`）。候选池缩小被 P.1 诊断为 `data_bug` 后，由人类明确接受继续并把本轮降格为 pipeline rehearsal（`outputs/alignment_stage_a/input_data/exclusion_reason_histogram.json`；`human_override_p1.json`）。

### Phase 2：集成测试、navtest cache 与 evaluator gate

Step `2.1`、`2.2` passed；原 Step `2.3`、`2.4` 在阶段汇总中记为 explicit authorization 下 skipped，随后其评测职责由 step5 文档取代。关键产物为 `step_2_1_tests.txt`、`step_2_2_navtest_metric_cache.log`、`evaluator_validation/`。旧 `SKIPPED.md` 对后续实际路线的描述已经失真，见第 12 节（`outputs/alignment_stage_a/phase_minus1_to_phase3_summary.json`；`outputs/alignment_stage_a/summary/requirement_coverage.json`，`CON-009`）。

### Phase 3：初始化、配置、sampler 与 callbacks

Step `3.1` 至 `3.4` passed。每个 seed 的两条件共享同一初始化 checkpoint；配置解析、显存 smoke、无 padding DDP sampler 和 callback 测试均落盘。关键产物为 `init_checkpoints/`、两份 `resolved_config_*.yaml`、`memory_smoke/`、`step_3_3_sampler_gate.json`、`step_3_4_pytest.xml`（`outputs/alignment_stage_a/phase_minus1_to_phase3_summary.json`）。

### Phase 4：两条件训练

按固定顺序执行 NoAlign 与 FullAlign，随后逐 seed 校验。三个 `validation_seed_*.json` 的 `status` 均为 `passed`，各自记录训练 token `2498`、验证 token `464`、effective global batch `128`、epochs `20` 和 optimizer steps `400`（`outputs/alignment_stage_a/validation_seed_0.json`、`validation_seed_1.json`、`validation_seed_2.json`）。六个训练目录保存 final checkpoint、逐 epoch 曲线、`loss_components.jsonl`、token checksum 和 reproducibility manifest（同上及 `outputs/alignment_stage_a/training/`）。

### Phase 5：按 S.1 至 S.8 执行的评测与汇总

| Step | 实际执行 | 关键产物 | 判定 |
|---|---|---|---|
| S.1 | 冻结 v1 协议并建立 token-to-log cluster key | `summary/summary_protocol.json`、`summary/token_to_log.json` | 映射覆盖 `12146` token、`136` logs（`outputs/alignment_stage_a/summary/token_to_log.json`） |
| S.1b | 审计三条收敛曲线可用性 | `summary/convergence_curve_audit.json` | `status=complete`，三条曲线齐全（同文件） |
| S.2 | 固定 official checkout，确认绝对 baseline 数值不存在 | `evaluator_validation/v1_commit.txt`、`v1_tag.txt`、`v1_paper_reference.json` | 原绝对值门禁作废（`docs/rap-alignment-stage-a-step5-v1-only.md`；上述产物） |
| S.3/S.3b | scorer 同源性检查、差异分类与 A/B 测量 | `v1_scorer_provenance.json`、`pdm_planner_diff_analysis.json`、`batch_lqr_ab_test.json` | 发现 `1` 个语义差异文件，A/B `passed`（上述 JSON） |
| S.3c | 冻结并复检评测代码基线 | `evaluation_v1/code_baseline/code_baseline.json`、`code_baseline_recheck.json` | 关键文件与子树摘要均 match，复检 `passed`（复检文件） |
| S.4 | 跑关系式门禁 G1–G7 | `v1_relational_gate.json`、`v1_regression.json` | failures 为空，`status=passed`（`v1_regression.json`） |
| S.4b | 为评测时全部缺失的 rendered 分支加入零占位，并验证数值惰性 | `eval_smoke/smoke_report.json`、评测代码基线记录 | smoke `rows=8`、`all_valid=true`、`ms_deformable_errors=0`（`smoke_report.json`） |
| S.5 | checkpoint 加载 smoke | `eval_smoke/` | schema、有效性与有限值检查通过（`smoke_report.json`） |
| S.6 | 完成六次完整 navtest v1 评测 | `evaluation_v1/*.csv`、`step_5_1_audit.json` | `n_csv=6`、token `12146`、log `136`，六份 score signature 不同，`status=passed`（审计文件） |
| S.7 | 记录 v2 延期 | `evaluator_validation/v2_deferral_record.json` | 人类决定只执行 v1.1；EPDMS `not_tested`（该记录及 `stage_a_report.json`） |
| S.8 | 由冻结统计脚本汇总 | `summary/stage_a_report.json`、`paired_deltas.csv` | `status=incomplete`（`stage_a_report.json`） |
| S.8 后描述性补充 | 汇总六个 run 的七列百分比均值；不由 `summarize_stage_a.py` 产生 | `summary/per_run_subscore_means.json` | 六个 CSV 各 `12146` scenes，源 CSV SHA256 全部记录（同文件） |
| S.8 后训练期诊断 | 只读分析 alignment loss 绝对量、加权占比、塌陷与实际 GRL 时序 | `summary/loss_component_diagnostics.json` | NoAlign 独立复核 `PASS`；方法限制与开放问题已落盘（同文件） |
| 覆盖审计 | 对四份规范文档的 `99` 项要求做覆盖分类 | `summary/requirement_coverage.json` | `57 IMPLEMENTED / 6 SUPERSEDED / 11 WAIVED_RECORDED / 3 WAIVED_UNRECORDED / 13 MISSING / 9 OUT_OF_SCOPE`（同文件，`status_summary`） |

## 3. 数据来源与代表性

### 3.1 协议要求披露的全部 caveat

以下十条逐字转录自 `outputs/alignment_stage_a/summary/summary_protocol_v2.json` 的 `data_provenance_caveats_required_in_report`：

> 1. Stage-A 子集是在合格候选池上按 log 规模比例分层的确定性哈希抽样，无 RNG、无 seed。
> 2. 合格候选池本身受 real sensor_blobs 可用性限制：real 三相机抽样覆盖率 20.887%，raster 100%；32,974/32,975 个被排除候选的原因是 real 图像缺失。
> 3. spec §3 的抽样分布审计比较的是「选中子集 vs 候选池」，结构上无法检出「候选池 vs 真实全集」的偏差。因此本次 run 的分布代表性不受该审计背书。
> 4. 缺失呈 log/chunk 块状分布（一个 val log 为 159/159 全缺），因此偏差与 drive、city、time-of-day 相关，不可视为随机缺失。
> 5. distribution_audit.json 的完整 summary 必须原样嵌入报告。
> 6. val split 的 route_command TVD = 0.023308 超出冻结阈值 0.02，已由人工接受；报告的外推章节必须按 runbook Step 5.3 声明该方向的代表性风险。
> 7. selected train logs 43/44、val logs 9/10，未满足 spec §7「每个 official train log 均有 token 被抽中」；四个 map_location 全部出现。
> 8. spec §4 已知偏离：DINO backbone 的实际 lr 为 2e-5（rap_agent.py configure_optimizers 中 vit_lr = lr * 0.2），spec §4 只记录了 lr = 1e-4。该偏离对两条件完全相同。
> 9. 评测使用 RAP fork 的 scorer，与 official v1.1 的唯一实质差异为 batch_lqr_utils.py 的伪逆实现；完整 navtest 实测 9/12,146 场景有非零差异，max |ΔPDMS| = 3.35e-12，均值差 1.39e-16。详见 scorer_equivalence_record.json。
> 10. constant_velocity baseline 在本流水线上得到 PDMS = 20.6517%，论文 Table 1 为 20.6，差 0.05 个百分点。该复现同时验证了 metric cache、地图版本、数据集版本、LQR 仿真与 PDMScorer 整条链路。

第 8 条是当前统治性 v2 的原文，但 `summary_protocol_v3.json` 草案指出它与源码不符：DINO backbone 在训练全程 `requires_grad=False`，vit optimizer 参数组被注释，`vit_lr` 只计算而未使用。v3 草案是在结果观测后起草的事实更正，尚待人工接受；其 freeze 状态、selection-on-result 披露和后续动作见第 10.2 节。本文保留 v2 原文以满足冻结协议披露，同时不把该原文当作准确的实际 optimizer 描述（`outputs/alignment_stage_a/summary/summary_protocol_v3.json`，`freeze_status`、`amendment.reason`、`changes_from_v2`）。

候选池 `21%` 的核心成因不是随机抽样本身，而是进入抽样前的文件可用性过滤：real 图像覆盖率只有 `20.887%`，排除项几乎全部是 real 图像缺失；缺失按 log/chunk 成块，甚至一个 val log `159/159` 全缺（`outputs/alignment_stage_a/input_data/image_coverage_probe.json`；`exclusion_reason_histogram.json`；`log_coverage_resolution.json`）。因此，分布审计即便显示“子集接近候选池”，也**结构上看不到“候选池相对真实全集”的偏差**。

### 3.2 `distribution_audit.json` 的完整 summary

以下对象原样嵌入 `outputs/alignment_stage_a/input_data/distribution_audit.json` 的 `summary`；相同对象也由 `stage_a_report.json` 转录为 `distribution_audit_summary`：

```json
{
  "train": {
    "full_count": 7495,
    "per_log_sampling_rate_max_min_ratio": 1.0443349753694582,
    "subset_count": 2498,
    "tvds": {
      "actor_density_bin": 0.013248864602021848,
      "endpoint_dx_bin": 0.001068286250080762,
      "endpoint_dy_bin": 0.01383842230555625,
      "endpoint_dyaw_bin": 0.0038495372682401732,
      "map_location": 0.0002553610600288178,
      "route_command": 0.003248976766469874
    }
  },
  "val": {
    "full_count": 1394,
    "per_log_sampling_rate_max_min_ratio": 1.0123456790123455,
    "subset_count": 464,
    "tvds": {
      "actor_density_bin": 0.014272992628506412,
      "endpoint_dx_bin": 0.001471825063078203,
      "endpoint_dy_bin": 0.01866682333151934,
      "endpoint_dyaw_bin": 0.006796368673625888,
      "map_location": 0.0004050610992925143,
      "route_command": 0.023308019591352123
    }
  }
}
```

val 的 `route_command` TVD 超过冻结 categorical 阈值，且决策记录要求披露：相对合格候选池，validation 子集略微过度代表 `TURN_LEFT`、不足代表 `GO_STRAIGHT`。这构成该方向的代表性和外推风险（`outputs/alignment_stage_a/input_data/distribution_audit_resolution.json`）。

## 4. 训练配置与受控性证据

### 4.1 冻结配置与实测预算

| 项目 | NoAlign / FullAlign | 来源 |
|---|---|---|
| train / val samples | `2498 / 464` | `outputs/alignment_stage_a/input_data/measured_training_budget.json` |
| epochs / optimizer steps | `20 / 400` | 同上 |
| effective global batch | `128`，即 devices `4`、per-device batch `32`、accumulation `1` | `outputs/alignment_stage_a/resolved_config_r100_noalign.yaml`；`resolved_config_r100_fullalign.yaml` |
| seeds / runs | `[0, 1, 2] / 6` | `outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`measured_budget` |
| task input / supervision / eval input | real / `100%` real / real | 两份 `resolved_config_*.yaml` |
| 唯一条件开关 | NoAlign spatial/global 均 `false`；FullAlign 均 `true` | 两份 `resolved_config_*.yaml` |
| alignment 权重 | spatial `0.002`；global `0.1` | 两份 `resolved_config_*.yaml` |
| optimizer / base lr / weight decay | AdamW / `1e-4` / `1e-4` | 两份 `resolved_config_*.yaml`；`docs/rap-alignment-stage-a-runbook.md` Step 3.2 |
| schedule | cosine decay；warmup `1` epoch；min lr `1e-5` | `docs/rap-alignment-stage-a-runbook.md` Step 3.2（训练配置冻结说明） |
| dropout / precision / gradient clipping | `0` / `16-mixed` / `0.0` | 两份 `resolved_config_*.yaml` |
| train/val shuffle | `true / false` | 两份 `resolved_config_*.yaml` |
| checkpoint | final optimizer step only | 两份 `resolved_config_*.yaml` |

统治性 v2 协议要求披露“DINO backbone 实际 lr = `2e-5`”；该原文已在第 3.1 节保留。实际源码/配置事实则是 backbone 参数全程冻结，vit optimizer 参数组被注释，optimizer 只有 `other_params` 一个参数组、lr=`1e-4`。这一事实更正已写入 `summary_protocol_v3.json`，但 v3 仍为待人工审阅的草案；本文因此同时记录“v2 仍统治”和“v2 此处事实表述与源码不符”，不代替人工冻结 v3（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，data caveat；`summary_protocol_v3.json`，`freeze_status`、`amendment.changes_from_v2`）。

### 4.2 运行时受控性

`validate_stage_a_seed.py` 对每个 seed 的两份 `loss_components.jsonl` 做逐行 gate；三个落盘结果均为 `status=passed`（`outputs/alignment_stage_a/validation_seed_0.json`、`validation_seed_1.json`、`validation_seed_2.json`）。这意味着实际运行满足以下已冻结校验：两条件行数相同且与 optimizer steps 一致；NoAlign 的 spatial/global effective weight 逐行为零且 total loss 逐行等于 task loss；FullAlign 的 effective weights 逐行为 `0.002` 与 `0.1`；GRL 系数单调并逐行满足冻结公式；total loss 可逐行重构；同 seed 两条件首个 optimizer step 的 task loss 相同（`docs/rap-alignment-stage-a-runbook.md` Step 4.1；上述 validation JSON 与对应 `training/**/loss_components.jsonl`）。

这些是“两条件唯一变量为 alignment loss”的运行时证据。它们不等同于规格书要求的直接参数张量/梯度测试；该测试覆盖缺口见第 12 节（`outputs/alignment_stage_a/summary/requirement_coverage.json`，`EXP-043`、`EXP-044`）。

### 4.3 alignment loss 分量诊断

本节只转录 `outputs/alignment_stage_a/summary/loss_component_diagnostics.json` 的训练期结果，不读取、不引用也不联系任何评测指标。六个 `loss_components.jsonl` 均为 `400` 行；分位数用线性插值，20 个等距采样点为 global step `1, 22, …, 400`；A4 的 `grl_lambda` 直接读取落盘值，未按公式重算。20 点完整轨迹、每个输入路径、行数与 SHA256 保存在该 JSON。

#### 4.3.1 原始 loss 绝对量

| Seed | 分量 | 首行 | 最小 | 末行 | p10 | p25 | p50 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | spatial | `0.0937586576` | `0.0937586576` | `0.202110589` | `0.139977182` | `0.168114062` | `0.194127820` | `0.214826446` | `0.264501882` |
| 0 | global | `0.699232101` | `0.000637958583` | `0.00135795027` | `0.00108435148` | `0.00202426722` | `0.00433863234` | `0.0122355730` | `0.110070540` |
| 1 | spatial | `0.112152956` | `0.0918136388` | `0.163278699` | `0.127212068` | `0.144616686` | `0.164315574` | `0.186484460` | `0.225431670` |
| 1 | global | `0.702869475` | `0.000683940947` | `0.00352528924` | `0.00119478661` | `0.00204361166` | `0.00365953392` | `0.0133341865` | `0.151601502` |
| 2 | spatial | `0.0939409584` | `0.0939409584` | `0.183248222` | `0.122894789` | `0.153581932` | `0.169899948` | `0.198088057` | `0.226060246` |
| 2 | global | `0.689047098` | `0.000726601225` | `0.00243591866` | `0.00117856690` | `0.00192535121` | `0.00341490144` | `0.0120433457` | `0.168103294` |

#### 4.3.2 加权后 loss 数值占比

占比逐 step 按 `spatial_weight_effective × spatial_loss_raw / total_loss`、`global_weight_effective × global_loss_raw / total_loss` 计算，`align_share` 为两者之和。下表已换算为百分比；时间平均是 400 step 的简单算术平均。

| Seed | 指标 | 首行 | 末行 | 中位数 | 最大值 | align 时间平均 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | spatial_share | `0.001152%` | `0.013708%` | `0.007154%` | `0.021679%` | — |
| 0 | global_share | `0.429609%` | `0.004605%` | `0.007784%` | `0.647362%` | — |
| 0 | align_share | `0.430761%` | `0.018314%` | `0.015960%` | `0.649412%` | `0.050907%` |
| 1 | spatial_share | `0.001340%` | `0.010523%` | `0.006305%` | `0.016237%` | — |
| 1 | global_share | `0.419807%` | `0.011359%` | `0.007547%` | `0.537331%` | — |
| 1 | align_share | `0.421147%` | `0.021882%` | `0.015318%` | `0.539265%` | `0.056108%` |
| 2 | spatial_share | `0.001085%` | `0.009314%` | `0.006049%` | `0.015144%` | — |
| 2 | global_share | `0.397735%` | `0.006190%` | `0.006618%` | `0.545722%` | — |
| 2 | align_share | `0.398820%` | `0.015504%` | `0.014519%` | `0.547301%` | `0.055393%` |

#### 4.3.3 塌陷、实际 GRL 时序与跨 seed 一致性

| Seed | global 首次到首行 50% / 10% / 1% | spatial 的 50% / 10% / 1% | `grl>=0.05` step / global 剩余 | `grl>=0.09` step / global 剩余 |
|---:|---|---|---|---|
| 0 | `27 / 44 / 80` | 均未达到 | `45 / 15.6620%` | `119 / 0.9481%` |
| 1 | `28 / 54 / 103` | 均未达到 | `45 / 21.4279%` | `119 / 1.3913%` |
| 2 | `30 / 53 / 107` | 均未达到 | `45 / 15.6004%` | `119 / 1.7824%` |

三个 FullAlign run 的 global loss 首次到 1% 时点 `80/103/107` 均早于实际 `grl_lambda>=0.09` 的 step `119`，源诊断据此将本次 run 标为时序错配。1% 时点跨 seed 相差 `27` steps，占 400 steps 的 `6.75%`；50% 与 10% 时点范围分别为 `3` 与 `10` steps。`domain_accuracy` 可按 epoch 取得：epoch 0 的三个原始值为 `0.909455–0.918470`，从 epoch 1 起均约为 `1.0`；少量聚合落盘值略高于 `1.0`，源诊断原样保留且未截断。NoAlign 三个 run 共 `1200` 行的独立复核结果为 `PASS`：两个 effective weight 的非零行均为 `0`，`|total_loss-task_loss|>1e-6` 的行均为 `0`，最大绝对差为 `0`（`outputs/alignment_stage_a/summary/loss_component_diagnostics.json`，A3–A6、`domain_accuracy`）。

方法学限制原文如下（同文件，`methodological_limitations`）：

> L1  **loss 量级不等于梯度贡献。** 一个数值很小的 loss 项仍可能产生较大梯度，取决于 ∂L/∂θ 而非 L 本身。A2 的占比是**代理指标，不是证明**。
>
> L2  训练期**未记录 gradient norm**（Phase 4 已确认）。因此从现有产物**无法给出 alignment 梯度贡献的确定性结论**。报告必须显式写明这一点，不得把 A2 的占比表述成「梯度贡献占比」。
>
> L3  即便发现 global alignment 贡献极小，这也只是**本次实现在 400 步、2,498 样本、且 real/raster 域极易区分条件下的观测**，不构成对 global alignment 方法本身的评价。措辞必须限定在本次 run。

开放问题原文为：“判别器的轻易取胜是训练预算过小的后果，还是 real/raster 两域本身就过易区分？现有产物无法区分这两种解释。”本报告不选择其中一种解释，也不把本节与评测指标的方向、符号或差值联系起来。

## 5. 收敛判定

全部判据、观测值和 verdict 均直接转录自 `outputs/alignment_stage_a/summary/stage_a_report.json` 的 `convergence_gate`：

| 判据 | 冻结要求 | 实测 | 结果 |
|---|---|---|---|
| 最小 val loss 位置 | 全部 `6` 个 run：`argmin(val_loss) <= max_epochs - 3`，不得落在最后两个 epoch | 全部 `6` 个 run 均为 epoch `19`（`max_epochs - 1`） | 未通过 |
| 末段斜率 | 全部 `6` 个 run：最后 `5` 个 epoch 的绝对斜率不超过 `0.02 × 首末差值 / epochs` | `[-0.048, -0.057, -0.046, -0.049, -0.052, -0.065]` | 未通过 |
| 总 verdict | 两项必须同时满足 | `FAILED` | under-trained regime |

`outputs/alignment_stage_a/summary/convergence_curve_audit.json` 的 `status=complete`，且 train loss、val loss、`val/score` 三条逐 epoch 曲线对全部运行齐全（同文件）。训练 loss 曲线在 v1 协议冻结前已被观测，因此阈值不是盲设；PDMS 统计规则冻结时尚未观测 PDMS（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`convergence_gate.blinding_disclosure`）。

按协议，**全部结论都带 under-trained regime 标注**，且不得为改善收敛追加 epoch（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`convergence_gate.consequence`）。

## 6. 评测器可信度

### 6.1 scorer 等价性

必须表述逐字引用如下：

> 使用 RAP fork 的 NAVSIM v1.1 scorer 评测。该 fork 与 official v1.1（commit 0811876c274e8b058ab2be9b3dcd4d37bd23f177, tag v1.1）在打分路径上的唯一实质差异为 batch_lqr_utils.py 中两处伪逆实现（numpy → torch）。完整 navtest 12,146 场景实测：9 个场景存在非零差异，max |ΔPDMS| = 3.35e-12，PDMS 均值差 1.39e-16。constant-velocity baseline 复现论文 Table 1 的 20.6 至 20.6517（差 0.05 个百分点）。

（`outputs/alignment_stage_a/evaluator_validation/scorer_equivalence_record.json`，`reporting_rules.required_statement`）

这应表述为“差异已测定且低于可报告精度”，不能表述为没有差异（同文件，`reporting_rules`）。constant-velocity 的 RAP 均值为 `0.20651651538608046`，official 均值为 `0.20651651538608060`；论文 Table 1 的 `20.6` 对应本流水线 `20.6516515386%`，绝对差 `0.0516515386` 个百分点（`outputs/alignment_stage_a/evaluator_validation/batch_lqr_ab_test.json`；`scorer_equivalence_record.json`）。论文模型行未用于比较；论文数值在这里仅作规则型 baseline 流水线校验。

### 6.2 关系式门禁 G1–G7

官方 checkout 内不存在 human / constant-velocity 的数值 baseline，只有命令、agent 描述和公式，因此原绝对值门禁无法建立；`v1_reference.json` 按替代文档要求不存在（`docs/rap-alignment-stage-a-step5-v1-only.md` Step S.2；`outputs/alignment_stage_a/evaluator_validation/v1_paper_reference.json`）。门禁改为不依赖外部绝对数值、只依赖 agent 定义和评分公式的关系式检查。

| Gate | 设计理由 | 通过证据 |
|---|---|---|
| G1 schema | 固定 v1 九列 schema、token 排序与全行 valid | `n_rows=12146`、`all_valid=true`（`outputs/alignment_stage_a/evaluator_validation/v1_relational_gate.json`；`v1_regression.json`） |
| G2 determinism | 同一规则型 agent 重跑必须完全一致 | 两次 constant-velocity DataFrame 完全相同（`v1_regression.json`） |
| G3 range | 全部 sub-score 与 PDMS 落在 `[0.0, 1.0]` | min `0.0`、max `1.0`（同上） |
| G4 non-degenerate | 排除全零、全常量或失真尺度 | CV open interval `[0.05, 0.6]`、human interval `[0.8, 1.0]`、distinct values `>10`（`v1_relational_gate.json`） |
| G5 ordering | human replay 应明显高于 constant velocity | human-minus-CV `0.7389975231460622`，门槛 `>0.3`（`v1_regression.json`；`v1_relational_gate.json`） |
| G6 human consistency | 专家轨迹按构造不应碰撞或驶离可行驶区 | NC/DAC/TTC 均值满足 `>=0.99`（上述两文件） |
| G7 formula | 逐行重构乘性 PDMS，避免用均值误验公式 | CV 最大残差 `2.220446049250313e-16`，human 最大残差 `3.3306690738754696e-16`，超容差行均为 `0`（`v1_regression.json`） |

`v1_regression.json` 的 failures 为空且 `status=passed`。这组门禁验证 schema、确定性、尺度、排序、专家一致性和公式重构；constant-velocity 对论文锚点的复现额外排除了关系式门禁看不到的共同尺度偏移（`outputs/alignment_stage_a/evaluator_validation/v1_regression.json`；`scorer_equivalence_record.json`）。

## 7. 结果

### 7.1 metric classification

按六份 CSV 中不同取值数自动分类：`EP`、`PDMS` 为 continuous；`NC`、`DAC`、`TTC`、`C`、`DDC` 为 discrete（`outputs/alignment_stage_a/summary/stage_a_report.json`，`metric_classification`）。PDMS 始终按 continuous 处理（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`metric_classification`）。

### 7.2 per-run sub-score 描述性均值

下表按 NAVSIM 论文 Table 1 的百分比口径转录自 `outputs/alignment_stage_a/summary/per_run_subscore_means.json`：每个 CSV 对七列逐 scene 取均值后乘 `100`、保留两位；`mean±sd` 对三个已保留两位的 seed 值计算，sd 为样本标准差 `ddof=1`。每份 CSV 均为 `12146` scenes，六份源 CSV 的 SHA256 也保存在该文件。该产物的 note 明确为 `descriptive per-run means; NOT produced by summarize_stage_a.py`。

| Condition | NC | DAC | TTC | C | DDC | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoAlign seed0 | `98.19` | `94.72` | `92.49` | `98.74` | `99.04` | `58.22` | `76.02` |
| NoAlign seed1 | `96.65` | `91.00` | `89.37` | `97.13` | `97.24` | `62.26` | `74.11` |
| NoAlign seed2 | `97.83` | `94.06` | `92.20` | `96.10` | `97.26` | `63.62` | `77.32` |
| NoAlign mean±sd | `97.56±0.81` | `93.26±1.98` | `91.35±1.72` | `97.32±1.33` | `97.85±1.03` | `61.37±2.81` | `75.82±1.61` |
| FullAlign seed0 | `98.18` | `95.93` | `92.15` | `97.74` | `98.56` | `59.62` | `76.90` |
| FullAlign seed1 | `97.71` | `92.93` | `90.61` | `98.06` | `97.85` | `63.33` | `76.35` |
| FullAlign seed2 | `97.25` | `90.94` | `90.82` | `98.13` | `97.38` | `60.49` | `74.15` |
| FullAlign mean±sd | `97.71±0.47` | `93.27±2.51` | `91.19±0.84` | `97.98±0.21` | `97.93±0.59` | `61.15±1.94` | `75.80±1.46` |

这些是描述性 per-run 绝对均值，不替代 scene-level paired delta、cluster bootstrap 或 noise-floor 判定。

### 7.3 连续指标 paired delta

delta 定义为逐 scene token 的 `FullAlign - NoAlign`，先 seed 内配对，再跨 seed 取均值（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`pairing`）。下表全部数值仅转录自 `outputs/alignment_stage_a/summary/stage_a_report.json` 的 `paired_delta.continuous`：

| metric | mean | median | q05 / q25 / q50 / q75 / q95 | cluster-bootstrap 95% CI | bootstrap p | Cohen's dz | paired dominance |
|---|---:|---:|---|---|---:|---:|---:|
| PDMS | `-0.00018424113877407014` | `0.0` | `-0.25935838752946544 / -0.011631759766594954 / 0.0 / 0.00862628219129391 / 0.25502147196209857` | `[-0.003212433863104941, 0.002802059539156341]` | `0.9096`，`descriptive_non_confirmatory` | `-0.0014983819876434753` | `-0.07253416762720238` |
| EP | `-0.0022013468213482436` | `0.0` | `-0.16736349128395256 / -0.023027702851402432 / 0.0 / 0.014681970907385062 / 0.1606138781305759` | `[-0.004547827487260247, 0.00007831810047990816]` | `0.1112` | `-0.021777400688088693` | `-0.10151490202535818` |

PDMS 的 per-seed mean delta 分别为 seed `0: 0.008793935302539609`、seed `1: 0.02241075099775581`、seed `2: -0.03175740971661763`；EP 分别为 seed `0: 0.013916319747570384`、seed `1: 0.010744744221275006`、seed `2: -0.03126510443289012`（`outputs/alignment_stage_a/summary/stage_a_report.json`，`paired_delta.continuous`）。这些 per-seed 项仅用于展示一致性，不是主分析。

### 7.4 same-condition noise floor

六条 null 序列均按与主分析相同的配对和 cluster bootstrap 流程处理。下表全部转录自 `outputs/alignment_stage_a/summary/stage_a_report.json` 的 `noise_floor.sequences`：

| 条件 / seed pair | mean | median | 95% CI | per-scene abs(delta) p95 |
|---|---:|---:|---|---:|
| FullAlign `0-1` | `0.005491968845292249` | `-0.011941789570635963` | `[0.000533470962016671, 0.010256652662609339]` | `0.7378786742936012` |
| FullAlign `0-2` | `0.027538508391868145` | `-0.006747715192254755` | `[0.019883213987453748, 0.03568740267769606]` | `0.7646267203124065` |
| FullAlign `1-2` | `0.022046539546575896` | `0.0` | `[0.01431975269494029, 0.03016921553733718]` | `0.795128479736763` |
| NoAlign `0-1` | `0.019108784540508454` | `-0.013624797760298124` | `[0.012440181612640187, 0.02605839801475713]` | `0.7792931462058118` |
| NoAlign `0-2` | `-0.013012836627289092` | `-0.01620062361157948` | `[-0.02032502424866329, -0.005488769045642727]` | `0.7545269214451198` |
| NoAlign `1-2` | `-0.032121621167797536` | `0.0` | `[-0.03856733384961941, -0.025521172295140632]` | `0.7993886681032597` |

practical regression threshold 为 `0.7993886681032597`，即六条 same-condition 序列 per-scene `|delta|` 的 `95%` 分位数取最大者（`outputs/alignment_stage_a/summary/stage_a_report.json`，`practical_regression_threshold`；`noise_floor`）。该阈值只用于标注实用可分辨范围，不能把连续指标二值化。

### 7.5 离散指标完整转移表

单元格格式为 `NoAlign 值 → FullAlign 值: count`；regression 指 FullAlign 值更低，rescue 指更高。所有单元格、regression 和 rescue 计数均逐项转录自 `outputs/alignment_stage_a/summary/stage_a_report.json` 的 `paired_delta.discrete`，没有从计数另算净值。

| metric | seed | 完整转移单元格 | regression / rescue |
|---|---|---|---:|
| C | `0` | `0→0:89; 0→1:64; 1→0:185; 1→1:11808` | `185 / 64` |
| C | `1` | `0→0:131; 0→1:218; 1→0:105; 1→1:11692` | `105 / 218` |
| C | `2` | `0→0:169; 0→1:305; 1→0:58; 1→1:11614` | `58 / 305` |
| C | merged | `0→0:389; 0→1:587; 1→0:348; 1→1:35114` | `348 / 587` |
| DAC | `0` | `0→0:387; 0→1:254; 1→0:107; 1→1:11398` | `107 / 254` |
| DAC | `1` | `0→0:653; 0→1:440; 1→0:206; 1→1:10847` | `206 / 440` |
| DAC | `2` | `0→0:483; 0→1:238; 1→0:617; 1→1:10808` | `617 / 238` |
| DAC | merged | `0→0:1523; 0→1:932; 1→0:930; 1→1:33053` | `930 / 932` |
| TTC | `0` | `0→0:690; 0→1:222; 1→0:264; 1→1:10970` | `264 / 222` |
| TTC | `1` | `0→0:863; 0→1:428; 1→0:278; 1→1:10577` | `278 / 428` |
| TTC | `2` | `0→0:632; 0→1:315; 1→0:483; 1→1:10716` | `483 / 315` |
| TTC | merged | `0→0:2185; 0→1:965; 1→0:1025; 1→1:32263` | `1025 / 965` |
| NC | `0` | `0→0:131; 0→0.5:0; 0→1:75; 0.5→0:1; 0.5→0.5:15; 0.5→1:11; 1→0:76; 1→0.5:12; 1→1:11825` | `89 / 86` |
| NC | `1` | `0→0:180; 0→0.5:0; 0→1:190; 0.5→0:0; 0.5→0.5:38; 0.5→1:35; 1→0:71; 1→0.5:16; 1→1:11616` | `87 / 225` |
| NC | `2` | `0→0:127; 0→0.5:0; 0→1:122; 0.5→0:0; 0.5→0.5:19; 0.5→1:11; 1→0:177; 1→0.5:40; 1→1:11650` | `217 / 133` |
| NC | merged | `0→0:438; 0→0.5:0; 0→1:387; 0.5→0:1; 0.5→0.5:72; 0.5→1:57; 1→0:324; 1→0.5:68; 1→1:35091` | `393 / 444` |
| DDC | `0` | `0→0:33; 0→0.5:4; 0→1:0; 0.5→0:2; 0.5→0.5:110; 0.5→1:47; 1→0:33; 1→0.5:101; 1→1:11816` | `136 / 51` |
| DDC | `1` | `0→0:47; 0→0.5:22; 0→1:7; 0.5→0:38; 0.5→0.5:261; 0.5→1:219; 1→0:3; 1→0.5:64; 1→1:11485` | `105 / 248` |
| DDC | `2` | `0→0:48; 0→0.5:41; 0→1:26; 0.5→0:16; 0.5→0.5:246; 0.5→1:173; 1→0:4; 1→0.5:214; 1→1:11378` | `234 / 240` |
| DDC | merged | `0→0:128; 0→0.5:67; 0→1:33; 0.5→0:56; 0.5→0.5:617; 0.5→1:439; 1→0:40; 1→0.5:379; 1→1:34679` | `475 / 539` |

离散侧存在明确的 per-seed 方向翻转：C 在 seed `0` 中 regression 多于 rescue，而 seed `1`、`2` 相反；DAC 在 seed `0`、`1` 中 rescue 多于 regression，而 seed `2` 相反；TTC、NC、DDC 也出现 seed 间次序翻转或接近持平（`outputs/alignment_stage_a/summary/stage_a_report.json`，上述完整转移表）。这是一条独立于连续 PDMS 的 seed 不一致证据。

## 8. 结论

本节受协议原文直接约束。noise floor 的解释规则为：

> 若 FullAlign - NoAlign 的 |mean delta| 不超过 null 序列 |mean delta| 的最大值，结论必须写成「效应不可与 seed 噪声区分」，不得写成「无效应」，也不得写成「有微小效应」。

（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`noise_floor.interpretation_rule`）

hypothesis family 的 partial-family 规则为：

> 禁止在 m=1 下施加 Holm。EPDMS 记为 not_tested，不得记为 pass 或 fail。PDMS 的 p 值按未校正值报告，并显式标注为 descriptive / non-confirmatory。任何显著性表述必须同时声明 co-primary family 不完整。

（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`hypothesis_family.partial_family_rule`）

在 `2498` 训练样本 / `400` optimizer steps 的条件下，**未能检出 paired real-raster alignment 对 navtest PDMS 的效应**（样本与预算：`outputs/alignment_stage_a/input_data/measured_training_budget.json`；效应统计：`outputs/alignment_stage_a/summary/stage_a_report.json`）。源文件记录的 PDMS mean delta 为 `-0.00018424113877407014`，六条 same-condition null 序列中绝对 mean 最大者对应的源值为 `-0.032121621167797536`；主分析的绝对 mean 未超过该 null 范围。按上述协议规则，**效应不可与 seed 噪声区分**（`outputs/alignment_stage_a/summary/stage_a_report.json`，`paired_delta.continuous.PDMS.mean`、`noise_floor.sequences.r100-noalign/1-2.mean`）。

全部结论适用 **under-trained regime** 标注；co-primary family 不完整，EPDMS 为 `not_tested`，PDMS 的 `p=0.9096` 仅为 `descriptive_non_confirmatory`，不施加 `m=1` 的 Holm（`outputs/alignment_stage_a/summary/stage_a_report.json`；`outputs/alignment_stage_a/summary/summary_protocol_v2.json`）。本报告 `status=incomplete`，不作方向性结论，不作 Stage-B 效应量外推，也不据本轮结果修改 Stage-B 设计。

本轮实际交付的是：端到端训练与评测链路可运行；两条件控制有逐行运行时证据；v1 评测器经过关系式门禁、代码冻结复检和 constant-velocity 已发表锚点复现；统计脚本与 artifact 链条完成演练；评测时间和显存测量已落盘（`outputs/alignment_stage_a/validation_seed_*.json`；`evaluator_validation/v1_regression.json`；`evaluation_v1/code_baseline/code_baseline_recheck.json`；`qsub_logs/s6-*.o*`；`evaluator_validation/rap_memory_probe2/rap_double_load_memory.json`）。

## 9. 已接受偏差清单

| 偏差 | 决策人 | 留痕 | 对结论的影响 |
|---|---|---|---|
| 候选池仅约 `21%`，real 三相机抽样覆盖率 `20.887%`，接受不补下载 | human，记录日期 `2026-08-19` | `outputs/alignment_stage_a/input_data/human_override_p1.json` | 本轮降格为 pipeline rehearsal；候选池非随机，不能作 Stage-B 效应量先导（同文件） |
| selected train logs `43/44`、val logs `9/10`，接受缺 log；四个 map 均覆盖 | human，记录写明“人工接受” | `outputs/alignment_stage_a/input_data/log_coverage_resolution.json` | 不满足 spec §7 的全 log 条款，代表性受限；缺失原因逐 log 留痕（同文件） |
| val `route_command` TVD `0.023308019591352123` 超阈值 `0.02` | human，决策时间 `2026-08-19T09:14:47+09:00` | `outputs/alignment_stage_a/input_data/distribution_audit_resolution.json` | validation 子集相对候选池的路线命令比例有偏；未重抽、未改算法和阈值（同文件） |
| spec §4 的样本/预算数字修正为实测 `2498 / 464 / 400` | human override 后按 P.3 冻结 | `outputs/alignment_stage_a/spec_budget_correction.patch`；`input_data/measured_count_differences.json`；`measured_training_budget.json` | 改变本轮解释口径；是否仍应称作相对旧 `1800` 的“偏离”存在文档冲突，见第 12 节（上述产物；`summary/requirement_coverage.json`，`CON-010`） |
| runbook Step 2.4 v2 gate 跳过，v2.2 EPDMS 延期 | human，记录日期 `2026-08-20` | `outputs/alignment_stage_a/evaluator_validation/v2_deferral_record.json` | co-primary family 不完整，EPDMS `not_tested`，报告恒为 `incomplete`（同文件） |
| 采用路线 A：RAP 本地 scorer，公开量化与 official 的差异 | human，协议修订时间 `2026-08-21` | `outputs/alignment_stage_a/evaluator_validation/scorer_equivalence_record.json`；`summary/summary_protocol_v2.json` | 不能无附注地称 official scorer；差异已测定且低于可报告精度（上述文件） |
| `v1_reference.json` 绝对值门禁作废，改用 G1–G7 关系式门禁 | step5 替代指令；路线决策由 human 接受 | `docs/rap-alignment-stage-a-step5-v1-only.md` Step S.2/S.4；`outputs/alignment_stage_a/evaluator_validation/v1_relational_gate.json`；`v1_regression.json` | 绝对值门禁无法建立；由关系式门禁加论文规则型 baseline 软核对支撑可信度（上述文件） |
| S.4b 对评测时全部缺失的 rendered 输入使用零占位 | step5 替代指令；记录未单列 `decided_by` | `docs/rap-alignment-stage-a-step5-v1-only.md` Step S.4b；`outputs/alignment_stage_a/eval_smoke/smoke_report.json`；`evaluation_v1/code_baseline/code_baseline.json` | 训练与评测代码状态不同；基线记录称逐位测试证明模型输入不变，但独立决策人字段缺失（上述文件） |

## 10. 协议修订记录

### 10.1 v2 已冻结修订

以下逐字嵌入 `outputs/alignment_stage_a/summary/summary_protocol_v2.json` 的完整 `amendment` 对象；特别保留修订时已观测与未观测信息：

```json
{
  "reason": "v1 的 forbidden_actions 含「用本地 scorer 替代官方 scorer」。Step S.2 核查确认 pinned official v1.1 worktree 内不存在任何 baseline 参考数值，且 Step S.3/S.3b 确认 RAP fork 与 official 在打分子树上存在一处实质差异（batch_lqr_utils.py 的伪逆实现）。人工决定采用路线 A：使用 RAP fork 的 scorer，并以实测量化其与 official 的等价程度。该决定与 v1 的该条禁令冲突，按 v1 的 amendment_rule 新建本文件。",
  "amended_by": "human",
  "amended_at": "2026-08-21",
  "observed_at_amendment_time": {
    "_this_is_the_key_disclosure": "v1 的 amendment_rule 要求记录修订时已观测到的数值范围，用以判定修订是否构成 selection-on-result。",
    "observed": [
      "constant_velocity_agent 在完整 navtest 上的 PDMS 均值：0.20651651538608046（RAP 版）/ 0.20651651538608060（official 版）",
      "batch_lqr A/B 的逐列差异统计：9/12,146 个 token 有非零差异，max |ΔPDMS| = 3.35e-12",
      "论文 Table 1 的 Constant Velocity = 20.6、Human = 94.8 等已发表基准值",
      "6 个 run 的训练 train/val loss 逐 epoch 曲线（在 v1 冻结时即已观测，见 convergence_gate.blinding_disclosure）"
    ],
    "not_observed": [
      "6 个 Stage-A checkpoint 的任何 PDMS 或 sub-score",
      "FullAlign 与 NoAlign 的任何对比信息",
      "任何 paired delta、CI、effect size 或 transition table"
    ],
    "selection_on_result_assessment": "本次修订所依据的全部观测均来自 constant_velocity / human 这两个**未经训练的规则型 baseline**，以及已发表的论文数值。它们与本实验的假设（alignment loss 是否改善 planning performance）零相关：这两个 agent 从未接触训练集，其分数不含任何关于 6 个 checkpoint 的信息。因此本修订**不构成 selection-on-result**。全部与实验假设相关的统计规则（pairing / bootstrap / effect size / practical threshold / hypothesis family）自 v1 冻结起至今未被观测，也未被修改。"
  },
  "changes_from_v1": [
    {
      "field": "forbidden_actions",
      "change": "「用本地 scorer 替代官方 scorer」→「在未量化并公开记录差异的情况下用本地 scorer 替代官方 scorer」",
      "justification": "禁令的实质目的是阻止**静默替换**。本轮的替换伴随完整的差异测定与公开记录（scorer_equivalence_record.json），满足该目的。"
    },
    {
      "field": "scorer_provenance",
      "change": "新增章节，指向 scorer_equivalence_record.json，并规定报告中的必需表述与禁止表述"
    },
    {
      "field": "data_provenance_caveats_required_in_report",
      "change": "新增两条：scorer 差异与实测等价程度；constant-velocity 对论文基准的复现结果"
    },
    {
      "field": "report_status.required_sections",
      "change": "新增「scorer 等价性章节」"
    },
    {
      "field": "convergence_gate.val_score_availability",
      "change": "补充引用 Step S.1b 的 convergence_curve_audit.json 作为落盘依据"
    }
  ],
  "unchanged_from_v1": [
    "run_scope（含 permitted_use / forbidden_use 全部条目）",
    "measured_budget",
    "convergence_gate 的判定标准与 verdict",
    "pairing",
    "cluster_bootstrap",
    "metric_classification",
    "continuous_metrics",
    "discrete_metrics",
    "noise_floor",
    "practical_regression_threshold",
    "hypothesis_family",
    "report_status.status"
  ]
}
```

修订时已经看到的只有规则型 baseline、scorer A/B 差异、论文锚点和训练 loss 曲线；尚未看到任何 checkpoint 分数、条件对比、paired delta、CI、effect size 或 transition table。协议据此判定该修订不构成 selection-on-result；与实验假设相关的统计规则未改变（`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，上述逐字对象）。

### 10.2 v3 未冻结草案与 post-observation 披露

`outputs/alignment_stage_a/summary/summary_protocol_v3.json` 当前不是统治性协议。其 `freeze_status` 原文为：

> DRAFT — 待人工审阅。在 amendment.amended_by / amended_at / human_assessment 三个字段被人工填写之前，本文件不得作为 summarize_stage_a.py 的 --protocol 输入，也不得视为已冻结协议。起草方不得自行填写这三个字段。

草案版本为 `3.0`，起草日期 `2026-08-24`，声明 supersede v2，并记录 v2 SHA256 `c24978869b9398e10e66735babc7fb6df7917db6c1e431a4c81f4e2a8d1d8cb7`；但三个必需人工字段当前均为空。待人工动作是：审阅 selection-on-result 风险；若接受则填写三个人工字段并把状态改为 `FROZEN`，若拒绝则删除草案、继续使用 v2；冻结后再同步到权威 outputs 目录（同文件，`human_action_required`）。本文不代填、不替人作决定。

草案更正的事实是：v2 caveat 8 所称 DINO backbone 实际 lr=`2e-5` 与代码不符；`rap_agent.py:571-572` 把 backbone 参数设为 `requires_grad=False`，`:576` 的 vit 参数组被注释，`:567` 的 `vit_lr` 只计算不使用。草案建议的新 caveat 因此表述 optimizer 只有 `other_params`、lr=`1e-4`，backbone 不接收梯度；alignment 梯度的作用面限于 DINO 后的可训练部分与下游 planner（同文件，`amendment.reason`、`changes_from_v2[0]`）。

该更正发生在结果被观测之后。草案逐项披露当时已经观测到：六个 run 的完整 sub-score/PDMS 均值；NoAlign PDMS `76.02 / 74.11 / 77.32`、FullAlign `76.90 / 76.35 / 74.15`；条件均值 `75.82 / 75.80`；scene-level PDMS mean delta `-0.018pp`、median `0`、Cohen's dz `-0.0015`、paired dominance `-0.073`；same-condition seed 对最大 `|mean delta|=3.21pp`；七个指标的跨条件 delta 均低于同条件 seed 噪声底。当时未观测 EPDMS，也未观测按协议 cluster bootstrap 计算的 CI/p；草案还记录当时 `stage_a_report.json` 尚未产出（同文件，`amendment.observed_at_amendment_time`）。最后一句是草案起草时的时点陈述；当前目录现已有 `stage_a_report.json`，本文不回写草案。

selection-on-result 评估没有回避风险：更正依据是早于本实验的源码事实，但“backbone 冻结使 alignment 梯度作用面窄”在方向上可能被读成对已观测 null 的事后解释；v2 的默认规则按字面将结果观测后的修订视为 selection-on-result。草案列出的缓解因素是统计规则完全不变、错误事实需要更正、源码事实可独立复核、结论地位不变、且新 caveat 自带 post-observation 标注；它明确把最终接受与否留给人工（同文件，`amendment.selection_on_result_assessment`）。

除 caveat 8 的事实更正和 required sections 新增 v3 披露外，草案声明 run scope、scorer、预算、收敛、pairing、bootstrap、指标分类、连续/离散规则、noise floor、practical threshold、hypothesis family、report status、forbidden actions 及其余 caveats 均与 v2 不变。其 downstream actions 尚未执行：分析脚本协议链扩展和改用 v3 都以人工冻结为前提（同文件，`unchanged_from_v2`、`downstream_actions_required`）。

## 11. 复现信息

### 11.1 代码、协议与环境

| 项目 | 冻结值 | 来源 |
|---|---|---|
| RAP git baseline commit | `6ed7a39bf9d2521b42b0edde0a28c49d8ca332e6` | `outputs/alignment_stage_a/baseline_commit.txt` |
| official evaluator commit / tag | `0811876c274e8b058ab2be9b3dcd4d37bd23f177` / `v1.1` | `outputs/alignment_stage_a/evaluator_validation/v1_commit.txt`；`v1_tag.txt` |
| `summary_protocol.json` SHA256 | `252994acb610806b59ee7653f6b7f27a1e3ab470d1efba7e9de045dba900945d` | `outputs/alignment_stage_a/summary/token_to_log.json`，`protocol_sha256` |
| `summary_protocol_v2.json` SHA256 | `c24978869b9398e10e66735babc7fb6df7917db6c1e431a4c81f4e2a8d1d8cb7` | `outputs/alignment_stage_a/summary/stage_a_report.json`，`protocol_v2_sha256` |
| `summary_protocol_v3.json` SHA256 / 状态 | `2315367bed4bdc0f432517042b17930e092b304989996f2a2934fa3cc965b2e0` / `DRAFT` | 对 `outputs/alignment_stage_a/summary/summary_protocol_v3.json` 只读 `sha256sum`；同文件 `freeze_status` |
| `scorer_equivalence_record.json` SHA256 | `34f298bc3bd61aaf66004da565015dfe946fcb6f130fd4930ebfce9a51255a94` | 报告生成时对该冻结文件只读执行 `sha256sum`；被校验文件为 `outputs/alignment_stage_a/evaluator_validation/scorer_equivalence_record.json` |
| `v2_deferral_record.json` SHA256 | `6f18c9b42574da9fa69fc4923c6c8091148736617037947657bd6aa5c0411c46` | 报告生成时对该冻结文件只读执行 `sha256sum`；被校验文件为 `outputs/alignment_stage_a/evaluator_validation/v2_deferral_record.json` |
| 评测代码基线 | head `6ed7a39bf9d2521b42b0edde0a28c49d8ca332e6`；pdm_planner digest `3f677f319dcac38e6f1acdc0ec3950eed839e50a3027acc50fc99dd0c31e2ba1`；worktree dirty | `outputs/alignment_stage_a/evaluation_v1/code_baseline/code_baseline.json` |
| 评测后复检 | critical files match、subtree match、`status=passed` | `outputs/alignment_stage_a/evaluation_v1/code_baseline/code_baseline_recheck.json` |

训练 checkpoint 产生于 S.4b 前，评测代码基线产生于 S.4b 后；基线记录将差异限定为 rendered 分支零占位，并记录逐位测试证明模型输入一致（`outputs/alignment_stage_a/evaluation_v1/code_baseline/code_baseline.json`）。

环境由 `outputs/alignment_stage_a/env_versions.json` 冻结：Python `3.9.25`，PyTorch `2.1.0+cu121`，torch CUDA `12.1`，PyTorch Lightning `2.2.1`，Hydra `1.2.0`，OmegaConf `2.3.0`，Ray `2.51.2`，Transformers `4.57.6`，timm `1.0.27`，mmcv `2.1.0`，mmdet `3.3.0`。GPU driver 查询落盘为 `CalledProcessError(9, ...)`，没有可用 driver version（同文件）。

### 11.2 成本实测

| 工作 | 实测 | 来源 |
|---|---|---|
| v1 评测 NoAlign seed `0 / 1 / 2` | `2030 / 1685 / 1653` 秒 | `outputs/alignment_stage_a/qsub_logs/s6-no0.o8457605`、`s6-no1.o8457845`、`s6-no2.o8457846` 的 `evaluation_elapsed_seconds` |
| v1 评测 FullAlign seed `0 / 1 / 2` | `1694 / 1713 / 1673` 秒 | `outputs/alignment_stage_a/qsub_logs/s6-fu0.o8457847`、`s6-fu1.o8457848`、`s6-fu2.o8457849` 的 `evaluation_elapsed_seconds` |
| RAP 双模型评测显存 probe | peak allocated `7315032576` bytes；peak reserved `7440695296` bytes | `outputs/alignment_stage_a/evaluator_validation/rap_memory_probe2/rap_double_load_memory.json` |
| 训练预算成本单位 | 每 run `400` optimizer steps、`20` epochs、`4` devices | `outputs/alignment_stage_a/input_data/measured_training_budget.json`；两份 `resolved_config_*.yaml` |

产物保存了训练进度日志，但没有单独的结构化“六次训练总 wall time / GPU-hours”记录；本报告不从时间戳自行计算该值，避免开启新的未审计成本统计路径（`outputs/alignment_stage_a/qsub_logs/rap-stage-a-p4.o8442095`；`outputs/alignment_stage_a/summary/requirement_coverage.json` 未列出结构化训练成本产物）。

## 12. 本报告未覆盖的已知问题与不一致

### 12.1 分析边界

- **选中样本的时序连续性未测量。** 哈希抽样不考虑时序；配置的时间间隔是 `0.5` 秒，scene 窗口是 `14` 帧（`4` history + `10` future），相邻 token 的两个窗口可共享 `13` 帧，因此有效样本量可能显著低于 `2498`。本轮没有落盘该依赖结构的审计，报告不对有效样本量作数值修正（`outputs/alignment_stage_a/resolved_config_r100_noalign.yaml`，trajectory sampling 与 scene filter；`docs/rap-navsim-v1-evaluation-pipeline.md` §4.A；`outputs/alignment_stage_a/input_data/measured_training_budget.json`）。
- **noise floor 没有 paired dominance。** `stage_a_report.json` 的六条 null 序列只有 mean、median、CI、bootstrap 内容和 abs-delta p95，没有 `paired_dominance`；因此主分析的 paired dominance 无 null 可比对（`outputs/alignment_stage_a/summary/stage_a_report.json`，`noise_floor`）。
- **训练总成本记录不完整。** 见第 11 节；不从日志时间戳补算（`outputs/alignment_stage_a/qsub_logs/rap-stage-a-p4.o8442095`）。

### 12.2 写作中发现的文档/产物不一致

以下不一致只披露、不修文件；主要转录自 `outputs/alignment_stage_a/summary/requirement_coverage.json` 的 `cross_document_contradictions`：

1. runbook 保留 `v1_reference.json` 前置条件，step5 要求该文件不存在；现由 step5 关系式门禁取代（`CON-001`）。
2. Phase 4 preflight 尾部仍要求 `v1_reference.json` 和两个 evaluator，而 step5 作废绝对 reference 并延期 v2（`CON-002`）。
3. runbook 禁止 v1-only downgrade，step5 与 `v2_deferral_record.json` 明确授权；旧 `SKIPPED.md` 仍声称没有 downgrade（`CON-003`、`CON-009`）。
4. step5 全局禁令仍写禁止本地 scorer，后续 protocol v2 amendment 与 scorer record 允许已量化的路线 A（`CON-004`）。
5. 规格书要求 v1+v2 co-primary，实际 v1-only；这是已记录偏差，导致 `incomplete`（`CON-005`）。
6. Phase 4 preflight 的 P.3/P.4 文本要求 `data_limit`，实际诊断为 `data_bug` 后由 human override 继续，preflight 未同步文字（`CON-006`）。
7. v2 协议要求写 DINO backbone 实际 lr `2e-5`；代码覆盖审计指出 backbone 参数被冻结、vit optimizer group 被注释。`summary_protocol_v3.json` 已起草事实更正，但仍为待人工审阅的 `DRAFT`，故统治性协议链尚未解决该冲突（`CON-007`；`EXP-052`；`summary_protocol_v3.json`，`freeze_status`）。
8. 规格书要求全部 metadata frame 硬链接；manifest/实现排除了 `3` 个硬编码 token，但没有找到具名审批人、日期、理由和影响记录（`CON-008`；`EXP-006`）。
9. 当前计划已把预算修正为 `400` steps，protocol v2 / `stage_a_report.json` 仍称 `400` 是相对冻结 `1800` 的偏离；这是未解决的事实表述冲突（`CON-010`）。
10. `env_versions.json` 的 GPU driver 版本缺失，只有查询错误；依赖环境记录不完整（`EXP-052`）。
11. 规格书点名的若干测试在 coverage review 中仍为 MISSING：完整 split 相等、original-drive 隔离、完整四相机 inventory、若干损坏输入失败路径、target field-by-field 重建、cross-log top-up、NoAlign 直接梯度、首步前参数/梯度分解、validation real-only、interrupt-resume（`EXP-026` 至 `EXP-031`、`EXP-037`、`EXP-043` 至 `EXP-045`、`RUN-007`，均见 `outputs/alignment_stage_a/summary/requirement_coverage.json`）。

### 12.3 requirement coverage 汇总

`outputs/alignment_stage_a/summary/requirement_coverage.json` 读取四份规范文档共 `2536` 行，分类 `99` 项要求，`unclassified_requirements=[]`。来源分布与状态如下：

| 来源 | 要求数 |
|---|---:|
| experiment plan | `57` |
| Phase 4 preflight | `6` |
| runbook | `22` |
| step5 v1-only | `14` |

| 状态 | 数量 |
|---|---:|
| IMPLEMENTED | `57` |
| SUPERSEDED | `6` |
| WAIVED_RECORDED | `11` |
| WAIVED_UNRECORDED | `3` |
| MISSING | `13` |
| OUT_OF_SCOPE | `9` |

三条 `WAIVED_UNRECORDED` 对应两个独立问题：`EXP-006` 与重复要求 `RUN-004` 指向完整 54-log 三相机硬链接中 3 个排除 token 缺少具名审批记录；`STEP5-011` 指向 v2 deferral 的留痕集合自相矛盾，需要非破坏性更正记录。13 条 `MISSING` 为 `EXP-026/027/028/030/031/035/037/043/044/045/050/052` 与 `RUN-007`，主题已在第 12.2 节第 10–11 项展开。审计还记录 `10` 组 cross-document contradictions；directory tree 没有“文档声明但缺失”的目录，额外目录均属于 provenance、diagnostics、retry、smoke 或更强评测控制，存在本身不构成缺陷（同文件，`status_summary`、`requirements`、`cross_document_contradictions`、`directory_tree_differences`）。

## 附录 A：产物清单

本清单由报告写作时实际只读遍历 `outputs/alignment_stage_a/` 生成；大规模 cache 按冻结 manifest/checksum 和路径模式列出，避免把数万条机械路径复制进正文。未依据记忆补项。

- 顶层 provenance/config：`.matplotlib/`、`baseline.patch`、`baseline_commit.txt`、`baseline_dirty.txt`、`baseline_tests.txt`、`baseline_untracked.txt`、`dino_reference_config.json`、`env_versions.json`、`metric_cache_count.txt`、`phase_minus1_gpu_gate.log`、`phase_minus1_to_phase3_summary.json`、`resolved_config_r100_fullalign.yaml`、`resolved_config_r100_noalign.yaml`、`spec_budget_correction.patch`、全部 `step_*` log/txt/json/xml/complete、`validation_seed_*.json`。
- `input_data/`：`b0_generation_manifest.jsonl`、`candidate_pool.json`、`dataset_manifest.json`、`distribution_audit.json`、`distribution_audit_resolution.json`、`distribution_audit_thresholds.json`、`exclusion_reason_histogram.json`、`frozen_counts.env`、`hardlink_manifest.jsonl`、`hardlink_manifest_summary.json`、`human_override_p1.json`、`image_coverage_probe.json`、`input_audit.json`、`log_coverage_resolution.json`、`measured_count_differences.json`、`measured_training_budget.json`、`paired_cache_checksum.json`、`resolved_target_cache.yaml`、`source_checksums.json`、`stage_a_token_manifest.json`、`target_cache_checksum.json`、`target_count.txt`、`topup_replacements.jsonl`。
- `target_cache/`：按 log/token 保存 `rap_target.gz`；冻结文件数 `41864`（`outputs/alignment_stage_a/input_data/target_cache_checksum.json`）。
- `paired_cache/`：按 log/token 保存 `rap_feature.gz` 与 `rap_target.gz`；冻结文件数 `5924`（`outputs/alignment_stage_a/input_data/input_audit.json`，`paired_cache_checksum.file_count`）。
- `init_checkpoints/` 与 `init_checkpoints_attempt1/`：各自含 seed checkpoint 与 `sha256.json`；正式运行使用前者（`outputs/alignment_stage_a/validation_seed_*.json`）。
- `memory_smoke/` 与 `memory_smoke_attempt1/`：Hydra code/config、训练日志、metrics、loss、token checksums、reproducibility manifest 与 checkpoint。
- `training/r100-noalign/seed_*`、`training/r100-fullalign/seed_*`：每个 run 含 Hydra code/config、`last.ckpt`、CSV metrics、epoch token checksums、loss components、训练主日志、DDP worker 日志和 reproducibility manifest。
- `evaluator_validation/`：`SKIPPED.md`、constant-velocity/human CSV 与运行目录、`v1_*` provenance/reference/gate/regression/commit/tag、`pdm_planner_*` diff、`batch_lqr_ab*`、两版 `batch_lqr_utils.py`、`scorer_equivalence_record.json`、`v2_deferral_record.json`、navtest cache checksum、RAP memory probe。
- `eval_smoke/`：smoke CSV/log/report、Hydra code/config、rendered placeholder worker record。
- `evaluation_v1/`：六份条件×seed CSV 及各自 Hydra code/config/log，`step_5_1_audit.json`，评测代码 baseline 与 recheck。
- `summary/`：`convergence_curve_audit.json`、`loss_component_diagnostics.json`、`paired_deltas.csv`、`per_run_subscore_means.json`、`requirement_coverage.json`、`stage_a_report.json`、`summary_protocol.json`、`summary_protocol_v2.json`、`summary_protocol_v3.json`、`token_to_log.json`。
- `qsub_logs/`：Phase 4、S.4 baseline、S.5 smoke、S.6 六次评测、LQR A/B、Ray 与显存 probe 的调度日志，包括失败重试日志；全部实际路径保留在该目录。

## 附录 B：`summary/` 结果台账

下表覆盖报告更新时 `outputs/alignment_stage_a/summary/` 内的全部 `10` 个文件。SHA256 由只读 `sha256sum` 取得；“本文覆盖”指聚合结果或状态在正文出现，不意味着把大规模逐 token 映射/明细机械复制进 Markdown。

| 文件 | SHA256 | 规模 / 核心状态 | 本文覆盖 |
|---|---|---|---|
| `convergence_curve_audit.json` | `93966e8b0eba916b7b8e29611e2d9a52d0647cf1737fdf62bb04dddd686550a6` | `6` runs；每 run 的 train/loss、val/loss、val/score 均 `20` 个非空点；`status=complete` | 第 5 节 |
| `loss_component_diagnostics.json` | `3f6d6041930dad99dd4f17c20d1ee7bae9b5b41c8c2ae4ba37febf49a3c23f93` | 六份 loss JSONL 各 `400` 行；三份 FullAlign metrics CSV 各 `40` 行、`20` 个 domain_accuracy 点 | 第 4.3 节 |
| `paired_deltas.csv` | `707a0775b1ebbbba6e348fe9959aea166ee835bbd8c73915514c8b1682afb99f` | `12146` token 行；`token + delta_NC/DAC/EP/TTC/C/DDC/PDMS` | 第 7.3–7.5 节 |
| `per_run_subscore_means.json` | `9b3f7a3f2a50bb0f2d39410009014985497c57aac74d42508a6c9afdf8b7bc37` | `6` runs × `7` metrics；各 `12146` scenes；含源 CSV SHA256 | 第 7.2 节 |
| `requirement_coverage.json` | `8bddf29e6702df837e6b922d96bc466fd97a87e7cddf4bef567edcded93f38c9` | `99` requirements；`10` contradictions；无未分类项 | 第 12、12.3 节 |
| `stage_a_report.json` | `8cf65120d3b73ea72f04aa10296ec36193bde951dff04bf4938123f601f0a0c1` | 冻结主分析；`operative_protocol=summary_protocol_v2.json`；`status=incomplete` | 第 0、3、5、7、8、10 节 |
| `summary_protocol.json` | `252994acb610806b59ee7653f6b7f27a1e3ab470d1efba7e9de045dba900945d` | v1 冻结协议 | 第 1、2、10、11 节 |
| `summary_protocol_v2.json` | `c24978869b9398e10e66735babc7fb6df7917db6c1e431a4c81f4e2a8d1d8cb7` | v2 冻结协议；当前 operative | 第 0–10 节 |
| `summary_protocol_v3.json` | `2315367bed4bdc0f432517042b17930e092b304989996f2a2934fa3cc965b2e0` | v3 `DRAFT`；三个人工决策字段为空，不得作为分析输入 | 第 0、3、4、10.2、11、12 节 |
| `token_to_log.json` | `29b64ae50d60d1c1e519440d3bca27fd482a389500245d538c9b86be49422fac` | `12146` tokens / `136` logs；源 schema=`file_name`；路径 schema=`<cache>/<log>/<scenario>/<token>/metric_cache.pkl` | 第 2、7、11 节 |

## 报告自查

### required sections 映射

协议要求的九项均已覆盖（要求清单来源：`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`report_status.required_sections`）：

| required section | 本报告位置 |
|---|---|
| run scope 与 permitted/forbidden use 原文 | 第 1 节 |
| data provenance caveats 全部条目 | 第 3.1 节 |
| scorer 等价性 required statement、差异和论文复现 | 第 6.1 节 |
| distribution audit 完整 summary | 第 3.2 节 |
| convergence gate 与 under-trained regime | 第 5 节 |
| PDMS 与全部 sub-score 的 paired delta、CI、effect size、transition table | 第 7 节 |
| noise floor 六条 null 序列 | 第 7.4 节 |
| co-primary family 不完整 | 第 0、8 节 |
| amendment 完整原文 | 第 10 节 |

追加的非统治性 summary 结果也已映射：`loss_component_diagnostics.json` → 第 4.3 节，`per_run_subscore_means.json` → 第 7.2 节，`requirement_coverage.json` → 第 12.3 节，v3 草案状态与 post-observation 披露 → 第 10.2 节。v3 仍为草案，因此其新增 required section 不是已生效的协议要求；本文提前完整披露不改变该状态。

### data provenance caveats 映射

协议的十条 caveat 已在第 3.1 节逐条逐字列出，并按下表逐项映射（条目及数量来源：`outputs/alignment_stage_a/summary/summary_protocol_v2.json`，`data_provenance_caveats_required_in_report`）：

| caveat | 主题 | 本报告位置 |
|---:|---|---|
| 1 | 确定性哈希抽样 | 第 3.1 节 |
| 2 | real 覆盖限制与排除原因 | 第 3.1 节 |
| 3 | 子集对候选池审计的结构盲区 | 第 3.1 节 |
| 4 | block 状非随机缺失 | 第 3.1 节 |
| 5 | 完整 distribution summary | 第 3.2 节 |
| 6 | val route command TVD 偏差 | 第 3.1、3.2、9 节 |
| 7 | train/val log 覆盖偏差 | 第 3.1、9 节 |
| 8 | DINO backbone lr 原文及 v3 草案更正 | 第 3.1、4.1、10.2、12.2 节 |
| 9 | RAP scorer 差异与等价程度 | 第 3.1、6.1 节 |
| 10 | constant-velocity 论文锚点复现 | 第 3.1、6.1 节 |

### 关键词扫描

扫描词为：`无效`、`有效`、`有帮助`、`有害`、`证明`、`显著改善`、`Stage-B 应`、`Stage-B 建议`。更新后只读全文扫描的命中数依次为 `3 / 7 / 1 / 1 / 6 / 1 / 1 / 1`（本报告自查扫描）。`无效` 的正文命中来自协议强制逐字引文；`有效` 的正文命中来自“有效性检查”和“有效样本量”；`证明` 的正文命中还包括 loss diagnostics 的强制方法限制“A2 的占比是代理指标，不是证明”，以及复述 S.4b 记录声称的逐位测试证据；其余命中只来自本自查词表。结论性断言未使用这些措辞。

### 数字溯源扫描

按十进制、科学计数法和百分数字面量的只读全文 regex 扫描，更新后报告共有 `1351` 个数字字面量，其中 `1351` 个处于明确溯源作用域，未溯源为 `0`（本报告自查扫描）。计数包含章节/步骤编号、版本标识和 SHA；量化事实由同段括号、表格“来源”列或引导逐字引文/JSON 的来源统一溯源，结构编号溯源至本报告结构，文件标识与 SHA 溯源至所指产物。
