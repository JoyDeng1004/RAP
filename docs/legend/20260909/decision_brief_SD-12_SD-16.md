# 裁决材料｜`SD-12`（合法训练集选哪个池）· `SD-16`（`4cam_v1` 可用性）

> **性质：决策材料，不是裁决。** 本文件不改变任何冻结项、不解除 `CP-2b`。
> 数据来源：`A1_clean_subset.md`、`A1_rap_datasets.md`、`A1_split_membership.md`、`NAVSIM_inventory.md`（Sprint Plan §3.7），以及本仓库代码。
> 依 Sprint Plan §2：`SD-15` 与 `SD-16` **必须一并裁决**；而 v2.6 已作废 `SD-15` 的既有数字（见 §3）。**本材料因此把 `SD-16` 拆成「可否使用」与「能用多少」两问**，前者现在可裁，后者须待重算。

---

## 0. 一句话结论

**这两条都不是"能不能跑 F0"的问题 —— 无论怎么裁，既有资产都达不到 `SD-3` 冻结的 10% 规模。**
`SD-12` 决定的是**未来重渲的抽样基数**；`SD-16` 决定的是**眼下有没有一份可用于 Smoke Test 的四相机资产**。把两者当成同一个问题，是过去两天判断反复的根源。

---

## 1. `SD-12`｜「NAVSIM 合法训练集」指哪个池

### 1.1 三个候选（`A1_clean_subset.md` 实测）

| 选项 | 来源 | 池规模 | ×10% | ∩ dataset | ∩`navtest` |
|---|---|---:|---:|---:|---:|
| **B1** `navtrain` | `scene_filter/navtrain.yaml` | 1,192 logs / 103,288 tok | ≈119 logs | 52 logs / 6,104 tok | **0 / 0** |
| **B2** `navall` | `scene_filter/navall.yaml` | 14,539 logs / 401,300 tok | ≈1,454 logs | 54 logs / 6,110 tok | **0 / 0** |
| **B3** split-config | `process_data/default_train_val_test_log_split.yaml` | 13,180 logs（**仅 log 级，无 token 列表**） | ≈1,318 logs | 44 logs / 35,890 tok | 0 logs |

### 1.2 v2.6 新增的三条决策材料

1. **`B1`/`B2` 的安全性完全等价。** 二者 ∩`navtest` 均为 **0/0**，且 `B1 ⊂ B2`。Sprint Plan v2.3 曾把 `B2` 写成"风险更高"，**该判断不成立**。选 `B1` 的理由只剩"规模小、跑得快"。
2. **官方定义已可信。** `NAVSIM_inventory.md` 证实 RAP 内的 `navtrain`/`navtest`/`navmini` membership 与官方 `v2.2-17-ga5f7110` checkout **相等**。`B1` 的 1,192/103,288 是官方真值。
3. **raw 侧支撑 `B1`。** `sensor_blobs/trainval` 对 `navtrain` 的定向核对为 current `826,304/826,304`、4-history `1,219,936/1,219,982`（仅缺 46 文件 / 3 logs）。**`B1` 在 raw 层面几乎完整。**

### 1.3 三个选项的实际代价

| 选项 | 优势 | 代价 |
|---|---|---|
| **B1** | 官方真值；最小；与 Human「前期优先小规模」倾向一致；raw 侧近乎完整 | 🔴 **需要一份 navtrain 版 `--split-config`（`train_logs`/`val_logs`/`test_logs` 格式），当前不存在**。`build_alignment_small_data.py` 只吃这种格式 |
| **B2** | 数据最多；安全性与 `B1` 等价 | 10% = 1,454 logs，规模与"快速验证"相悖 |
| **B3** | **`build_alignment_small_data.py` 实际消费的就是它**，无需新建 split-config | 🔴 **只有 log 级、无 token 列表** → 一切 token 级判定（含污染核验）都退化为 log 级反推，`A1_clean_subset` 明示「保证更弱」 |

### 1.4 建议

**裁 `B1`。** 三条理由：

- 安全性与 `B2` 等价（已 discharge），规模最小，方向上符合已表明的倾向；
- `B3` 的 log 级局限会**直接削弱 `A1` 硬门的证明强度** —— 在一个已经因污染 FAIL 的项目里，主动降低核验粒度是错误方向；
- 缺失的 navtrain split-config 是**一次性工程**（从 `navtrain.yaml` 的 1,192 logs 生成 `train_logs`/`val_logs` 切分），不是研究判断。

> ⚠️ 裁 `B1` 后须立即登记一项工程任务：**产出 navtrain 版 `--split-config`**，并按 `CP-CODE` 审。这是 `B1` 唯一的额外成本，不应被遗漏。
> ⚠️ `SD-3` 的 10% 在既有资产上兑现不了（见 §2.3），但这**不是换池能解决的** —— §2 `SD-12` 决策材料第 2 条已明示「用哪个池」与「抽多大比例」是两个正交旋钮。

---

## 2. `SD-16`｜`rendered_sensor_blobs_4cam_v1` 能否作为 F0/F1 输入

### 2.1 为什么会有这个问题：根因已定位

`A1_clean_subset` 的现象是「主 raster 根 `CAM_B0` 命中率 0%」。**代码层面的原因是确定的：**

| 证据 | 内容 |
|---|---|
| `process_data/helpers/renderer.py:695` | `def __init__(self, camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0'], ...)` —— 默认**三相机，无 `CAM_B0`** |
| `process_data/create_openscene_metadata.py:218` | `renderer = ScenarioRenderer()` —— **不传参**，直接吃默认值 |
| `renderer.py:119` | `camera_params` 字典里 `CAM_B0` **是定义好的**，只是从未进入 `camera_channel_list` |

**推论（决策相关）**：主根的三相机不是数据损坏，是**生成脚本的结构性行为**。而 `4cam_v1` 有 2,962 张 `CAM_B0` —— 它**不可能**由当前 `create_openscene_metadata.py` 原样产出。

> 这把 `SD-16` 的 provenance 问题从「文档缺失」升级为「**代码路径不同**」：`4cam_v1` 是由某个**与当前 `main` 不一致的代码版本或改动**生成的，而该版本未知。`A1_rap_datasets.md` 的 `UNKNOWN` 判定因此是**保守而正确**的。

### 2.2 资产实测

| 项 | 值 | 出处 |
|---|---|---|
| `4cam_v1` 覆盖 | **54 个 log**；`CAM_F0/L0/R0` 各 43,429，`CAM_B0` **2,962** | `A1_rap_datasets.md` |
| 主根按 channel | `B0 48 / F0 51,898 / L0 51,898 / L1 0 / L2 0 / R0 51,900 / R1 0 / R2 0` | `NAVSIM_inventory.md` |
| `C_d` 齐全 token（token 级剔除） | `B1` **1,878** ／ `B2` **1,878** ／ `B3` **2,387** | `A1_clean_subset.md` |
| `C_d` 齐全 token（log 级剔除） | **全部为 0** | 同上 |
| provenance | **UNKNOWN**（生成 commit / job / 原始输入均无记录） | `A1_rap_datasets.md` |

> 🔴 主根那 **48 张孤立 `CAM_B0`** 同样无解释。它们与 `4cam_v1` 是什么关系，目前无证据。

### 2.3 决定性的算术：批准 `SD-16` 也救不了 F0

| | `SD-3` 要求（`B1`） | `4cam_v1` 能给 |
|---|---:|---:|
| logs | **≈119** | 52（`A1_clean_subset` 的 ∩ 值） |
| `C_d` 齐全 token | —— | **1,878** |

**即使 `SD-16` 全票通过，F0 仍然不满足 `SD-3` 冻结的 10% budget。** 差距不是 10%，是量级。

**这改变了 `SD-16` 的性质**：它不是「F0 能不能跑」的开关，而是「**手上有没有一份四相机资产可以拿去做 Smoke Test / H0 几何检查**」的开关。

### 2.4 三个选项

| 选项 | 含义 | 后果 |
|---|---|---|
| **(a) 允许作为 F0/F1 输入** | 承认 `4cam_v1` 为合法 raster 源 | 🔴 不建议。provenance 是**代码路径级**未知（§2.1），且规模仍不满足 `SD-3`。用它跑出的 F0 无法在 `A5` 里填出 renderer/`Cd_hash` 行 |
| **(b) 完全不允许** | F0 必须重渲 `C_d` 四相机 raster | 立场最干净，但**当下没有任何四相机资产**，H0 与 Smoke 都无料可用 |
| **(c) 限定用途** ⭐ | 允许用于 **Smoke Test / H0 几何诊断**；**禁止**进入任何产生科学结论的 run；registry 中标 `INVALID for science` | 保留工程推进能力，同时不污染科学结论。**符合 §2 `SD-12` 决策材料第 3 条**：「Smoke 仅约 50 步、不产生科学结论，不受 `SD-2`/`SD-3` 约束」 |

### 2.5 建议

**裁 (c)，并附三项前置核验**（§2 `SD-16` 原文已建议前两项，第三项为本材料新增）：

1. 与主根**同名文件的像素级一致性抽样** —— 验证 `F0/L0/R0` 三路在两根之间是否等价；
2. **渲染配置 `Cd_hash` 反推** —— 从产物推断其相机内外参，与 `renderer.py:63–127` 的 `camera_params` 比对；
3. 🆕 **解释主根那 48 张 `CAM_B0`** —— 若它们与 `4cam_v1` 同源，则 `4cam_v1` 更可能是一次中断的重渲；若不同源，则存在第三个未知代码路径。

> ⚠️ **(c) 不自动解除 `CP-2b`。** §9 `CP-2b` 的措辞是「禁止 Smoke Test 与任何训练」，没有为 Smoke 留豁免口。若要在 `CP-2b` PASS 之前跑 Smoke，需要 Human **另开一条明确的豁免决议**，并在 registry 与 PPT 中标注该 run 的资产为 `provenance UNKNOWN`。

---

## 3. `SD-15` 的现状（影响上面两条，故须一并知悉）

Sprint Plan v2.6 已**作废** `SD-15` 的既有数字：

- `A1_clean_subset` 判 log 级归零的直接原因是「`warmup_test_e2e` 命中 64 个 log 中的 62 个」；
- 而 `NAVSIM_inventory.md` 证实 **`warmup_test_e2e.yaml` 是旧定义**，v2.2 真值 `warmup_two_stage` 对 `dataset_norm` 只命中 **1 log / 2 tokens**。

**方向性推论（尚待实测确认）**：在 v2.2 定义下扣除的评测侧 token 更少，因此重算后的干净池**只会 ≥ 现值**，`1,878 / 2,387` 是下界。

**但 log 级是否仍然归零，取决于一个未裁决的口径问题**：`navmini`（62 logs / 396 tokens）算不算评测侧？

| 口径 | 依据 | log 级后果 |
|---|---|---|
| **算** | `A1_clean_subset` 把它当评测侧扣除 | 64 个 log 中 62 个被命中 → 仍然归零 |
| **不算** | §4.1 的禁读清单**只列** `navtest` / `navhard_two_stage` / `warmup_two_stage`，**未列** `navmini` | log 级可能不归零，须重算 |

> **建议**：`SD-15` 暂不裁，先做一次 v2.2 口径下的重算（可扩展已入库的 `scripts/audit/a1_clean_subset.py`，加 v2.2 filter 源与 `navmini` 开关）。**但请注意 §2.3 的算术 —— 重算不会把 52 logs 变成 119 logs，`SD-3` 的缺口与 `SD-15` 无关。**

---

## 4. 建议的裁决顺序

| 序 | 条目 | 现在可裁？ | 建议 |
|---|---|---|---|
| 1 | `SD-12` | ✅ 材料齐全 | **`B1`**，并登记「产出 navtrain 版 split-config」工程任务 |
| 2 | `SD-16` 之「可否使用」 | ✅ 材料齐全 | **(c) 限定用途**，附 §2.5 三项核验 |
| 3 | `SD-16` 之「能用多少」 | ❌ 待 v2.2 重算 | 与 `SD-15` 一并，重算后裁 |
| 4 | `SD-15` | ❌ 待 v2.2 重算 + `navmini` 口径 | 同上 |
| 5 | **`SD-3` 的 10% budget** | ⚠️ **本材料未覆盖，但已被上述算术逼到台前** | 既有资产给不出 119 logs。按 §2，下调 budget 须 Human **新开决议 + 新 `run_id` + 重做 `A5`** |

> 第 5 项是本材料最重要的副产物：`SD-12`/`SD-15`/`SD-16` 怎么裁都绕不开它。**它才是 F0 规模问题的真正开关。**
