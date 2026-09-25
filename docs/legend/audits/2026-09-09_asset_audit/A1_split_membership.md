# A1｜Split Membership 与污染判定

- 审计时间：2026-09-09 17:01–17:14 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex

## 判定

**CONTAMINATED / FAIL。禁止训练。**

`dataset_norm` 与 `navtest` 的交集不是 0：**10 logs / 1,365 tokens**。这已触发 Sprint Plan §3.1 的硬门。另有 `navhard_two_stage` split 定义缺失，不能核验其交集。

## 实测集合

`dataset_norm/navsim_logs/mini/*.pkl`：64 files，51,867 records，51,867 unique tokens，64 unique logs；无不可读文件、无重复 token、无缺失 token/log 字段。

| 官方 split | 官方 logs | 官方 tokens | dataset log 交集 | dataset token 交集 | 状态 |
|---|---:|---:|---:|---:|---|
| `navtrain` | 1,192 | 103,288 | 52 | 6,104 | MIXED |
| `navtest` | 136 | 12,146 | **10** | **1,365** | **CONTAMINATED** |
| `navmini` | 62 | 396 | 62 | 396 | PRESENT |
| `warmup_test_e2e` | 62 | 563 | 62 | 563 | EVAL-SIDE OVERLAP |
| `navhard_two_stage` | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | **BLOCKED：定义缺失** |

在当前存在的上述官方 split 合集中，仍有 2 logs / 44,389 tokens 未命中。大量 token 未命中是因为官方 YAML 只列经过 scene filter 的 token 子集，不能把它们自动解释成合法训练 token。

### 10 个 navtest 重叠 logs

1. `2021.05.25.14.16.10_veh-35_01690_02183`
2. `2021.06.03.12.02.06_veh-35_00233_00609`
3. `2021.06.03.13.55.17_veh-35_00073_00426`
4. `2021.06.28.15.02.02_veh-38_02398_02848`
5. `2021.06.28.16.29.11_veh-38_01415_01821`
6. `2021.06.28.16.29.11_veh-38_03263_03766`
7. `2021.06.28.16.57.59_veh-26_00016_00484`
8. `2021.08.30.14.54.34_veh-40_00439_00835`
9. `2021.09.16.15.12.03_veh-42_01037_01434`
10. `2021.10.06.07.26.10_veh-52_00006_00398`

### 当前 split 定义外的 2 个 logs

- `2021.06.23.20.43.31_veh-16_03607_04007`
- `2021.08.24.13.12.55_veh-45_00386_00472`

未自行创建 `navhard_two_stage` 定义，也未把 `navmini` 当作训练许可。
