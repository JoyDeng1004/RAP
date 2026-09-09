# A4｜Source-pixel Boundary

- 审计时间：2026-09-09 17:36–17:54 JST
- 执行主机：`r4n11`
- Git HEAD：`6bca83e5864715ac6a1de5aab2d25c60024a19be`
- RAP_ROOT：`/gs/bs/tga-RLA/qdeng/RAP`
- 执行者：OpenAI Codex

## 判定

**BLOCKED for P1 F1；代码属于 P2。**

审计开始时当前分支没有对象文件；审计期间 Human 将 `tools/render_nuscenes_camera_cross.py` 暂存为新增文件。最终快照如下：

- Git 状态：`A  tools/render_nuscenes_camera_cross.py`（staged、未 commit）
- SHA-256：`e3220c8cbbd91a3aba6bb240c71e34c41b700aeb39d64907e5b8f21a4198a72d`
- `tools/canonical_bev/`：不存在
- `outputs/poster_pairs/nuscenes_cross_camera/`：不存在

## 四个必答问题

| 问题 | 证据 | 结论 |
|---|---|---|
| 是否读取 source calibration？ | `main()` 读取 `calibrated_sensor`、`sensor`，并取 native rotation/translation/camera_intrinsic | **YES** |
| 是否读取 source RGB？ | `real_source = root / native_sd["filename"]`；随后 `cv2.imread()` 与 `shutil.copy2()` | **YES** |
| raster 使用哪套 rig？ | Render A 注入 NAVSIM CAM_F0；Render B 注入 nuScenes native CAM_FRONT | **同时输出 C_d 与 C_s** |
| 既有产物属于哪种？ | 当前输出目录不存在，无法审计实际产物 | **NO ARTIFACT / BLOCKED** |

该文件与 `exp/rap-alignment-regression` 的 commit `27b1b6fec8e4cd7008f3332aa21c1324d73849be` 内容来源一致。无论 Render A 是否使用 C_d，这个程序的同一执行路径仍会读取 source calibration 和 RGB，因此整体访问计数不可能满足 P1 的全零约束。

此外，默认 `--nuscenes-root` 指向本仓库 `nuscenes-mini/nuscenes`，而 A2 证明该副本的 CAM_* 目录为空；本轮没有执行 renderer，也不推测其运行结果。
