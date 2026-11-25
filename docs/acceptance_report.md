# 验收报告（Phase 1–4 实际结果）

本文件只写**实际跑出来的东西**：命令、输出、失败与限制。所有数字都可以用列出的命令复现。

## 0. 环境

| 项 | 值 |
|---|---|
| OS / Shell | Windows / pwsh |
| Python | 3.12.10（`C:\Users\zz\AppData\Local\Programs\Python\Python312\python.exe`） |
| torch | 2.14.0+cpu（**CPU only**，无 CUDA） |
| numpy / scipy / matplotlib / pyyaml / pytest | 2.4.6 / 1.18.1 / 有 / 有 / 有 |
| 说明 | 系统默认 `C:\Python314\python.exe` 的 torch 是坏的（`ImportError: DLL load failed while importing _C`），全部命令用 3.12 解释器 |

## 1. 交付清单

```text
GeoPath-Diffuser/
├── configs/default.yaml             全量配置（对齐文档 §12）
├── configs/c64_longpath.yaml        长路径用 C=64 覆盖
├── references/{Astar-Diffusion,PathSafe-Ellipse,TrajSafe-Diffuser}   三个只读克隆（clean）
├── src/gp_diffuser/
│   ├── config.py frames.py factory.py
│   ├── map_encoder/{resunet.py, frozen_encoder.py}
│   ├── guidance/{grid,config,propagation,field,policies,paths,collision,sources,normalize}.py + search.py
│   ├── safety/{psenet,sampling,convex_region,safety_corridor,bspline_constraints}.py
│   │        + safety_features.py, psenet_inputs.py, local_regions.py
│   ├── trajectory/{bspline.py, dense_path.py}
│   ├── diffusion/{schedule.py, bspline_alm.py, sampler_conditional_dit.py}
│   ├── models/{conditional_dit.py, condition_encoders.py}
│   ├── training/{dataset.py, safety_cache.py, losses.py, dual_branch_trainer.py}
│   └── evaluation/{metrics.py, visualize.py}
├── scripts/{audit_references, build_safety_cache, check_bspline_fit, smoke_pipeline,
│            train_dual_branch, run_inference}.py
├── tests/{test_map_encoder, test_conditional_dit, test_dual_branch_training, test_trajectory,
│          test_guidance_search, test_safety_geometry, test_alm, test_inference_pipeline}.py
└── docs/{reference_code_audit, module_provenance, design_review, acceptance_report}.md
```

## 2. 参考仓库状态

```
Astar-Diffusion     main                          39c31d42ebb13baf4f644e0a9136641232fd30f8  clean
PathSafe-Ellipse    feature/v2.1-centerline       90d1f3ca8f7b5d6ffe359ce93380f07e732c5575  clean
TrajSafe-Diffuser   v4-map16-fbdelta-safeloss     14a5a9763781ae0c491a9f6bee732da7d5ee68cb  clean
```

`python scripts/audit_references.py` → `outputs/reports/reference_audit.json`：

```
checkpoint: 5041093 params, 20.25 MB, epoch 0, step 1250
features:  occ_bottleneck [1,192,32,32]   sdf_bottleneck [1,192,32,32]
           occ_full       [1,128,256,256] sdf_full       [1,128,256,256]
frozen=True  rays=16  d_clip=8.0  a∈[0.15,6.0]  b_min=0.05  ray_margin=0.01
```

## 3. 测试结果

```
$ python -m pytest tests -q
81 passed, 1 skipped in 108.60s
```

（skip 的是 `tests/test_cuda_device.py`：本机 `torch 2.14.0+cpu`，CUDA 用例只在有 GPU 的机器上执行。）

| 测试文件 | 覆盖的验收点 |
|---|---|
| `test_map_encoder.py` (9) | 瓶颈 `[192,32,32]` / 全分辨率 `[128,256,256]`；`bottleneck` 就是 `enc3` 直出（与手工重算逐元素相等）；一次调用只跑一次接一次 encoder；`requires_grad=False`；checkpoint 参数 5,041,093；**训练一步后 encoder 权重逐元素不变**；DiT 内无任何 `Conv2d`；`map_projection` 只吃 384=192+192 通道 |
| `test_conditional_dit.py` (9) | `D=384 / 12 blocks / 6 heads / FFN=1536`；AdaLN-Zero 9 参数零初始化；输出 `[B,32,2]`；两模式共用同一组参数对象；条件记忆 1026（map+SG）→ 1218（+192 安全）；path / ellipse 模态嵌入可分；源码与模块名中无 V5 组件；全 False 掩码时安全分支 ≡ Map-only 分支 |
| `test_dual_branch_training.py` (6) | 两次前向用**同一** `q_t` 与同一 `t`；只有 1 次 `backward` + 1 次 `optimizer.step`；联合损失等于文档公式（误差 <1e-4）；`pred_c.grad is None`（stop-grad）且下降方向指向安全分支；冻结模块无梯度；源码无 `warmup/base_only/safety_activation/stage_/phase` |
| `test_trajectory.py` (6) | clamped 端点 `C(0)=Q_0`、`C(1)=Q_31`；端点约束最小二乘端点精确；光滑曲线拟合误差 <0.02；硬端点写回；C 可改而不动网络；导数基对直线方向恒定 |
| `test_guidance_search.py` (5) | 搜索路径 128 点、端点精确、全部落在自由格；直线弧长重采样间距均匀（相对标准差 <1e-4）；引导源全在障碍内仍能出可行路径；无通路时返回 None；两个洞时路径跟随引导走廊（穿越点距引导口 <3 m） |
| `test_safety_geometry.py` (7) | 64 椭圆 + `[cx,cy,a,b,cos2θ,sin2θ]`；`cos2θ/sin2θ` 单位化；轴长在上游界限内且 `b≤a`；查询点首尾等于路径首尾；无碰撞 → 只有 64 个 box cell（4 面）；碰撞曲线 → 产生 ≥`min_queries` 的查询跨度；pack 面掩码与单位法向；padding 行 `b=+inf` 不生效；`SceneTransform` 往返一致且格心映射到 `grid_sample` 坐标 |
| `test_alm.py` (7) | ALM 降最大违反量；端点保持；返回对偶且可继承（warm start 后有限且不增）；全掩码 pack 为 no-op（`q_safe==q`，`lam=0`）；**追加 7 行 padding 不改变修正结果**；密集验证能识别碰撞（0 → 1）；单样本与 2 样本 pack 一致 |
| `test_review_regressions.py` (24) | 详见 `docs/code_review_verification.md` §0：P0-1..P1-7、P2-1、P2-3 每条缺陷的修复验收 |
| `test_cuda_device.py` (1, skip) | codec / heading / pack 跟随设备（需 CUDA 机器） |
| `test_inference_pipeline.py` (9) | 一次运行只编码一次地图；安全几何只生成一次（`search` 1 次、椭圆网络 1 次）；**所有引导步共用同一个 pack 对象**；网络从不接收 `feedback/history/delta/correction`；`safety_condition` 对象跨步稳定；`map_only` 模式无安全条件/无 ALM；输出有限、端点精确；`steps` 可配 |

## 4. 端到端最小验证

```
$ python scripts/smoke_pipeline.py --steps 10
16/16 checks passed
```

单样本（CPU，10 步 DDIM）耗时分解（`outputs/reports/smoke_checks.json`）：

| 阶段 | 秒 | 次数 |
|---|---|---|
| 地图编码（冻结，两次 ResUNet） | 0.127 | **1** |
| Map Projection | ≈0.001 | 每次前向复用 |
| 初始 Map-only 前向（5 步） | 0.219 | 5 |
| 引导场构建 + 网格搜索 | 0.199 | **1** |
| PathSafe 椭圆（64 个） | 0.023 | **1** |
| 局部凸区域 + 约束包 | 0.017 | **1** |
| ALM（5 个引导步） | 0.008 | 5 |
| 最终密集验证（512 点） | 0.005 | 1 |
| **合计** | **0.863** | |

（scipy 后端已装：引导场走 compiled sparse Dijkstra。另验证 heap / scipy 两后端等价：最大绝对差 7.8e-16、可达集完全一致，见 §7。）

## 5. B-spline 表示（长路径）实测

```
$ python scripts/check_bspline_fit.py
| path length (m) | n | max err C=32 | max err C=48 | max err C=64 | mean err C=32 |
| 20-40 |  9 | 0.2069 | 0.1475 | 0.0874 | 0.1370 |
| 40-60 |  9 | 0.3290 | 0.2089 | 0.1387 | 0.1791 |
| 60-inf| 18 | 0.6162 | 0.3930 | 0.2799 | 0.1923 |
```

结论：C=32 在 >60 m 路径上最大误差 0.62 m，与 0.5 m 机器人半径同量级；C=64 降到 0.28 m。详见 `docs/design_review.md` §3.2（需你决定正式训练是否统一切 64）。

## 6. 安全几何与走廊闭环率

```
$ python scripts/build_safety_cache.py --split train --size 16
corridor built for 8/16 samples; mean local cells 32.6; bridges 12
corridor failure reasons: {'bridge_failed': 7, 'bridge_region_invalid': 1}

$ python scripts/build_safety_cache.py --split val --size 4
corridor built for 1/4 samples; mean local cells 16.5; bridges 4
corridor failure reasons: {'bridge_failed': 3}
```

凸单元尺寸实测（val 6 个场景，64 单元/场景）：面积中位数 **30.6–156.3 m²**、`bounded` 合法率 **100 %**；对照上游 `results/figures/grid/summary.json` 的 `area_p50_m2 = 103.8` 与 `corridor_success_rate = 0.542`。

即：凸单元几何与上游同量级，走廊闭环率（≈50 %）也与上游自身水平（54 %）一致。椭圆偏薄（`b` 中位数 0.13 m）是冻结网络在合成地图上的域偏移，未做任何补偿（不放大椭圆、不改 PathSafe 输出）。未闭环的样本退化为「仅地图边界 box cell」；ALM **不会**自动跳过（见下方更正）。

> **更正（代码复审后）**：本报告此前写作「ALM 自动 no-op（`pack.piece_mask` 全 False）」，这是**错的**。`build_region_table()` 在没有任何局部凸单元时仍为 64 个查询点生成地图边界 box cell，`piece_mask` 并非全 False，ALM 仍会执行（实测：`local_cells = 0` 时 ALM 仍跑 5 步）。实测证据见 `docs/code_review_verification.md` 的 P1-2。

## 7. 双路联合训练（历史记录：本轮修复未训练）

> **状态说明（代码复审后 + 网络结构修订后）**：本节数字来自修复**之前**的一次 16 步烟测，
> 且该权重属于旧网络结构，已随结构变更一并删除（现仓库内无任何 Diffusion 权重）。
> 复审意见是"先修 P0/P1，再谈训练"，因此本轮修复过程**没有**执行任何训练命令；
> 那次误跑的产物（`outputs/checkpoints/toy.pt`、`last.pt`、`training_curves.png`）
> 已删除。下表仅作为"双路联合训练能跑通"的历史证据保留，不再代表当前代码，
> 也不能用于任何精度结论。正式训练待复审通过后再安排。

```
$ python scripts/train_dual_branch.py --steps 60 --epochs 4 --batch-size 4 --size 16 \
      --save outputs/checkpoints/toy.pt
trainable parameters: 44940290 (frozen encoders: 0 trainable)
epoch 4 step 16 | loss 0.56640 (u 0.17710 c 0.17688 align 0.00000 col 0.17374 corr 0.03867)
```

* 可训练参数 44,940,290 = Conditional DiT 44,792,450 + Map Projection 147,840；冻结模块 0 可训练参数。
* 每次迭代恰好 2 次前向、1 次反向、1 次更新（测试用 spy 断言）。
* `align` 在初始阶段接近 0（输出头零初始化 → 两分支同值），随训练才分化；这是预期行为，不是未实现。
* **仅 16 步，模型远未收敛，任何精度结论都不成立**。按你的要求没有启动长时间训练。
* 本次网络结构修订（双 Adapter 地图融合 + 独立 Safety Cross-Attention）之后，
  该 checkpoint 与当前代码**不兼容**（`architecture_version` 校验会拒绝加载），文件已删除。

## 8. 推理三配置对照（历史记录，数字已作废）

> **状态说明（代码复审后）**：下表由已删除的 toy checkpoint 产生，且修复前
> `collision_loss` 单位错误、ALM 无门控、碰撞检测只查点、碰撞区间用索引映射，
> 因此**这些数字全部作废**，不能引用。修复后的推理脚本已扩展为报告
> 朝向误差与 ALM 接受/拒绝步数（`scripts/run_inference.py`），
> 但按"先复审、再训练/评测"的顺序，本轮没有重新生成对照表。
> 复现命令已写在 README §3.4；重新生成时请一并更新本节。

```
$ python scripts/run_inference.py --samples 3 --steps 10 --checkpoint outputs/checkpoints/toy.pt
```

| mode | 碰撞率 | 成功率 | 最小净空 (m) | 路径长 (m) | 平均曲率 | 走廊违反 (m) | ALM 步数 | 单样本耗时 (s) |
|---|---|---|---|---|---|---|---|---|
| map_only | 0.1544 | 0.33 | −1.673 | 58.07 | 2.4389 | 0.0000 | 0 | 0.57 |
| no_alm | 0.1553 | 0.33 | −1.686 | 58.27 | 2.4700 | 42.1363 | 0 | 0.83 |
| full | 0.2153 | 0.00 | −2.187 | 86.00 | 2.1171 | 35.9648 | 5 | 0.89 |

读法（重要，不要误读）：

* ALM **确实在起作用**：固定走廊违反量从 42.14 m 降到 35.96 m。
* 但 toy 模型轨迹本身是垃圾（碰撞率 15–22 %，最小净空 −2 m），ALM 把它往 box cell 内部推，路径被拉长到 86 m、碰撞率反而升高。这是"未收敛模型 + 硬约束"的正常现象，**不能作为方法有效性的证据**。
* `map_only` 的走廊违反恒为 0 只是因为它没有 pack（不是它更安全）。
* 三种模式都已经跑通并产出可视化图（`outputs/figures/{map_only,no_alm,full}_000_10.png`）。

## 9. 可视化

`outputs/figures/smoke_pipeline.png` 与 `outputs/figures/{mode}_000_10.png` 三联图：

* (a) 统一安全地图 + GT 路径 + 初始 Map-only 引导轨迹 + 单条搜索路径 + 起终点；
* (b) 64 个 PathSafe 椭圆（紫）+ 实际构建的凸单元（青=局部、灰=地图边界 box cell）；
* (c) 最终轨迹 + B-spline 控制多边形 + `valid / collision / ALM 步数` 角标。

`outputs/figures/training_curves.png`：联合损失与四个分量的 symlog 曲线。

## 10. 已知限制（未完成项，全部显式记录）

1. **没有正式训练**：只有 16 步烟测模型；没有任何收敛性、精度或对比实验结论。文档 §14 的 A–I 实验矩阵只搭好了开关（`safety_mode ∈ {map_only, no_alm, full}`），未执行。
2. **数据是合成的**：原数据集不在工作区，指标不可与论文对比。
3. **PathSafe 在合成域上椭圆退化**（b≈0.13 m）→ 走廊闭环率 ≈50 %；未做任何补偿性修改。
4. **训练期走廊用全路径 scope**（`corridor.training_scope: full`），推理期用碰撞区间局部 scope；原因见 `docs/design_review.md` §3.1，待你确认。
5. **C=32 vs 64** 未定（见 §5）。
6. **未迁移 `ellipse_densify`**：局部缺口只靠上游 `bridge` 处理。
7. **CPU only**：单样本 10 步 0.86 s，50 步约 4 s（估）；无 GPU 上的吞吐数据。
8. **多 batch 的 `_merge_packs`** 只在 B>1 的 `sample()` 里用到，已跑通（2 样本 smoke），但没有针对"不同样本 piece/face 数差异很大"的压力测试。
