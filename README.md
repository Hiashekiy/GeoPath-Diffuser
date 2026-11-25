# GeoPath-Diffuser

基于**冻结地图瓶颈特征**的安全条件扩散轨迹规划 —— 独立新项目。

扩散主干是全新的标准 **Conditional DiT-S**（384 维 / 12 层 / 6 头 / AdaLN-Zero），条件来自：

* **PathSafe-EllipseNet 已训练的两个 ResUNet 的 `enc3` 瓶颈特征**（冻结，只编码一次）；
* 起终点 / 朝向；
* 由 Astar-Diffusion 引导场搜索出的**单条路径** + PathSafe 预测的 **64 个椭圆**；
* 由搜索路径与障碍物构建的**局部凸安全区域**，通过 **ALM** 在推理时做硬约束修正。

设计依据：`docs/安全条件扩散轨迹规划方案_Conditional-DiT网络修正版.md`。

> **参考仓库只读**：`references/` 下的三个仓库仅作参考代码来源，从未被修改、从未在其上开发，也不是本项目的主体。逐文件迁移记录见 `docs/module_provenance.md`。

---

## 1. 架构与数据流

### 推理（一次地图编码、一次搜索、一次安全几何）

```text
occupancy + sdf  (metres)
      │  frozen PathSafe ResUNet ×2        ← 只执行一次
      ├── enc3 bottleneck [B,192,32,32] ──→ MapConditionEncoder ──→ map memory [B,1024,384]
      │                                    （双 Adapter + 残差空间融合，可训练）
      └── full feature    [B,128,256,256] ─→ PathSafe-EllipseNet 的局部查询
                                                  │
Q_T ~ N(0,I) ──→ Map-only DDIM 前段 ──→ 初始全局引导轨迹
                                                  │
                              Astar-Diffusion 引导场搜索（1 次）
                                                  │  单条可行路径 [128,2] m
                              PathSafe-EllipseNet（1 次）→ 64 椭圆
                                                  │
                    SafetyConditionEncoder（1 次）→ safety memory [B,64,384]
                              局部凸区域 + 约束包（1 次，碰撞区间 ±4，≥8 查询）
                                                  │
        Safety-conditioned DDIM 后段（固定复用以上几何）
        Map-only 分支：只用 Map Context；安全分支：每层额外一条
        Safety Cross-Attention（独立参数 + 可学习门控 α_l，初始 0.1）
                                                  │
                          每步 ALM：min ‖Q−Q_raw‖² + λs·L_smooth   s.t. A_jE_{l,r}Q ≤ b_j
                                                  │
                                      DDIM 状态更新 → 最终控制点 → 128 点轨迹
                                                  │
                                     密集碰撞验证（512 点）
```

* **没有历史反馈**：ALM 结果只进入当前 DDIM 状态更新，不编码、不缓存、不回灌网络。
* **不重复昂贵操作**：搜索 / 椭圆 / 走廊各只做一次，后续反向步复用同一个 pack（测试断言了对象同一性）。

### 训练（单阶段双路联合，一次反向）

```text
同一样本、同一噪声 ε、同一时间步 t：
  Qt = √ᾱt·Q0 + √(1−ᾱt)·ε
  pred_u = f_θ(Qt, t, S, G, M_map)                       ← Map-only 分支
  pred_c = f_θ(Qt, t, S, G, M_map, H_safe)               ← Safety-conditioned 分支（同一套权重）
  L = λc·‖pred_c−Q0‖² + λu·‖pred_u−Q0‖²
      + λa·‖pred_u − sg(pred_c)‖²
      + λcol·L_collision(pred_c) + λcorr·L_corridor(pred_c)
  optimizer.zero_grad(); loss.backward(); optimizer.step()      ← 只有这一次
```

没有 warm-up、没有 Safety Activation 阶段、没有第二套网络。安全条件（路径 / 椭圆 / 凸区域）由冻结模块**离线生成并缓存**（`scripts/build_safety_cache.py`）。

## 2. 目录

```text
configs/                 default.yaml（全量配置）、c64_longpath.yaml
references/              Astar-Diffusion / PathSafe-Ellipse / TrajSafe-Diffuser（只读克隆）
src/gp_diffuser/
  config.py frames.py factory.py
  map_encoder/           resunet.py（迁移）、frozen_encoder.py
  guidance/              Astar-Diffusion 全量迁移 + search.py 适配层
  safety/                psenet.py sampling.py convex_region.py safety_corridor.py bspline_constraints.py（迁移）
                         safety_features.py psenet_inputs.py local_regions.py（新写）
  trajectory/            bspline.py（迁移）、dense_path.py
  diffusion/             schedule.py bspline_alm.py（迁移）、sampler_conditional_dit.py（新写）
  models/                conditional_dit.py condition_encoders.py（新写）
  training/              dataset.py safety_cache.py losses.py dual_branch_trainer.py
  evaluation/            metrics.py visualize.py
scripts/                 audit / cache / fit-study / smoke / train / inference
tests/                   56 个 pytest 用例
docs/                    reference_code_audit.md module_provenance.md design_review.md acceptance_report.md
outputs/                 figures/ reports/ cache/ checkpoints/（运行产物）
```

## 3. 安装与运行

### 3.1 环境

```powershell
# 解释器（本机默认 3.14 的 torch 是坏的，必须用 3.12）
$py = "C:\Users\zz\AppData\Local\Programs\Python\Python312\python.exe"
cd D:\ProjectDirectory\GeoPath-Diffuser

& $py -m pip install torch numpy pyyaml matplotlib pytest
& $py -m pip install scipy        # 可选；装上后引导场走 compiled sparse Dijkstra
```

### 3.2 拉取三个只读参考仓库（必需，`references/` 被 git 忽略）

```powershell
& $py scripts\setup_references.py          # 克隆缺失的仓库（main / feature/v2.1-centerline / v4-map16-fbdelta-safeloss）
& $py scripts\setup_references.py --check   # 仅校验：分支、工作区 clean、checkpoint 5,041,093 参数 + sha256
```

`setup_references.py` 退出码非 0 就说明参考环境不完整，后面所有命令都会因为缺
`references/PathSafe-Ellipse/checkpoints/v3_best.pt` 而失败，**不要跳过这一步**。

### 3.3 校验与测试

```powershell
& $py scripts\audit_references.py    # 分支 / commit / 工作区 / checkpoint 张量形状 / 特征形状
& $py -m pytest tests -q              # 82 个用例（81 passed + 1 CUDA skip）
```

### 3.4 运行

```powershell
# 1) 端到端最小验证（18 项检查 + 三联图）
& $py scripts\smoke_pipeline.py --steps 10

# 2) B-spline 长路径拟合误差统计
& $py scripts\check_bspline_fit.py

# 3) 生成训练用安全条件缓存（冻结模块离线跑；带指纹校验）
& $py scripts\build_safety_cache.py --split train --size 16
& $py scripts\build_safety_cache.py --split val   --size 4

# 4) 双路联合训练（烟测；正式训练请自己加 epochs/数据量）
& $py scripts\train_dual_branch.py --steps 60 --epochs 4 --size 16 --save outputs/checkpoints/toy.pt

# 5) 推理 + 三配置对照 + 出图（碰撞率 / 朝向误差 / 走廊违反 / ALM 接受-拒绝）
#    仓库不附带任何权重；--checkpoint 指向第 4 步自己产出的文件（可省略 = 随机初始化）
& $py scripts\run_inference.py --samples 3 --steps 10 --checkpoint outputs/checkpoints/last.pt
```

缓存文件（`outputs/cache/safety_*.npz`）带指纹（配置子集 + 数据集形状 + checkpoint
大小/mtime 的 sha256）；换配置或换 checkpoint 后 `load_safety_cache` 会拒绝旧缓存并提示重建，
不会静默用错数据。

## 4. 配置要点（`configs/default.yaml`）
| 段 | 关键值 | 说明 |
|---|---|---|
| `map_encoder` | `checkpoint: references/PathSafe-Ellipse/checkpoints/v3_best.pt`，`freeze: true` | 架构从 checkpoint 内嵌 `cfg` 重建，不猜 YAML |
| `map_condition` | `encoder_type: dual_adapter_residual_fusion`：occ/sdf 各一个 Adapter（Conv1x1+GN16+SiLU）→ concat 384 → 残差空间融合（DW3x3+PW1x1）→ 1024 token | 分辨率始终 32×32；不新增编码器 |
| `diffusion` | DiT-S：`hidden_dim 384 / depth 12 / num_heads 6 / mlp_ratio 4`，`x0` 预测，`squaredcos_cap_v2`，T=1000 | 与 V5 无关 |
| `diffusion.condition_fusion` | `separate_map_safety_cross_attention`：每层 SA → Map CA → Safety CA → MLP | Map CA 独立参数/LayerNorm，始终启用；Safety CA 只在安全模式执行 |
| `diffusion.safety_residual_gate_init` | `0.1`（每层可学习 α_l） | 不额外叠加零初始化门控 |
| `trajectory` | `degree 3 / num_controls 32 / curve_points 128` | 控制点数是轨迹表示，不是网络深度 |
| `guidance` | `lam 0.01 / weight linear / policy B3_linear_fusion / alpha 0.25 / path_points 128` | 上游实测最优组合；λ 随地图尺度调整 |
| `safety_condition` | **仅 64 椭圆** `[cx,cy,a,b,cos2θ,sin2θ]`；`path_tokens: false` | 搜索路径仍用于椭圆中心线/碰撞投影/凸区域，但不进入 Diffusion |
| `corridor` | `min_overlap_ratio 0.10`、`bridge`、`region.*`、`local.pad_queries 4`、`local.min_queries 8`、`training_scope full` | 数值沿用上游 `config_v5_c32q64.yaml` |
| `alm` | `rho 5.0 / step_size 0.05 / inner_steps 6 / tol 1e-3 / max_curve_step_scene 0.20 / inherit_dual true / dense_verify 512` | 完全沿用上游，不另设 |
| `sampling` | `steps 50`、`mid_forward_fraction 0.5` | 安全条件在第 ⌊0.5·(N−1)⌉ 次前向之后生成一次 |
| `loss_weights` | `λu 1.0 / λc 1.0 / λalign 0.5 / λcol 1.0 / λcorr 1.0` | 待验证集调 |

### 4.1 网络规模（实测，CPU 构建）

| 模块 | 参数量 | 是否训练 |
|---|---|---|
| FrozenMapEncoder（两个 ResUNet，enc3 前向） | 3,157,168 | 冻结 |
| PathSafe-EllipseNet 全网络（含安全几何头） | 5,041,093 | 冻结 |
| MapConditionEncoder（occ/sdf Adapter + 融合） | 226,944 | 训练 |
| Conditional DiT（含 Safety Encoder 150,912 + Safety CA 7,096,320） | 55,435,406 | 训练 |
| 其中每层 Safety Cross-Attention / Map Cross-Attention / Self-Attention / MLP | 各 7,096,320 / 7,096,320 / 7,096,320 / 14,178,816 | — |
| 每层可学习安全门控 α_l | 12（12×1） | 训练 |
| **可训练参数合计** | **55,662,350** | |

张量接口：`map_memory [B,1024,384]` → `map_context [B,1026,384]`；
`ellipse [B,64,6]` → `safety_memory [B,64,384]` + `safety_mask [B,64]`。
checkpoint 带 `architecture_version`（`conditional_dit_dual_map_fusion_dual_cross_attention_v2`），
旧结构权重加载会直接报错提示重新训练。

## 5. 数据说明（重要）

原数据集**不在本工作区**（PathSafe 的 `data/` 被 git 忽略；TrajSafe 的 `processed_root` 指向外部目录）。本项目自带合成场景生成器（`training/dataset.py`），严格对齐 PathSafe 文档化的数据契约：

* 256×256 栅格、`resolution_m = 0.25`（64 m × 64 m）；
* `occupancy ∈ {0,1}`，障碍按机器人半径（2 格 = 0.5 m）**只膨胀一次**，该膨胀后地图是搜索 / SDF / PathSafe / 走廊 / 碰撞度量的唯一地图；
* `sdf` 米制、自由正、障碍内负、越界 = blocked（`SDF_OUTSIDE_M = -1.0`）；
* 16 条 LiDAR 射线、`angle_k = 2πk/16`、量程 9 m、到第一个 blocked 格**边界**的距离、`ray_valid = lidar < max_range − 1 mm`。

因此**所有指标都是合成数据上的，不可与论文数字直接对比**。

## 6. 已知限制

1. **仓库内没有任何 Diffusion 权重**：条件网络结构已变更
   （`conditional_dit_dual_map_fusion_dual_cross_attention_v2`），旧结构 checkpoint 全部作废并已删除；
   正式实验必须从头训练。`scripts/train_dual_branch.py` 只是训练入口。
2. 冻结 PathSafe 在合成地图上椭圆偏薄（`b` 中位数 0.13 m）→ 走廊闭环率 ≈50 %（上游自身 54 %）；未闭环样本自动退化为"仅地图边界约束"，ALM no-op。
3. 长路径（>60 m）C=32 拟合误差 0.62 m，建议切 `configs/c64_longpath.yaml`（待确认）。
4. 训练用走廊是全路径 scope，推理用碰撞区间局部 scope（理由见 `docs/design_review.md` §3.1）。
5. CPU only。

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/reference_code_audit.md` | 三个仓库的结构、真实接口、张量形状、依赖、复用决策、缺失模块取证 |
| `docs/module_provenance.md` | 逐文件迁移表（仓库 / 分支 / commit / 原路径 / 新路径 / 适配 / 是否改算法） |
| `docs/design_review.md` | 文档内部不一致、文档与代码不一致、需你确认的取舍、与"明确禁止"的逐条对照 |
| `docs/acceptance_report.md` | 实际测试与实测数字（56 tests、逐阶段耗时、拟合误差、走廊统计、三配置对照） |
