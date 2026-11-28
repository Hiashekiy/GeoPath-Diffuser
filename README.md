# GeoPath-Diffuser

基于**冻结地图瓶颈特征**的安全条件扩散轨迹规划 —— 独立新项目。

扩散主干是全新的标准 **Conditional DiT-S**（384 维 / 12 层 / 6 头 / AdaLN-Zero），条件来自：

* **PathSafe-EllipseNet 已训练的两个 ResUNet 的 `enc3` 瓶颈特征**（冻结，只编码一次）；
* 起终点 / 朝向；
* 由 Astar-Diffusion 引导场搜索出的**单条路径** + PathSafe 预测的 **64 个椭圆**；
* 由搜索路径与障碍物构建的**局部凸安全区域**，通过 **CPCG（基于干净轨迹预测的约束梯度引导）+ 末端凸 QP** 在推理时修正。

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
                  每步 CPCG：q0_guided = q0 − α·η·∇_q0 J(q0)，J = 1/(2M) Σ relu(A_j(l)N_lQ − b_j(l))²
                  （解析梯度：不反传 DiT；均匀弧长采样点，每点只受它所属那一个凸区域约束；
                   端点/朝向保持；backtracking 只接受 J 严格下降的步，位移不再归一化到固定米数）
                                                  │
                                       DDIM 状态更新（q0_used = q0_guided）
                                                  │
                        全部反向步结束 → 一次末端凸 QP（OSQP）：min ‖Q−Q_guided‖²
                                          s.t. A_j(l)N_lQ ≤ b_j(l)（同一批 256 个采样点）,
                                               Q_0=S, Q_{C−1}=G
                                                  │
                                      最终控制点 → 128 点轨迹
                                                  │
                                     密集碰撞验证（512 点，在 QP 之后）
```

* **没有历史反馈**：CPCG 的干净预测修正只进入当前 DDIM 状态更新（`q0_used`），不编码、不缓存、不回灌网络，也不增加模型前向次数。
* **不用 ALM**：`diffusion/bspline_alm.py` 仅作历史/对照保留，活动推理链路不再 import 它；对带噪状态不施加碰撞半空间，也不对 DiT 反传。
* **不重复昂贵操作**：搜索 / 椭圆 / 走廊各只做一次，后续反向步复用同一个 pack（测试断言了对象同一性）。
* **约束表达（2026 改造）**：推理期不再做 Bezier 抽取。安全几何生成时用参考控制点 `Q_ref` 求一次均匀**弧长**采样
  `u_l`（`M = sampling.constraint_points`），并把 `N_l = N(u_l)`、每个采样点唯一归属的凸区域 `j(l)` 一起冻结；
  CPCG 与末端 QP 共用同一个 sample pack，约束恒为 `A_{j(l)} N_l Q ≤ b_{j(l)}`。
  采样只保证这 M 个点满足各自区域，**不构成整条连续曲线的数学安全证书**，最终仍由 512 点独立碰撞验证判定。

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
configs/                 base.yaml（全量基线，T=12）、c64_t12.yaml、k4p_160m_t12_c32.yaml、c64_longpath.yaml
references/              Astar-Diffusion / PathSafe-Ellipse / TrajSafe-Diffuser（只读克隆）
src/gp_diffuser/
  config.py frames.py factory.py
  map_encoder/           resunet.py（迁移）、frozen_encoder.py
  guidance/              Astar-Diffusion 全量迁移 + search.py 适配层
  safety/                psenet.py sampling.py convex_region.py safety_corridor.py bspline_constraints.py（迁移，
                         训练/历史 Bezier 包）safety_features.py psenet_inputs.py local_regions.py
                         bspline_sample_constraints.py（新写，推理期均匀弧长采样包）qp_projection.py（新写）
  trajectory/            bspline.py（迁移）、dense_path.py
  diffusion/             schedule.py sampler_conditional_dit.py clean_guidance.py（新写）
                         bspline_alm.py（历史/对照，不在活动链路）
  models/                conditional_dit.py condition_encoders.py（新写）
  training/              dataset.py safety_cache.py losses.py dual_branch_trainer.py
  evaluation/            metrics.py visualize.py
scripts/                 audit / cache / fit-study / smoke / train / inference
tests/                   pytest 用例（含 clean_guidance / qp_projection / 推理集成）
docs/                    reference_code_audit.md module_provenance.md design_review.md acceptance_report.md
outputs/                 figures/ reports/ cache/ checkpoints/（运行产物）
```

## 3. 安装与运行

### 3.1 环境（训练用 GPU 环境）

```powershell
# 推荐：conda 环境 GGMPC（Python 3.10, torch 2.9.1+cu126, CUDA 12.6, RTX 4070）
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
cd D:\ProjectDirectory\GeoPath-Diffuser

# 备用：系统 Python 3.12 + torch 2.14.0+cpu（仅 CPU，速度约慢 10 倍）
#   C:\Users\zz\AppData\Local\Programs\Python\Python312\python.exe
```

* 依赖（GGMPC 已全部具备）：`torch` / `numpy` / `scipy` / `pyyaml` / `matplotlib` / `pytest`。
* 训练/推理加 `--device cuda`；不带该参数默认 `cpu`。
* 系统默认的 `C:\Python314` 里 torch 是坏的（`DLL load failed`），不要用。
* 设备一致性已由测试覆盖：`tests/test_cuda_device.py` 校验 codec buffer、heading、约束包、
  **噪声调度表**（`q_sample` 用时间步索引它）都在同一设备上。

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

# 5) 推理 + 三配置对照 + 出图（碰撞率 / 朝向误差 / 走廊违反 / CPCG 每步违反量 / 末端 QP）
#    仓库不附带任何权重；--checkpoint 指向第 4 步自己产出的文件（可省略 = 随机初始化）
#    默认 8 个执行步（DDIM 子采样，权重不变、不重训）
& $py scripts\run_inference.py --dataset real --config configs/c64_longpath.yaml \
        --samples 8 --modes map_only,safety_only,cpcg --checkpoint outputs/checkpoints/last.pt

# 6) 走廊配置迁移等价性（ALM -> corridor，几何必须逐位相同）
& $py scripts\verify_corridor_equivalence.py --index 2
```

缓存文件（`outputs/cache/safety_*.npz`）带指纹（配置子集 + 数据集形状 + checkpoint
大小/mtime 的 sha256）；换配置或换 checkpoint 后 `load_safety_cache` 会拒绝旧缓存并提示重建，
不会静默用错数据。

## 4. 配置要点（`configs/base.yaml`）
| 段 | 关键值 | 说明 |
|---|---|---|
| `map_encoder` | `checkpoint: references/PathSafe-Ellipse/checkpoints/v3_best.pt`，`freeze: true` | 架构从 checkpoint 内嵌 `cfg` 重建，不猜 YAML |
| `map_condition` | `encoder_type: dual_adapter_residual_fusion`：occ/sdf 各一个 Adapter（Conv1x1+GN16+SiLU）→ concat 384 → 残差空间融合（DW3x3+PW1x1）→ 1024 token | 分辨率始终 32×32；不新增编码器 |
| `diffusion` | DiT-S：`hidden_dim 384 / depth 12 / num_heads 6 / mlp_ratio 4`，`x0` 预测，`squaredcos_cap_v2`，`train_diffusion_steps 1000`（训练噪声表长度，**权重绑定**；推理步数由 `sampling.steps: 8` 控制） | 与 V5 无关 |
| `diffusion.condition_fusion` | `separate_map_safety_cross_attention`：每层 SA → Map CA → Safety CA → MLP | Map CA 独立参数/LayerNorm，始终启用；Safety CA 只在安全模式执行 |
| `diffusion.safety_residual_gate_init` | `0.1`（每层可学习 α_l） | 不额外叠加零初始化门控 |
| `trajectory` | `degree 3 / num_controls 32 / curve_points 128` | 控制点数是轨迹表示，不是网络深度 |
| `guidance` | `lam 0.01 / weight linear / policy B3_linear_fusion / alpha 0.25 / path_points 128` | 上游实测最优组合；λ 随地图尺度调整 |
| `safety_condition` | **仅 64 椭圆** `[cx,cy,a,b,cos2θ,sin2θ]`；`path_tokens: false` | 搜索路径仍用于椭圆中心线/碰撞投影/凸区域，但不进入 Diffusion |
| `corridor` | `mode trajsafe`、`min_overlap_ratio 0.10`、`bridge`、`region.*`（含 `safety_margin 0.02`）、`constraint_margin_scene 0.0`、`local.*`、`training_scope full` | 几何参数已从 `alm` 解耦（`scripts/verify_corridor_equivalence.py` 证明逐位不变） |
| `clean_guidance` | `enabled true`、`method analytic_squared_hinge_unormalised`、`step_size 50.0`、`backtrack_factors [1,.5,.25,.125,.0625]`、`margin_scene 0.0` | CPCG：`J = 1/(2M)·Σrelu(g+margin)²`，`M` 固定为采样点数（**不除违反个数**）+ 解析梯度 `∇J = (1/M)Σ relu(g+δ)a_{j(l),f}N_l` → BoundaryDecoder 重投影 → **不再**把位移归一化到固定 `max_delta_m`，`step_size` 就是真实梯度步长 → backtracking 只接受 `J` 严格下降的步 |
| `final_projection` | `enabled true`、`solver osqp`、`eps 1e-5`、`max_iter 10000`、`polish true`、`feasibility_tol_scene 1e-4` | 采样结束后一次末端凸 QP |
| `alm` | `enabled false`（历史/对照，不在活动链路） | 只为 `tests/test_alm.py` 保留 |
| `sampling` | **`steps 8`**（默认执行 8 个反向步）、`mid_forward_fraction 0.5`、`constraint_points 256`、`arc_length_lookup_points 1024`、`dense_verify_points 512` | 8 步 = 在训练好的 1000 点噪声表上均匀取 8 个时刻（DDIM 子采样），无需重训；`steps >= train_diffusion_steps` 才是满表 |
| `loss_weights` | `λu 1.0 / λc 1.0 / λalign 0.5 / λcol 1.0 / λcorr 1.0` | 待验证集调 |

### 4.1 网络规模（实测，CPU 构建）

| 模块 | 参数量 | 是否训练 |
|---|---|---|
| FrozenMapEncoder（两个 ResUNet，enc3 前向） | 3,157,168 | 冻结 |
| PathSafe-EllipseNet 全网络（含安全几何头） | 5,041,093 | 冻结 |
| MapConditionEncoder（occ/sdf Adapter + 融合） | 226,944 | 训练 |
| Conditional DiT（含 Safety Encoder 150,912 + Safety CA 7,096,320） | 55,435,406 | 训练 |
| 其中 Self-Attention（12 层合计） | 7,096,320 | 训练 |
| 其中 Map Cross-Attention（12 层合计） | 7,096,320 | 训练 |
| 其中 Safety Cross-Attention（12 层合计） | 7,096,320 | 训练 |
| 其中 MLP（12 层合计） | 14,178,816 | 训练 |
| 其中 AdaLN 调制（12 层合计，含 bias：12×11×(384×384+384)） | **19,514,880** | 训练 |
| 其中每层可学习安全门控 α_l | 12（12×1） | 训练 |
| **可训练参数合计** | **55,662,350** | |

张量接口：`map_memory [B,1024,384]` → `map_context [B,1026,384]`；
`ellipse [B,64,6]` → `safety_memory [B,64,384]` + `safety_mask [B,64]`。
checkpoint 带 `architecture_version`（`conditional_dit_dual_map_fusion_dual_cross_attention_v2`），
旧结构权重加载会直接报错提示重新训练。

### 4.2 梯度启动过程（实测，3 次真实 optimizer.step）

结构不变的前提下，零初始化策略决定了前几步哪些参数能拿到梯度
（`tests/test_gradient_onset.py` 用**全新**模型和真实 `optimizer.step` 逐帧记录）：

| 步骤 | 获得有效梯度的部分 | 仍为 0 的部分 |
|---|---|---|
| 第 1 步 | `final_norm` + 输出头 | 其余全部（`head.weight = 0` → `dL/dx_final = 0`，AdaLN 残差门控也因 `gate = 0` 不产生权重梯度） |
| 第 2 步 | AdaLN 调制（shift/scale/gate 的参数）+ 安全分支（Safety Encoder / Safety CA / α_l）+ 轨迹嵌入 | Self-Attention / Map CA / MLP 的权重、MapConditionEncoder（仍被 `gate = 0` 挡住） |
| 第 3 步 | Self-Attention、**Map Cross-Attention**、MLP、时间步嵌入、起终点编码、**MapConditionEncoder** | —— |

原因：输出头零初始化让第 1 步只有头部有梯度；AdaLN-Zero 的 gate 在第 1 步拿到梯度后变成非零，
第 3 步才让被门控的子层权重开始训练。安全分支由 α_l = 0.1 直接门控，因此第 2 步就有梯度。
这是既有零初始化策略的必然结果，不是接线错误；冻结编码器在任何一步都没有梯度。

## 5. 数据说明（重要）

### 5.1 真实数据集 geopath_v1（推荐路径）

真实数据在 `data/{trajectories,distance_fields,maps}`（PathSafe-Ellipse V3，只读，已 gitignore）。
按《数据集制作方案》制作：

```powershell
# 1) 盘点 + 固定规格筛选（0.25 m/cell、256x256、64 m；48 m/80 m 全部排除）
& $py scripts\audit_real_dataset.py

# 2) 划分(map_group 80/10/10, seed 0) + GT 转换 + B-spline/碰撞审计
& $py scripts\prepare_real_dataset.py --config configs/c64_longpath.yaml --pilot \
        --limit-train 1000 --limit-val 200
#    全量：去掉 --pilot

# 3) 安全条件缓存（冻结 PathSafe-EllipseNet，指纹含样本 ID 摘要）
& $py scripts\build_real_safety_cache.py --split train --config configs/c64_longpath.yaml
& $py scripts\build_real_safety_cache.py --split val   --config configs/c64_longpath.yaml

# 4) Pilot 验收 A–D（16 项检查）
& $py scripts\validate_real_dataset.py --config configs/c64_longpath.yaml

# 5) 训练（真实数据，C=64）
& $py scripts\train_dual_branch.py --config configs/c64_longpath.yaml --dataset real \
        --real-split train --val-split val --steps 900 --epochs 4 --batch-size 4 \
        --save outputs/checkpoints/pilot_c64.pt
```

实测（Pilot）：16,756 条合格轨迹 / 5,702 张 0.25 m 地图 / 30 组；C=64 保留率 91.3%（C=32 为 77.3%）；
train 915 + val 181 的安全缓存搜索成功率 100%、走廊闭环 86–90%；验收 16/16 通过。

**训练现状**：Pilot 900 步（4 epoch，CPU 34 min）损失明确下降（total −52%，安全分支 −19%，走廊 −27%），
验证集安全分支 −23%；但模型**远未收敛**（推理碰撞率 9–38%，长度可达 GT 3 倍），
不构成任何性能结论。全量数据制作与正式训练待确认。详见 `docs/dataset_v1_report.md` §7。

### 5.2 合成数据集（无真实数据时的回退）



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
2. 冻结 PathSafe 在合成地图上椭圆偏薄（`b` 中位数 0.13 m）→ 走廊闭环率 ≈50 %（上游自身 54 %）；未闭环样本自动退化为"仅地图边界约束"：CPCG 的解析代价恒为 0（不做无意义移动），末端 QP 仍会给出结论，但**不会被标记为有走廊安全证书**。
3. 长路径（>60 m）C=32 拟合误差 0.62 m，建议切 `configs/c64_longpath.yaml`（待确认）。
4. 训练用走廊是全路径 scope，推理用碰撞区间局部 scope（理由见 `docs/design_review.md` §3.1）。
5. CPU only。

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/reference_code_audit.md` | 三个仓库的结构、真实接口、张量形状、依赖、复用决策、缺失模块取证 |
| `docs/module_provenance.md` | 逐文件迁移表（仓库 / 分支 / commit / 原路径 / 新路径 / 适配 / 是否改算法） |
| `docs/design_review.md` | 文档内部不一致、文档与代码不一致、需你确认的取舍、与"明确禁止"的逐条对照 |
| `docs/acceptance_report.md` | 实际测试与实测数字（pytest 结果、逐阶段耗时、拟合误差、走廊统计、A/B/C/D 配对实验） |
| `docs/cpcg_report.md` | 本轮 CPCG + 末端 QP 的实施与验收报告（解析梯度校验、QP 诊断、8 步 / 20 场景配对实验） |
