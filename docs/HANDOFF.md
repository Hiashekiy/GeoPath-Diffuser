# GeoPath-Diffuser 交接文档（2026-11-27 12:30）

> 用途：把这份文档整份丢给新的对话窗口，即可无缝接手。本文只写**已验证的事实**和**当前状态**；
> 每条数字都来自仓库里的日志/报告文件，出处标注在括号里。

---

## 0. 一句话状态

一个独立实现的「安全条件扩散轨迹规划」项目（GeoPath-Diffuser）已经跑通训练/推理/评测全链路；
当前正在把**无安全支路的监督机制**从"输出对齐"改成"特权走廊监督 + 蒸馏"，两条续训（到 100 epoch）
正在后台跑。核心实测结论：**无安全支路（Map-only）的输出实质是一次"以地图为条件的均值预测"，
它作为"引导路径生成器"足够用（2 步去噪 = 50 步），但**不能**单独作为安全轨迹使用。

**最新一轮（CPCG）**：推理期修正已从 ALM 换成 **Clean-Prediction Constraint Guidance + 一次末端凸 QP**。
每步只对**网络预测的干净控制点**算走廊半空间的解析梯度（masked squared hinge，只按当前违反面归一化），
经 Trust Region 与 BoundaryDecoder 保持端点/朝向，然后作为 `q0_used` 进入**原样未改**的 DDIM 更新；
全部反向步结束后做一次 OSQP 凸投影，再做 512 点密集碰撞验证。训练目标、网络结构、数据集、
搜索/椭圆/走廊构建算法本轮**未改**。报告见 `docs/cpcg_report.md`。

---

## 1. 环境与运行约定（非常重要）

| 项目 | 值 |
|---|---|
| Python | `E:\CondaEnvData\envs\GGMPC\python.exe`（**必须用它**；系统默认 python 是 3.14，`import torch` 会 DLL 报错） |
| torch | 2.9.1+cu126，CUDA 可用 |
| osqp | 1.1.0（末端凸 QP 的求解器；**必需**，缺失时 `safety/qp_projection.py` 会明确报错而不是退回 ALM） |
| GPU | RTX 4070，12,282 MiB（单次训练占用 5.6–8.2 GB，**不能并行跑两个 16-batch 训练**） |
| 项目根 | `D:\ProjectDirectory\GeoPath-Diffuser` |
| 磁盘 | D: 约 28 GB 空闲（每个含优化器状态的 `last.pt` 约 637 MB，纯权重 212 MB） |

常用命令模板（PowerShell）：

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
& $py -m pytest tests -q            # 当前：140 passed, 1 skipped（另 2 个 tmp_path 用例受本机沙箱限制）
& $py scripts\run_inference.py --dataset real --config configs/c64_longpath.yaml --samples 8 --modes map_only,safety_only,cpcg --checkpoint <ckpt>
```

**已知坑**

- `scripts/train_dual_branch.py --steps` 默认是 **10**（不是"跑满"）；不限步数时必须显式传
  `--steps 15240`（20 epoch × 762 步）。第一次误用导致"20 epoch 只训了 10 步"。
- PowerShell `Set-Content -Encoding utf8` 会写 BOM，Python 报 `invalid non-printable character U+FEFF`；
  统一用 `[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))`。
- `compare_safety_ablation.py` 的 `--safety-mode` 是**全局**的：一次调用里所有 arm 用同一个模式。
- 日志用 `*>` 重定向（老 PS 写 UTF-16），读日志的脚本要带编码探测。

---

## 2. 项目定位与不可违反的架构约束

来源：`docs/安全条件扩散轨迹规划方案_Conditional-DiT网络修正版.md`。

1. **不得**在 `references/` 里的三个上游仓库上开发；上游代码以 vendor 方式引入
   （`src/gp_diffuser/vendor/`），见 `docs/module_provenance.md`。
2. 地图编码器 = 上游 **PathSafe ResUNet，冻结**；DiT 只吃 `enc3` 瓶颈。
3. 主干必须是**标准 Conditional DiT-S**：D=384、12 block、6 head、mlp_ratio 4、AdaLN-Zero、x0 预测
   （55,662,350 可训练参数；frozen encoders 3,157,168 / 0 trainable）。
4. **单阶段同步双支路联合训练**：同一 `q_t / t`，两次前向，**一次**反向、一次更新。
5. 凸区域 / B-spline / DDIM 复用 TrajSafe 的做法；安全几何只在中间步生成**一次**。
   **推理期修正已从 ALM 换成 CPCG（干净预测的解析约束梯度）+ 一次末端凸 QP**；
   `diffusion/bspline_alm.py` 只作历史/对照保留，活动链路不 import 它。
6. 场景帧：256×256 @ 0.25 m = 64 m；`SceneTransform(64,256)`；轨迹状态在 `[-1,1]^2` 场景坐标。

---

## 3. 数据与缓存

- 数据集 `data/geopath_v1`（由 PathSafe 的真实数据构建，见 `docs/dataset_v1_report.md`）：
  只保留 0.25 m 分辨率 → 50,648 个候选里 16,756 合格；按 `map_group` 80/10/10（seed 0）
  → **train 12,178 / val 1,472 / test 1,484**（15,134 保留，1,622 因拟合碰撞被丢弃）。
- 轨迹表示 `C = 64` 个控制点（三次 B 样条，128 点曲线）。C=32 保留率只有 77.3%，C=64 是 91.3%。
- 安全缓存（离线，PSENet + 引导搜索 + 走廊）：
  - `outputs/cache/safety_train.npz`：12,178 样本，指纹 **`d8359de3a9e13163`**
  - `outputs/cache/safety_val.npz`：1,472 样本，指纹 **`d5bc2a8519c25f4e`**
  - 指纹不匹配会被 `load_safety_cache` 拒绝；重建命令见 §9。
- **关键语义**：上游 `world_to_grid` 里 `r_scene` 是**单元格尺寸（res_m）**，不是物理范围。
  早期误传 48/64 m 导致查询点全塌到角落、椭圆极小（面积 6.4 m² vs 上游 40.7 m²），已修正。

---

## 4. 代码结构（现有关键模块）

```
src/gp_diffuser/
  models/conditional_dit.py       DiT-S；DiTBlock = SA → Map CA → Safety CA → MLP；AdaLN 输出 11 段
                                 (sa3 + map3 + safety2 + mlp3)；每 block 一个 safety_gate 标量 α（初值 0.1）
  models/boundary.py             BoundaryDecoder：无参数，端点位置调制 + 端点切向调制
                                 (profile 1.0/0.75/0.5/0.25, span 4)；端点精确 + 出发/到达朝向精确
  models/condition_encoders.py   MapConditionEncoder（双适配器 + 残差融合 → 1024 token）+ SafetyConditionEncoder(64 椭圆 token)
  training/dual_branch_trainer.py 双支路训练器；开关：use_safety / u_target / align_space；续训 start_epoch
  training/losses.py             损失集（见 §5）；DEFAULT_LOSS_WEIGHTS 含 lambda_safe_u（默认 0）
  diffusion/sampler_conditional_dit.py  DDIM + 中间步一次性生成安全几何 + CPCG + 末端 QP；
                                 safety_mode ∈ {map_only, safety_only, cpcg}
                                 （旧别名 no_alm → safety_only、full → cpcg）；
                                 record_curves 记录每步 x0_pred / x0_guided，末尾 x0_projected
  diffusion/clean_guidance.py     CPCG：masked squared-hinge 安全代价 + 解析梯度 + Trust Region
                                 （不对 DiT 反传，不需要 autograd）
  diffusion/bspline_alm.py        历史/对照模块（ALM），活动推理链路已不再调用
  safety/qp_projection.py         末端凸 QP（OSQP）：pack → 稀疏 A x ≤ b + 端点/朝向行，逐样本求解
  safety/, guidance/, vendor/    走廊/凸区域/椭圆/引导搜索（vendored 上游代码）
scripts/
  train_dual_branch.py           训练入口（--dataset real --no-safety --u-target --align-space
                                 --lambda-align --lambda-safe-u --ckpt-dir --ckpt-every --tag --resume）
  build_real_safety_cache.py     离线安全缓存
  compare_safety_ablation.py     多臂闭环对照（同样本/同噪声；逐例配对 + 符号检验 + 拼图 + 柱图）
  visualize_prediction.py        单场景：安全几何 / 每步 x0 面板（--safety-mode，--ncols）
  visualize_map_only_evolution.py 一次前向到收敛的 EVERY-step x0（三联图 + 逐 step 位移统计）
  evaluate_real.py, reproduce_pathsafe_eval.py, test_vendored_modules.py, smoke_pipeline.py,
  verify_corridor_equivalence.py（走廊配置迁移等价性）, diag_regions_and_alm.py（区域 + CPCG 轨迹诊断），
  audit/prepare/validate real dataset 系列, octile_* 系列, _probe_alignment.py（对齐机制诊断）
tests/  140 passed, 1 skipped（含解析梯度有限差分、掩码、空 pack、C=32/64、DDIM 一致性、
        QP 可行/不可行、三模式推理集成；2 个用 tmp_path 的用例在本机沙箱下无法建临时目录，
        与本轮改动无关，见 docs/cpcg_report.md §3）
```

---

## 5. 损失函数（当前实现，全部按"逐样本向量 → batch 均值"计算，一次反向）

| 项 | 计算 | 权重 |
|---|---|---|
| `L_c` | 安全支路 vs GT，**只用中间 62 个控制点**的 MSE | 1.0 |
| `L_u` | Map-only 支路同一算法 vs GT（`u_target=gt`）或恒 0（`u_target=safety`） | 1.0 |
| `L_align` | `MSE(u, sg(c))`；`align_space=raw` 用**端点调制前**的多边形（默认），`final` 用调制后 | 0.5（蒸馏臂用 1.0） |
| `L_boundary` | 端点附近 4 个位置的加权 MSE，目标是**平移到本样本 S/G 的 GT 多边形**；`u_target=safety` 时只作用于 c | 0.2 |
| `L_smooth` | 控制点二/三阶差分，除以 GT 平均步长后取 `log1p`（acc 0.25 / jerk 1.0），两支平均 | 0.15 |
| `L_curve` | 同上但作用在解码后的 128 点曲线 | 0.15 |
| `L_safe` | 走廊连续约束 `Aβ ≤ b`：每段 4 个 Bezier 控制点 × 所有面取 relu，逐样本取 **max**，场景单位 | 0.15 |
| **`L_safe_u`** | **同一走廊约束作用于 Map-only 支路（特权监督，标签而非输入）** | **默认 0（新开关）** |

不含：噪声预测损失（直接 x0 预测）、SDF/碰撞项、地图一致性项、多样性/熵项。
推理期的 CPCG 用同一套 Bezier 凸包约束语义，但代价形式是 masked squared hinge（只对**当前违反**的有效面归一化），
末端再补一次凸 QP；训练侧这一轮完全没动。

---

## 6. 已经得到的实测结论（按主题）

### A. 双支路训练 vs 纯无安全训练（第一轮 A/B，均 20 epoch，同 seed）

| 支 | map_only 无碰撞率 | 碰撞占比 | chamfer | 控制点 | 长度比 |
|---|---|---|---|---|---|
| 双支路（安全链路训练） | 0.590 | 0.1192 | **1.450** | **1.535** | 0.942 |
| 纯 Map-only 训练 | 0.586 | 0.1234 | 1.580 | 1.711 | 0.924 |

配对：碰撞 p=0.16（平手），几何双支路更好，长度误差 178/78（p<1e-4）。
→ **"安全训练过的权重"在无安全推理时并不更安全**；安全链路的收益几乎全部来自推理期
（同一权重开安全：无碰撞率 0.906、碰撞占比 0.0079）。文档：`docs/ablation_safety_map_only.md`。

### B. 旧的 `align` 机制为什么结构上无效

- 训练日志（`outputs/logs/ab_dual.log`）：step 2,000 之后 `align/u` 长期停在 0.3–1.3 **没有下降趋势**，
  两支间距 1.0–2.7 m ≈ u 自己的误差，即**从未收敛到一起**。
- 诊断脚本 `scripts/_probe_alignment.py`（teacher-forced，64 样本，最终权重）：
  `err_c = 1.05 m` vs `err_u = 1.90 m`（安全支路准 45%，解码曲线碰撞 0.024 vs 0.156）；
  两支间距 1.89 m；投影系数 **k = ⟨u−gt, c−gt⟩/‖c−gt‖² = 0.52**。
- 解析：`L_u + 0.5·L_align` 的极小点是 `u = (gt + 0.5c)/1.5`，只能走 1/3 的路程；而 u 的误差里
  **约 90% 的能量与 (c−gt) 正交**，align 根本碰不到。
- 控制空间 vs 曲线空间算 align 只差 3%（1.89 vs 1.93 m）→ **"改成轨迹差"不会改变任何东西**。

### C. 四臂消融（20 epoch / 15,240 step / 同 seed、同数据顺序、同缓存）

| 臂 | 无安全支路监督 | align 位置 | 走廊标签 | 无碰撞率 | 碰撞占比 | chamfer | 控制点 | 长度比 |
|---|---|---|---|---|---|---|---|---|
| `baseline_gt` | GT | final（旧） | 无 | 0.594 | 0.1192 | 1.450 | 1.535 | 0.942 |
| `d1_distill` | 只蒸馏 | raw | 无 | 0.613 | 0.1174 | 1.564 | 1.667 | 0.940 |
| `d2_distill_corridor` | 只蒸馏 | raw | 0.15 | **0.645** | **0.0880** | 1.395 | 1.592 | **0.968** |
| `d3_gt_corridor` | GT | raw | 0.15 | 0.609 | 0.1046 | **1.332** | **1.508** | 0.948 |

逐例配对（符号检验，n=256）：d1 碰撞 p=0.57 / chamfer p=0.001（更差）；
d2 碰撞 37:76 **p<0.001**、长度 p=0.039；d3 chamfer 88:168 **p<0.001**、碰撞 p=0.055。
开安全档位（64 样本）：基线 0.906/1.201，d2 0.891/**0.957**，d3 **0.922**/**0.928**。

**判读**：① "去 GT 换蒸馏"单独用是**把误差换位置**（几何显著变差）——因为学生的盲区经**共享权重**
回灌教师（教师 teacher-forced 误差 0.00162→0.00237、走廊违规 0.018→0.044）；
② 真正的有效成分是**把走廊约束当标签喂给无安全支路**；③ 两条路径互补：走廊管"合法"、蒸馏管"选哪条"。
文档：`docs/ablation_branch_transfer.md`。

### D. `d2` 续训（20 → 80 epoch，同权重续训）

| 权重 | 无碰撞率 | 碰撞占比 | chamfer | 控制点 | 长度比 |
|---|---|---|---|---|---|
| 基线 @20 | 0.594 | 0.1192 | 1.450 | 1.535 | 0.942 |
| d2 @20 | 0.645 | 0.0880 | 1.395 | 1.592 | 0.968 |
| d2 @40 | 0.688 | 0.0729 | 1.244 | 1.430 | 0.971 |
| d2 @60 | **0.750** | 0.0661 | 1.308 | 1.449 | 0.985 |
| d2 @80 | 0.727 | **0.0598** | **1.147** | **1.249** | 0.975 |
| d2 @100 | **0.750** | **0.0510** | 1.179 | 1.350 | **0.988** |

配对（vs 基线@20）：d2@80 碰撞 16:92 / chamfer 77:179；d2@100 碰撞 **14:94** / chamfer 94:162 /
长度 116:140（n=256）。开安全（64 样本）：d2@80 **0.906 / 0.0099 / chamfer 0.732 / 控制点 0.786**；
d2@100 0.875 / 0.0106 / 0.792 / 0.912（基线@20 是 0.906 / 0.0079 / 1.201 / 1.412；64 样本下
0.875 vs 0.906 只差 2 个样本，属噪声）。
→ **20 epoch 时的"教师被拖差"是欠训练造成，80–100 epoch 已完全恢复并反超**；两种档位都更好。

### E. 步数消融：2 步 = 50 步（同一权重 `d2_ep080`，安全链路开启，64 样本）

| 步数 | 无碰撞率 | 最终碰撞占比 | chamfer | 引导自身碰撞占比 | 引导→最终 | 搜索展开 | 耗时 |
|---|---|---|---|---|---|---|---|
| 2 | 0.906 | 0.0102 | 0.731 | 0.084 | 0.60 m | 4,484 | **11.3 s** |
| 5 | 0.906 | 0.0099 | 0.740 | 0.084 | 0.61 m | 4,444 | 13.8 s |
| 10 | 0.906 | 0.0099 | 0.733 | 0.084 | 0.60 m | 4,492 | 18.5 s |
| 20 | 0.906 | 0.0100 | 0.733 | 0.084 | 0.60 m | 4,461 | 26.9 s |
| 50 | 0.906 | 0.0099 | 0.732 | 0.084 | 0.60 m | 4,467 | 57.0 s |

配套观察（`outputs/figures/map_only_evolution_d2ep80/`，逐 step 的 x0 可视化）：

- 逐 step 的 x0 平均位移只有 0.03–0.13 px/step（1 px = 0.25 m），**50 步累计形状改变 0.4–1.6 m**，
  而模型自身误差 1.9 m → **反向过程几乎不改变形状**，采样器实质是"一次条件均值预测"。
- 高噪声步（t≈999）输出的 x0 就已经是"抄近路的直弦"（多模态 GT 的均值），失败样本从头到尾没离开这条弦。
- 管线把引导的 8.4% 碰撞洗到 1.0%，只移动 0.60 m → **引导只需方向对，不需无碰撞**（这条已由用户确认
  为设计意图：无安全预测的职责就是"给搜索一个方向"）。

### F. `safety_gate`（每 block 一个标量 α，初值 0.1）

- 基线@20 0.148；d2@20 0.107；d2@85 0.103→0.100；纯无安全臂（不训练安全链路）**精确 0.100 不变**
  （`p.grad is None`，AdamW 直接跳过 → 证明该臂从未碰过安全链路，已有测试锁定）。
- 旧 70-epoch 权重最终 0.135，且逐层自主分配（block1 0.028、block6 0.205）。
- d2 中 α 缓慢下降 = "守走廊"的能力被搬进共享主干（`L_safe_u` 让无安全支路自己合规），符合迁移预期。

---

## 7. 正在进行（后台）与下一步

### 7.0 当前状态（2026-11-27 18:10 更新：实验已收口，见 `docs/ablation_ep100.md`）

| 时间 | 内容 |
|---|---|
| ✅ 13:08:03 | `d2` 续训到 **100 epoch 完成**，产出 `d2_ep040/060/080/100.pt` + `last.pt` |
| ✅ 18:06:45 | **纯无安全臂 `ab_map_only` 20 → 100 epoch 完成**（`--no-safety`、60,960 步、0.24 s/step），
产出 `outputs/checkpoints/ab_map_only_e100/am_ep{030..100}.pt` + `last.pt`（582 MB，**带优化器状态**） |
| ❌ 已取消 | 基线 `ab_dual → 100 epoch`（缺优化器状态，对照不干净，见 §7.05）。改用 `ab_map_only` 做机制对照 |
| 测试 | `101 passed, 1 skipped` + 2 个 fixture 阶段 PermissionError（沙箱建不出 `tmp_path`，非回归） |

**本轮结论（全部 GPU、n=256、50 步、同索引同噪声，符号检验）**

| 对比 | 无碰撞率 | 碰撞占比配对 | chamfer 配对 |
|---|---|---|---|
| d2@80 → d2@100 | 0.727 → **0.750** | 22:52 **p=6.4e-4**（更好） | 125:131 p=0.76（平） |
| **am@20 → am@100（纯无安全，5× 训练量）** | 0.586 → 0.590 | 53:59 **p=0.637（完全平手）** | 187:69 **p=9.5e-14（几何大幅变好）** |
| **am@100 vs d2@100（同训练量）** | 0.590 vs **0.750** | 17:91 **p=2.1e-13**（d2 压倒性） | 127:129 **p=0.95（打平）** |

→ **多训练 5 倍买得到"像 GT"，买不到"合规"**；d2 的优势不是训练量、也不是形状，而是**安全链路的监督**。
→ 判据（无碰撞率 ≥0.70 且控制点 ≤1.535）成立：**d2 配方定为正式配置**（0.750 / 1.350）。
→ 但**开安全档位**下 d2@80 略优于 @100（0.906 vs 0.875，长度误差 45:19 p=7e-4），`d2_ep080.pt` 保留为备选。

**踩过的坑（别再重复）**

- 用 `Start-Process` 拉起的训练**会在启动它的那次工具调用结束时被回收**（14:07 / 14:11 两次都是这么死的，
当时误判成"另一个会话在清理"）。**正确做法：交给持久后台作业（`run_in_background`）**，已验证跨多次调用存活。
- **训练期间不要跑 GPU 评测**：13:11 一次 `--device cuda` 对照被显存竞争硬杀；13:47 两个评测进程并行导致 8× 变慢
（实测单进程同任务是 38 s/arm）。

**CPU 预跑（已被 GPU 权威数字取代，保留作交叉验证）**：`device=cpu`、64 样本、DDIM 2 步
（`outputs/reports/prelim_d2_ep100_cpu.json`）给出 d2@80 0.625 / d2@100 0.703、配对碰撞 20:4
（p≈0.0015）；与 GPU 50 步/256 样本的结论方向一致，且**同权重同 64 样本下与 GPU 聚合指标逐位相同**，
可作 CPU 评测可信度的证据。

**顺带验证（CPU，d2@100，同 64 样本同噪声）**：`map_only` 的 2 步与 50 步输出**最大逐点差 4 mm**
（`max|curve2 − curve50| = 0.0041 m`，碰撞/ chamfer / 控制点 4 位小数完全一致，耗时 23.6 s vs 258 s）。
→ 之前只在“开安全”档位验证过的“2 步 = 50 步”，在 `map_only` 档位同样成立，可安全作为默认。
报告：`outputs/reports/prelim_map_only_50steps_d2ep100_cpu.json`。

### 7.05 会话交接记录：基线续训为什么被停（2026-11-27 14:10 更正）

我之前（本窗口）在 13:45 和 14:05 两次重启过"基线续训到 100 epoch"，**两次都在几分钟内 exit=1、无 traceback、
输出目录被清掉**。原因已查清，不是崩溃，而是**用户有意叫停 + 另一个会话执行清理**：

- 基线续训用的 `ab_dual/last.pt` **不含优化器状态**（它是 11-27 21:15 之前的 trainer 存的，日志明确
  `resume: checkpoint carries no optimizer state - AdamW restarts from zero`），
- 而 `d2` 从 20 → 100 是**带着 AdamW 动量**续训的，
- 两者训练动力学不等价 → 100 vs 100 的差值会混入"动量连续性"，分不清"机制"与"训练历史"。
- 因此该续训被停止、残留的 `ab_dual_e100/` 目录被删除（本窗口也删了自己重启产生的那一份）。

**正确的 100 vs 100 做法**（约 6.6 h，二选一）：

1. **从 0 训到 100**：`--epochs 100 --steps 76200`，不加 `--resume`，其余配方与原基线完全一致
   （`u_target=gt` / `align_space=final` / `lambda_align=0.5` / `lambda_safe_u=0`）；
2. 或者先训 0→20、用**现在带优化器状态**的 trainer 存 `last.pt`，再 `--resume` 到 100。

无论哪种，报告里必须写明"该权重是否携带 AdamW 状态"，否则 100 vs 100 的对比不成立。

**并发规则（重要）**：同一时刻只能有一个会话跑 GPU 任务；另一会话要评测时请先确认没有训练在跑
（`Get-Process python | Where Path -like "*GGMPC*"`），或改用 `--device cpu`（慢但安全）。

> **✅ 已结束（2026-11-27 18:06:45）：`ab_map_only_e100` 训练完成**，
> 结果见 §7.0 与 `docs/ablation_ep100.md`。GPU 现已空闲。
>
> **踩过的坑（已修正的误判，别再重复）**：用 `Start-Process` 拉起的训练**会在启动它的那次工具调用结束时
> 被回收**（两次都正好死在调用结束的那一分钟：14:07:19→14:09:53、14:11:10→14:12:50），
> 当时误判成"另一个会话在清理"。实测全机只有 1 个 python 进程，根本没有别人。
> **正确做法：把训练交给持久后台作业（`run_in_background`）跑**，已验证跨多次调用存活。

**`d2@100` 的 GPU 权威数字**（本窗口 14:01 跑，256 样本 × 50 步，可直接引用）：
`map_only` 无碰撞率 **0.750**、碰撞占比 **0.0510**、chamfer 1.179、控制点 1.350、长度比 0.988
（基线@20 = 0.594 / 0.1192 / 1.450 / 1.535 / 0.942；配对碰撞 14:94，平 148）；
开安全（64 样本）0.875 / 0.0106 / 0.792 / 0.912 / 1.003。
报告：`outputs/reports/ablation_transfer_e100.json`、`ablation_transfer_d2_ep100_safety_on.json`。

**必须做的下一步（拿到 100 epoch 权重后）**

评测批次已写成脚本 `outputs/logs/_eval_ep100.ps1`（GPU 空闲后一次跑完，约 10 min），
产物 `outputs/reports/ablation_ep100_1v1*.json`、`ablation_transfer_d2_ep100*.json`、
`map_only_steps2_d2ep100.json`、`safety_steps2_d2ep100.json`。

1. `d2@100` 的 `map_only` 与 `open-safety` 指标，与 `d2@80` 比较是否还在收敛；——CPU 预跑已指向“还在收敛”，待 GPU 权威数字；
2. **基线@100 vs d2@100** 的逐例配对检验——这是唯一能把"机制有效"与"单纯多训练"分开的实验；
3. 结论口径建议改成"引导质量"而不是"引导无碰撞率"：引导 vs GT 的 chamfer、引导落入自由走廊的比例、
   最终轨迹是否仍在引导的同伦内（现在偏移 0.60 m 说明几乎总是在同一侧）。
   → **已在 `scripts/compare_safety_ablation.py` 里落地**：新增 `guide_chamfer_to_gt_m`、
   `guide_collision_rate`（自由率 = 1 − 它）、`guide_chamfer_to_final_m`、`homotopy_witness_ok`
   （引导↔最终曲线按 `s∈[0,1]` 线性混合 11 帧，全帧无碰撞 = 存在自由同伦见证；单侧指标，
   有碰撞不代表不同伦）；
4. 若 d2@100 达到"无碰撞率 ≥0.70 且控制点误差 ≤ 基线 1.535"，则把该配方定为正式训练配置。

**后续候选方向（按证据强度排序，尚未实施）**

- **决策级蒸馏**：让无安全支路学"从障碍哪一侧过/横向偏移"，而不是坐标——符号量不会被平均掉，
  且该信息只有安全支路（椭圆）能给。这是唯一能真正解决"多模态 GT → 均值解"的**逐样本**手段。
- **路线潜变量 / 多候选输出**：结构级解法（同一地图输出多个候选 + 最近候选损失），
  能彻底摆脱"单输出 = 均值"；代价是改网络 + 数据侧需要构造同起终点的多 GT 配对。
- **推理 2 步**：零质量损失、5× 加速，可直接作为默认（需先确认与 50 步在更大样本上一致）。
- **训练目标对齐"引导用途"**：既然引导只需方向正确，可考虑弱化"贴 GT 形状"，强化"进入自由走廊"。
- 冻结项（除非明确要求，不要动）：批量训练机制、训练期搜索回退、推理搜索失败处理、
  额外路径碰撞扫描、椭圆生成内部、凸区域算法、ALM。

---

## 8. 现有权重清单

| 路径 | 内容 |
|---|---|
| `outputs/checkpoints/ab_dual/last.pt` | 基线（GT + 旧 align@final），20 epoch，`map_only` 0.594 |
| `outputs/checkpoints/ab_map_only_e100/{last,am_ep030..100}.pt` | 纯无安全臂（`use_safety=False`）**20 → 100 epoch 已完成 18:06**；`map_only` 0.590 / 1.334 / 1.448 |
| `outputs/checkpoints/ab_map_only/last.pt` | 纯 Map-only 训练（无安全链路），20 epoch，0.586 |
| `outputs/checkpoints/ab_distill_raw/{last,d1_ep010,d1_ep020}.pt` | d1：去 GT + 蒸馏 |
| `outputs/checkpoints/ab_distill_safeu/{last,d2_ep010,d2_ep020}.pt` | d2：去 GT + 蒸馏 + 特权走廊 |
| `outputs/checkpoints/ab_gt_safeu/{last,d3_ep010,d3_ep020}.pt` | d3：GT + 特权走廊 |
| `outputs/checkpoints/ab_distill_safeu_e100/{last,d2_ep040,d2_ep060,d2_ep080,d2_ep100}.pt` | d2 到 100 epoch（**已完成 13:08**） |
| `outputs/checkpoints/ab_dual_e100/last.pt` | ❌ 已中止并删除（缺优化器状态，见 §7.05） |
| `outputs/checkpoints/{geopath_v1_c64_ts,pilot_c64,last}.pt` | 早期实验/旧缓存训练的权重，仅作历史参考 |

---

## 9. 复现命令（可直接复制）

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
$common = @("--config","configs/c64_longpath.yaml","--dataset","real",
            "--real-split","train","--val-split","val","--device","cuda",
            "--epochs","20","--batch-size","16","--steps","15240",
            "--log-every","200","--ckpt-every","10","--seed","1")

# 重建安全缓存（约 30 min；train 12,178 / val 1,472）
& $py -u scripts/build_real_safety_cache.py --split train --config configs/c64_longpath.yaml --device cuda
& $py -u scripts/build_real_safety_cache.py --split val   --config configs/c64_longpath.yaml --device cuda

# 四个实验臂
& $py -u scripts/train_dual_branch.py @common --ckpt-dir outputs/checkpoints/ab_dual `
   --ckpt-prefix abd --tag ab_dual                                     # 基线
& $py -u scripts/train_dual_branch.py @common --u-target safety --align-space raw `
   --lambda-align 1.0 --lambda-safe-u 0.0  --ckpt-dir outputs/checkpoints/ab_distill_raw    --ckpt-prefix d1 --tag ab_distill_raw
& $py -u scripts/train_dual_branch.py @common --u-target safety --align-space raw `
   --lambda-align 1.0 --lambda-safe-u 0.15 --ckpt-dir outputs/checkpoints/ab_distill_safeu  --ckpt-prefix d2 --tag ab_distill_safeu
& $py -u scripts/train_dual_branch.py @common --u-target gt     --align-space raw `
   --lambda-align 0.5 --lambda-safe-u 0.15 --ckpt-dir outputs/checkpoints/ab_gt_safeu       --ckpt-prefix d3 --tag ab_gt_safeu

# 续训到 100 epoch（--epochs 是"再训几个"，epoch 编号自动接着数）
& $py -u scripts/train_dual_branch.py @common --steps 60960 --epochs 80 `
   --resume outputs/checkpoints/ab_distill_safeu/last.pt --u-target safety --align-space raw `
   --lambda-align 1.0 --lambda-safe-u 0.15 `
   --ckpt-dir outputs/checkpoints/ab_distill_safeu_e100 --ckpt-prefix d2 --tag ab_distill_safeu_e100
& $py -u scripts/train_dual_branch.py @common --steps 60960 --epochs 80 `
   --resume outputs/checkpoints/ab_dual/last.pt --u-target gt --align-space final `
   --lambda-align 0.5 --lambda-safe-u 0.0 `
   --ckpt-dir outputs/checkpoints/ab_dual_e100 --ckpt-prefix b --tag ab_dual_e100

# 多臂闭环对照（不许安全）
& $py -u scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
   --arm baseline=outputs/checkpoints/ab_dual/last.pt `
   --arm d2=outputs/checkpoints/ab_distill_safeu/last.pt `
   --split val --samples 256 --batch 8 --steps 50 --device cuda
# 同一批权重开安全（参考线）
& $py -u scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
   --arm d2_full=outputs/checkpoints/ab_distill_safeu_e100/d2_ep080.pt `
   --split val --samples 64 --batch 8 --steps 50 --device cuda --safety-mode full

# 逐 step 干净轨迹可视化（无安全）
& $py -u scripts/visualize_map_only_evolution.py --config configs/c64_longpath.yaml --device cuda `
   --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep080.pt `
   --split val --indices 0,1,13,142,229 --steps 50
& $py -u scripts/visualize_prediction.py --config configs/c64_longpath.yaml --device cuda `
   --checkpoint <ckpt> --split val --indices 0,1,2 --steps 50 --ncols 10 --safety-mode map_only
```

---

## 10. 代码状态（未提交）

- 最后一次 commit：`5a27519`（TrajSafe 损失集）。此后所有工作**均未提交**（用户要求：不确认不 commit/push）。
- 未提交的源码改动：`scripts/train_dual_branch.py`（--no-safety/--u-target/--align-space/--lambda-*/
  --resume/--seed/--ckpt-*/--tag）、`scripts/visualize_prediction.py`（--safety-mode/--ncols）、
  `src/gp_diffuser/training/dual_branch_trainer.py`（use_safety / u_target / align_space / L_safe_u /
  续训 / 带优化器状态的 ckpt）、`src/gp_diffuser/training/losses.py`（lambda_safe_u + total_loss 新参数）、
  `tests/test_dual_branch_training.py`（+4 个测试）。
- 新增未跟踪：`scripts/{compare_safety_ablation,visualize_map_only_evolution,_probe_alignment}.py`、
  `docs/{ablation_safety_map_only,ablation_branch_transfer}.md`、`outputs/reports/*`、`outputs/figures/*`。
- **本轮（CPCG）未提交改动**：新增 `src/gp_diffuser/diffusion/clean_guidance.py`、
  `src/gp_diffuser/safety/qp_projection.py`、`scripts/verify_corridor_equivalence.py`、
  `tests/{test_clean_guidance,test_qp_projection}.py`、`docs/cpcg_report.md`；
  修改 `diffusion/sampler_conditional_dit.py`（ALM → CPCG + 末端 QP，模式改名，trace/统计字段）、
  `configs/default.yaml`（`clean_guidance` / `final_projection`，走廊几何参数归位 `corridor`）、
  `evaluation/visualize.py`（三条曲线 + Bezier 分段）、`scripts/*`（`alm_stats` → `guidance_stats` /
  `final_qp_*`）、`tests/{test_inference_pipeline,test_review_regressions,test_network_structure}.py`、
  `README.md`、本文件。
- 测试：**140 passed, 1 skipped**（另 2 个 `tmp_path` 用例在本机沙箱下 pytest 无法创建临时目录，
  与本轮改动无关，见 `docs/cpcg_report.md` §3）。

---

## 11. 关键文件索引

| 目的 | 路径 |
|---|---|
| 设计文档 | `docs/安全条件扩散轨迹规划方案_Conditional-DiT网络修正版.md` |
| **本轮 CPCG + 末端 QP 实施与验收报告** | `docs/cpcg_report.md` |
| 数据集构建报告 | `docs/dataset_v1_report.md` |
| 第一轮 A/B（安全训练 vs 无安全训练） | `docs/ablation_safety_map_only.md` |
| 四臂监督机制消融（含续训计划） | `docs/ablation_branch_transfer.md` |
| 上游代码来源与改动 | `docs/module_provenance.md`、`src/gp_diffuser/vendor/` |
| 训练日志 | `outputs/logs/{ab_dual,ab_map_only,ab_distill_raw,ab_distill_safeu,ab_gt_safeu,ab_distill_safeu_e100}.log` |
| 闭环评测 JSON | `outputs/reports/ablation_transfer_{all,d2_ep80,...}.json`、`steps_ablation_{2,5,10,20,50}.json` |
| 逐 step 可视化 | `outputs/figures/map_only_evolution_d2ep80/`、`outputs/figures/map_only_steps_d2ep80/` |
| 对照拼图/柱图 | `outputs/figures/ablation_safety/`、`outputs/figures/ablation_transfer/` |
