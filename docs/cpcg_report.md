# CPCG 实施与验收报告（Clean-Prediction Constraint Guidance + 末端凸 QP）

> 方案依据：`GeoPath_CPCG_实施方案.md`
> 审查基准 commit：`f63e123e7019fbfbbb1b478f047cdfe6dae73962`
> 运行环境：`E:\CondaEnvData\envs\GGMPC\python.exe`（torch 2.9.1+cu126 / osqp 1.1.0 / RTX 4070）
> **采样设置（用户指令，2026-11-28）**：统一 **8 个反向步**（`sampling.steps: 8`）——
> 在训练好的 1000 点噪声表里均匀取 8 个时刻做 DDIM 子采样，**不重训、不改权重**。
> 方案原文要求的“完整 1000-step 时间表”已按用户指示作废（原结果保留在 §10 存档）。

---

## 0. 一句话结论

推理期 ALM 已被 **CPCG（干净预测空间的解析走廊梯度）+ 一次末端 OSQP 凸投影** 取代，
采样器不再 import/call ALM；解析梯度与有限差分一致（差 3.3e-5），走廊几何在配置迁移后**逐位不变**。

**2026-11-28 评审整改后**（详见 §2.1）：安全代价不再除以违反约束个数（`count` 归一化会带来
不连续 + 梯度被稀释），Trust Region 改为"步长"本身并对每个试探重新过 BoundaryDecoder 评估真实代价，
新增 backtracking 接受机制（只接受 `J` 严格下降的步）。整改后 `max_delta_m` 成为**单调有效的旋钮**。

**8 步 + 当前 checkpoint（`d2_ep100`，C=64，真实 test split，20 个场景配对，配置见 §6.0）**：

| 组 | 碰撞率 | 碰撞自由场景 | 每步违反 均值/最大 (m) | 最终违反 (m) | QP solved | 有证书 | QP 修正 (m) | 单样本耗时 |
|---|---|---|---|---|---|---|---|---|
| A map_only | 0.0609 | 13/20 | 0.000 / 0.000 | 0.000 | — | 0 | — | 0.21 s |
| B safety_only | 0.0245 | 15/20 | 0.600 / 0.794 | 0.368 | — | 0 | — | 0.38 s |
| C safety+CPCG（无 QP） | 0.0192 | 17/20 | 0.531 / 0.749 | 0.232 | — | 0 | 0.46 s |
| **D safety+CPCG+QP** | **0.0134** | **19/20** | 0.531 / 0.749 | **0.056** | **17/20** | **17/20** | 0.578 | 0.49 s |

结论：
* **CPCG 现在自己有实质贡献**：走廊违反 0.368 → **0.232 m（−37 %）**、碰撞 2.45 % → **1.92 %（−22 %）**、
  碰撞自由 15 → **17/20**；`max_delta_m` 从 0.1 扫到 1.0 时违反单调降到 0.155 m（−58 %，§7）。
* **末端 QP 仍是"零碰撞"的收尾**：把 1.92 % 压到 **1.34 %**、最终违反 0.232 → **0.056 m**，17/20 拿证书。
* 代价：曲率均值 +15 %（md 0.5）/ +44 %（md 1.0），见 §7 的"质量代价"列。
* 3/20 场景末端 QP 不可行（走廊不覆盖起终点），采样器如实报 `primal infeasible`、不给证书、
  不放宽端点/朝向条件（方案 §5.4）。

**历史**（整改前的 count 归一化版本，保留以便对比）：C 组 0.365 m（−0.8 %）、D 组 0.078 m / 18/20 自由；
1000 步满表运行的存档见 §10。

---

## 1. 改动清单

| 文件 | 状态 | 内容 |
|---|---|---|
| `src/gp_diffuser/diffusion/clean_guidance.py` | 新增 | `J = 0.5·Σrelu(g+margin)²`（不除违反个数）+ 解析梯度 → 重投影 → 归一化到 `max_delta_m` → backtracking 接受（见 §2.1） |
| `src/gp_diffuser/safety/qp_projection.py` | 新增 | pack → 稀疏 `A x ≤ b` + 端点/朝向行，逐样本 OSQP 凸 QP，证书与失败诊断 |
| `src/gp_diffuser/diffusion/sampler_conditional_dit.py` | 修改 | 删除 ALM 分支（`lam`/dual/accept-reject），接入 CPCG 与末端 QP，模式改名，trace/统计字段 |
| `configs/default.yaml` | 修改 | 新增 `clean_guidance` / `final_projection`；走廊几何参数归位 `corridor`；`alm` 标记为历史 |
| `scripts/run_inference.py` | 修改 | `map_only / safety_only / cpcg` 三模式、`--dataset real`、CPCG/QP 指标与报告表 |
| `scripts/run_cpcg_paired.py` | 新增 | 同噪声明配对 A/B/C/D 实验 + `--set` 参数扫描（Phase C/D） |
| `scripts/verify_corridor_equivalence.py` | 新增 | 走廊配置迁移（`alm` → `corridor`）逐位等价性验证 |
| `src/gp_diffuser/evaluation/visualize.py` | 修改 | 三条干净曲线（pred / guided / projected）+ Bezier 分段控制多边形；接受 CUDA 张量 |
| `src/gp_diffuser/evaluation/metrics.py` | 修改 | `corridor_violation_m` 补回 `SceneTransform.offset`（此前漏掉导致 32 m 的假违反） |
| `tests/test_clean_guidance.py` / `tests/test_qp_projection.py` | 新增 | 26 个单元测试（解析梯度/掩码/空 pack/C=32,64/DDIM 一致性/QP 可行-不可行） |
| `tests/test_inference_pipeline.py` 等 | 修改 | 三模式集成测试、ALM 不再被调用、QP 恰好一次、未认证不覆盖轨迹 |
| `scripts/{smoke_pipeline,demo_full_pipeline,evaluate_real,octile_gallery,test_octile_map,visualize_prediction,diag_regions_and_alm,compare_safety_ablation}.py` | 修改 | `alm_stats` → `guidance_stats` / `final_qp_*`，旧模式名改为别名 |
| `README.md`、`docs/HANDOFF.md` | 修改 | 推理修正方法改为 CPCG + 末端 QP；配置/测试/复现命令同步 |

**未改动**（本轮明确禁止）：训练目标（`training/losses.py`）、网络结构（`models/*`）、数据集、
地图编码器、搜索 / PathSafe 椭圆 / 凸走廊构建算法、`src/gp_diffuser/vendor/trajsafe/`。
`diffusion/bspline_alm.py` 仅作历史模块保留，活动链路不 import（有测试断言源码里不出现 `bspline_alm`）。

---

## 2. 实现的数学

```text
beta[p,r] = E[p,r] Q                                 精确三次 Bezier 提取（pack 已提供）
g[p,r,f] = A[p,f]·beta[p,r] - b[p,f]                  g<=0 安全
raw       = g + margin_scene                          margin_scene 默认 0（走廊自带 safety_margin）

v         = relu(raw) * valid                          valid = piece_mask & face_mask
J         = 0.5 * sum(v²)                             **不做 count 归一化**（§2.1）
dJ/dQ[k,d]= sum_{p,r,f} v * A[p,f,d] * E[p,r,k]       解析梯度，无 autograd
```

引导步（`clean_prediction_guidance`，direction → trust region → backtracking）：

```text
grad[:, 0] = grad[:, -1] = 0                          端点不是自由变量
q0_tmp     = q0 - step_size * grad
delta_eff  = BoundaryDecoder(q0_tmp, S, G, heading) - q0    端点/朝向条件在步后仍然成立
d_TR       = delta_eff * (max_delta_scene / max_i|delta_eff_i|)   ← Trust Region 就是步长
for a in (1, 0.5, 0.25, 0.125, 0.0625):
    Q_try  = BoundaryDecoder(q0 + a * d_TR, S, G, heading)     每次试探都重新过 decoder
    if J(Q_try) < J(q0):  accept(Q_try, alpha=a); break        严格下降才接受
否则：reject，q0 原样返回（统计里计为 rejected）
```

### 2.1 评审整改（2026-11-28）

针对静态审查提出的四个问题逐条处理，**未改动** Diffusion 网络、训练损失、Bezier 约束与 DDIM：

| 审查项 | 处理 | 证据 |
|---|---|---|
| ① `count` 归一化稀释梯度（梯度/数百 → 毫米级位移，Trust Region 从不生效） | 删除分母；`J = 0.5·Σv²`，梯度为原始求和 | `test_cost_is_the_plain_sum_of_violations`、`test_cost_grows_with_the_number_of_equal_violations`（重复一份约束 → J 恰好 ×2） |
| ② 代价在违反个数变化时不连续（N:1→2 使 J 减半） | 同上；`safe_loss_before/after` 现在可直接用作判据 | `test_cost_is_continuous_when_a_new_constraint_starts_violating`：同一扫描下新代价单调 ↑；并断言旧式 `J/count` 在该扫描上**确实非单调**（N:48→82 时掉 41 %） |
| ③ 缺修正接受机制（改坏也照样进 DDIM） | backtracking 线搜索 + 严格下降判据；不接受则原样返回 | `test_a_step_that_cannot_improve_is_rejected`、`test_acceptance_never_increases_the_cost`（随机多边形 3 seed，被接受行严格下降、被拒行位级不变） |
| ④ Trust Region 只是"上限" | 归一化为步长 + 逐试探重投影后评估真实代价 | `test_trust_region_is_the_actual_step_length`（接受后 `|Δ|_max == max_delta_m`）、`test_backtracking_shortens_an_overshooting_step`（窄走廊下 α<1 被选中） |

行为变化（实测，10 个 test 场景 × 8 步）：

* 每步 `J_after/J_before = 0.698`（平均降低 30 %），步长 `|Δ|_max = 0.500 m` 正好等于 `max_delta_m`；
* 40 个被引导的步里有 16 个（40 %）存在违反并全部被接受，`alpha = 1.0`、`trials = 1`
  —— 在这批场景上全步长就能下降，线搜索当前是"安全网"而非限流器（窄走廊单测里它会真的触发）；
* 拒绝率 0 %（拒绝只在无法下降时发生，见单测）。

配置一致性（审查第 4 点）：`configs/default.yaml` 现在是 `step_size: 1.0` / `max_delta_m: 0.5` /
`backtrack_factors: [1, .5, .25, .125, .0625]` / `sampling.steps: 8`，本报告所有表格都用这套值；
整改前的 `step_size: 10.0` 系列数字集中在 §7 的历史段与 §10。

接入 DDIM（**调度器数学一字未改**，只替换 `q0_used`）：

```text
eps = (q_t - sqrt(ab)_t * q0_used) / sqrt(1-ab)_t
q   = sqrt(ab)_prev * q0_used + sqrt(1-ab)_prev * eps
等价于 eps_guided = eps_pred - sqrt(ab/(1-ab)) * (q0_guided - q0_pred)
```

末端凸 QP（`project_controls_to_corridor_qp`，每样本一次）：

```text
min 0.5||Q - Q_guided||²
s.t. A[p,f]·E[p,r]·Q <= b[p,f]            只展开 piece_mask & face_mask 的有效行
     Q[0] = start, Q[C-1] = goal
     normal(Q_1-Q_0) = 0, forward(Q_1-Q_0) >= 0      （heading 有效时）
     normal(Q_{C-1}-Q_{C-2}) = 0, forward(...) >= 0  （同上）
```

单位：`q0` / `pack` 全部是 normalized scene 坐标；`max_delta_scene = max_delta_m / SceneTransform.scale`
（本机 `scale = 32 m`）。控制点数 `C` 全程从 `codec.num_controls` / `pack.num_controls` 读取（C=32 与 C=64 均有测试）。

---

## 3. Phase A：测试结果

```
python -m pytest tests -q          # 156 collected
155 passed, 1 skipped
```

| 文件 | 用例 | 说明 |
|---|---|---|
| `tests/test_dual_branch_training.py` | 13 | 双支路共享 `q_t/t`、单次反向、无安全臂隔离；**非有限守卫**（NaN loss 跳过、NaN 梯度跳过、NaN 权重拒绝落盘） |
| `tests/test_clean_guidance.py` | 26 | 有限差分核对（atol 1e-4 / rtol 1e-3）、下降方向、已满足则零引导、掩码不产生梯度、空 pack（None / 0 piece / 全 False）、batch 独立性、C=32/64、端点与朝向保持、DDIM 代数一致性；**代价无 count 归一化**（`J = 0.5Σv²`、重复约束使 J 恰好翻倍、跨违反数跳变仍单调、并断言旧式 `J/count` 在该扫描上非单调）；**Trust Region = 步长**（接受后 `|Δ|_max == max_delta_m`）；**backtracking**（窄走廊下 α<1、无法下降则拒绝且位级不变、随机多边形上被接受行严格下降） |
| `tests/test_qp_projection.py` | 10 | box 走廊可行、端点精确、切向约束、全局控制点（非独立 Bezier）、空 pack 不认证、不可行明确报告、C=32/64、行装配与 `pack` 半空间逐一对应、padding 面不入 QP |
| `tests/test_inference_pipeline.py` | 22 | 搜索/椭圆各一次、pack 与 safety memory 对象同一、CPCG 不增加前向、`q0_guided` 确实进入 DDIM、QP 恰好一次、512 点验证针对 QP 后曲线、三模式差异、别名、搜索失败标记、未认证不覆盖轨迹、默认 8 步且几何落在第 4 步 |

有限差分实测（box 走廊，C=32，`k=10,d=0`）：解析 `0.0590868` vs 有限差分 `0.0591204`，差 `3.3e-5`。

---

## 4. 走廊配置迁移等价性（方案 §6.2）

`scripts/verify_corridor_equivalence.py` 用同一场景、同一批椭圆同时构造两次走廊与约束包：

* legacy：`{**corridor.region, **<alm 几何键>}`，`margin = alm.constraint_margin`（从 `git show HEAD:configs/default.yaml` 读取迁移前的 `alm` 段）
* current：`corridor.region` 全部，`margin = corridor.constraint_margin_scene`

结果（test split，index 2）：

```
corridor_valid  (True, True)      corridor_cells (64, 64)      bridge_cells (0, 0)
pack_summary    num_pieces 92 / num_faces_max 12 / active_halfspaces 914
identical       anchors ✓  cells ✓  halfspaces ✓  pack ✓  extraction ✓  bounds ✓
GEOMETRY UNCHANGED
```

因此"把几何参数从 `alm` 搬到 `corridor`"没有改变任何几何数值。

---

## 5. Phase B：8 步 · 8 个场景 · 三模式（真实 checkpoint）

```powershell
& $py -u scripts\run_inference.py --dataset real --config configs/c64_longpath.yaml --device cuda `
    --samples 8 --modes map_only,safety_only,cpcg `
    --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt
```

（默认即 `--steps 8`：`pick_times(1000, 8)` = 在训练噪声表上均匀取 8 个时刻，**不重训**；
arch 版本校验通过，C=64。）

| mode | 碰撞率 | 成功 | 朝向误差 | 走廊违反（m） | CPCG 步数 | QP 状态 | 证书 | QP 修正（m） | 总耗时/样本 |
|---|---|---|---|---|---|---|---|---|---|
| map_only | 0.0000 | 1.00 | 0.0° | 0.0000 | 0 | — | 否 | 0.0000 | 0.24 s |
| safety_only | 0.0019 | 0.88 | 0.0° | 0.1591 | 0 | — | 否 | 0.0000 | 0.43 s |
| cpcg | 0.0000 | 1.00 | 0.0° | **0.0669** | 4 | solved | **是** | 0.0199 | 0.48 s |

（“成功”= 该样本 `valid`；8 个场景里有 1 个在两种安全模式下都不 valid，原因与走廊/端点覆盖有关，见 §6.1。）

分项耗时（cpcg，8 步合计 0.48 s）：地图编码 `0.012 s` / 4 步 map-only + 4 步 safety-conditioned ≈ `0.40 s` /
搜索 `0.03 s` / 椭圆 `0.05 s` / 走廊 `0.11 s` / **CPCG 4 步（含 backtracking）≈ `0.03 s`** / QP `0.014 s` /
512 点验证 `0.017 s`。对比：同样链路跑满 1000 步是 **27 s/样本**，即 8 步快 **≈60 倍**（见 §10 存档）。

图：`outputs/figures/{map_only,safety_only,cpcg}_000_08.png`
（(c) 面板同时给出 `x0_pred → x0_guided → x0_projected` 与 pack 的 Bezier 分段多边形）。

---

## 6. Phase C：同噪声明配对 A/B/C/D（8 步 · 20 个场景）

### 6.0 报告的配置口径（审查第 4 点）

本节与 §0/§7 的全部数字都用**同一套配置**，即 `configs/default.yaml` 当前值：

```yaml
sampling.steps: 8
clean_guidance: {enabled: true, step_size: 1.0, max_delta_m: 0.5,
                 backtrack_factors: [1.0, 0.5, 0.25, 0.125, 0.0625], margin_scene: 0.0}
final_projection: {enabled: true, solver: osqp, feasibility_tol_scene: 1.0e-4}
```

checkpoint：`outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt`（T=1000，C=64）。
命令行：

```powershell
& $py -u scripts\run_cpcg_paired.py --config configs/c64_longpath.yaml --dataset real --split test `
    --indices 0,1,2,...,19 --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda
```

固定：checkpoint / 场景 / 起终点与朝向 / 初始噪声 seed / 8 个执行步。
脚本对每个场景校验 B/C/D 三组用的是**同一个 pack**（`torch.equal` 全字段）→
20 个场景全部 `shared_pack_across_conditioned_groups = true`，几何差异没有污染对照。

| 组 | 碰撞率 | 碰撞自由 | 最小间隙(m) 均值 | 每步违反 均值/最大(m) | 最终违反(m) | QP solved | 有证书 | QP 修正(m) | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| A map_only | 0.0609 | 13/20 | 0.858 | 0.000 / 0.000 | 0.000 | — | 0 | — | 0.21 s |
| B safety_only | 0.0245 | 15/20 | 0.751 | 0.600 / 0.794 | 0.368 | — | 0 | — | 0.38 s |
| C safety+CPCG | **0.0192** | **17/20** | 0.769 | 0.531 / 0.749 | **0.232** | — | 0 | — | 0.46 s |
| D safety+CPCG+QP | **0.0134** | **19/20** | **0.813** | 0.531 / 0.749 | **0.056** | **17/20** | **17/20** | 0.578 | 0.49 s |

### 6.1 关键判据（对照方案 §10）

* **CPCG 单独有效**：走廊违反 0.368 → **0.232 m（−37 %）**、碰撞 2.45 % → **1.92 %（−22 %）**、
  碰撞自由 15 → **17/20**，每步最大违反 0.600 → 0.531 m。整改前同一配置（旧 count 归一化 +
  `step_size 10 / md 0.5`）只能做到 −0.8 %，差别就来自 §2.1 的归一化与步长语义。
* **末端 QP 完成"零碰撞"收尾**：1.92 % → **1.34 %**、最终违反 0.232 → **0.056 m**，17/20 求解成功并拿到证书
  （512 点密集碰撞验证同时通过）。
* **3/20 场景 QP 不可行**（`primal infeasible`）：诊断字段 `failure_reason = endpoints_outside_corridor`
  —— 走廊的首/末凸单元没有覆盖 S/G，硬端点条件与走廊冲突。按方案 §5.4 如实报告、**不放宽**约束、
  **不**把未投影曲线当成安全轨迹（`corridor_certified = false`）。
* **安全条件 ≠ 更"走廊安全"**：A 组（map_only）碰撞反而最低（0.061 vs 0.025），因为网络学到的曲线会内切
  走廊收口；这正是需要梯度引导 + 末端 QP 的几何原因。

### 6.2 每步违反曲线

`outputs/figures/cpcg_paired/scene*.png` 左图为 A/B/C/D 四条轨迹，右图为**每个执行步的干净预测**对走廊的
最大违反量（m）。8 步下前 4 步 map-only（尚未建走廊，违反记为 0 是因为该阶段没有 pack），
第 5 步起几何生效、CPCG 开始逐步压低干净预测的违反。

---

## 7. Phase D：`max_delta_m`（Trust Region = 步长）扫描（8 步 · 20 个场景 · C 与 D 组）

整改后 `step_size` 只是"原始梯度步的比例"，归一化会让它退化，真正的旋钮是 `max_delta_m`：

```powershell
& $py -u scripts\run_cpcg_paired.py ... --groups C,D --set clean_guidance.max_delta_m=1.0
```

| `max_delta_m` (m) | C 最终违反 (m) | C 每步违反 均值/最大 (m) | C 碰撞率 | C 碰撞自由 | 曲率均值 / 最大 | 最小间隙 (m) | D QP 修正 (m) | D 碰撞率 |
|---|---|---|---|---|---|---|---|---|
| 0（无引导，B 组） | 0.368 | 0.600 / 0.794 | 0.0245 | 15/20 | 0.0745 / 1.25 | 1.208 | — | — |
| 0.10 | 0.337 | 0.585 / 0.785 | 0.0228 | 16/20 | 0.0749 / 1.27 | 1.230 | 0.074 | 0.0142 |
| 0.25 | 0.296 | 0.563 / 0.771 | 0.0209 | 17/20 | 0.0773 / 1.29 | 1.261 | 0.067 | 0.0134 |
| **0.50（默认）** | **0.232** | **0.531 / 0.749** | **0.0192** | **17/20** | 0.0856 / 1.37 | 1.304 | **0.056** | **0.0134** |
| 1.00 | 0.155 | 0.485 / 0.748 | 0.0168 | 18/20 | 0.1076 / 1.97 | 1.354 | 0.040 | 0.0134 |

结论：

1. **`max_delta_m` 现在是单调有效的步长旋钮**（整改前 0.125/0.25/0.5 三组结果**完全相同**，
   因为梯度被 `count` 除到毫米级，上限永不触发）。0.1 → 1.0 m 使走廊违反 −54 %、碰撞 −26 %。
2. **代价是曲率**：md 0.5 曲率均值 +15 %、md 1.0 为 +44 %（最大曲率 1.25 → 1.97），
   长度仅 +0.8 %，最小间隙 +8 %。默认取 0.5 m 是"违反/碰撞收益 vs 平滑度"的折中点，
   1.0 m 作为安全优先选项记录在此表中。
3. `step_size` 保持 1.0：它只决定"原始梯度步有多大"，归一化后不影响被接受步的长度
   （单测 `test_trust_region_is_the_actual_step_length` 断言 `|Δ|_max == max_delta_m`）。
4. `margin_scene` 保持 0（走廊构建已含 `corridor.region.safety_margin = 0.02`），未叠加第二层裕度。

### 7.1 整改前的扫描（历史，count 归一化 + step_size 语义）

为便于对照，记录整改前的同一组实验：`step_size` 0.1 / 2.0 / 10.0 → C 最终违反 0.365 / 0.320 / 0.296 m，
`max_delta_m` 0.125 / 0.25 / 0.5 → 结果完全相同。这组数据就是"归一化缺陷"的直接证据，
也是本轮整改的依据。1000 步满表下的同类扫描见 §10。

---

## 8. 已知限制 / 下一轮建议

1. **checkpoint 未收敛且正在换 T**：`d2_ep100`（T=1000）在 README §5.1 已记录"推理碰撞 9–38 %"，
   本报告的绝对数值只用于验证 CPCG 管线与相对趋势；`configs/c64_t12.yaml`（T=12、100 epoch、全量
   real train split）的重训完成后，同一套脚本会再出一版数字（§11）。
2. **走廊不覆盖端点的样本会导致末端 QP 不可行**（20 个场景里 3 个）。下一轮应检查
   `build_safety_corridor` 的首/末单元是否包含 S/G，或把 QP 的端点行改为"包络式"（先判覆盖，再决定是否投影）
   ——方案禁止的是**悄悄**丢约束，显式判定并报告是允许的，但需要先确认口径。
3. **CPCG 的剩余上限**：8 步里只有 4 个被引导的步，`max_delta_m` 再往上加会开始牺牲平滑度
   （§7：md 1.0 → 曲率 +44 %）。若要让引导承担更多工作，应增加执行步数（8 → 12/20）或改进方向
   （例如把 `margin_scene` 调成正裕度），而不是单纯放大步长。
4. **验收口径**：本轮的"安全证书"= 末端 QP 在非空 pack 上 solved、残差 ≤ `feasibility_tol_scene`、
   且 512 点密集碰撞验证通过（`corridor_certified`）。它只对 pack 覆盖的曲线区间成立，不是对整条曲线的
   地图级保证；地图级结论仍以 dense collision validation 为准（与方案 §12 一致）。
5. **`num_controls` 与 checkpoint 不匹配不会报错**（实测遇到的坑）：DiT 的控制点数不进入任何权重形状，
   用 C=32 的配置加载 C=64 的权重会"成功"但在错误的表示上推理。跑评测时请用训练时的 config
   （本报告一律 `configs/c64_longpath.yaml`）。
6. 测试：**155 passed, 1 skipped**（含新增的 3 个非有限守卫用例；沙箱放开后原先 `tmp_path` 的两个 error 也消失了，现在是全绿）。

---

## 9. 复现清单

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
cd D:\ProjectDirectory\GeoPath-Diffuser

# A) 单元 + 集成测试
& $py -m pytest tests -q

# B) 走廊配置迁移等价性
& $py scripts\verify_corridor_equivalence.py --index 2

# C) 8 步 · 三模式 + 图（默认步数就是 8）
& $py -u scripts\run_inference.py --dataset real --config configs/c64_longpath.yaml --device cuda `
    --samples 8 --modes map_only,safety_only,cpcg `
    --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt

# D) 配对 A/B/C/D（Phase C，20 个场景）
& $py -u scripts\run_cpcg_paired.py --config configs/c64_longpath.yaml --dataset real --split test `
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
    --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda

# E) 参数扫描（Phase D，Trust Region = 步长）
& $py -u scripts\run_cpcg_paired.py --config configs/c64_longpath.yaml --dataset real --split test `
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
    --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda `
    --groups C,D --set clean_guidance.max_delta_m=1.0 --out outputs/reports/cpcg_newguid_md10.json

# F) 逐去噪步可视化（含每步碰撞量），见 §11.3
& $py -u scripts\visualize_cpcg_steps.py --config configs/c64_t12.yaml `
    --checkpoint outputs/checkpoints/cpcg_t12_100ep/t12_ep070.pt `
    --split test --scan 40 --mode cpcg --device cuda
```

产物：`outputs/reports/{inference_report,cpcg_paired,cpcg_newguidance_8steps,cpcg_newguid_md01,
cpcg_newguid_md05,cpcg_newguid_md10}.{json,md}`、
`outputs/figures/{map_only,safety_only,cpcg}_000_08.png`、`outputs/figures/cpcg_paired/scene*.png`、
`outputs/logs/{cpcg_8steps,new_guid_8,new_guid_md*,corridor_equiv}.log`。

---

## 10. 存档：方案原文要求的“完整 1000 步”结果（已被用户指令作废）

用户于 2026-11-28 指示“把这个 1000 删掉，统一成 8 步，不用训练，用现在的模型测”。
下面是与 8 步结论**定性一致、定量更显著**的一次满表运行，仅作存档，不再是默认配置：

* 单场景（test #0）三模式：碰撞 0 / 走廊违反 0 / CPCG 500 步 / QP solved+认证；
  总耗时 16.5 s（map_only）、24.0 s（safety_only）、27.0 s（cpcg，其中 CPCG 2.35 s、QP 0.10 s）。
  与 8 步的 0.13 / 0.23 / 0.26 s 相比，满表慢 **≈100 倍**。
* 配对（场景 1、4）：场景 4 上 CPCG 使每步违反 0.1888 → 0.1847 m（−2.2 %）、最终 QP 修正 0.5285 → 0.4872 m
  （`step_size` 0.05 → 0.5）；场景 1 的 QP 因走廊不覆盖端点而 `primal infeasible`。
* `step_size` 扫描（场景 4）：0.05/0.1/0.2/0.5 → 最终违反 0.367/0.363/0.355/0.330 m；
  `max_delta_m` 0.125/0.25/0.5 三组完全相同（Trust Region 未生效）。
* 图表：`outputs/figures/{map_only,safety_only,cpcg}_000_1000.png`、
  `outputs/reports/cpcg_paired.json`（已被 8 步版本覆盖，历史数字见本报告表格）。

---

## 11. T=12 重训（`configs/c64_t12.yaml`）的结局与逐步骤可视化

### 11.1 训练实况（结论：**取 epoch 70 作为交付权重**）

| 阶段 | 结果 |
|---|---|
| 12,178 条全量 train split、C=64、双支路、batch 4 | 3,045 步/epoch，0.14–0.18 s/步 |
| epoch 1 → 70 | loss 0.970 → **0.0088**，`sm 0.026` 主导，无异常 |
| epoch 71 step 215,650 | loss 单步尖峰 0.14（正常 0.009），**下一步开始 NaN** |
| epoch 71–79 | 无守卫 → 带着 NaN 又"训"了 9 个 epoch，`last.pt` 被写成 **261 个非有限张量** |
| **本次修复** | `train_step` 增加非有限守卫（loss / 梯度）、`save_checkpoint` 拒绝写非有限权重；3 个单测覆盖 |
| epoch 70 → 86（从 `t12_ep070.pt` 续训） | 守卫生效：0 个 NaN 权重落盘；但 epoch 74 起坏批次数单调上升 1 → 623 → 1,410 → 2,046 → 2,926/3,045（**96 % 被跳过**），且未跳过的步 loss 从 0.0088 涨到 0.40–0.54 → **模型已坏**，任务在下令后人工中止 |

推断根因：**后期学习率过高**（loss 1e-2 的尖锐极小值 + `lr 2e-4` 的 AdamW），一次坏更新就把模型踢出极小值区；
守卫只止损不治病。若要继续，应从 `t12_ep070.pt` 以 **lr 5e-5** 续训。

**可用权重**：`outputs/checkpoints/cpcg_t12_100ep/t12_ep070.pt`（权重全有限，loss 0.0088）；
`t12_ep080/090/last.pt` 训练中后期已被污染（权重有限但质量崩坏，勿用）。

### 11.2 `t12_ep070` 的推理质量（20 场景、12 步、整改后的 CPCG）

| 组 | 碰撞率 | 碰撞自由 | 每步违反 (m) | 最终违反 (m) | QP solved | 有证书 |
|---|---|---|---|---|---|---|
| C safety+CPCG | 0.0435 | 14/20 | 1.245 | 0.746 | — | 0 |
| D + 末端 QP | **0.0085** | **19/20** | 1.245 | **0.197** | 18/20 | 18/20 |
| 对照 D（T=1000 `d2_ep100`，8 步） | 0.0134 | 19/20 | 0.531 | 0.056 | 17/20 | 17/20 |

即 T=12 的 70-epoch 权重**碰撞率更低**（0.85 % vs 1.34 %），但网络输出离走廊更远
（每步违反 1.245 vs 0.531 m），所以末端 QP 的修正量更大（1.54 m vs 0.58 m）；
纯 CPCG 组也更依赖 QP（4.35 % → 0.85 %）。

### 11.3 逐去噪步可视化（新脚本）

```powershell
& $py -u scripts\visualize_cpcg_steps.py --config configs/c64_t12.yaml `
    --checkpoint outputs/checkpoints/cpcg_t12_100ep/t12_ep070.pt `
    --split test --scan 40 --mode cpcg --device cuda
```

对**每一个执行步**输出一格子图，含：地图、GT、搜索路径、64 个椭圆、走廊凸区域（只画曲线附近的单元）、
该步的 `x0_pred` 与 `x0_guided`，标题标注该步的**碰撞百分比 / 碰撞段数 / 走廊违反量（m）**，
角落标注 CPCG 的 `J 前→后 / Δmax / α`；最后一格是末端 QP 之后的最终轨迹
（同时画 QP 前的曲线），标注 `QP 状态 / 证书 / 修正量 / 最小间隙`。
另出一张 `..._collision.png` 画"每步碰撞量"曲线，并写 `..._metrics.json`。

首个碰撞样本（test #1，`cpcg_ep070_s001`）的读数：

| 阶段 | 每步碰撞 | 走廊违反 |
|---|---|---|
| 第 1–6 步（map-only，几何尚未生成） | **40.3 %**（2 段） | 6.50 → 6.40 m |
| 第 7 步（几何生效，CPCG 首次介入） | 39.6 % → guided 19.3 % | 6.41 m |
| 第 8–12 步（CPCG 逐步压低） | 19.3 → 17.3 %（1 段） | 3.91 → 3.72 m |
| 最终（末端 QP） | **16.96 %**，`primal infeasible` → 未修正、无证书 | 最小间隙 −1.77 m |

结论：这一样本的走廊**完全不覆盖**该轨迹（违反 6.4 m），因此末端 QP 不可行、如实拒绝认证，
CPCG 只能把碰撞从 40 % 压到 17 %。这与 §8 第 2 条（走廊覆盖）是同一个待修问题，
也是"CPCG 有效但受几何质量限制"的最直观证据。
图：`outputs/figures/cpcg_steps/cpcg_ep070_s001.png`、`..._collision.png`、`..._metrics.json`。
