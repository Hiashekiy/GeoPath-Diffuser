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

**8 步 + 当前 checkpoint（`d2_ep100`，C=64，真实 test split）实测效果（20 个场景配对）**：

| 组 | 碰撞率 | 碰撞自由场景 | 每步违反 均值/最大 (m) | 最终违反 (m) | QP solved | 有证书 | QP 修正 (m) | 单样本耗时 |
|---|---|---|---|---|---|---|---|---|
| A map_only | 0.0609 | 13/20 | 0.000 / 0.000 | 0.000 | — | 0 | — | 0.13 s |
| B safety_only | 0.0245 | 15/20 | 0.600 / 0.794 | 0.368 | — | 0 | — | 0.23 s |
| C safety+CPCG（无 QP） | 0.0244 | 15/20 | 0.599 / 0.793 | 0.365 | — | 0 | — | 0.22 s |
| **D safety+CPCG+QP** | **0.0149** | **18/20** | 0.599 / 0.793 | **0.078** | **17/20** | **17/20** | 0.763 | 0.26 s |

即：**末端 QP 是主力**（碰撞率 2.45 % → 1.49 %，碰撞自由 15 → 18/20，最终走廊违反 0.368 → 0.078 m，
17/20 拿到证书）；**CPCG 本身在 8 步下贡献很小**（0.368 → 0.365 m，−0.8 %），
把 `step_size` 放大到 10 后可到 −19 %（0.296 m，碰撞自由 17/20）。原因见 §7：方案 §2.2 的
`count` 归一化把每步位移压到毫米级，Trust Region（0.25 m）从未生效。


**实测效果（1000 步、真实 checkpoint `d2_ep100`）**：
在唯一"CPCG 有事可做"的场景（test #4）上，CPCG 使后续干净预测的走廊违反量**单调下降**
（0.1888 → 0.1847 m，`step_size` 扫到 0.5 时为 0.1682 m），末端 QP 的修正需求量**同步下降 7.8 %**
（0.5285 → 0.4872 m）；最终 QP 在该场景求解成功并给出证书（违反量 0，修正 0.5239 m）。
场景 #1 上末端 QP **不可行**（走廊不覆盖起终点），采样器如实报告 `primal infeasible` 并**不**给出证书、
也**不**放宽端点/朝向条件——不把原始预测冒充安全轨迹。

---

## 1. 改动清单

| 文件 | 状态 | 内容 |
|---|---|---|
| `src/gp_diffuser/diffusion/clean_guidance.py` | 新增 | masked squared-hinge 安全代价、解析梯度、Trust Region 引导步 |
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

## 2. 实现的数学（与方案逐条对应）

```text
beta[p,r] = E[p,r] Q                                 精确三次 Bezier 提取（pack 已提供）
g[p,r,f] = A[p,f]·beta[p,r] - b[p,f]                  g<=0 安全
raw       = g + margin_scene                          margin_scene 默认 0（走廊自带 safety_margin）

v         = relu(raw) * valid                          valid = piece_mask & face_mask
violated  = valid & (raw > 0)
count     = sum(violated).clamp_min(1)                只统计"当前违反"的有效标量
J         = 0.5 * sum(v²) / count                     masked squared hinge
dJ/dQ[k,d]= sum_{p,r,f} v * A[p,f,d] * E[p,r,k] / count   解析梯度，无 autograd
```

引导步（`clean_prediction_guidance`）：

```text
grad[:, 0] = grad[:, -1] = 0                          端点不是自由变量
q0_tmp     = q0 - step_size * grad
q0_bound   = BoundaryDecoder(q0_tmp, S, G, heading)   端点/朝向条件在步后仍然成立
delta_eff  = q0_bound - q0
scale      = min(1, max_delta_scene / max_i|delta_eff_i|)
q0_guided  = q0 + where(has_violation, scale*delta_eff, 0)   无违反 → 原样返回
```

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
python -m pytest tests -q          # 142 collected
140 passed, 1 skipped, 2 errors    # 两个 error = pytest 的 tmp_path fixture 在本机沙箱下无法创建临时目录
```

| 文件 | 用例 | 说明 |
|---|---|---|
| `tests/test_clean_guidance.py` | 16 | 有限差分核对（atol 1e-4 / rtol 1e-3）、下降方向、已满足则零引导、掩码不产生梯度、空 pack（None / 0 piece / 全 False）、batch 独立性、C=32/64、端点与朝向保持、Trust Region 上限、DDIM 代数一致性 |
| `tests/test_qp_projection.py` | 10 | box 走廊可行、端点精确、切向约束、全局控制点（非独立 Bezier）、空 pack 不认证、不可行明确报告、C=32/64、行装配与 `pack` 半空间逐一对应、padding 面不入 QP |
| `tests/test_inference_pipeline.py` | 22 | 搜索/椭圆各一次、pack 与 safety memory 对象同一、CPCG 不增加前向、`q0_guided` 确实进入 DDIM、QP 恰好一次、512 点验证针对 QP 后曲线、三模式差异、别名、搜索失败标记、未认证不覆盖轨迹、**默认 8 步且几何落在第 4 步** |

有限差分实测（box 走廊，C=32，`k=10,d=0`）：解析 `0.0590868` vs 有限差分 `0.0591204`，差 `3.3e-5`。

**两个 error 与本轮改动无关**（`test_checkpoint_version_gate`、`test_cache_fingerprint_rejects_stale`）：
它们使用 pytest 的 `tmp_path` fixture；在当前的 DSH 沙箱下 `tmp_path` 的目录创建/扫描被拒绝
（用一个只写 `tmp_path` 的空测试单独复现，同样失败）。命令行上给 `--basetemp` 也会在 pytest 会话结束时
于 `cleanup_dead_symlinks` 报同样的权限错误。

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
| map_only | 0.0000 | 1.00 | 0.0° | 0.0000 | 0 | — | 否 | 0.0000 | 0.15 s |
| safety_only | 0.0019 | 0.88 | 0.0° | 0.1591 | 0 | — | 否 | 0.0000 | 0.24 s |
| cpcg | 0.0018 | 0.88 | 0.0° | **0.1075** | 4 | solved | **是** | 0.0691 | 0.25 s |

（“成功”= 该样本 `valid`；8 个场景里有 1 个在两种安全模式下都不 valid，原因与走廊/端点覆盖有关，见 §6.1。）

分项耗时（cpcg，8 步合计 0.25 s）：地图编码 `0.012 s` / 4 步 map-only + 4 步 safety-conditioned ≈ `0.20 s` /
搜索 `0.03 s` / 椭圆 `0.05 s` / 走廊 `0.11 s` / **CPCG 4 步合计 ≈ `0.02 s`** / QP `0.013 s` / 512 点验证 `0.017 s`。
对比：同样链路跑满 1000 步是 **27 s/样本**，即 8 步快 **≈100 倍**（见 §10 存档）。

图：`outputs/figures/{map_only,safety_only,cpcg}_000_08.png`
（(c) 面板同时给出 `x0_pred → x0_guided → x0_projected` 与 pack 的 Bezier 分段多边形）。

---

## 6. Phase C：同噪声明配对 A/B/C/D（8 步 · 20 个场景）

```powershell
& $py -u scripts\run_cpcg_paired.py --config configs/c64_longpath.yaml --dataset real --split test `
    --indices 0,1,2,...,19 --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda
```

固定：checkpoint / 场景 / 起终点与朝向 / 初始噪声 seed / 8 个执行步。
脚本对每个场景校验 B/C/D 三组用的是**同一个 pack**（`torch.equal` 全字段）→
20 个场景全部 `shared_pack_across_conditioned_groups = true`，几何差异没有污染对照。

| 组 | 碰撞率 | 碰撞自由 | 最小间隙(m) 均值 | 每步违反 均值/最大(m) | 最终违反(m) | QP solved | 有证书 | QP 修正(m) | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| A map_only | 0.0609 | 13/20 | 0.858 | 0.000 / 0.000 | 0.000 | — | 0 | — | 0.13 s |
| B safety_only | 0.0245 | 15/20 | 0.751 | 0.600 / 0.794 | 0.368 | — | 0 | — | 0.23 s |
| C safety+CPCG | 0.0244 | 15/20 | 0.751 | 0.599 / 0.793 | 0.365 | — | 0 | — | 0.22 s |
| D safety+CPCG+QP | **0.0149** | **18/20** | **0.813** | 0.599 / 0.793 | **0.078** | **17/20** | **17/20** | 0.763 | 0.26 s |

### 6.1 关键判据（对照方案 §10）

* **末端 QP 是这轮的主力**：碰撞率 2.45 % → 1.49 %，碰撞自由 15 → 18/20，最终走廊违反 0.368 → 0.078 m，
  17/20 场景求解成功且拿到证书（512 点密集碰撞验证同时通过）。
* **CPCG 自身影响很小但方向正确**：每步最大违反 0.600 → 0.599 m、最终 0.368 → 0.365 m（−0.8 %）。
  放大 `step_size` 可放大到 −19 %（§7）。**这是"梯度太弱"，不是"接线错误"**：
  解析梯度与有限差分一致，且效果随 `step_size` 单调。
* **3/20 场景 QP 不可行**（`primal infeasible`）：诊断字段 `failure_reason = endpoints_outside_corridor`
  —— 走廊的首/末凸单元没有覆盖 S/G，硬端点条件与走廊冲突。按方案 §5.4 如实报告、**不放宽**约束、
  **不**把未投影曲线当成安全轨迹（`corridor_certified = false`）。这 3 个场景里 CPCG/QP 都改不了碰撞，
  下一轮要先修走廊覆盖。
* **安全条件 ≠ 更"走廊安全"**：A 组（map_only）碰撞反而最低（0.06 vs 0.025），因为网络学到的曲线会内切
  走廊收口；这正是需要末端 QP 的几何原因，也是 A→B 走廊违反从 0 升到 0.368 m 的来源。

### 6.2 每步违反曲线

`outputs/figures/cpcg_paired/scene*.png` 左图为 A/B/C/D 四条轨迹，右图为**每个执行步的干净预测**对走廊的
最大违反量（m）。8 步下前 4 步 map-only（尚未建走廊，违反记为 0 是因为该阶段没有 pack），
第 5 步起几何生效、CPCG 开始逐步压低干净预测的违反。

---

## 7. Phase D：`step_size` 扫描（8 步 · 20 个场景 · C 与 D 组）

```powershell
& $py -u scripts\run_cpcg_paired.py ... --groups C,D --set clean_guidance.step_size=10.0
```

| step_size | C 最终违反 (m) | C 每步违反 均值/最大 (m) | C 碰撞率 | C 碰撞自由 | D QP 修正 (m) | D 碰撞率 |
|---|---|---|---|---|---|---|
| **0.1（默认）** | 0.365 | 0.599 / 0.793 | 0.0244 | 15/20 | 0.763 | 0.0149 |
| 2.0 | 0.320 | 0.575 / 0.779 | 0.0225 | 16/20 | 0.669 | 0.0140 |
| 10.0 | **0.296** | **0.563 / 0.771** | **0.0209** | **17/20** | 0.708 | **0.0134** |

（1000 步下的同款扫描见 §10 存档，结论一致：单调变好、幅度有限。）

结论（如实记录，未改方案公式）：

1. **效果随 `step_size` 单调**：0.1 → 10（100 倍）使 C 组最终违反 −19 %、碰撞自由 +2/20。
2. **Trust Region 从未生效**：`max_delta_m = 0.125 / 0.25 / 0.5` 结果完全相同（1000 步扫描）；
   本数据实测单步位移只有毫米级。根因是方案 §2.2 规定的 `count` 归一化（只按当前违反面数平均，
   违反面常有数百个），把梯度压到 ~1e-3。
3. 若想让 Trust Region 真正起作用，可考虑 `step_size ∈ [2, 10]` 或按 `sqrt(count)` 归一化——
   后者是对方案**有意偏离**，需要你确认；本轮默认值仍是方案给的 `0.1`。
4. `margin_scene` 保持 0（走廊构建已含 `corridor.region.safety_margin = 0.02`），未叠加第二层裕度。


结论（如实记录，未擅自改公式）：

1. **效果随 `step_size` 单调变好但没有饱和点被触及**：0.1 → 10（100 倍）只带来 −19 % 的走廊违反改进。
2. **Trust Region 从未生效**：`max_delta_m = 0.125 / 0.25 / 0.5` 三组结果**完全相同**，
   实测单步最大位移只有毫米级。原因是方案 §2.2 规定的 `count` 归一化（只按当前违反面数平均）
   在本数据上把梯度压到 ~1e-3 量级：`delta ≈ step_size * v / count ≈ 0.1 * 0.01/500`。
   若下一轮想真正用满 Trust Region，可考虑 `step_size ∈ [2, 10]` 或改为按 `sqrt(count)` 归一化——
   后者是对方案的**有意偏离**，需要先确认，本轮默认值仍是方案给的 `0.1`。
3. `margin_scene` 保持 0（走廊构建已含 `corridor.region.safety_margin = 0.02`），未叠加第二层裕度。

---

## 8. 已知限制 / 下一轮建议

1. **checkpoint 未收敛**：`d2_ep100` 在 README §5.1 已记录"推理碰撞 9–38 %"，本报告的绝对数值只用于
   验证 CPCG 管线与相对趋势，不构成性能结论。
2. **走廊不覆盖端点的样本会导致末端 QP 不可行**（20 个场景里 3 个）。下一轮应检查
   `build_safety_corridor` 的首/末单元是否包含 S/G，或把 QP 的端点行改为"包络式"（先判覆盖，再决定是否投影）
   ——方案禁止的是**悄悄**丢约束，显式判定并报告是允许的，但需要先确认口径。
3. **CPCG 的梯度被归一化稀释**：8 步只有 4 个被引导的步，每步位移毫米级，因此它的贡献远小于末端 QP。
   若要让 CPCG 成为主力，需要（a）提高 `step_size`/改归一化，或（b）增加执行步数（8 → 20/50）。
4. **验收口径**：本轮的"安全证书"= 末端 QP 在非空 pack 上 solved、残差 ≤ `feasibility_tol_scene`、
   且 512 点密集碰撞验证通过（`corridor_certified`）。它只对 pack 覆盖的曲线区间成立，不是对整条曲线的
   地图级保证；地图级结论仍以 dense collision validation 为准（与方案 §12 一致）。
5. 2 个 `tmp_path` 用例在本机沙箱下无法运行（见 §3），不影响本轮任何结论；README/HANDOFF 里的
   测试数量以本报告为准（**140 passed, 1 skipped**，8 步改动后重跑）。

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

# E) 参数扫描（Phase D）
& $py -u scripts\run_cpcg_paired.py --config configs/c64_longpath.yaml --dataset real --split test `
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
    --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda `
    --groups C,D --set clean_guidance.step_size=10.0 --out outputs/reports/cpcg_8steps_ss100.json
```

产物：`outputs/reports/{inference_report,cpcg_8steps_20,cpcg_8steps_ss200,cpcg_8steps_ss100}.{json,md}`、
`outputs/figures/{map_only,safety_only,cpcg}_000_08.png`、`outputs/figures/cpcg_paired/scene*.png`、
`outputs/logs/{cpcg_8steps,cpcg_8_20,corridor_equiv}.log`。

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
