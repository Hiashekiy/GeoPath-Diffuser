# 代码复审逐条核验（2026-xx）

复审提出 20 余条问题。本文档逐条给出**核验结论 + 可复现证据**：哪些是真缺陷、哪些只在特定条件下成立、哪些我判断为设计取舍而非缺陷。
不预设结论：全部结论都由本次运行的数值或代码行支撑。

可复现的验收载体：`tests/test_known_defects.py` —— 8 个 `xfail(strict=True)` 用例，每个用例编码"修复后应满足的行为"。
当前 `8 xfailed`（缺陷全部存在）；一旦修复，它们会 `XPASS`，`strict=True` 会让套件**失败**，从而强制把标记和修复一起清掉。

```
$ python -m pytest tests/test_known_defects.py -q
xxxxxxxx                                                                 [100%]
8 xfailed in 6.45s
```

---

## 0. 修复状态（本轮）

| ID | 状态 | 修复位置 | 验收测试（`tests/test_review_regressions.py`） |
|---|---|---|---|
| P0-1 | **已修** | `training/losses.py::collision_loss`（`margin_m` 直接与米制 SDF 比较） | `test_collision_loss_uses_metre_margin`、`test_collision_loss_analytic` |
| P0-2 | **已修** | `safety/safety_features.py::ellipse_shape4_to_scene`；`sampler._generate_safety` 与 `training/safety_cache.py` 均改用场景坐标 | `test_ellipse_shape4_to_scene_units`、`test_shape4_metre_frame_hits_the_axis_clamp`、`test_shape4_scene_axes_are_physically_consistent`、`test_region_builder_receives_scene_axes` |
| P0-3 | **已修** | `safety/local_regions.py`：约束表行数 = `len(cells)`（含桥接），不再是 `Q` | `test_no_bridge_cell_is_truncated` |
| P0-4 | **已修** | `safety/local_regions.py::pack_from_tables` 逐样本构包 + 新增 `safety/pack_utils.py::merge_packs` 按 piece/face padding 合并 | `test_batch_pack_uses_per_sample_anchors`、`test_different_row_counts_merge` |
| P1-1 | **已修** | 新增 `safety/collision_query.py`（Amanatides-Woo 线段穿越），`sampler._collision_fractions`、`evaluation/metrics.py`、`local_regions` 全部改用线段级 | `test_thin_wall_detected_by_segment_sweep`、`test_sampler_collision_is_segment_level` |
| P1-2 | **已修** | `sampler`：ALM 门控改为 `local_cells > 0`；`trainer`：`L_corridor` 用 `include_box=False` 的 pack | `test_alm_runs_only_with_local_convex_cells`、`test_corridor_loss_ignores_box_cells` |
| P1-3 | **已修** | `sampler.sample`：每次前向后立刻 `hard_control_endpoints(pred, start_goal)` | `test_prediction_endpoints_locked` |
| P1-4 | **已修** | `sampler._generate_safety`：搜索失败 → `search_failed`，不再用引导轨迹冒充路径；该样本不生成安全条件、`valid=False` | `test_failed_search_is_not_a_path` |
| P1-5 | **已修** | `sampler.accept_correction`：碰撞率或最大违反量变差即拒绝该步修正 | `test_accept_correction_rejects_worse` |
| P1-6 | **已修** | `collision_query.collision_spans_on_path`：碰撞位置投影到搜索路径的弧长 → 查询索引 | `test_local_spans_follow_the_projected_position` |
| P1-7 | **已修** | `local_regions` info 新增 `dropped_spans` / `dropped_length_m`（不再静默丢弃） | 同上（返回字段） |
| P2-1 | **已修** | `sampler._heading_errors_deg`（用 `basis_derivative_at` 精确切向）→ 输出 `heading_error_deg`，并纳入 `valid`（`sampling.heading_tol_deg = 20°`） | `test_heading_errors_measured` + smoke 检查 |
| P2-2 | **已修（CUDA 未实测）** | `factory` 把 codec `to(device)`；`sampler` 搬 heading；pack 显式给 device | `test_pipeline_shapes_and_devices_on_cuda`（本机 `torch 2.14.0+cpu` → skip） |
| P2-3 | **已修** | `scripts/setup_references.py` + README §3.2；`safety_cache.fingerprint`（配置 + 数据形状 + checkpoint 大小/mtime 的 sha256），加载时校验 | `test_cache_fingerprint_rejects_stale` |

测试现状（本机 CPU）：`80 passed, 1 skipped`（skip 为 CUDA 用例）。

**本轮没有跑训练。** 之前那次 `train_dual_branch.py` 是修复过程中的一次误操作（16 个合成样本、4 个 epoch、共 16 步、约 30 秒、CPU），产出已删除，也没有任何基于它的结论。正式训练仍按你的要求，等代码复审通过后再谈。

### 0.1 复查时修正了我自己的两个判断

1. **P0-2 的影响不是"完全等价的接口问题"**。修复前 `shape4` 以米进入构建器，会命中
   `convex_region` 的 `_ELLIPSE_LOG_CLAMP=( -6.0, 0.7)` 与 `axis_max=2.0`：
   任何半轴大于 `exp(0.7)=2.01 m` 的椭圆被截断到 2.0 m。实测 test split 场景 0 的 a 达 2.18 m，
   截断后邻域点排序与面法向改变（`cell_A` 最大差 1.99）。所以它**确实改变输出**，
   但只在轴长越界的样本上发生。
2. **走廊统计在修复后数值未变**（train 8/16、bridges 12、失败原因相同）。原因：
   本数据集多数样本轴长未越界，而"两轴等比除以同一常数"只会把 Mahalanobis 度量整体放大
   32² 倍，排序不变（float32 下仅并列点次序可能变化）。也就是说：P0-2 是真实的单位/契约
   缺陷，但在这个数据集上的影响是局部的，不能靠"统计数字没变"判断它无所谓。

## 一、P0（必须修）：4/4 全部确认

### P0-1 `collision_loss` 安全距离单位错误 —— **确认，比描述更严重**

`src/gp_diffuser/training/losses.py:40-41`：

```python
margin_scene = float(margin_m) / float(scene_scale)   # 0.30 / 32 = 0.009375
violation = torch.relu(margin_scene - sampled)        # sampled 是米
```

`sdf` 网格存的是**米**，`grid_sample` 查出来的 `sampled` 也是米，而这里把 0.30 m 除以 32 变成 0.0094，
于是只有 `sdf < 0.0094 m`（几乎等于"已经进入障碍物"）才会产生惩罚。

实测（8 个点，常数 SDF，margin 0.30 m）：

| 真实净空 | 当前 loss | 应有 loss |
|---|---|---|
| 0.05 m | **0.00000000** | 0.06250000 |
| 0.10 m | **0.00000000** | 0.04000001 |
| 0.20 m | **0.00000000** | 0.01000000 |
| 0.28 m | **0.00000000** | 0.00040000 |
| 0.31 m | 0.00000000 | 0.00000000 |

结论：`L_collision` 实际退化成"穿透惩罚"，`λ_col` 与 0.3 m 安全裕度**完全失效**。
比复审说的"缩小 32 倍"更糟：在 0.0094 m 以上整段区间内恒为 0，梯度也是 0。

修复：`violation = torch.relu(float(margin_m) - sampled)`（`scene_scale` 参数保留但不再用于此处），
并加解析测试（`test_collision_loss_uses_metre_margin`，已就位）。

### P0-2 椭圆 `shape4` 未从米制转到场景坐标 —— **确认，误差正好 32.0×**

调用点：`sampler_conditional_dit.py:317,326` 与 `training/safety_cache.py:95`。
两处都把 `safety["ellipse_shape4"]`（`[log a_m, log b_m, cos2θ, sin2θ]`）直接送进 `build_region_table`，
而**中心点**已经 `tf.to_scene(...)` 转换过。`ellipse_shape4_to_quadratic()` 里 `axes = exp(shape4[...,0:2])`，
于是构建器把**米**当成**场景单位**。

实测（test 场景 0，前 3 个查询）：

| | a | b |
|---|---|---|
| 构建器实际收到（当成场景单位） | 0.77917 | 0.13796 |
| 应收到（场景单位） | 0.024349 | 0.004311 |
| 倍率 | **32.00×** | **32.00×** |

后果：Mahalanobis 度量的 `Q(x,x)` 从应有的 1.41e4 变成 13.79（差 3 个数量级），
邻域点的排序与分离面选取全部改变——**凸单元的几何是错的**，即使尺寸看起来仍和上游同量级
（面积主要由障碍面与边界环决定，所以此前的面积统计没能暴露它）。

修复：在推理与训练缓存两处统一构造 `shape4_scene = [log(a_s), log(b_s), cos2θ, sin2θ]`；
传给 DiT 的 `ellipse` Token 已用正确的 `ellipse_scene`，**不要重复转换**。
验收测试 `test_region_builder_receives_scene_axes` 通过拦截 `build_region_table` 的入参断言（已就位）。

### P0-3 桥接单元被 `cells[:Q]` 截断 —— **确认**

`local_regions.py:188-193`：`cell_A = np.zeros((Q, max_faces, 2))`，随后 `for i, (s, A, b) in enumerate(cells[:Q])`。

实测（test 场景 0，`scope="full"`）：

```
corridor valid=True  base=64 bridge=1  cells=65  Q=64
table rows (Q) = 64 ; cells lost by cells[:Q] = 1
anchors dropped (largest -> near goal): [1.0]
```

被丢掉的正是 anchor = 1.0 的桥接单元——也就是**终点附近**的安全区域，最不该丢的那个。
本项目的配置本来就开着 `corridor.bridge.enabled`，验收报告也记录过桥接区域，因此这不是理论风险。

修复：约束表的行数改为 `len(cells)`（含桥接），batch 时按最大行数 padding 并用 mask 保证 padding 不产生约束；
增加"桥接场景下 `table 行数 == corridor.num_cells`"的测试（`test_no_bridge_cell_is_truncated`，已就位）。

### P0-4 训练 batch 中 per-sample anchors 被覆盖 —— **确认**

`local_regions.py:233-238`：循环里 `anchors = np.asarray(table["anchors"])` 每次覆盖，
最后把**最后一个样本**的 anchors 传给整个 batch 的 `build_constraint_pack_from_regions`。

实测（两张 anchors 不同的表 A/B）：

```
sample 0 pack == pack built with table_a anchors ? False
sample 0 pack == pack built with table_b anchors ? True
```

即样本 0 的约束被按样本 1 的参数区间分配。`test_pack_from_region_table_matches_batch_build`
之所以没发现，是因为它把同一张表复制了两份（复审这条判断正确）。

修复：每个样本独立构包后再按 piece/face padding 合并（`_merge_packs` 已有能力），
或把 `build_constraint_pack_from_regions` 的 anchors 扩成 `[B,R]`。
验收测试 `test_batch_pack_uses_per_sample_anchors`（已就位）。

---

## 二、P1（安全逻辑）：5 条确认，1 条部分成立

### P1-1 碰撞检测只查离散点，不查线段 —— **确认**

`_generate_safety` 用 `self._decode(pred)`（128 点）与 `collision_mask`（逐点查栅格）。

实测：128 点平均间距 **0.348 m**，而地图分辨率 **0.25 m**。
间距大于一格 → 相邻两点都在自由格、中间却穿过一堵 0.25 m 的墙时**检测不到**，于是不会产生碰撞区间、
不会构建对应凸区域、ALM 也没有那一段的约束。最终 512 点检查（间距 0.086 m）能发现，但那时安全几何已经生成完了。

修复方向（按复审建议取第二方案）：改用 Astar-Diffusion 已有的连续线段穿越检测
（`guidance/collision.py::segment_collision_intervals` / `trace_segment_cells`，已迁移可用），
在曲线上按线段求碰撞区间，再统一换算成弧长比例。
验收测试 `test_thin_wall_between_samples_is_detected`（已就位）。

### P1-2 无有效局部凸区域时 ALM 仍执行 —— **确认，且我此前的文档写错了**

`build_region_table()` 即使没有任何局部凸单元，也会给 64 个查询点生成**地图边界 box cell**，
所以 `pack.piece_mask.sum() > 0` 恒成立。

实测：`corridor_info.local_cells = 0`、`spans = []`，`alm_stats` 仍有 **5** 条。

`docs/acceptance_report.md` 原写"ALM 自动 no-op（`piece_mask` 全 False）"，**已更正**（见该文档 §6 的更正框）。
这不只是文档问题，行为本身也不符合设计文档 §8"没有有效局部凸约束时，跳过 ALM，直接执行 DDIM 更新"。

修复：门控改为 `corridor_info["local_cells"] > 0`（或显式区分"障碍凸约束"与"地图边界约束"两类行）；
`L_corridor` 也应对二者分别统计。验收测试 `test_alm_skipped_when_no_local_convex_cell`（已就位）。

### P1-3 ALM 输入的预测端点未锁定 —— **确认，实测漂移极大**

`pred = self.model(...)` 之后直接 `bspline_alm_correct(pred, pack, ...)`；
`hard_control_endpoints` 只作用在 DDIM 状态 `q` 上（第 143、229 行），不在 `pred` 上。

实测（test 场景 1，12 步，ALM 5 步，每一步 ALM 输入端）：

```
step 0..4: 起点误差 0.31662 scene = 10.13 m ; 终点误差 1.04625 scene = 33.48 m
最终 controls 端点 vs S/G: 0.000000 / 0.000000
```

也就是说：ALM 一直在修正一条**起点偏 10 m、终点偏 33 m** 的曲线，其约束验证针对的也不是本任务的起终点；
之后 DDIM 再把端点硬扳回去，ALM 的可行性与证书不再成立。
（当前是 toy 权重，漂移幅度会随训练下降，但"网络没有端点硬约束"这一结构性缺陷与权重无关。）

修复：每步前向之后立刻 `pred = self.codec.hard_control_endpoints(pred, start_goal)`，再做解码/搜索/ALM。
验收测试 `test_prediction_endpoints_locked_before_alm`（已就位）。

### P1-4 搜索失败被静默当成"可行搜索路径" —— **确认**

`sampler_conditional_dit.py:290-293`：`path_m is None` → `path_m = guide_m; status = "fallback_to_guide"`，
随后照常做椭圆、安全 Token、凸区域、ALM。而 `guide_m` 是无安全条件分支的预测轨迹，**可能本身有碰撞**。

修复：搜索失败应标记 `search_failed`，不生成安全条件、不把该样本标记为安全，走明确的失败分支。
验收测试 `test_failed_search_is_not_reported_as_a_search_path`（已就位）。

### P1-5 ALM 验证未形成安全门控 —— **事实成立，"违反设计"部分成立**

代码事实：`valid_after`（约束可行率 ≥0.999 且密集碰撞 = 0）只写入 `alm_stats`，
`q0_used = q_safe` 无条件执行（`sampler_conditional_dit.py:207-219`）。

设计文档 §7.3 的原文是："未通过验证：记录约束违反和碰撞结果，**不能将其标记为安全轨迹**；
**可按现有采样器的失败处理策略继续执行**，但不得生成任何历史反馈条件。"
→ 文档**允许**继续执行，只要求"不得标记为安全"。
本实现的最终 `valid` 只由最终 512 点碰撞 + 端点位置决定，碰撞了就是 False，因此**没有谎报安全**；
但确实没有"ALM 让事情变坏就回退"的机制，也没有把"本步 ALM 是否通过"记进输出。

我把它定为**部分成立**：不是违规，而是"缺一层保守回退"。建议（不改 ALM 算法）：
- 记录 `alm_accepted` / `alm_rejected` 计数；
- 若 `dense_collision_after > dense_collision_before` 或 `max_violation_after > before`，保留 `pred` 而非 `q_safe`。

### P1-6 碰撞区间 → 查询索引用"索引比例"映射 —— **确认为风险，未造成崩溃**

`local_regions.collision_intervals` 假设"当前轨迹的参数进度"与"搜索路径的弧长进度"一致；
而两者分别是 B-spline 参数采样与弧长重采样，几何不同。所以可能出现"碰撞在 A，安全区域建在 B"。

修复方向（采纳复审建议）：把碰撞点投影到**搜索路径的最近点**，用该点的弧长参数取查询索引，
而不是用碰撞点在自身轨迹上的序号比例。

### P1-7 `max_spans` 静默截断 —— **确认**

`spans = collision_intervals(...)[:max_spans]`（默认 3），超过 3 段的碰撞区间被直接丢弃且无计数。
修复：记录 `dropped_spans`、`dropped_collision_length`；超限时可合并为覆盖区间或提高上限。

---

## 三、P2（工程）：4 条中 3 条确认、1 条补充说明

### P2-1 端点朝向没有验收 —— **确认**

网络接收 `heading/heading_valid`，但最终验收 `_endpoint_error()` 只查位置；`valid` 只查碰撞率 + 位置。
`codec.basis_derivative_at()` 已经可用，可直接算 `Ṗ(0)=B'(0)Q`、`Ṗ(1)=B'(1)Q` 并与输入朝向比较。
修复：输出 `heading_error_start/goal_deg`，纳入 `valid` 与报告。

### P2-2 CUDA 设备迁移 —— **确认（按代码静态核验；本机无 GPU 无法实跑）**

```
codec buffer device: cpu      factory moves codec?  False
sampler moves heading?  False  cache fingerprint?    False
```

具体缺口：`heading/heading_valid` 未 `.to(device)`；`BSplineCodec` 的 `basis/pinv` 是 CPU buffer 且从未 `.to(device)`；
`pack_from_tables` 产出 CPU 张量而 `corridor_loss` 用 GPU 控制点；`build_safety_cache` 构造的张量均在 CPU。
本机 torch 是 `2.14.0+cpu`，**所以 `--device cuda` 从未被验证过**。README/报告已写明"CPU only"，但这不等于支持 CUDA。
修复：pipeline 初始化统一设备；输入统一搬移；pack 参与计算前显式 `.to(device)`；补一个 CUDA smoke（有 GPU 时运行）。

### P2-3 README 缺少参考仓库初始化步骤 + 缓存无指纹 —— **确认**

`.gitignore` 里有三个克隆命令（注释形式），但 README 的"运行"章节直接从 `audit_references.py` 开始，
新机器 clone 后会因为缺 `references/` 与 `v3_best.pt` 直接报错。
缓存 `safety_*.npz` 也没有记录配置/数据集/checkpoint 指纹，改配置后容易误用旧缓存。

修复：README 增加"环境 → 拉参考仓库 → 校验 checkpoint → 配置检查 → 测试 → 运行"完整步骤，并提供一键脚本；
缓存写入 `config_hash + dataset_split + checkpoint_md5 + codec 参数` 元信息，加载时校验不匹配即重建。

### P2-4 碰撞检测采样密度（同 P1-1）—— 确认；训练侧同样受影响

`collision_loss` 用 `codec.decode_controls(pred_c)` 默认 128 点（间距 ~0.35 m），
与推理侧同一问题。修复 P1-1 时一并处理（训练侧可对线段中点补采样，或同样用线段-栅格穿越的软惩罚）。

---

## 四、复审中我判断"不是缺陷"或需要澄清的点

| 复审说法 | 我的核验 |
|---|---|
| "`shape4` 应转换，但椭圆 Token 已正确" | 正确。`sampler` 里 `ellipse_scene = [centers_scene, axes_scene, θ]` 确实用了转换后的轴长；问题只在 `shape4` 这一条旁路。 |
| "ALM 在使用错误端点下的曲线做约束验证" | 成立（P1-3），且实测漂移达 10 m / 33 m，比复审估计严重。 |
| "搜索模块把端点替换成原始 S/G，可能又变碰撞" | 成立。`search.py` 先对**栅格节点**做 `path_is_valid`，再把 `world[0]/world[-1]` 覆盖成原始 S/G，**没有对最终重采样路径复查**。同意加入"重采样后再验证一次"。 |
| "训练 full / 推理 local 是设计问题，不是 bug" | 同意，已记录在 `docs/design_review.md` §3.1，且列为待确认项。 |
| "C=32/64 是设计问题" | 同意，`docs/design_review.md` §3.2 已给实测表格与建议（长路径 64），等待决策。 |
| "`test_pack_from_region_table_matches_batch_build` 用同一表复制所以漏检" | 完全正确，是测试设计缺陷，已在新 harness 里用两张不同 anchors 的表覆盖。 |

---

## 五、建议的修复顺序（与复审一致，略作细化）

**P0（先修，再重新生成缓存与重跑统计）**

1. `losses.collision_loss` 单位（1 行 + 解析测试）。
2. `shape4` → 场景坐标（推理 + 训练缓存两处；抽一个 `ellipse_shape4_to_scene()` 公共函数避免再次漏改）。
3. 约束表行数改为 `len(cells)`（含桥接）+ padding mask。
4. `pack_from_tables` 支持 per-sample anchors。

**P1（安全语义）**

5. 碰撞区间改用线段穿越检测 + 投影到搜索路径求查询索引。
6. `max_spans` 截断计数。
7. 前向后立刻锁端点（P1-3）。
8. `local_cells == 0` 时跳过 ALM（P1-2）；`L_corridor` 区分障碍约束与边界约束。
9. ALM 保守回退 + `alm_accepted/rejected` 记录（P1-5）。
10. 搜索失败 → `search_failed`，不生成安全条件（P1-4）；重采样后再验证路径。

**P2（工程/实验）**

11. 朝向误差纳入验收；12. 设备统一（或明确 CPU-only）；13. README 初始化步骤 + 缓存指纹；14. 重跑全部测试与统计；15. 小规模收敛验证后再谈正式训练。

**完成判据**：`tests/test_known_defects.py` 中 8 个 `xfail` 全部转为通过（标记删除），
且 `docs/acceptance_report.md` 的所有数字由修复后的缓存重新生成。
