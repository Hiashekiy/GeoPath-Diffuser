# 参考代码审查报告（Phase 1）

审查对象：`references/` 下三个仓库（只读）。审查时间点即下表 commit；审查内容包括目录结构、被引用文件、真实接口签名、张量形状、依赖、可直接复用模块与必须适配的接口。

配套机器可读结果：`outputs/reports/reference_audit.json`（由 `python scripts/audit_references.py` 生成）。

## 1. 仓库清单与工作区状态

| 仓库 | 分支（实际 = 期望） | Commit | 最近提交 | 工作区 | 文件数 | 体积 |
|---|---|---|---|---|---|---|
| Astar-Diffusion | `main` | `39c31d42ebb13baf4f644e0a9136641232fd30f8` | `39c31d4` 2026-11-19 `core: allow a per-call distance override in GuidanceField.search` | clean | 37 | 0.28 MB |
| PathSafe-Ellipse | `feature/v2.1-centerline` | `90d1f3ca8f7b5d6ffe359ce93380f07e732c5575` | `90d1f3c` 2026-11-24 `Convex regions: build one for every ellipse, including the ones that collide` | clean | 106 | 32.57 MB |
| TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a9763781ae0c491a9f6bee732da7d5ee68cb` | `14a5a97` 2026-11-18 `docs(paper): RAL v2 正文与数据更新…` | clean | 1543 | 384.28 MB |

三个仓库全部完整克隆、分支与要求一致、工作区 clean，未做任何修改。

## 2. Astar-Diffusion（`main`）

### 2.1 结构

```text
src/guidance_field/
  config.py        Config dataclass（lam=0.02/64 格、num_source_samples=64、source_method=M0、
                   alphas=(0.25,0.5,0.75)、BUCKETS、FIELD_CLIP、COLLISION_BANDS）
  grid.py          Grid：occ==0 → free；预计算 4 邻接表；BFS 测地距离、连通分量、叶剪枝、detour 剖面
  propagation.py   F(x)=max_i[w_i - lam*d_free] 的 5 个后端：heap / vectorized / bucket / scipy / auto
  field.py         build_weighted_field(grid, sources, kind, lam)、source_weights（progress 不重归一化）
  normalize.py     normalize_field（按 FIELD_CLIP）、normalize_distance（Manhattan / Dmax）、prepare_scalars
  policies.py      B0 astar_search；B1..B5 的 priority(r,c) + priority_search（不读 g）；path_is_valid
  paths.py         astar / weighted astar / greedy；resample_arc_length（**吸附到栅格节点**）；候选池
  collision.py     Amanatides-Woo 段-格穿越、detect_collision_intervals、measure_collision_ratio、
                   project_to_nearest_free、traversed_obstacle_cells
  sources.py       M0 drop / M1 project / M2 bridge（每段桥接继承原始 progress，u 轴留洞）
  __init__.py 导出，cli/viz/metrics/experiment/scenes/trajectories 属于实验脚手架
field-guided-search/  同一算法的第二种打包（pyproject 包名 field_guided_search）——未使用
```

### 2.2 关键接口（真实签名）

```python
Grid(occupancy int8)                                   # FREE=0, OCCUPIED=1
build_weighted_field(grid, {"positions_rc": (K,2) int, "progress": (K,) float},
                     kind="linear", lam=0.02, method="scipy") -> {"field","owner","weights",...}
prepare_scalars(grid, field, weight, goal_rc) -> {"F_hat","d_hat","distance","D_max","clip"}
LinearFusionPolicy(scalars, alpha=0.25).priority(r, c)          # B3
priority_search(grid, start_rc, goal_rc, policy, cutoff=0) -> {"status","expanded","path": list[int],...}
astar_search(grid, start_rc, goal_rc)                          # B0 fallback
resample_arc_length(nodes, width, m) -> (m,2) int              # 吸附栅格
detect_collision_intervals(grid, path_rc_float) -> {intervals: [{s_in,s_out,u_in,u_out,entry_cell,exit_cell}]}
```

### 2.3 与原设计的对应关系

* 设计文档 §5.4 的 `F(x)=max_i[w(u_i)-λ d_free]`、progress 不重归一化、"障碍内点不得作源"——分别对应 `field.py`、`field.source_weights`、`sources.build_sources_drop`（M0，默认）。
* 文档 §5.4 "搜索失败时使用普通 A* 作回退" → `policies.astar_search`（B0）。
* 文档 §5.5 "按弧长重采样、首尾点严格对应 S/G" → 上游 `resample_arc_length` 是**栅格吸附**版本，且首尾取 `t=0,t=total` 后仍 `rint`；本项目在世界坐标里做连续弧长重采样（`guidance/search.py::resample_polyline`），并把首尾强制写成精确 S/G。
* 上游 `docs/results.md` 的实测最优组合（B3 linear fusion、`linear` 场权、α=0.25、扩张 0.204×A*、路径长 4.2 %）被选为本项目默认策略。
* λ 依赖地图尺度（上游 64 格用 0.02）；本项目 256 格，默认 λ=0.01（λ·L 落在上游说明的 1…4 区间）。

## 3. PathSafe-Ellipse（`feature/v2.1-centerline`）

### 3.1 结构

```text
ellipsenet/models/resunet.py   ResBlock/Down/Up/ResUNet(base=24, depths=[1,1,1,1], out_ch=128)
ellipsenet/models/psenet.py    PSENet：两个 ResUNet + 9 点占据查询 + 16 射线特征 + ray/path 两层 Transformer
                               + Direction(16)/Ratio(1)/Shape(4) 三个头 + 16 候选（arg-max 选方向）
ellipsenet/geometry/sampling.py  world_to_grid / sample_feature / sample_sdf（越界=-1 m）/ 椭圆模板
ellipsenet/geometry/{losses,polytopes,exact}.py  几何损失与多面体工具（未迁移）
ellipsenet/{train,evaluate,visualize*}.py        训练/评估脚手架（未迁移）
configs/v3.yaml, checkpoints/{v3_best.pt,v3_config.json,v3_run_meta.json,v3_train_log.csv}
docs/report.md (57 KB), results/metrics/data_contract.json
```

### 3.2 实测（不是文档转述）

* `checkpoints/v3_best.pt`：20.25 MB，keys `['model','cfg','epoch','step','metrics','seed']`，231 个张量，**参数 5,041,093**，与 `v3_run_meta.json.params` 一致；checkpoint 内嵌完整 `cfg`（训练配置），因此新项目直接从 checkpoint 重建网络，不猜 YAML。
* 编码器通道：`stem (24,1,3,3)` → `down1 (48,24,…)` → `down2 (96,48,…)` → `down3 (192,96,…)` → `enc3 (192,192,…)`；`head (128,24,1,1)`。
* 因此 256×256 输入下：**瓶颈 `enc3` = `[B,192,32,32]`**，**全分辨率 `head` 输出 = `[B,128,256,256]`**——与设计文档 §3.2 表格完全一致。
* `configs/v3.yaml`：`ch=128, resunet_base=24, resunet_depths=[1,1,1,1], lidar_rays=16, d_clip=8.0, attn_heads=4, ray_attn_layers=2, path_attn_layers=2, occ_token_spacing_m=0.5, ray_sample_lambdas=[0.25,0.5,0.75], candidate_ray_margin_m=0.01, shape_a_min_m=0.15, shape_a_max_m=6.0, shape_b_min_m=0.05`（与 `checkpoints/v3_config.json` 一致）。
* 数据契约（`results/metrics/data_contract.json`）：`occupancy [256,256] {0,1}`、`signed_distance_m [256,256]` 米、`path_xy [N,2]` 米、`lidar_ranges [N,16]` 米（`max_range 9.0`）、`ray_directions [16,2]`、`ray_valid [N,16]`、`resolution_m 0.25`。
* 文档化语义（`docs/report.md`）：`SDF_OUTSIDE_M = -1.0 m`；LiDAR 到**第一个 blocked 格边界**；`ray_valid = lidar < max_range − 1 mm`；LiDAR 真实命中比例均值 0.338。
* 上游自身测试集成绩（`results/figures/grid/summary.json`）：`exact_safe_rate_per_point 0.950`、`corridor_success_rate 0.542`、`boundary_cover 0.447`、`area_p50 103.8 m²`。
  → 记住 `corridor_success_rate ≈ 0.54`：**该模块本身**就有近一半样本完不成凸走廊闭环，这为第 5 节的实测结果提供了参照。

### 3.3 缺失与风险

| 现象 | 证据 | 处理 |
|---|---|---|
| `ellipsenet/data/` 未跟踪 | `.gitignore` 的 `data/`；`train.py` import `ellipsenet.data.dataset` 失败 | 数据加载器/射线生成按 §3.2 语义在新项目重写 |
| `SDF_OUTSIDE_M` 缺失 | `sampling.py` 从 `..data.sdf` 导入 | 以 -1.0 补回（`build_distance_fields.py:167`、`report.md:59`） |
| 数据集不在仓库 | `data/` 被忽略 | 新项目用合成数据集（对齐上述契约） |
| 权重在合成地图上椭圆退化 | 本项目实测 `a≈0.7–1.1 m, b≈0.13 m` | 记录为已知限制（见 `docs/design_review.md` §4） |

## 4. TrajSafe-Diffuser（`v4-map16-fbdelta-safeloss`）

### 4.1 与"只作参考"要求相符的部分（已迁移）

```text
src/geometry/bspline.py             BSplineCodec(degree=3, num_controls=32, curve_points=128)
                                    basis/derivative_op/interior_pinv 均为 buffer；fit_curve_to_controls 端点约束最小二乘
src/geometry/bspline_constraints.py responsibility_intervals(tau) → exact_subdivision → Bezier 抽取
                                    E_{l,r} → A_j E_{l,r} Q <= b_j；build_constraint_pack(_from_regions)
src/geometry/convex_region.py       EllipseRegionBuilder(map[1,1,H,W], cfg).build_from_ellipse/build_from_metric
                                    Neural-IRIS Mahalanobis 排序 + 贪心分离面 + 局部边界环 + 地图边界 4 面
src/geometry/safety_corridor.py     重叠判据 min_overlap_ratio、点种子桥接（GCOPTER convexCover 思路）、
                                    SafetyCorridor(cells, anchors(), valid, failure_reason)
src/diffusion/bspline_alm.py        bspline_alm_correct(q_ref, pack, codec, lam, config, inner_steps)
                                    目标 = lp‖B(Q-Qref)‖² + ls‖D2 B(Q-Qref)‖²；proximal 内迭代 + Cholesky；
                                    曲线位移限幅 max_curve_step_scene；对偶 lam 按 pack 排序可跨反向步继承
src/diffusion/schedule.py           NoiseSchedule（squaredcos_cap_v2 / linear），q_sample
src/models/trajsafe/geometry.py     dense_arclength / gather_dense_path_points（桥接种子用）
```

关键张量形状（实测）：`pack.extraction [B,P,4,C]`、`pack.piece_A [B,P,F,2]`、`pack.piece_b [B,P,F]`、`pack.face_mask [B,P,F]`、`pack.piece_mask [B,P]`；`lam [B,P,4,F]`。

### 4.2 明确不继承的部分

`src/models/trajsafe/planner_v5.py`（TrajBlock / Control Backbone / Final Denoiser / Skeleton Encoder）、`planner.py`、`blocks.py`、`fusion.py`、`heads.py`、`feedback.py`（历史修正反馈）、`src/models/common/scene_cnn.py`（第二套地图编码器）、`src/diffusion/sampler.py`（V5 采样器，含 feedback/warmup 逻辑）、`alm_guidance.py`（旧 waypoint ALM）、`configs/config_v5_c32q64.yaml` 的 `model:` 段（D=128、4 heads、layer 划分）与 `diffusion: {timesteps:16, inference_steps:4}`。

仅沿用该 YAML 中与"算法"相关的数值：`corridor`（`min_overlap_ratio 0.10`、`bridge.max_bridge_per_gap 1`、`local.pad_queries 4`、`local.min_queries 8`、`region.safety_margin 0.02`、`obstacle_window_half 0.35`、`ellipse_axis_min 0.002`、`ellipse_axis_max 2.0`）与 `alm`（`rho 5.0`、`step_size 0.05`、`inner_steps 6`、`proximity_weight 1.0`、`correction_smooth_weight 0.1`、`constraint_tol 1e-3`、`max_curve_step_scene 0.20`、`inherit_dual_across_reverse_steps true`、`dense_verify_points 512`）。

### 4.3 依赖

`numpy`、`torch`、`scipy`（仅 `convex_region` 的 qhull 参考实现与 `propagation` 的 scipy 后端；两者都可选）。本机 Python 3.12 环境**没有 scipy**，因此迁移时把两处改成可选导入，`auto` 传播后端回退 `heap`（同一 exact 算法）。

## 5. 复用决策汇总

| 需求 | 来源 | 方式 |
|---|---|---|
| 冻结地图编码器 + 瓶颈特征 | PathSafe `resunet.py`/`psenet.py` | 迁移 + 新增 `forward_with_features`；权重来自 `v3_best.pt` |
| 椭圆安全特征 | PathSafe `psenet.py` + `sampling.py` | 迁移 + `forward_from_features` 子类；`no_grad` |
| 引导场搜索 | Astar-Diffusion 全 `src/guidance_field` | 迁移 + 新写 `guidance/search.py` 适配层 |
| B-spline 表示 | TrajSafe `geometry/bspline.py` | 迁移（逐字节） |
| 凸区域 / 走廊 / 约束包 | TrajSafe `convex_region`/`safety_corridor`/`bspline_constraints` | 迁移（仅 import） |
| ALM | TrajSafe `diffusion/bspline_alm.py` | 迁移（仅 import），配置沿用上游 |
| DDIM 更新 | TrajSafe `schedule.py` + `sampler.py` 的 `q = sa_s*q0 + s1_s*eps` | 调度迁移；更新式在新采样器中重写（V5 采样器不可复制） |
| Diffusion 主干 | 无（禁止复用 V5） | 新建标准 Conditional DiT-S |

详细逐文件记录见 `docs/module_provenance.md`。
