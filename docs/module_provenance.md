# 模块迁移来源记录（Module provenance）

本文件记录每一个从 `references/` 复制到 `src/gp_diffuser/` 的源文件。规则：

1. 参考仓库只读，任何改动都不发生在 `references/` 内（三个仓库都是 clean 状态，见 `docs/reference_code_audit.md`）。
2. 只复制需要的文件，不复制无关代码与依赖。
3. 复制后只做 **接口适配**（import 路径、可选依赖、新增入口），不改变原算法实现；改动逐条列出。
4. 不通过 `sys.path` 注入、软链接或运行时 import `references/` 来"假装"迁移。

## 1. 迁移总表

| # | 原始仓库 | 分支 | Commit | 原始文件 | 新文件 | 是否改算法逻辑 |
|---|---|---|---|---|---|---|
| 1 | Astar-Diffusion | `main` | `39c31d42ebb13baf4f644e0a9136641232fd30f8` | `src/guidance_field/grid.py` | `src/gp_diffuser/guidance/grid.py` | 否（逐字节复制） |
| 2 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/config.py` | `src/gp_diffuser/guidance/config.py` | 否 |
| 3 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/propagation.py` | `src/gp_diffuser/guidance/propagation.py` | 否（仅 scipy 改为可选导入 + `auto` 后端回退 heap） |
| 4 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/field.py` | `src/gp_diffuser/guidance/field.py` | 否 |
| 5 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/policies.py` | `src/gp_diffuser/guidance/policies.py` | 否 |
| 6 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/paths.py` | `src/gp_diffuser/guidance/paths.py` | 否 |
| 7 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/collision.py` | `src/gp_diffuser/guidance/collision.py` | 否 |
| 8 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/sources.py` | `src/gp_diffuser/guidance/sources.py` | 否 |
| 9 | Astar-Diffusion | `main` | `39c31d4…` | `src/guidance_field/normalize.py` | `src/gp_diffuser/guidance/normalize.py` | 否 |
| 10 | PathSafe-Ellipse | `feature/v2.1-centerline` | `90d1f3ca8f7b5d6ffe359ce93380f07e732c5575` | `ellipsenet/models/resunet.py` | `src/gp_diffuser/map_encoder/resunet.py` | 否（仅**新增** `forward_with_features`，原 `forward` 未改） |
| 11 | PathSafe-Ellipse | `feature/v2.1-centerline` | `90d1f3c…` | `ellipsenet/models/psenet.py` | `src/gp_diffuser/safety/psenet.py` | 否（仅 import 路径） |
| 12 | PathSafe-Ellipse | `feature/v2.1-centerline` | `90d1f3c…` | `ellipsenet/geometry/sampling.py` | `src/gp_diffuser/safety/sampling.py` | 否（缺失常量 `SDF_OUTSIDE_M` 按仓库自有证据补回，见 §3.1） |
| 13 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a9763781ae0c491a9f6bee732da7d5ee68cb` | `src/geometry/bspline.py` | `src/gp_diffuser/trajectory/bspline.py` | 否（逐字节复制） |
| 14 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/geometry/convex_region.py` | `src/gp_diffuser/safety/convex_region.py` | 否（scipy 改为可选；生产路径 `halfspaces_to_vertices` 本就不用 scipy） |
| 15 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/geometry/safety_corridor.py` | `src/gp_diffuser/safety/safety_corridor.py` | 否（仅 `_default_gamma_fn` 的 import 目标） |
| 16 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/geometry/bspline_constraints.py` | `src/gp_diffuser/safety/bspline_constraints.py` | 否（仅 import 路径） |
| 17 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/diffusion/schedule.py` | `src/gp_diffuser/diffusion/schedule.py` | 否（逐字节复制） |
| 18 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/diffusion/bspline_alm.py` | `src/gp_diffuser/diffusion/bspline_alm.py` | 否（仅 `SCENE_TO_METER` 的 import 路径） |
| 19 | TrajSafe-Diffuser | `v4-map16-fbdelta-safeloss` | `14a5a97…` | `src/models/trajsafe/geometry.py`（只取两个函数） | `src/gp_diffuser/trajectory/dense_path.py` | 否（`dense_arclength` + `gather_dense_path_points` 原样，V5 模型类未复制） |

**未复制**（有意排除）：TrajSafe V5 网络（`src/models/trajsafe/planner_v5.py`、`planner.py`、`blocks.py`、`fusion.py`、`heads.py`、`feedback.py`、`boundary.py`、`encoders.py`）、`src/models/common/*`、`src/diffusion/sampler.py`、`sampler_v5.py`、`alm_guidance.py`（旧 waypoint ALM）、`sdf_guidance.py`、`carla_*`、`grid_search/*`（focal A*，本项目用 Astar-Diffusion）、`losses/losses.py`（V5 损失集合）；PathSafe 的 `train.py`/`evaluate.py`/`losses.py`/`polytopes.py`/`visualize*.py`；Astar 的 `scenes.py`/`trajectories.py`/`experiment.py`/`metrics.py`/`viz.py`/`cli.py`/`field-guided-search/`（同一算法的第二份打包）。

## 1.1 新增（非迁移）的文件

这些文件是评审后为修复缺陷新写的，**不含参考仓库代码**，因此不在迁移表内：

| 新文件 | 作用 | 对应缺陷 |
|---|---|---|
| safety/collision_query.py | 线段级碰撞区间（Amanatides-Woo，复用迁移来的 	race_segment_cells）+ 碰撞位置到搜索路径的弧长投影 | P1-1 / P1-6 |
| safety/pack_utils.py | 逐样本 BSplineConstraintPack 的 padding 合并、设备迁移 | P0-3 / P0-4 / P2-2 |
| safety/psenet_inputs.py | LiDAR 射线重建（PathSafe 数据加载器不在仓库内） | — |
| safety/local_regions.py、safety/safety_features.py、models/*、guidance/search.py、diffusion/sampler_conditional_dit.py、	raining/*、evaluation/* | 新项目自有模块（详见 README §2） | — |

## 2. 逐项适配明细

### 2.1 Astar-Diffusion → `gp_diffuser/guidance/`

* **import**：模块名与文件名保持与上游一致（`grid/config/propagation/field/policies/paths/collision/sources/normalize`），所有相对 import 原样可用，**零改动**。
* **`propagation.py`**：`scipy.sparse` 改为可选导入。本机 Python 3.12 环境没有 scipy，`propagate_auto` 在缺少 scipy 时回退到 `propagate_heap`（同一 exact max-Dijkstra，只是更慢）。算法、计数、所有权语义不变。
* **新增**：`guidance/search.py`（本文件不属于参考代码）负责世界坐标 ↔ 栅格坐标换算、把引导轨迹交给上游 `sources.build_sources_drop`（M0）、构建加权场、选择策略、执行 `priority_search`、失败回退 `astar_search`、再按弧长重采样为 128 点。上游求解器未被改写。

### 2.2 PathSafe-Ellipse → `gp_diffuser/map_encoder/` + `gp_diffuser/safety/`

* **`resunet.py`**：新增 `forward_with_features(x)`，返回 `{"bottleneck": enc3 输出, "full": head(up(...)), "skips": ...}`。原 `forward()` 一行未改，checkpoint 兼容性不变（新增方法无参数）。
* **`psenet.py`**：`from ..geometry.sampling` → `from .sampling`；`from .resunet` → `from ..map_encoder.resunet`。类名、层名、前向数学不变，`v3_best.pt` 仍可 `strict=True` 加载。
* **`sampling.py`**：上游 `from ..data.sdf import SDF_OUTSIDE_M`，而 `ellipsenet/data/` 在 PathSafe-Ellipse 里被 `.gitignore` 排除（`data/` 整目录未跟踪），该模块在克隆中不存在。此处按仓库自有证据补回常量（见 §3.1），公式不变。
* **`safety_features.py`（新增）**：`PathSafeEllipseNet(PSENet)` 子类提供
  - `encode_maps(occ, sdf)`（一次前向给出两组瓶颈 + 两组全分辨率特征）；
  - `forward_from_features(batch, features)`（逐行镜像 `PSENet.forward`，只是把两次 encoder 调用换成外部特征）；
  - `predict_ellipses(...)`（arg-max 方向 + 形状头，`decode_argmax` 语义不变）。
  用**子类**而不是改上游文件，保证迁移文件与上游一致、权重加载严格匹配。
  另有 `SafetyConditionGenerator`（把搜索路径 → 64 椭圆 → `[cx,cy,a,b,cos2θ,sin2θ]`）与 `resample_arclength`。

### 2.3 TrajSafe-Diffuser → `gp_diffuser/trajectory/` + `gp_diffuser/safety/` + `gp_diffuser/diffusion/`

* **`bspline.py`**：逐字节复制。`_DEFAULT_KNOTS` 指向上游 `data/carla_v1/bspline_knots.npy`，本仓库不存在该文件，`knots: auto`（本项目的用法）会走 `resolve_knots` 生成 clamped-uniform 结点向量——这正是上游为"改 C 而无需重新生成结点文件"设计的路径，行为不变。
* **`convex_region.py`**：`scipy.spatial` 改可选导入。生产路径 `halfspaces_to_vertices` 使用两两面求交的向量化实现（上游注释明确说明 qhull 只作参考实现），因此无 scipy 时功能完整。
* **`safety_corridor.py`**：`_default_gamma_fn()` 原本延迟 import `..models.trajsafe.geometry`（V5 模型包），改为 `..trajectory.dense_path`（同一个函数）。桥接区域数学、重叠判据、`SCENE_TO_METER` 全部不变。
* **`bspline_constraints.py`**：`from .bspline import ...` → `from ..trajectory.bspline import ...`。分段/Bezier 抽取/掩码逻辑不变。
* **`bspline_alm.py`**：`from ..geometry.safety_corridor import SCENE_TO_METER` → `from ..safety.safety_corridor import ...`。ALM 目标函数、proximal 内迭代、Cholesky 求解、曲线位移限幅、对偶变量更新与继承全部不变；本文档不改 `rho / step_size / inner_steps / tol / max_curve_step_scene` 的语义，配置值直接沿用上游 `configs/config_v5_c32q64.yaml` 的 `alm:` 段。
* **`schedule.py`**：逐字节复制（`squaredcos_cap_v2` = 上游 cosine 调度，T=1000）。

## 3. 参考仓库自身的缺失与取证

### 3.1 `SDF_OUTSIDE_M`（PathSafe-Ellipse）

* 事实：`ellipsenet/geometry/sampling.py` 与 `scripts/build_distance_fields.py` 都 `from ellipsenet.data.sdf import SDF_OUTSIDE_M`，但 `ellipsenet/data/` 未被 git 跟踪（`.gitignore` 的 `data/` 规则）。
* 取证（不靠印象）：
  * `scripts/build_distance_fields.py:167` → `"sdf_outside_value_m": -1.0`；
  * `docs/report.md:59` → “地图外一律视为 blocked：… 固定常数 `SDF_OUTSIDE_M = -1.0 m`”。
* 处理：在迁移后的 `sampling.py` 内以 `SDF_OUTSIDE_M = -1.0` 定义，并注明来源。

### 3.2 数据加载器（PathSafe-Ellipse）

* `ellipsenet/train.py` 依赖 `ellipsenet.data.dataset`（`TrajectoryDataset` / `collate` / `load_index`），该模块同样不在仓库中（`data/` 被忽略）。因此 `lidar` / `ray_dir` / `ray_valid` 三个张量无法直接从代码读出。
* 取证：
  * `docs/report.md:82-83`：`ray_directions` 由 loader 派生，`angle_k = 2πk/16`、`u=(cos,sin)`；`ray_valid = lidar < max_range − 1 mm`；
  * `docs/report.md:105-109`：`lidar_ranges` 是“沿射线走到**第一个 blocked 格边界**的距离”，量程即点机器人沿该方向可走距离；
  * `results/metrics/data_contract.json`：`lidar_ranges` `[N,16]` 米、`max_range_m = 9.0`、`resolution_m = 0.25`、`sdf` 米、`path_xy` 米；
  * `docs/report.md:101`：LiDAR 真实命中比例均值 0.338。
* 处理：`src/gp_diffuser/safety/psenet_inputs.py` 按上述语义实现 Amanatides-Woo 射线步进（到第一个 blocked 格**边界**的距离、越界即 blocked）。本项目合成场景实测命中比例 0.383，与上游审计值 0.338 同一量级。

### 3.3 数据集（两仓库）

* PathSafe 的 `data/` 与 TrajSafe 的 `processed_root`（`D:/ProjectDirectory/Neural-IRISDiffuser/data/...`）都不在本工作区。因此本项目生成**合成数据集**（`training/dataset.py`），严格对齐 PathSafe 文档化的数据契约（256²、0.25 m、SDF 米、16 射线、9 m 量程、`SDF_OUTSIDE_M=-1`），并在 README 中标注"非原始数据集"。

## 4. 迁移后的验证

* `tests/test_map_encoder.py`：瓶颈/全分辨率形状、`enc3` 直出、一次前向只调用一次 encoder、冻结、训练一步后权重不变、checkpoint 参数数 5,041,093。
* `tests/test_alm.py`：ALM 降违反量、端点保持、对偶变量返回并可继承、全掩码 pack 为 no-op、padding 行不影响结果。
* `tests/test_safety_geometry.py`：椭圆 64 个、`cos2t/sin2t` 单位化、轴长在上游边界内、凸区域法向单位化、无碰撞时只有 4 面 box cell、碰撞区间映射到 ≥`min_queries` 的查询跨度、padding 不改变 pack。
* `scripts/audit_references.py` → `outputs/reports/reference_audit.json`：三分支/commit/工作区状态、被引用文件存在性、checkpoint 张量形状、特征形状。
