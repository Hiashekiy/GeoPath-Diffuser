# 设计文档审查与决策记录

审查对象：`docs/安全条件扩散轨迹规划方案_Conditional-DiT网络修正版.md`（全文 1055 行源文本 / 1519 行渲染，已逐节读完）。
本文件只记录**文档内部不一致**、**文档与参考代码不一致**、以及**实施中必须做的取舍**。按你的要求，冲突不自行掩盖：每条都写明"文档怎么说 / 代码怎么说 / 本项目怎么做 / 需要你确认的点"。

## 1. 文档内部不一致

### 1.1 DDIM 步数：4 步 vs 50 步

* §8 开头：“采用当前四步 DDIM 推理结构。”
* §4.1 表格：`采样调度 DDIM 反向步数 50（初始）`，并注明“不沿用旧版 4 步设置”。
* §12 配置：`sampler: ddim`、`inference_steps: 50`，说明“后续可对比 20、50、100；不要未经验证直接恢复 4 步采样”。

**判断**：§8 的那句是 V5 文档残留（V5 用 `timesteps:16 / inference_steps:4`）。
**本项目**：默认 `sampling.steps = 50`，`--steps` 可覆盖（测试里覆盖过 4/6/8/10/12）；文档 §4.1/§12 的禁令优先。
**需你确认**：是否同意把 §8 的“四步”视为笔误。

### 1.2 "两个分支共享 Control Encoder" vs "不得保留 Control Backbone"

* §1.1 第 5 条：“两个分支共享同一套 Diffusion Backbone、**Control Encoder** 和轨迹输出头。”
* §4.6：“不存在两套独立主干，也不需要额外的 Safety Backbone。”
* §4.8：明确“不得在实现时继续沿用或改名保留 `Control Backbone / TrajBlock`”。

**判断**：§1.1 的 “Control Encoder” 是 V5 术语残留，与 §4.8 的禁令直接冲突。
**本项目**：共享的是「控制点 Token 投影 + 12 层 DiT Block + 输出头」，不存在独立的 Control Encoder / Safety Backbone。`tests/test_conditional_dit.py::test_no_v5_modules_are_used` 断言了这一点。

### 1.3 §7.1 的公式与 §7 说明"以代码为准"的差异

* §7.1 写 `A_j B_k Q ≤ b_j`（用 B-spline 基矩阵在某采样点 $k$ 上取值）。
* 参考实现（`bspline_constraints.py`）用的是**精确 Bezier 控制点抽取**：把曲线按"责任区间 ∪ 结点"精确细分，每段取 4 个 Bezier 控制点 `b_{l,r}=E_{l,r}Q`，再施加 `A_j E_{l,r} Q ≤ b_j`。由于 Bezier 凸包性，这给出**连续**安全证书，而不是 128 点采样近似。

**本项目**：按你"第 7 章数学表达只是设计说明，实现以 TrajSafe 指定分支代码为准"的要求，采用代码实现（`build_constraint_pack_from_regions`），ALM 的 `constraint_state` 消费的正是 `pack.extraction / piece_A / piece_b`。差异记录在此。

### 1.4 §6.4 "densify 仅用于缺口" 的具体载体

* §6.4 提到 `pad_queries=4`、`min_queries=8`、`corridor.min_overlap_ratio`、`densify` 仅在局部区域存在连接缺口时使用。
* 上游有两种补缺手段：`safety_corridor.build_safety_corridor(bridge=...)`（点种子桥接区域）和 `ellipse_densify.densify_ellipses(...)`（插值加密椭圆）。

**本项目**：只迁移并启用 `bridge`（点种子桥接），**未迁移** `ellipse_densify`。理由：桥接区域就是为"相邻凸单元重叠不足"设计的；加密椭圆会改变 PathSafe 输出集合，与"安全几何只生成一次、不重复生成椭圆"的约束更易冲突。`corridor.densify` 未在 `configs/default.yaml` 中出现（不静默忽略，明确缺失）。

## 2. 文档与参考代码不一致（以代码为准，已核实）

| 文档表述 | 参考代码实际 | 处理 |
|---|---|---|
| §3.2 表格：瓶颈 `[B,192,32,32]`、全分辨率 `[B,128,256,256]` | 实测一致（`enc3` 192 通道 @32²，`head` 128 通道 @256²） | 直接使用 |
| §3.1：“输出特征通道：128” | 128 是 `head` 输出通道；`enc3` 是 192 | 两个数都保留，分属不同层级 |
| §4.3 建图条件 `Conv1x1([B_occ;B_sdf])`，输入 384 → 384 | 与实现一致（`MapFeatureProjection`） | 直接使用 |
| §6.2 “从 16 个候选方向中选取” | `dir_head: Linear(128,16)`，`decode_argmax` 取 arg-max | 一致 |
| §9.3 一致性损失 `‖Q̂_u − sg(Q̂_c)‖²` | 本项目 `alignment_loss` 完全一致，`tests/test_dual_branch_training.py::test_alignment_target_is_stop_gradient` 验证 `pred_c.grad is None` | 一致 |
| §11.4 模型接口 `model(q_t, map_memory, cond, t, safety_condition, heading, heading_valid)` | 本项目 `ConditionalDiT.forward` 签名完全一致（另加可选的 `condition_memory/condition_mask` 以避免两路重复构建条件记忆） | 一致 |
| §12 `beta_schedule: cosine` | 上游 `squaredcos_cap_v2`（即 cosine） | 用 `squaredcos_cap_v2` |
| 文档 §5.1 训练用 GT 路径作引导源 | 与实现一致（`safety_cache.build_safety_cache` 第一步） | 一致 |

## 3. 实施中必须做的取舍（需要你知道）

### 3.1 训练用走廊是"全路径"，推理用走廊是"碰撞区间局部"

* 文档 §6.4 要求"凸区域只在碰撞区间及其邻域构建"；§9.1 又说训练时由搜索路径 + 椭圆 + 障碍物构建凸区域并缓存。
* 现实：训练样本用的是 **GT 轨迹**（§5.1 明确 GT 只用于构造安全条件），而 GT 曲线本身无碰撞 → 按"碰撞区间"取局部会得到**空约束集**，`L_corridor` 恒为 0，安全损失失效。

**本项目**：`corridor.training_scope: "full"`（每个椭圆一个凸单元，离线缓存）；推理严格 `scope="local"`（碰撞区间 + `pad_queries=4`，`min_queries=8`，最多 `max_spans=3`）。
**需你确认**：是否接受"训练全量 / 推理局部"；替代方案是"训练时用 GT + 扰动轨迹求碰撞区间"，但那会引入一个额外的超参（扰动量）并让缓存与模型行为耦合。

### 3.2 控制点数 32 → 长路径拟合误差实测偏大

`scripts/check_bspline_fit.py`（输出 `outputs/reports/bspline_fit.md`）：

| 路径长度 (m) | n | max err C=32 | max err C=48 | max err C=64 | mean err C=32 |
|---|---|---|---|---|---|
| 20–40 | 9 | 0.2069 | 0.1475 | 0.0874 | 0.1370 |
| 40–60 | 9 | 0.3290 | 0.2089 | 0.1387 | 0.1791 |
| 60–∞ | 18 | **0.6162** | 0.3930 | 0.2799 | 0.1923 |

（单位米；含项目数据集路径与合成波浪长路径。）

**判断**：长路径（>60 m）最大重建误差 0.62 m，超过机器人半径 0.5 m 的量级，可能把曲线推过墙角。文档 §4.1 明确说"如果长路径拟合误差明显增大，应统一提高控制点数（例如 48 或 64）"。
**本项目**：`configs/default.yaml` 保持文档的初始值 **32**（不改你已定的初始配置）；同时提供 `configs/c64_longpath.yaml`（C=64，其余不变）供长路径实验。
**需你确认**：正式训练前是否统一切到 64。

### 3.3 数据集：原数据不可得，改用合成场景

PathSafe 的 `data/`、TrajSafe 的 `processed_root`（指向 `Neural-IRISDiffuser/data/...`）都不在本工作区。本项目自带合成场景生成器，严格对齐 PathSafe 文档化契约（256²、0.25 m、SDF 米、16 射线、9 m 量程、越界 -1 m）。所有实验数字都在合成数据上取得，**不可与论文数字直接比较**。

### 3.4 冻结 PathSafe 在合成地图上椭圆偏"薄"

实测（合成 val 场景，64 个查询）：`a` 中位数 0.69–1.08 m，`b` 中位数 **0.13 m**；而上游在自有数据上的凸单元面积 `area_p50 = 103.8 m²`，本项目实测凸单元面积中位数 **30–156 m²**、单元合法率 100 %（`bounded` 全通过）。也就是说：椭圆本身偏薄，但由障碍物+边界环+Mahalanobis 排序生成的**凸单元尺寸与上游同量级**。

走廊闭环率实测：train 缓存 8/16、val 1/4（约 50 %），失败原因 `bridge_failed`（7）与 `bridge_region_invalid`（1）。上游自己的 `corridor_success_rate = 0.542`，属同一水平。
**处理**：不修改 PathSafe 输出、不人为放大椭圆（那会篡改冻结模块语义）；走廊未闭环的样本退化为"仅地图边界 box cell"，ALM 对它们自动 no-op（`pack.piece_mask` 全 False），采样继续进行。

### 3.5 scipy 缺失

本机 Python 3.12 环境无 scipy（`pip install scipy` 在此网络下超时）。迁移时把 `propagation.py` 的 scipy 后端与 `convex_region.py` 的 qhull 参考实现改为可选导入；引导场搜索走 `heap`（同一 exact max-Dijkstra），凸多边形走 `halfspaces_to_vertices` 的两两面求交（上游生产路径）。已安装 scipy 的环境会自动走 scipy 后端，无需改代码。

### 3.6 一处主动补齐：`collision_expand` 语义

文档 §5.3 要求"如果 `blocked` 已经是机器人中心可行性地图，不能重复膨胀"。本项目合成数据里 `blocked` 是**已按机器人半径膨胀过一次**的地图（`dataset.robot_radius_cells = 2` → 0.5 m），因此 `guidance.inflation_cells = 0`，不做二次膨胀。真实数据集接入时应按数据元信息设置该值。

## 4. 与"明确禁止"的逐条对照

| 禁止项 | 本项目的状态 | 证据 |
|---|---|---|
| 先训练无安全分支再训练有安全分支 | 不存在 | 训练器单循环双前向；`test_no_phased_training_switches` 断言源码无 `warmup/base_only/safety_activation/stage_/phase` |
| Base-only Warmup / Safety Activation 阶段 | 不存在 | 同上 |
| 两套独立 Diffusion 网络 | 不存在 | 同一 `model` 实例两次调用；`test_both_branches_share_q_t_and_timestep` |
| 历史修正反馈训练 | 不存在 | 采样器不向网络传 `feedback/history/delta/correction`；`test_no_history_feedback_reaches_the_network` |
| 只同步更新两分支、无显式联合损失 | 不是 | `L = λc·Lc + λu·Lu + λa·Lalign + (λcol·Lcol + λcorr·Lcorr)`；`test_joint_loss_is_the_documented_weighted_sum` |
| 沿用 V5 网络结构 | 不存在 | 新建 `models/conditional_dit.py`；迁移清单里没有 `planner_v5/blocks/fusion/heads` |
| 重新建立一套地图编码器 | 不存在 | 只有一份 `PathSafeEllipseNet` 实例；DiT 内无任何 `Conv2d` |
| 用 Decoder 上采样后的全分辨率特征作地图条件 | 不存在 | `map_projection` 只吃 `[192,32,32]` 瓶颈；`test_bottleneck_is_enc3_not_decoder_output` |
| 重新实现 ALM / QP / 投影 | 不存在 | 直接调用迁移来的 `bspline_alm_correct`，配置沿用上游 |
| 每个去噪步重新搜索 / 重新生成椭圆 / 重建走廊 | 不存在 | `test_search_and_ellipses_are_never_recomputed`、`test_same_pack_is_reused_by_every_guided_step` |
| 在任何旧仓库上开发 | 不存在 | 三个仓库工作区 clean；所有新代码只在 `GeoPath-Diffuser/` 内 |
