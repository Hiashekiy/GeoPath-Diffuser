# GeoPath-Diffuser 数据集 v1（geopath_v1）制作报告

依据：`GeoPath-Diffuser_数据集制作方案.md`（Dataset v1）。原始数据：`data/{trajectories,distance_fields,maps}`（PathSafe-Ellipse V3，只读）。

## 1. 盘点与筛选（stage 1）

| 项 | 值 |
|---|---|
| 轨迹文件（磁盘） | 50,648 |
| SDF 索引记录 | 50,648 |
| 唯一地图（SDF npz） | 17,038 |
| 分辨率分布（全部） | 0.3125 m: 17,134 / 0.1875 m: 16,758 / **0.25 m: 16,756** |
| 0.25 m 候选 | 16,756 |
| **合格样本** | **16,756（0 排除）** |
| 合格 map_group | 30 |
| occupancy 与 SDF `blocked` 一致 | 16,756 / 16,756 |
| 耗时 | 154.7 s |

筛选依据：`distance_fields/index.jsonl` 的 `resolution_m`（容差 1e-6）＋每条轨迹实际 `occupancy` 形状 ＋ SDF npz 的 `resolution_m`/`blocked`，**不依据文件名**；同时校验 `meta_json.physical_field_m == 64.0` 与 `source_resolution_m == 0.25`。48 m / 80 m 规格全部排除，未做任何 resize/插值。

产物：`data/geopath_v1/{manifest.json, eligible_samples.jsonl, rejected_samples.jsonl}`、`reports/data_audit.{json,md}`。

## 2. 划分（map_group 80/10/10，seed 0）

| split | 样本 | 地图组 |
|---|---|---|
| train | 13,435 | 24 |
| val | 1,673 | 3 |
| test | 1,648 | 3 |

同一 `map_group` 的全部轨迹只进入一个集合；`split_manifest.json` 记录实际分组映射，可由 seed 0 复现。

## 3. GT 转换与 B-spline 质量审计

流程：`path_xy` → 128 点弧长重采样（米）→ `SceneTransform(64, 256)` 转场景坐标 → `BSplineCodec.fit_curve_to_controls` → 解码 → 稠密碰撞/误差审计。

### 3.1 控制点数评估（方案 §6.3 要求）

Pilot（1000 train + 200 val 分层抽样）在 C=32 与 C=64 下的对照：

| 控制点 | 保留 | 拟合碰撞排除 | fit max err (m) mean / p90 / p99 / max |
|---|---|---|---|
| 32 | 928 (77.3%) | 272 | 0.053 / 0.096 / 0.129 / 0.146 |
| **64** | **1,096 (91.3%)** | **104** | **0.023 / 0.043 / 0.069 / 0.102** |

原因用数值确认：被排除样本的**原始路径**沿 512 点采样最小 SDF 恒为正（0.06–0.13 m），而**拟合曲线**的 SDF 掉到 0.012 m 甚至 −0.094 m；拟合最大偏差 67–272 mm > 走廊净空 60–130 mm。

结论：C=32 在真实数据上损失 22.7% 样本，C=64 降到 8.7% 且误差减半，因此**首版统一采用 C=64**（`configs/c64_longpath.yaml`），不做 C=32/C=64 混合。该决定可由一个 `--config` 参数回退。

### 3.2 Pilot 数据集（C=64）

| 项 | 值 |
|---|---|
| 转换 | 1,200 → 保留 1,096（train 915 / val 181） |
| 拟合碰撞排除 | 104（8.7%） |
| fit RMSE / max error | 均值 0.023 m / 最大 0.078 m（抽样 96 个） |
| 端点误差 | 0（控制点首尾强制等于 S/T） |
| 朝向误差 | 起终点切向由导数基精确计算，计入质量字段 |
| 稠密碰撞 | 保留样本全为 0 |
| 误差阈值 | `max(0.5 m, p99)`，本次 0.5 m（无样本触发） |

产物：`processed/{train,val}/*.npz`（occupancy/sdf/valid_mask/path_xy_m/path_xy_scene/controls_scene/start_goal_*/heading/heading_valid/meta_json）、`train.jsonl`、`val.jsonl`、`reports/{bspline_fit.json,collision_check.json,conversion_rejections.jsonl,dataset_summary.md}`。

## 4. 安全条件缓存（冻结模块，`training_scope: full`）

| split | 样本 | 搜索成功 | 走廊闭环 | 局部凸单元均值 | 桥接 | 凸单元总数 | 指纹 |
|---|---|---|---|---|---|---|---|
| train | 915 | **100%** (915/915) | 86.1% | 55.3 | 151 | 50,565 | `335b9cdde7ec9a58` |
| val | 181 | **100%** (181/181) | 90.1% | 57.8 | 35 | 10,462 | `740cec0b07661f29` |

* 缓存指纹包含：数据集版本、split、**样本 ID 集合摘要**（train `7e2600138df6729a` / val `de9c39dccf865a9e`）、地图规格、控制点规格、PathSafe checkpoint 身份、搜索/椭圆/走廊配置。换数据集或改配置后旧缓存会被拒绝（已由 `test_cache_fingerprint_rejects_stale` 与 pilot 校验 D 覆盖）。
* 与合成数据（走廊闭环 ~50%）相比，真实地图上冻结 PathSafe-EllipseNet 的走廊闭环率显著更高（86–90%），说明安全几何在真实数据上可用。
* 搜索失败样本数为 0；实现仍保留 `search_failed` 标记路径（合成数据测试覆盖）。

## 5. Pilot 验收（`scripts/validate_real_dataset.py`）

**16/16 通过**（`reports/pilot_validation.md`）：

* A. 每张地图 256×256、SDF 有符号且与 occupancy 一致；
* B. 路径 128 点、控制点 64、首尾控制点精确等于起终点、朝向有效；拟合曲线无碰撞、误差 < 0.30 m；分辨率 0.25；
* C. train/val 缓存指纹匹配、样本数与索引一一对应（915/915、181/181）、椭圆 64×6 且有限；
* D. 真实样本上完成一次真实 `forward/backward/optimizer.step`（loss 0.36，双分支均非零），且换数据集后缓存被拒绝。

## 6. 新增加载器与脚本

| 文件 | 作用 |
|---|---|
| `src/gp_diffuser/training/real_dataset.py` | `RealTrajectoryDataset`：`__getitem__`（训练张量）与 `raw_scene`（安全缓存几何），带按 `map_key` 的 LRU 地图缓存与 split digest |
| `scripts/audit_real_dataset.py` | 盘点 + 规格筛选 + SDF 一致性 |
| `scripts/prepare_real_dataset.py` | 划分 + GT 转换 + 拟合/碰撞审计（支持 `--pilot`、`--config`、断点续跑） |
| `scripts/build_real_safety_cache.py` | 真实数据安全缓存（复用现有 `build_safety_cache`） |
| `scripts/validate_real_dataset.py` | Pilot 验收 A–D |
| `scripts/train_dual_branch.py` | 新增 `--dataset real --real-split/--val-split/--limit` |

网络主体、损失定义、双分支优化策略、PathSafe checkpoint、凸区域与 ALM 均未修改。

## 7. Pilot 训练结果（真实数据，C=64，CPU）

```powershell
& $py scripts\train_dual_branch.py --config configs/c64_longpath.yaml --dataset real \
        --real-split train --val-split val --steps 900 --epochs 4 --batch-size 4 \
        --save outputs/checkpoints/pilot_c64.pt
```

* 900 步 / 4 epoch / 2,014 s（≈2.2 s/步，CPU），可训练参数 55,662,350，冻结编码器 0 可训练参数。
* checkpoint：`outputs/checkpoints/pilot_c64.pt`（含 `architecture_version`），训练史：`outputs/reports/train_history.json`。

### 7.1 训练损失（前 60 步 vs 末 60 步，训练集）

| 指标 | 起始 | 结束 | 变化 |
|---|---|---|---|
| total | 4.462 | 2.158 | −52% |
| L_u (Map-only) | 0.2137 | 0.1936 | −9% |
| L_c (Safety) | 0.2155 | 0.1743 | −19% |
| L_align | 0.0008 | 0.0817 | 分支开始分化 |
| L_collision | 3.987 | 1.716 | −57% |
| L_corridor | 0.0452 | 0.0329 | −27% |
| α_l 均值 | 0.1002 | 0.1085 | 门控缓慢打开 |

`L_c` 的下降幅度大于 `L_u`：安全分支确实开始利用椭圆条件；一致性损失同时把 Map-only 分支往安全分支拉（`L_align` 由 ~0 升到 0.08）。

### 7.2 验证集对照（同 seed，64 样本，`eval_step`）

| 指标 | 未训练 | 900 步 | 变化 |
|---|---|---|---|
| L_u | 0.2193 | 0.2021 | −7.8% |
| L_c | 0.2193 | **0.1683** | **−23.3%** |
| L_align | 0.0000 | 0.1357 | 分化 |
| L_collision | 1.2492 | 1.4270 | **+14%（变差）** |
| L_corridor | 0.0512 | 0.0384 | −25% |

### 7.3 推理端检查（训练后 checkpoint，val 前 4 条，DDIM 20 步）

| idx | valid | 碰撞率 | 最小净空 (m) | 长度 (m) / GT | ALM 步 | 朝向误差 max |
|---|---|---|---|---|---|---|
| 0 | False | 0.09 | −1.42 | 42.9 / 47.0 | 9 | 158° |
| 1 | False | 0.09 | −1.45 | 143.7 / 30.2 | 9 | 179° |
| 2 | False | 0.09 | −1.42 | 54.7 / 48.2 | 0 | 54° |
| 3 | False | 0.38 | −3.30 | 147.1 / 52.8 | 9 | 152° |

**结论：训练管线与数据接口完全打通且损失明确下降，但 900 步 / 915 样本远未收敛**——
控制点空间误差仍约 0.2（scene²），推理轨迹碰撞率 9–38%、长度可达 GT 的 3 倍。
因此本轮**不宣称任何性能结论**，`pilot_c64.pt` 只能作为"接口可用"的证据。

已知训练动力学问题（损失定义未改，仅记录）：

1. `L_collision` 出现尖峰（训练中单批最高 20.1），平方惩罚在某些批次主导联合损失，导致 `L_u/L_c` 抖动；
2. `α_l` 900 步只从 0.100 升到 0.109，安全分支的注入强度增长很慢；
3. 验证集 `L_collision` 略升（+14%），说明当前学习率/权重下碰撞项尚未稳定收敛。

建议下一步（需你确认）：① 全量数据（约 1.28 万 train 样本）＋ 更多步数；② 针对碰撞项的学习率/权重调参（属于配置，不改损失定义）；③ 或先做小规模超参搜索再决定。

## 8. 未完成 / 下一步

1. **全量制作未执行**：当前只做了 Pilot（1,096 样本）。全量需 `prepare_real_dataset.py`（无 `--pilot`，约 15k 样本，预计 5–8 min 转换）＋ `build_real_safety_cache.py`（train 约 13k × 0.4 s ≈ 90 min）。
2. 104 个拟合碰撞样本（8.7%）已排除；若要求更高保留率，需要进一步评估 C=96/更高或允许极小穿透（本方案未擅自放开安全 GT）。
3. 训练观察：`loss_collision` 在训练早期出现尖峰（0 → 5.3 → 0.8），说明预测偶尔深穿障碍物，平方惩罚主导联合损失。损失定义未改，建议后续通过学习率/损失权重调参解决。
