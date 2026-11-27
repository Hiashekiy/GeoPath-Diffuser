# 分支间监督机制重设计：为什么旧的对齐无效，以及四种替代方案的结果（20 epoch）

## 1. 问题回顾：旧的 `align` 结构上无效

原机制：`L_align = MSE(u_interior, sg(c_interior))`，权重 0.5，端到端一次反向。

三处硬伤（都有实测支撑）：

1. **点目标 → 必然被平均**。它与 `L_u = MSE(u, gt)`（权重 1.0）作用于同一个量，极小点是
   `u = (gt + 0.5·c)/1.5`，即沿 c 的方向只能走到 1/3；实测投影系数 `k = ⟨u−gt, c−gt⟩/‖c−gt‖² = 0.52`。
2. **90% 的误差不在这个方向上**：u 的误差 1.90 m 里只有约 0.55 m 落在 `(c−gt)` 方向，即使完全复制 c
   也只能拿走那一小块。
3. **无条件蒸馏**：老师的输出不总是更好，噪声也会被复制。

训练日志里 `align/u` 在 step 2,000 之后长期停在 0.3–1.3 且没有下降趋势（间距 1.0–2.7 m，
与 u 自己的误差同量级），说明两支从未真正收敛到一起。

## 2. 四个实验臂（均 20 epoch / 15,240 step，同 seed、同数据顺序、同安全缓存）

| 臂 | 无安全支路的监督 | 对齐位置 | 无安全支路的走廊标签 |
|---|---|---|---|
| `baseline_gt` | GT（权重 1.0） | 端点修正后（旧） | 无 |
| `d1_distill` | **只有蒸馏**（`u_target=safety`，权重 1.0） | **原始输出** | 无 |
| `d2_distill_corridor` | 只有蒸馏（权重 1.0） | 原始输出 | **0.15** |
| `d3_gt_corridor` | GT（权重 1.0） | 原始输出 | **0.15** |

设计要点：

- **去掉 GT 锚**（`d1`/`d2`）让蒸馏成为唯一的位置目标，理论上消除“两目标平均”的均衡；
- **对齐移到原始输出**：端点调制会把两支同时拽到同一个 S/G，在修正后的多边形上算对齐等于在测“修正项”，
  掩盖真实分歧；
- **特权走廊**（`d2`/`d3`）：把训练期才有的走廊约束（`Aβ ≤ b`）作为**标签**加到无安全支路上——它是集合
  约束，不会被平均掉，而与“像 GT”不冲突（一个管位置偏好，一个管可行域）。推理时该支路仍然只吃地图。

## 3. 闭环结果（val 256 样本、DDIM 50 步、两支都不给安全，`map_only`）

| 臂 | 无碰撞率 | 碰撞占比均值 | chamfer (m) | 控制点 RMSE (m) | 长度比 |
|---|---|---|---|---|---|
| `baseline_gt` | 0.594 | 0.1192 | 1.450 | 1.535 | 0.942 |
| `d1_distill` | 0.613 | 0.1174 | 1.564 | 1.667 | 0.940 |
| `d2_distill_corridor` | **0.645** | **0.0880** | 1.395 | 1.592 | **0.968** |
| `d3_gt_corridor` | 0.609 | 0.1046 | **1.332** | **1.508** | 0.948 |

逐例配对（对 `baseline_gt`，n=256；符号检验）：

| 臂 | 碰撞率 | chamfer | 长度误差 |
|---|---|---|---|
| `d1_distill` | 52/59/145，p=0.57（无差别） | 154/102，**p=0.001（更差）** | 157/99，**p<0.001（更差）** |
| `d2_distill_corridor` | 37/**76**/143，**p<0.001（更好）** | 123/133，p=0.57 | 111/**145**，**p=0.039（更好）** |
| `d3_gt_corridor` | 44/**65**/147，p=0.055（临界） | 88/**168**，**p<0.001（更好）** | 113/143，p=0.070（临界） |

## 4. 参考：同一批权重把安全打开（`safety_mode=full`，64 样本）

| 臂 | 无碰撞率 | 碰撞占比 | chamfer | 控制点 RMSE | 长度比 |
|---|---|---|---|---|---|
| `baseline_gt` | 0.906 | 0.0079 | 1.201 | 1.412 | 1.020 |
| `d2_distill_corridor` | 0.891 | 0.0130 | **0.957** | **1.074** | 1.001 |
| `d3_gt_corridor` | **0.922** | 0.0074 | **0.928** | **1.080** | 1.003 |

## 5. 结论

1. **“去 GT 换蒸馏”本身不成立**：`d1` 的碰撞改善在噪声内（p=0.57），几何显著变差（chamfer p=0.001）。
   机制上它也解释得通——去掉 GT 锚后学生没有上界约束，它的盲区会经**共享权重**回灌给老师：
   安全支路的 teacher-forced 误差从 0.00162 涨到 0.00237、走廊违规从 0.018 涨到 0.044，
   `safety_gate` 从 0.148 掉到 0.161→实测 0.161（`d1`）/0.107（`d2`）。
2. **真正的有效成分是“把走廊约束当标签喂给无安全支路”**：`d3`（保留 GT + 走廊）在几何上全面变好
   （chamfer −8%、控制点 −1.8%，都显著），碰撞也降了 12%（p=0.055 临界）；`d2`（再叠加蒸馏）
   把碰撞进一步压到 −26%（p<0.001），代价是控制点误差 +3.7%、安全支路本身被拖差一点。
3. 两条路径的差异说明：**走廊标签管“合法”，蒸馏管“选哪条”**，两者互补但会互相牵扯。
   `d2` 偏安全、`d3` 偏准确；`d3` 在“开安全”档位也是最好的（92.2%），且教师没有被拖差。
4. 都还没到“无安全推理可以替代安全链路”的程度（最好 64.5%，而开安全是 89–92%），
   但方向与统计显著性是明确的，值得继续训练。

## 6. 后续

- `d2` 正在用 `--resume` 从 epoch 20 续训到 **100 epoch**（AdamW 动量延续，epoch 从 21 接着数）；
- 其后把 `baseline_gt` 同样续到 100 epoch，保证 100 epoch 的对照公平；
- 若 `d2` 在 100 epoch 时仍领先但几何有代价，则考虑“部分 GT 锚 + 蒸馏 + 走廊”的组合臂。

## 7. 复现

```powershell
# 基线（20 epoch，已存在）
python scripts/train_dual_branch.py --config configs/c64_longpath.yaml --dataset real `
  --real-split train --val-split val --device cuda --epochs 20 --batch-size 16 `
  --steps 15240 --ckpt-dir outputs/checkpoints/ab_dual --ckpt-every 10 --ckpt-prefix abd `
  --tag ab_dual --seed 1
# d1：去 GT，只蒸馏
python scripts/train_dual_branch.py ... --u-target safety --align-space raw `
  --lambda-align 1.0 --lambda-safe-u 0.0 --ckpt-dir outputs/checkpoints/ab_distill_raw `
  --ckpt-prefix d1 --tag ab_distill_raw
# d2：去 GT + 特权走廊
python scripts/train_dual_branch.py ... --u-target safety --align-space raw `
  --lambda-align 1.0 --lambda-safe-u 0.15 --ckpt-dir outputs/checkpoints/ab_distill_safeu `
  --ckpt-prefix d2 --tag ab_distill_safeu
# d3：保留 GT + 特权走廊
python scripts/train_dual_branch.py ... --u-target gt --align-space raw `
  --lambda-align 0.5 --lambda-safe-u 0.15 --ckpt-dir outputs/checkpoints/ab_gt_safeu `
  --ckpt-prefix d3 --tag ab_gt_safeu
# 续训到 100 epoch
python scripts/train_dual_branch.py ... --epochs 80 --steps 60960 `
  --resume outputs/checkpoints/ab_distill_safeu/last.pt --u-target safety `
  --align-space raw --lambda-align 1.0 --lambda-safe-u 0.15 `
  --ckpt-dir outputs/checkpoints/ab_distill_safeu_e100 --ckpt-prefix d2 `
  --tag ab_distill_safeu_e100
```

产物：`outputs/reports/ablation_transfer_{all,d2_safety_on,d3_safety_on}.json`、
`outputs/figures/ablation_transfer/`、`outputs/logs/ab_{distill_raw,distill_safeu,gt_safeu}.log`、
权重 `outputs/checkpoints/ab_{distill_raw,distill_safeu,gt_safeu}/`。
