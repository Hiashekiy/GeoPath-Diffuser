# 100 epoch 定级实验：机制 vs 训练量（2026-11-27）

对应 `docs/HANDOFF.md` §7「必须做的下一步」。所有数字来自 `outputs/reports/*.json`，
配对检验一律为**同 split、同索引、同噪声**的逐例符号检验（双侧精确二项，平局剔除），
由 `scripts/report_ablation.py` 直接从 JSON 生成，不手工誊写。

---

## 0. 结论先行

1. **d2 到 100 epoch 仍在收敛，但只收敛在"合规"这一个维度**：无碰撞率 0.727 → **0.750**、
   碰撞占比 0.0598 → **0.0510**（配对 22:52，**p=6.4e-4**）；chamfer（125:131，p=0.76）与
   长度误差（121:135，p=0.42）**都没有变化**。
2. **几何代价是真实存在的**：控制点 RMSE 1.249 → 1.350（配对 157:99，**p=3e-4**，@100 更差）。
3. **关键对照做出来了（替代被叫停的 100 vs 100）：把"从没碰过安全链路"的纯无安全臂
   `ab_map_only` 同样训到 100 epoch（20 → 100），它的无碰撞率纹丝不动**：
   0.586 → 0.590（配对 53:59，**p=0.637**，即完全平手），只有几何变好
   （chamfer 1.580 → **1.334**，配对 187:69，**p=9.5e-14**；控制点 1.711 → 1.448）。
   → **多训练 5 倍买得到"像 GT"，买不到"合规"；合规来自安全链路的监督机制。**
4. **同一批 256 个样本、同一噪声下，d2@100 对 am@100（纯无安全、同样 100 epoch）**：
   无碰撞率 **0.750 vs 0.590**（配对 **17:91，p=2.1e-13**），而 chamfer **打平**
   （127:129，p=0.95）。→ 差距**不是**来自"训练量"或"形状更好"，而是来自合规性本身。
5. **判据成立 → d2 配方定为正式配置**：无碰撞率 0.750 ≥ 0.70 ✓、控制点 1.350 ≤ 1.535 ✓。
   但在**开安全**档位下 @80 略优于 @100（0.906 vs 0.875；长度误差 45:19，p=7e-4），
   所以 `d2_ep080.pt` 保留为"追求形状/最终质量"时的另一档权重。

---

## 1. 方法

- 数据：`data/geopath_v1` val（1,472 样本），固定取索引 0..255（`--seed 0`）。
- 采样：`safety_mode=map_only`（两支都不给安全），DDIM **50 步**，batch 8；`--device cuda`。
- 脚本：`scripts/compare_safety_ablation.py`（一次调用多臂 ⇒ 同数据序、同初始噪声）。
- **等价性核对**：`--device cpu`（2 步）与 `--device cuda`（50 步）在 d2@80/d2@100 上给出
  **完全相同的聚合指标**（0.727/0.0598/1.147/1.249 与 0.750/0.0510/1.179/1.350），
  所以 CPU 预跑的趋势表可以直接采信。

## 2. d2 收敛趋势（ep40 → ep100）

自洽四臂同次运行（`outputs/reports/prelim_d2_trend_cpu256.json`，n=256、2 步、CPU）：

| 权重 | 无碰撞率 | 碰撞占比 | chamfer (m) | 控制点 RMSE (m) | 长度比 |
|---|---|---|---|---|---|
| d2@40 | 0.688 | 0.0730 | 1.244 | 1.430 | 0.971 |
| d2@60 | 0.750 | 0.0661 | 1.308 | 1.448 | 0.984 |
| d2@80 | 0.727 | 0.0597 | **1.147** | **1.249** | 0.975 |
| d2@100 | **0.750** | **0.0510** | 1.179 | 1.350 | 0.988 |

碰撞占比单调下降；无碰撞率在 0.69–0.75 抖动；chamfer **无**单调趋势，ep80 最好。
配对（对 d2@40）：碰撞 34:52（p=0.066）、29:61（**p=9.7e-4**）、23:63（**p=1.9e-5**）。

## 3. 权威数字：d2@80 vs d2@100（GPU，50 步，n=256）

`outputs/reports/ablation_transfer_d2_ep100.json`：

| 权重 | 有效 | 无碰撞率 | 碰撞占比 | chamfer (m) | 控制点 RMSE (m) | 长度比 |
|---|---|---|---|---|---|---|
| d2@80 | 0.727 | 0.727 | 0.0598 | **1.147** | **1.249** | 0.975 |
| d2@100 | 0.750 | **0.750** | **0.0510** | 1.179 | 1.350 | 0.988 |

| 指标 | 更好 / 更差 / 平 | p | 判读 |
|---|---|---|---|
| 碰撞占比 | 22 / 52 / 182 | **6.4e-4** | @100 显著更合规 |
| chamfer | 125 / 131 / 0 | 0.76 | 无差别 |
| 控制点 RMSE | 157 / 99 / 0 | **3e-4** | @100 显著更差 |
| 长度误差 | 121 / 135 / 0 | 0.42 | 无差别 |

## 4. 纯无安全臂 20 → 100：训练量买不到合规（本轮的机制判据）

`outputs/checkpoints/ab_map_only/last.pt`（`use_safety=False`，20 epoch，日志
`safety branch: OFF (ablation arm)`、`gate` 恒 0.100）用 `--no-safety` 续训到 100 epoch
（60,960 步，0.24 s/step，`outputs/checkpoints/ab_map_only_e100/`，日志 `ab_map_only_e100.log`）。
同上评测（`outputs/reports/ablation_map_only_ep100.json`，n=256、50 步）：

| 权重 | 无碰撞率 | 碰撞占比 | chamfer (m) | 控制点 RMSE (m) | 长度比 |
|---|---|---|---|---|---|
| am@20（纯无安全） | 0.586 | 0.1234 | 1.580 | 1.711 | 0.924 |
| am@100（纯无安全，5× 训练量） | 0.590 | 0.1207 | **1.334** | **1.448** | 0.922 |
| d2@100（安全链路训练） | **0.750** | **0.0510** | **1.179** | **1.350** | 0.988 |

| 对比 | 指标 | 更好 / 更差 / 平 | p | 判读 |
|---|---|---|---|---|
| am@100 vs am@20 | 碰撞占比 | 53 / 59 / 144 | 0.637 | **完全无进步** |
| am@100 vs am@20 | chamfer | 187 / 69 / 0 | **9.5e-14** | 几何大幅变好 |
| am@100 vs am@20 | 控制点 RMSE | 210 / 46 / 0 | **~1e-30** | 几何大幅变好 |
| am@100 vs d2@100 | 碰撞占比 | 17 / 91 / 148 | **2.1e-13** | d2 压倒性更合规 |
| am@100 vs d2@100 | chamfer | 127 / 129 / 0 | 0.95 | **打平** |
| am@100 vs d2@100 | 长度误差 | 105 / 151 / 0 | 0.0048 | d2 更准 |

**这是本轮最重要的一张表**：它把「机制」与「训练量」分开了——

- 纯无安全臂多训 5 倍，**几何显著变好、合规完全不动**（碰撞 p=0.637）；
  ​这与 HANDOFF §6E 的解释一致：它在学"以地图为条件的均值"，训练越久越准，但均值本身就会切障碍。
- 因此 d2 的合规优势（0.590 → 0.750）不可能由"多训练"解释：
  两臂训练量完全相同（都是 100 epoch / 60,960 步续训），差别只有监督配方。
- 且 d2 优于 am **不是**靠"形状更贴 GT"（chamfer 打平，p=0.95）——它靠的是"绕开障碍"。

## 5. 引导质量（HANDOFF §7 第 3 项）

`scripts/compare_safety_ablation.py` 新增逐样本字段：`guide_collision_rate`（自由率 = 1 − 它）、
`guide_chamfer_to_gt_m`、`guide_chamfer_to_final_m`、`homotopy_witness_ok`。
同伦见证 = 把引导曲线与最终曲线按 `(1-s)·guide + s·final`（`s∈[0,1]`，11 帧）线性混合，
**每帧都无碰撞**即证明两者可由一条自由同伦相连；有碰撞**不能**反证不同伦（单侧指标）。

`safety_mode=full`、64 样本、50 步（`outputs/reports/ablation_transfer_d2_ep100_safety_on_paired.json`）：

| 权重 | 引导自由率 | 引导↔GT chamfer | 引导↔最终 | 同伦见证率 |
|---|---|---|---|---|
| d2@80 | 0.916 | **1.046** | 0.598 | 0.594 |
| d2@100 | **0.936** | 1.102 | **0.583** | **0.703** |

→ @100 的引导**更"自由"、离最终轨迹更近、同伦见证率更高**（+11 个百分点），
但"像 GT"略差。按 HANDOFF 建议的"引导质量"口径，@100 作为**引导生成器**更好；
按"最终形状"口径，@80 更好。两者都是同一配方下的可选档位。

**顺带验证：2 步 = 50 步在 `map_only` 档位也成立**——同 64 样本同噪声，
`max|curve₂ − curve₅₀| = 0.0041 m`（轨迹长约 30 m），聚合指标 4 位小数一致，耗时 23.6 s vs 258 s。

## 6. 开安全档位（参考线，`safety_mode=full`，64 样本，50 步）

| 权重 | 无碰撞率 | 碰撞占比 | chamfer (m) | 控制点 RMSE (m) | 长度比 |
|---|---|---|---|---|---|
| d2@80 | **0.906** | **0.0099** | **0.732** | **0.786** | **0.995** |
| d2@100 | 0.875 | 0.0106 | 0.792 | 0.912 | 1.003 |

配对：碰撞 5:4（p≈1，平 55）、chamfer 38:26（p=0.077，临界）、长度误差 45:19（**p=7e-4**，@80 更好）。
→ **@100 在 map_only 档位学的"合规"，开安全后并没有额外收益**；推论：安全链路一旦接管，
引导只需"方向对"，@80 的引导已经够用（且更短、更贴 GT）。

## 7. 判据判定与配置定级（HANDOFF §7 第 4 项）

判据：无碰撞率 ≥ 0.70 且控制点误差 ≤ 1.535。

| 条件 | d2@100 | 结论 |
|---|---|---|
| 无碰撞率 ≥ 0.70 | **0.750** | ✅ |
| 控制点 RMSE ≤ 1.535 | **1.350** | ✅ |

→ **`d2` 配方定为正式训练配置**：`u_target=safety`、`align_space=raw`、`lambda_align=1.0`、
`lambda_safe_u=0.15`、100 epoch ⇒ `outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt`；
`d2_ep080.pt` 作为"开安全档位更优"的备选权重一并保留。

边界条件（不要外推）：0.750 是**引导**质量，同权重开安全是 0.906，两者不可互换。

## 8. 没做的事：`ab_dual` 的 100 vs 100

原计划的基线臂（`ab_dual`，GT + 旧 align@final，x3 权重 0.5）续训到 100 被**用户叫停**：
其 20 epoch 的 `last.pt` 不带优化器状态（`AdamW restarts from zero`），与 d2 的动量续训不等价，
对照会被污染（详见 `docs/HANDOFF.md` §7.05）。残留说明在 `outputs/logs/ab_dual_e100.log`。

**替代方案已实施**（§4）：改用同样"从没碰过安全链路"的 `ab_map_only` 续到 100，
它与 d2 的训练量、数据、步数完全一致，且两条臂的 20 epoch 起点都不带优化器状态以外的差别——
因此"机制 vs 训练量"的分离在这个对照里是干净的。

若仍要 `ab_dual` 的 100 vs 100：从 0 训 100 epoch（`--epochs 100 --steps 76200`，不 `--resume`），
约 6.6 h。

## 9. 复现命令

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
# 纯无安全臂 20 → 100（约 4.2 h；共 60,960 步）
& $py -u scripts/train_dual_branch.py --config configs/c64_longpath.yaml --dataset real `
   --real-split train --val-split val --device cuda --epochs 80 --batch-size 16 --steps 60960 `
   --log-every 200 --ckpt-every 10 --seed 1 `
   --resume outputs/checkpoints/ab_map_only/last.pt --no-safety `
   --ckpt-dir outputs/checkpoints/ab_map_only_e100 --ckpt-prefix am --tag ab_map_only_e100
# 机制 vs 训练量（三臂同次，n=256、50 步）
& $py -u scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
   --arm am_ep100=outputs/checkpoints/ab_map_only_e100/am_ep100.pt `
   --arm am_ep020=outputs/checkpoints/ab_map_only/last.pt `
   --arm d2_ep100=outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt `
   --split val --samples 256 --batch 8 --steps 50 --device cuda `
   --out outputs/reports/ablation_map_only_ep100.json
# 引导质量 / 开安全档位（只能对 d2 这类训练过安全链路的臂）
& $py -u scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
   --arm d2_ep080=outputs/checkpoints/ab_distill_safeu_e100/d2_ep080.pt `
   --arm d2_ep100=outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt `
   --split val --samples 64 --batch 8 --steps 50 --device cuda --safety-mode full `
   --out outputs/reports/ablation_transfer_d2_ep100_safety_on_paired.json
# 表格（只读 JSON，含符号检验）
& $py -u scripts/report_ablation.py outputs/reports/ablation_map_only_ep100.json `
   outputs/reports/ablation_transfer_d2_ep100.json `
   outputs/reports/ablation_transfer_d2_ep100_safety_on_paired.json
```

## 10. 产物索引

| 目的 | 路径 |
|---|---|
| 机制 vs 训练量（三臂） | `outputs/reports/ablation_map_only_ep100.json` |
| d2@80 vs d2@100（map_only） | `outputs/reports/ablation_transfer_d2_ep100.json` |
| 引导质量 + 开安全档位 | `outputs/reports/ablation_transfer_d2_ep100_safety_on_paired.json` |
| d2 四臂趋势（CPU） | `outputs/reports/prelim_d2_trend_cpu256.json` |
| 2 步 vs 50 步（CPU） | `outputs/reports/prelim_d2_ep100_cpu.json`、`prelim_map_only_50steps_d2ep100_cpu.json` |
| 柱图 / 拼图 | `outputs/figures/ablation_map_only_ep100/`、`ablation_transfer_e100*/` |
| 权重 | `outputs/checkpoints/ab_map_only_e100/am_ep{030..100}.pt`、`outputs/checkpoints/ab_distill_safeu_e100/d2_ep{040..100}.pt` |
| 日志 | `outputs/logs/{ab_map_only_e100,ab_distill_safeu_e100,eval_ep100}.log` |
