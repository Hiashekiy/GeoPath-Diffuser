# 消融：安全特征链路训练的模型，在"不给安全"时是否更好？

问题：模型 A 用**双支路 + 安全链路**（64 个椭圆 token + 对齐项 + 走廊约束）训练；模型 B
**完全没有安全特征指导**（只训 Map-only 支路）。如果推理时**都不给安全**（`safety_mode="map_only"`：
无搜索、无椭圆 token、无 ALM），A 会不会比 B 更好？

结论（20 epoch / 15,240 step，各一支，同一份数据、同一随机流）：**避障能力上没有差别，
几何保真上 A 略好**。安全链路的收益几乎全部发生在**推理时**，不是被学进权重里的。

---

## 1. 协议（两支唯一差别是训练目标）

| 项目 | 臂 A `safety_trained` | 臂 B `no_safety` |
|---|---|---|
| 训练目标 | `λc·Lc + λu·Lu + λalign·Lalign + λbound + λsmooth + λcurve + λsafe` | `λu·Lu + λbound + λsmooth + λcurve` |
| 前向次数/step | 2（Map-only + Safety） | 1（Map-only） |
| 椭圆 token | 64 个（只进 c 支路） | 不构造（`safety_memory=None`） |
| 走廊 pack / `L_safe` | 有 | 无 |
| batch / steps / lr / 裁剪 | 16 / 15,240 / 2e-4 / 1.0 | 同 |
| 初始化 | `build_pipeline(seed=0)` | 同（完全相同的初始权重） |
| 数据顺序、噪声、t | `fit(seed=1)` | 同 |

两条控制点，保证这是受控对照：

1. **初始化相同**：两支都走 `factory.build_pipeline(seed=0)`，在分叉前权重完全一致；
2. **随机流相同**：step 200/400 两支的 loss 打印到小数点后 4 位**完全相同**（B 0.1730/0.1920，
   A 0.1733/0.1921），说明每步抽到的是同一批样本、同一个 `eps`、同一个 `t`。

训练开销：B 0.249 s/step（190 s/epoch），A 0.398 s/step（303 s/epoch）——两次前向 + pack 构造
+ 三项额外 loss，反向仍只有一次。

## 2. 训练侧（同 step、同批样本、同噪声，可直接对比）

`u` = Map-only 支路的内控制点 MSE。安全支路通过**共享权重 + 对齐项**把这条支路一起带下去：

| step | B `u` | A `u` | B smooth | A smooth |
|---|---|---|---|---|
| 1,000 | 0.1708 | **0.0227** | 0.3153 | 0.2983 |
| 2,000 | 0.0873 | **0.0032** | 0.2646 | 0.1002 |
| 3,000 | 0.0429 | **0.0019** | 0.2208 | 0.0723 |
| 4,800 | 0.0384 | **0.0032** | 0.2141 | 0.0821 |

即：A 用 ~1,000 step 达到 B 需要 ~5,000–8,000 step 的拟合水平（**收敛速度差 5–8 倍**），
到 20 epoch 末差距收窄。

20 epoch 末的 teacher-forced val（`trainer.eval_step`，128 样本）：

| | `loss_u` | `loss_c` | `align` | `bnd` | `smooth` | `curve` | `safe` |
|---|---|---|---|---|---|---|---|
| A | **0.00598** | 0.00162 | 0.00679 | 0.00069 | 0.0509 | 0.0181 | 0.0182 (max 0.0954) |
| B | 0.0084 | — | — | 0.00019 | 0.0373 | 0.0097 | — |

> 注：A 的 `smooth`/`curve` 是**两支均值**，与 B 的单支数值不可直接比较；`loss_u` 可比较，
> A 低 29%。两支的 `safe` 只在 A 里有定义。

`safety_gate`（每个 block 一个标量 α，初值 0.1，见 `models/conditional_dit.py`）：
A 从 0.100 → **0.148**（12 个 block 全部上抬，最小 0.103、最大 0.222），旧 70-epoch 权重最终
也在 0.135 附近（block1 掉到 0.028、block6 到 0.205，逐层自主分配）；B 恰好是 0.100/0.100 全部
不动 —— 因为臂 B 里 `safety_memory=None`，安全支路整个不执行，那 7.1 M 参数与 12 个 α
**从未进入计算图**，`p.grad is None`，AdamW 直接跳过（`tests/test_dual_branch_training.py::
test_map_only_arm_never_touches_the_safety_chain` 把这条钉住了）。

## 3. 闭环对照：两支都用 `map_only`（完全不给安全）

val 前 256 个样本，DDIM 50 步，同一批样本、同一初始噪声（`compare_safety_ablation.py`）：

| 指标 | A `safety_trained` | B `no_safety` | 逐例配对 A/B/tie | 符号检验 p |
|---|---|---|---|---|
| valid | 0.590 | 0.586 | — | — |
| **collision-free** | **0.594** | **0.586** | — | — |
| collision fraction（均值） | 0.1192 | 0.1234 | 65 / 49 / 142 | 0.16 |
| chamfer vs GT (m) | **1.450** | 1.580 | 140 / 116 / 0 | 0.15 |
| 控制点 RMSE (m) | **1.535** | 1.711 | — | — |
| 长度比 | 0.942 | 0.924 | — | — |
| \|长度比 − 1\| | **更接近 GT** | — | 178 / 78 / 0 | <1e-4 |

读法：
- **避障是平手**：无碰撞率 59.4% vs 58.6%（+0.8 pp，配对 65:49，p=0.16 不显著）。
- **几何保真是 A 稳定更好**：控制点 RMSE −10%、chamfer −8%（趋势级）、长度误差在 70% 的样本上
  更小（p<1e-4）。这与 `align` 项的设计意图一致——它把安全支路的**轨迹形状/长度**统计量传给
  Map-only 支路，但**传不了避障**，因为安全支路的优势来自推理时那 64 个椭圆 token。

## 4. 参考点：同一份 A 权重，把安全打开

同样 64 个 val 样本、50 步，`safety_mode="full"`（搜索 + 椭圆 + corridor + ALM）：

| | valid | collision-free | coll fraction | chamfer | ctrl RMSE | 长度比 |
|---|---|---|---|---|---|---|
| A + 安全 | **0.906** | **0.906** | **0.0079** | 1.201 | 1.412 | 1.020 |
| A 不给安全 | 0.590 | 0.594 | 0.1192 | 1.450 | 1.535 | 0.942 |

安全链路把无碰撞率从 59% 抬到 **91%**、碰撞率压低 **15 倍**、长度几乎与 GT 相等（1.02）。
图 `outputs/figures/ablation_safety/safety_on/montage.png` 里能直接看到：给了安全时曲线沿 GT
绕障；`outputs/figures/ablation_safety/montage.png` 里两支都在切障碍。

## 5. 结论与实践含义

1. **不能指望"安全训练过的权重"在无安全推理时更安全**——20 epoch 的对照里它与"从未见过安全"
   的模型在碰撞指标上等价。
2. **但也没有代价**：A 的 Map-only 支路不比 B 差，几何/长度还更好（并且收敛快 5–8 倍）。
   所以双支路训练不会损害"无安全推理"这条路径。
3. 安全链路的收益是**推理时注入**的（椭圆 token → Safety Cross-Attention；corridor → ALM），
   `align` 只能传递轨迹级统计量。若希望"无安全推理"也变安全，需要在 Map-only 支路自己的目标里
   加与地图障碍直接相关的项（当前 `L_safe` 只作用于 c 支路），而不是依赖对齐项。

## 6. 复现

```powershell
# 臂 B：只训 Map-only
python scripts/train_dual_branch.py --config configs/c64_longpath.yaml --dataset real `
  --real-split train --val-split val --device cuda --epochs 20 --batch-size 16 `
  --steps 15240 --no-safety --ckpt-dir outputs/checkpoints/ab_map_only `
  --ckpt-every 10 --ckpt-prefix abm --tag ab_map_only --seed 1

# 臂 A：双支路（安全链路）
python scripts/train_dual_branch.py --config configs/c64_longpath.yaml --dataset real `
  --real-split train --val-split val --device cuda --epochs 20 --batch-size 16 `
  --steps 15240 --ckpt-dir outputs/checkpoints/ab_dual `
  --ckpt-every 10 --ckpt-prefix abd --tag ab_dual --seed 1

# 对照（两支都不给安全）
python scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
  --arm safety_trained=outputs/checkpoints/ab_dual/last.pt `
  --arm no_safety=outputs/checkpoints/ab_map_only/last.pt `
  --split val --samples 256 --batch 8 --steps 50 --device cuda

# 参考点：A 开安全
python scripts/compare_safety_ablation.py --config configs/c64_longpath.yaml `
  --arm safety_trained_WITH_safety=outputs/checkpoints/ab_dual/last.pt `
  --split val --samples 64 --batch 8 --steps 50 --device cuda --safety-mode full
```

产物：
- `outputs/reports/ablation_safety_map_only_n256.json`（两支闭环 + 逐例记录 + 配对统计）
- `outputs/reports/ablation_safety_safety_on.json`（参考点）
- `outputs/figures/ablation_safety/{summary,montage}.png`、`.../safety_on/{summary,montage}.png`
- 训练日志 `outputs/logs/ab_map_only.log`、`outputs/logs/ab_dual.log`
- 权重 `outputs/checkpoints/ab_map_only/{last,abm_ep010,abm_ep020}.pt`、
  `outputs/checkpoints/ab_dual/{last,abd_ep010,abd_ep020}.pt`
