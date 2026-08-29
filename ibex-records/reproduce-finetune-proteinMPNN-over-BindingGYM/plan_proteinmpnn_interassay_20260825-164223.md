# 复现 BindingGYM 的 ProteinMPNN 微调结果（inter-assay split）

- **project_name**: `reproduce-finetune-proteinMPNN-over-BindingGYM`
- **worktree**: `/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce`（分支 `reproduce`）
- **ibex**: `/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/`
- **与既有项目的关系**：与 `bindingGYM-reproduce`（复现 H3-DDG）**同分支、不同 project**，共享已验证的数据与 fold 划分，按 ibex-usage §1c-4 不新建 worktree。

## 1. 目标与交付物

在 BindingGYM 的 inter-assay cluster 5-fold split 上复现**朴素 ProteinMPNN** 的两组数值，并与论文对齐。

交付一张表：**f0 / f1 / f2 / f3 / f4 / ALL folds × {Spearman, AUC}**，四行：

| 行 | 内容 | 状态 |
|---|---|---|
| (1) | **我们跑的** pretrained ProteinMPNN（zero-shot）| 待跑 |
| (2) | **我们跑的** fine-tuned ProteinMPNN（inter-assay）| 待跑 |
| (3) | **论文的** pretrained ProteinMPNN（zero-shot）| ✅ 已取得 |
| (4) | **论文的** fine-tuned ProteinMPNN（inter-assay）| ✅ 已取得 |

### 已取得的参照值 (3)(4)

来源：`BindingGYM/results/ProteinMPNN_{zero_shot,finetune_inter_cluster}_metric.csv`，逐 assay 25 行；
论文对应 Table（zero-shot）`0.40 / 0.69` 与 Table 5（微调）`0.42 / 0.70`，与 csv 均值逐位吻合。

| | f0 | f1 | f2 | f3 | f4 | **ALL** |
|---|---|---|---|---|---|---|
| **(3) 论文 zero-shot** Spearman | 0.4537 | 0.3522 | 0.2354 | 0.5200 | 0.4160 | **0.3970** |
| **(3) 论文 zero-shot** AUC | 0.7009 | 0.7260 | 0.5385 | 0.7604 | 0.7172 | **0.6879** |
| **(4) 论文 微调** Spearman | 0.5542 | 0.2719 | 0.3035 | 0.5550 | 0.3916 | **0.4217** |
| **(4) 论文 微调** AUC | 0.7328 | 0.6843 | 0.5669 | 0.8016 | 0.7066 | **0.6995** |

⚠️ **两点口径说明，必须随表附带**：
- **逐 fold 分项是本项目重构的**，官方 csv 只有 `DMS_id,Spearman,AUC,MCC,NDCG,AP` 六列、**无 fold 列**，两篇论文都未发布逐 fold 数字。重构依据：微调结果是 OOF（每 assay 恰好 held-out 一次，已核 25 行/各 1 次/5 fold 全覆盖），且我们的 fold 划分与官方在其自身 pin 下逐行一致（`diff` 0 行）。
- **zero-shot 本身没有 fold 概念** —— 它是对全部 25 个 assay 的一次打分。按 fold 分组只是为了和微调结果并排看，不代表 5 次独立实验。

## 2. 方案：跑 BindingGYM 官方代码本身，不重新实现

**理由**：这是复现**他们的**结果。若自行实现，「有没有复现出来」会和「我们的实现对不对」混在一起，无法归因。我们在 `bindingGYM-reproduce` 项目里已有一套验证过的 H3-DDG 侧管线，但那用的是热力学循环 readout；官方是 **mask 全部突变位点 + Σmt_logit − Σwt_logit**，模型也是**无 hypergraph / 无 triplet attention 的朴素 ProteinMPNN**。两者不可互换。

两个入口：

| 交付项 | 入口 | 依赖 |
|---|---|---|
| (1) zero-shot | `baselines/protein_mpnn/compute_fitness_multi_pdb.py --dms_index i`（逐 assay，i=0..24）| 仅 torch + pandas + 本地 `protein_mpnn_utils` |
| (2) 微调 | `training/main.py --model_type structure --mode inter --split cluster --batch_size 8` | **额外**需 torch_geometric + torch_scatter + esm + peft |

(2) 的额外依赖是因为 `main.py` 在模块级 import 了 `DEMEmodel`（→ `torch_geometric.utils`, `torch_scatter`）、`torch_geometric.loader`、以及第 56–59 行的 `esm` / `peft` —— **即使走 structure 路径也会执行这些 import**。

## 3. 环境

按官方 `install.sh` 建**本项目专用** env `bgym-official`（**不得改动任何既有 env**，尤其 `unibind` 虽有 PyG 也不能借用 —— ibex-usage §1d 的 2026-07-01 事故就是别的 agent 动了正在跑的实验的 env）：

```
python 3.8 · torch 1.13.1+cu117 · torch-scatter 2.1.0+pt113cu117 · torch-geometric 2.2.0
numpy 1.24.4 · scikit-learn 1.3.2 · pandas 2.0.3 · scipy 1.10.1 · biopython 1.83 · peft 0.12.0 · fair-esm
```

全是预编译 wheel，无需编译。numpy/sklearn 的 pin 取自 `BindingGYM.yml`，与我们固化 fold 划分时用的官方 pin 一致（§5.16 已验证该组合复现出同一份划分）。

## 4. 数据（按 §1c-3 复制进分支，不引用外部路径）

| 项 | 来源 | 目标 |
|---|---|---|
| DMS 数据 25 个 csv + `BindingGYM.csv` + 22 个结构 | 本分支 `data/input/`（已有，376,446 行，已通过全量审计）| `bgym_official/input/` |
| ProteinMPNN 权重 `v_48_020.pt` | `BindingGYM/training/cache/` | `bgym_official/cache/` |
| cluster 表 `BindingGYM_cluster.tsv` | 同上（与本分支 `data_splits/` 里那份同源）| `bgym_official/cache/` |
| 官方代码 `training/` + `utils/` + `baselines/protein_mpnn/` | `/home/guoj0f/repos/BindingGYM` | `bgym_official/`（vendored，注明 upstream commit）|

**为什么 vendor 官方代码**：§1c-3 要求任务消耗的一切都物理复制进当前分支，不得引用外部仓库路径 —— 否则 Ibex 同步与分支隔离都会破。会在目录里留 `UPSTREAM.md` 记录来源 commit。

## 5. 阶段与风险

| 阶段 | 内容 | 主要风险 |
|---|---|---|
| P1 | 建 env、vendor 代码、复制数据、本地 smoke（各跑 1 个 assay / 极少 step）| torch-scatter 与 torch/cu 版本不匹配 |
| P2 | (1) zero-shot：25 个 assay 全量打分 → 逐 assay 指标 | 逐 assay 调用 25 次；吞吐未测 |
| P3 | (2) 微调：5 个 fold。官方 `main.py` 一次调用内 `for fold in range(5)` 串跑，**需评估是否加 `--fold` 参数拆成 5 个作业**（参照我们给 `train_skempi.py` 加 `--fold` 的做法）| 单作业可能远超 walltime |
| P4 | 用官方 `calc_metric.ipynb` 的口径（`calc_zero_shot_metric`，≥100 行过滤）算指标，出交付表 | — |

**尚未测量、故不预先承诺的量**：朴素 ProteinMPNN 的前向吞吐（我们只测过 H3-DDG 的 15.4 行/s，后者每行 2 次前向且多一层 O(K²) attention，不可直接外推）。P1 的 smoke 会给出实测值，再定 walltime 与拆分方式。

## 6. 与 `bindingGYM-reproduce` 项目的边界

本项目**不涉及 H3-DDG**，不改动 `bindingGYM-reproduce` 的任何代码、配置或作业。共享的只有：本分支 `data/input/` 的原始数据、以及已固化的 `data_splits/inter_assay_folds.tsv`（用于把官方逐 assay 结果重构成逐 fold 分项）。

---

## 附：突变位点 masking 的跨方法对照

官方微调在**训练时**把所有突变位点置 `'X'`（`training/dataset.py:97`，评测时不置）。这一步在论文
与代码里都没有任何论证，且对多点突变有实际后果。

完整的四套对照（BindingGYM 官方微调 / H3-DDG over SKEMPIv2 / StaB-ddG over Megascale /
StaB-ddG over SKEMPIv2），含实测证据，写在姊妹项目的记录里：
`ibex-records/bindingGYM-reproduce/bindinggym_interassay_h3ddg_20260817-092000.md` **§5.21**。

对本项目最要紧的两条：

1. 我们**照做**了这个 masking（跑的就是他们的代码，未改），所以复现结果继承它的全部性质，包括
   训练/评测的不一致。
2. 实测表明 `'X'` 并**不**影响被打分位点自身（自回归掩码已保证 Δ = 0），它屏蔽的是**突变位点
   之间的互见** —— 这使 score 在序列层面被强制成可加的，多点突变的 epistasis 不可学。而
   BindingGYM 是多点突变主导的库。

## Change log
- 2026-08-28 18:53: f3/f4 已从 Ibex 迁到 workstation 串行跑（PID 3235074，14:13 启动）。理由：walltime
  实测已不影响排队（05:00/07:00/12:00/24:00 返回同一预估开始时间），分段 --resume 的收益前提消失。
  f3 进度：预处理缓存重建约 2h（25 个 pkl / 289 MB），epoch 0 零样本评估 valid_spearman **0.513917**
  （pooled 口径），现处 epoch 1 训练 88%。**口径提醒：这里的 epoch 是遍历全量训练数据的 17,864 步，
  不是 H3-DDG 那条线的 256 步**；f3 单 epoch 约 1h20m（训练 ~40min + 142,905 行评估 ~40min）。
- 2026-08-28 18:53: Ibex `50960860` pmpnn_f0fast 仍 PENDING，预估开始 **2026-08-31 06:22**（较上次的
  ~9 天提前到 ~3 天）。f0 用临时变体 main_f0_fast.py（EVAL_EVERY=3 / STOP_AT=0.52），选择规则与
  其余 fold 的 patience-3 不同。

---

## ⚠️ 口径 NOTE：f0/f3/f4 用 `EVAL_ENSEMBLE = 1`，f1/f2 用上游原值 5

**这是 2026-08-28 用户批准的临时提速，必须在任何汇报里带上。**

### 上游原本做什么
ProteinMPNN 每次前向都要抽一个自回归解码顺序 —— `protein_mpnn_utils.py:1069-1072` 抽出
`randn`，`:1097` 用 `argsort((chain_M+0.0001)*|randn|)` 把它变成 `decoding_order`。同一结构换个
顺序，打出的分数就不同。

`main.py:564-585` 因此在**每个 epoch 的验证**里跑 **5 遍**完整 held-out 集，每遍用一个新抽的
解码顺序（`model.randn = None` 强制重抽），再 `valid_pred /= 5` 平均：

```python
if args.model_type == 'structure':
    all_randn = [model.randn.clone()]
    for _ in range(4):
        model.randn = None          # 换一个解码顺序
        ...完整扫一遍 held-out...
        valid_pred += valid_pred1
        all_randn.append(model.randn.clone())
    valid_pred /= 5
```

这 5 个 order 被存进 `all_randn`，测试阶段（`:611`、`:636`）再复用，保证选模型与最终测试同序。

**关键机制**：`if self.randn is None or self.training` —— 训练时每次 forward 都重抽（粒度是
per-batch，因为 `randn` 形状 `(1, L)` 在 batch 内广播）；`model.eval()` 后走 else 分支复用缓存，
所以**一次 pass 内全部 batch 共用一个 order**。ensemble 的 5 个成员是 5 个固定 order，不是每
batch 乱抽。

### 我们改了什么、为什么
`EVAL_ENSEMBLE = 1`：只保留 1 个解码顺序，不做平均。

成本实测：fold 3 held-out 142,905 行 = 17,864 个 batch，单遍约 40 min；一个 epoch 是
**31 秒训练 + 5 × 40 min 验证 ≈ 3.4 h**，验证占 99%。整个 fold 3 约需 34–48 h，f3+f4 共 2–2.5 天。
降到 1 后每 epoch 约 40 min。

### 后果（汇报时必须说明）
1. **f0/f3/f4 的 held-out 分数带单个解码顺序的方差**，不是 5 次平均；f1/f2 是 5 次平均。
2. **可能选出不同的 epoch** —— patience-3 用的正是这个更抖的数，early stop 的触发点会变。
3. 因此 **f0/f3/f4 与 f1/f2 不是严格同口径**，逐 fold 表并排时要标注。
4. 这是**复现性检查**（能否大致复现官方量级），不是严格的跑数汇报。用户 2026-08-28 明确接受此取舍。

### 变体与作业
| 用途 | 文件 | 旋钮 |
|---|---|---|
| f0（Ibex） | `bgym_official/training/main_f0_fast.py` | `EVAL_ENSEMBLE=1`、`EVAL_EVERY=3`、`STOP_AT=0.52` |
| f3/f4（workstation） | `bgym_official/training/main_ens1.py` | `EVAL_ENSEMBLE=1`（其余全同上游：每 epoch 评、patience-3） |

- Ibex：`50960860` 取消（PENDING，无损失）→ **`50966710`** pmpnn_f0fast，24h。
- workstation：ensemble=5 那轮（PID 3235074，跑了 7h、2 个 epoch）已 kill；其 `resume_fold3.pt` +
  `train_fold3.log` **归档**到 `results/_archived_pmpnn_ensemble5_20260828/`（未删）。新 PID **81495**。
- 启动断言踩过一次坑：`find .` 递归把归档里的 `resume_fold3.pt` 当成活状态而拒绝启动。已改成
  排除 `_archived*`，并把归档移出训练树 —— 递归检查保留，它本来就该抓到任何位置的活状态。

### fold 3 完成（2026-08-29 02:49，workstation PID 81495，`EVAL_ENSEMBLE=1`）
patience-3 早停于 **ep7**，best 在 **ep4**（pooled valid_spearman 0.537918）。5h32m 跑完 8 个 epoch
（约 43 min/epoch，对比 ensemble=5 时的 3.4 h/epoch）。

逐 epoch（pooled）：
```
ep0 0.459567(零样本)  ep1 0.446486 NIE1  ep2 0.477360  ep3 0.533826
ep4 0.537918(best)    ep5 0.471995 NIE1  ep6 0.534487 NIE2  ep7 0.510092 NIE3 -> STOP
```

BindingGYM 官方口径 per-DMS（从 5 个 `*_oof.csv` 复算）：

| assay | n | Spearman | AUC |
|---|---|---|---|
| SARS2-RBD_ACE2_deltaKd_6M0J | 21,872 | 0.7091 | 0.8527 |
| GB1_IgG-Fc_fitness_1FCC | 92,891 | 0.5798 | 0.7203 |
| CD19_FMC63_Fitness_7URV | 3,886 | 0.5664 | 0.7116 |
| GB1_IgG-Fc_fitness_1FCC_2016 | 22,176 | 0.4955 | 0.7613 |
| 4D5_HER2_fitness_1N8Z | 2,080 | 0.3389 | 0.8320 |
| **per-DMS mean** | | **0.5379** | **0.7756** |

**对官方 fold 3 参照 0.5550：达到 96.9%**（0.5379 vs 0.5550，差 0.017）。是目前三个完成 fold 里
与官方最接近的一个。注意本 fold 用 `EVAL_ENSEMBLE=1`，官方与我们的 f1/f2 用 5 —— 见上方口径 NOTE。

**ensemble=1 的方差实测**：同一零样本模型、同一 held-out 集，ep0 在 ensemble=5 下是 **0.513917**，
在 ensemble=1 下是 **0.459567**，差 **0.054**。这是单个解码顺序相对 5 次平均的抽样偏离，量级不小，
是解释 f0/f3/f4 与 f1/f2 差异时必须计入的一项。

### fold 4 完成（2026-08-29 03:25，同一 workstation 作业）—— f3+f4 `ALL DONE`, `EXIT=0`
patience-3 早停于 **ep4**，best 在 **ep1**（pooled 0.404624）。仅 36 分钟（约 7 min/epoch）。

```
ep0 0.392536(零样本)  ep1 0.404624(best)  ep2 0.365440 NIE1  ep3 0.378435 NIE2  ep4 0.401879 NIE3 -> STOP
```

| assay | n | Spearman | AUC |
|---|---|---|---|
| BH3_Mcl-1_normed_3KZ0 | 518 | 0.6622 | 0.8861 |
| 5A12_VEGF_fitness_4ZFF | 29,981 | 0.4594 | 0.6525 |
| HLA-A2_TAPBPR_meanscore_5WER | 3,344 | 0.3617 | 0.6375 |
| 5A12_Ang2_fitness_4ZFG | 944 | 0.1349 | 0.6338 |
| **per-DMS mean** | | **0.4046** | **0.7025** |

对官方 fold 4 参照 0.3916：**103.3%**。

### 逐 fold 汇总（4/5 完成，19/25 assays）

| fold | n | 官方 | 我们 | 达成 |
|---|---|---|---|---|
| 0 | 6 | 0.5542 | — | 待跑 |
| 1 | 5 | 0.2719 | 0.4132 | 152.0% |
| 2 | 5 | 0.3035 | 0.2839 | 93.5% |
| 3 | 5 | 0.5550 | 0.5379 | 96.9% |
| 4 | 4 | 0.3916 | 0.4046 | 103.3% |

已完成 19 个 assay 的加权平均：**我们 0.4102 / 官方同 19 个 0.3799 = 108.0%**。
（官方 25-assay 总平均 0.4217；缺的 fold 0 是官方最强的一折之一，0.5542。）

⚠️ f1/f2 用上游 `ensemble=5`，f3/f4 用 `ensemble=1` —— 见上方口径 NOTE。
