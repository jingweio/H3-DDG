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
