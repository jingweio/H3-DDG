# 参考：pretrained ProteinMPNN 在 BindingGYM 上的 zero-shot 评测

**用途**：推进任何 BindingGYM zero-shot 评测时复用。本文只讲 zero-shot 那条路径；微调路径见
姊妹项目 `ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/`，masking 的跨方法对照见
本目录 `bindinggym_interassay_h3ddg_20260817-092000.md` §5.21。

**已复现**：作业 `50891332`，25/25 assay，**1:09:42**，ALL Spearman **0.3840** vs 官方 **0.3970**（96.7%）。

---

## 1. 入口与依赖

```
bgym_official/baselines/protein_mpnn/compute_fitness_multi_pdb.py --dms_index i     # i = 0..24
```

⚠️ **这是原版 ProteinMPNN**（`forward(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)`），
与 `bgym_official/training/protein_mpnn_utils.py` 里那份**改过的**（`forward(data)` 收 PyG batch、
带 mask+logit-diff readout）**不是同一个类**，不可互换。

**依赖极轻**：只要 `torch` + `pandas` + 同目录的 `protein_mpnn_utils.py`。
**不需要** `torch_geometric` / `torch_scatter` / `esm` / `peft` —— 那些是 `training/main.py` 的模块级
import 拖进来的，zero-shot 完全不碰。所以 zero-shot 可以在任何有 torch 的 env 里跑，不必等
`bgym-official` 那套重环境。

实际用的命令（见 `sh/zeroshot_proteinmpnn.sh`）：

```bash
python compute_fitness_multi_pdb.py \
  --dms_mapping      ../input/BindingGYM.csv \
  --dms_input        ../input/Binding_substitutions_DMS \
  --structure_folder ../input/structures \
  --dms_index        "$i" \
  --model_location   ../../training/cache/v_48_020.pt \
  --dms_output       "$OUT" \
  --backbone_noise   0.00 \
  --suppress_print   1
```

---

## 2. 打分方式（逐行核实过）

```python
S[:, start:start+len] = S_input                  # S 被【突变序列】覆盖
log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
scores = _scores(S, log_probs, mask_for_loss)    # 读回【同一条 S】；_scores 是 NLL
design_score_list.append(-1 * ns_mean)           # 取负 → +Σ log p
```

$$\texttt{score} = \sum_{\text{auto-regressive}} \log p(\text{mut} \mid \text{backbone},\ \text{ctx}=\text{mut-seq})$$

**没有 wt 相减**，就是突变序列自身的自回归对数似然。方向：越大 = 越像天然 = 结合越强，与
`DMS_score` 同向，直接算相关即可，不要翻符号。

### 四条容易踩的细节

**① 全程没有任何 mask。** 微调路径训练时会把突变位点置 `'X'`，zero-shot **不会**。

**② 解码顺序按【结构】缓存，同一 assay 的所有变体共用同一份**：

```python
if POI not in randn_1_dic:
    randn_1 = torch.randn(chain_M.shape)
    randn_1_dic[POI] = randn_1          # ← 按 POI 缓存，不是每个变体重抽
```

这是刻意的方差控制 —— assay 内所有变体在**同一个因子分解**下打分，而 within-assay 排序正是
Spearman 度量的东西。**改动或绕过这一点会显著抬高噪声。**

**③ 求和跑遍全序列，突变的影响会向下游传播。** 因为没有 wt 相减，非突变位点**不是常数偏移**：
位点 *j* 的突变会改变解码顺序里排在它之后的所有位点的条件分布（实测：单点突变可让 116 个下游位点
的 log-prob 变化最大 2.348）。所以 zero-shot 的 score 含「这个突变让整条序列多不像天然」的全局效应。
**微调路径相减后只剩突变位点，这一项被消掉了 —— 两者不可直接类比。**

**④ 输出两列 `design_score` / `global_score`，两者恒等。** 差别只在掩码
（`mask*chain_M*chain_M_pos` vs `mask`），但本脚本 `designed_chain_list = chain_ids`（全部链都是
designed）且无 fixed positions，所以 `chain_M ≡ 1`、`chain_M_pos ≡ 1`，两个掩码恒等。实测两列
Spearman 逐位相同。**官方 notebook 只读成品 csv、不含生成脚本，所以「他们用哪一列」本是歧义 ——
此歧义不影响任何结论。**

---

## 3. 指标口径

用 `bindinggym_metrics.py::bindinggym_metrics_one_assay`，逐字移植官方
`calc_metric.ipynb::calc_zero_shot_metric(top_test=False)`：

| 指标 | 定义 |
|---|---|
| Spearman | 秩相关 |
| **AUC** | 正类阈值 = **label 自身的 90 分位**（不是固定阈值）|
| MCC | 预测的 90 分位做阈值 |
| NDCG | `k = n // 10` |
| AP | 同 AUC 的正类定义 |

切片：`ALL` / `<3` / `≥3`，每切片 **≥100 行**才计入。zero-shot 的 25 个 assay 全部进 ALL。

---

## 4. 我们的复现结果

作业 `50891332`（7h walltime，实际 **1:09:42**）。逐 assay 存于
`ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/results/our_zeroshot_per_assay.csv`。

| fold | 我们 Spearman | 官方 Spearman | Δ | 我们 AUC | 官方 AUC | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.4391 | 0.4537 | −0.0146 | 0.7013 | 0.7009 | +0.0004 |
| 1 | 0.3337 | 0.3522 | −0.0185 | 0.6843 | 0.7260 | −0.0416 |
| 2 | 0.2222 | 0.2354 | −0.0133 | 0.5328 | 0.5385 | −0.0057 |
| 3 | 0.5148 | 0.5200 | −0.0052 | 0.7592 | 0.7604 | −0.0012 |
| 4 | 0.4033 | 0.4160 | −0.0127 | 0.7206 | 0.7172 | +0.0034 |
| **ALL** | **0.3840** | **0.3970** | **−0.0130** | **0.6789** | **0.6879** | **−0.0090** |

⚠️ **zero-shot 本身没有 fold 概念** —— 它对全部 25 个 assay 打一次分。按 fold 分组只为与微调结果
并排看，**不代表 5 次独立实验**。分组用本仓库固化的 `data_splits/inter_assay_folds.tsv`
（已直接确认与官方代码运行时算出的划分一致）。

**残差来源是随机解码顺序，不是实现差异。** `--seed 0` 按脚本 help 是「随机挑一个种子」，而模型每次
前向都抽解码顺序。逐 assay 平均绝对偏差 **0.035**，24/25 在 0.05 以内；唯一大偏离是
`Z-domain_ZSPA-1_LL1`（官方 0.3066 / 我们 0.0888）—— 那个 assay 恰好也是官方**微调后**表现最差的
（0.0121），是全库最难的一个。

**若要减小这个残差**：把 `--seed` 固定成一个具体值，或对同一 assay 多次打分取平均
（StaB-ddG 评测时用 `ensemble=20` 就是这个思路）。官方没有这么做，所以复现时也没做。

---

## 5. 官方参照值的出处

```
/home/guoj0f/repos/BindingGYM/results/
  ProteinMPNN_zero_shot_metric.csv                    ← 本文对标的那份（25 行）
  ProteinMPNN_finetune_inter_cluster_metric{,_oneORtwo,_multi}.csv   ← 微调，非 zero-shot
  ProteinMPNN_R_*                                     ← 随机初始化（--use_weight native）
```

列：`DMS_id, Spearman, AUC, MCC, NDCG, AP`，**无 fold 列**，两篇论文都未发布逐 fold 数字 ——
本文表中的逐 fold 分项是本项目的重构。

论文对应值（zero-shot 表）：ProteinMPNN `0.40 / 0.69 / 0.15 / 0.72 / 0.22`，与 csv 的
`0.3970 / 0.6879 / …` 逐位吻合。

---

## 6. 数据与运行

**数据**：`bgym_official/input/`（本分支自有副本，595 MB，gitignored）。

⚠️ `Binding_substitutions_DMS/` 里有 **28** 个 csv、合计 **508,962 行**，而 `BindingGYM.csv` 只登记
**25** 个、**376,446 行**。多出的 3 个是未登记的抗体-流感 logKd assay
（`CR9114_FluAH3` 65,535 / `CR9114_FluAH1` 65,094 / `CR6261_FluAH1` 1,887 = 132,516）。
**官方 pipeline 遍历 `BindingGYM.csv` 的 index，所以从不碰它们** ——
`--dms_index` 的取值范围就是 0..24。（H3-DDG 论文 §4.1 声称的 "508,962 curated entries"
正是把 28 个全数了。）

**sbatch**：`sh/zeroshot_proteinmpnn.sh`（在姊妹项目目录下），**幂等** —— 每 assay 一个输出 csv，
已存在就跳过，超时直接重投即续。产出在
`bgym_official/output/zeroshot/{DMS_id}.csv`（原始行 + `design_score` + `global_score`）。

**成本**：25 个 assay / 376,446 行 / a100 / `--batch_size 1` → **1:09:42**（约 90 rows/s 端到端）。
7h walltime 绰绰有余；若只跑部分 assay，2:30 的 walltime 排队快得多（本账号 ≤2.5h 实测排 0.1h，
≥4h 排 11–15h）。

---

## 7. 解读 zero-shot 结果时要记住的两件事

**① label 方向全库统一，但参照点不统一。** 25 个 assay 的 `DMS_score` 都是「越大 = 结合越强」
（BindingGYM 在建库时用 `DMS_score = raw * DMS_directionality` 归一化过，已独立验证：22 个有 WT 行的
assay 中 18 个 WT 分位 ≥50%，flipped-sign 候选 0 个）。**但有 4 个 gain-of-function assay，其野生型
是很差的结合体**：

| assay | WT 分位 | ρ(突变数, score) | fold |
|---|---|---|---|
| `5A12_VEGF_fitness_4ZFF` | **0.02%** | **+0.417** | 4 |
| `CD19_FMC63_Fitness_7URV` | 29.7% | **+0.735** | 3 |
| `hYAP65_peptide` | 35.4% | **+0.495** | 2 |
| `ACE2_SARS2-RBD_enrich_6M17` | 44.3% | — | 2 |

inverse-folding 的似然奖励「像天然序列」，而这些库里天然序列恰恰最差 ——
**zero-shot 在它们上按构造是不利的**。这是 BindingGYM zero-shot 天花板偏低的一个结构性原因，
不是实现问题。（微调能翻转它：官方微调后在 `5A12_VEGF` 上拿到 +0.4460。）

**② 预训练权重贡献了绝大部分。** 官方对照：`ProteinMPNN`（pretrained）微调后 **0.4217**，
`ProteinMPNN-R`（随机初始化、其余配置完全相同）只有 **0.1585** —— 差 **0.263**。
而 zero-shot 不训练就有 0.3970。所以在 BindingGYM 上，**预训练 >> 微调带来的增量**。

---

## 8. 复现步骤（最小）

```bash
# 1. env：任何有 torch + pandas 的即可（h3ddg-reproduce 也行，不必用 bgym-official）
# 2. 提交（幂等，可重复投）
cd ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/sh
sbatch --time=02:30:00 zeroshot_proteinmpnn.sh
# 3. 拉回并算指标
rsync -a guoj0f@glogin.ibex.kaust.edu.sa:/ibex/user/guoj0f/H3-DDG/reproduce/bgym_official/output/zeroshot/ ./zs/
python -c "
import glob,os,pandas as pd
from bindinggym_metrics import bindinggym_metrics_one_assay
rows=[]
for f in sorted(glob.glob('zs/*.csv')):
    d=pd.read_csv(f); m=bindinggym_metrics_one_assay(d, pred_col='global_score', label_col='DMS_score')
    rows.append(dict(DMS_id=os.path.basename(f)[:-4], **m))
t=pd.DataFrame(rows); print(t.to_string(index=False))
print('ALL:', t.Spearman.mean(), t.AUC.mean())"
```
