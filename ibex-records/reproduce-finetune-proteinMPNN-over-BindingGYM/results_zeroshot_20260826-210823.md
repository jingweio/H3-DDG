# 交付项 (1)：zero-shot pretrained ProteinMPNN

作业 `50891332`，COMPLETED，**1:09:42**，25/25 assay。跑的是 vendored 的官方
`baselines/protein_mpnn/compute_fitness_multi_pdb.py`（**原版** ProteinMPNN，`forward(X, S, ...)`），
`--backbone_noise 0.00`，权重 `v_48_020.pt`。

## 结果

| fold | 我们 Spearman | 官方 Spearman | Δ | 我们 AUC | 官方 AUC | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.4391 | 0.4537 | −0.0146 | 0.7013 | 0.7009 | +0.0004 |
| 1 | 0.3337 | 0.3522 | −0.0185 | 0.6843 | 0.7260 | −0.0416 |
| 2 | 0.2222 | 0.2354 | −0.0133 | 0.5328 | 0.5385 | −0.0057 |
| 3 | 0.5148 | 0.5200 | −0.0052 | 0.7592 | 0.7604 | −0.0012 |
| 4 | 0.4033 | 0.4160 | −0.0127 | 0.7206 | 0.7172 | +0.0034 |
| **ALL** | **0.3840** | **0.3970** | **−0.0130** | **0.6789** | **0.6879** | **−0.0090** |

Spearman 达官方的 **96.7%**，AUC 达 **98.7%**。逐 assay 结果见 `results/our_zeroshot_per_assay.csv`。

## 三点方法学说明

**① `design_score` 与 `global_score` 的 Spearman 完全相同**（两列秩等价）。官方 notebook 只读
`results/ProteinMPNN_zero_shot_metric.csv`、不含生成脚本，所以「他们用哪一列」本来是个歧义 ——
实测证明这个歧义不影响任何结论。

**② 残差来源是随机解码顺序，不是实现差异。** ProteinMPNN 每次前向都抽一个随机解码顺序
（`decoding_order = argsort((chain_M+0.0001) * |randn|)`），而 `--seed 0` 按脚本 help 是
「随机挑一个种子」。逐 assay 平均绝对偏差 0.035，24/25 在 0.05 以内；唯一大偏离是
`Z-domain_ZSPA-1_LL1`（官方 0.3066 / 我们 0.0888）—— 那个 assay 恰好也是官方**微调后**表现最差的
（0.0121），是全库最难的一个。

**③ zero-shot 没有 fold 概念。** 它对全部 25 个 assay 打一次分；上表按 fold 分组只为与微调结果并排看，
不代表 5 次独立实验。fold 归属用的是本仓库固化的 `data_splits/inter_assay_folds.tsv`
（已验证在官方自身 pin 下逐行一致）。

## 指标口径

`bindinggym_metrics.py::bindinggym_metrics_one_assay`，逐字移植官方
`calc_metric.ipynb::calc_zero_shot_metric(top_test=False)`：Spearman 为秩相关，AUC 的正类阈值取
**label 自身的 90 分位**。与官方发布数值的对比即用此口径，双方一致。
