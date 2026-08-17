# bindingGYM-reproduce — 主实验：H3-DDG 在 BindingGYM inter-assay split 上的复现 (Table 2)

> created 2026-08-17 09:20 ｜ **status: PLANNED**
> 前置验证：[skempiv2_cv3_h3ddg_20260817-092000.md](skempiv2_cv3_h3ddg_20260817-092000.md)

## 1. Goal / hypothesis

复现 H3-DDG (NeurIPS 2025) **Table 2** 的 H3-DDG 行 —— 在 BindingGYM 的 **inter-assay split** 上、按 **per-DMS** 口径、在 ALL / <3 / ≥3 三个 mutation-depth 切片上的表现。

**论文目标数值（Table 2，H3-DDG 行）**

| Mutations | Pearson↑ | Spearman↑ | AUROC↑ | RMSE↓ |
|---|---|---|---|---|
| ALL | 0.3057 | 0.2725 | 0.5703 | 1.1294 |
| <3  | 0.3322 | 0.3031 | 0.5745 | 1.0758 |
| ≥3  | 0.2472 | 0.2755 | 0.6734 | 2.4976 |

baseline 行（ProteinMPNN / BA-Cycle / Prompt-DDG / BA-DDG）**直接引用论文数值，不重跑**（用户决定）。

## 2. ⚠️ 复现的根本前提：官方没有发布 BindingGYM 代码

`biomed-AI/H3-DDG` **只有 master 一个分支**（`git ls-remote` 已核实，与本地 fork 同一 commit `3a752b3`），其中：

- `train_skempi.py` / `skempi.py` / `dataset.py` / `config/train_h3-ddg.json` —— **全部是 SKEMPI 专用**
- 全 repo grep `binding.?gym` → **0 命中**

因此 Table 2 不是「跑一下脚本」就能复现的，必须**自己重实现整条 BindingGYM pipeline**：dataset loader + inter-assay split + per-DMS metrics + train/eval 脚本。用户决定：**参考 BindingGYM 官方 repo（`/home/guoj0f/repos/BindingGYM`，branch `analysis` @ `752c612`）来构建 data-loader**。

## 3. 数据层（已实测核实，非推测）

- 来源：Zenodo `12514160` 的 `input.zip`（166 MB），解压到本 worktree 的 `data/input/`（§1c-3 数据自包含，已 gitignore）
- `input/BindingGYM.csv`：**25 行**（25 个 shipped assay）；`input/Binding_substitutions_DMS/`：**28 个 csv**（多出的 3 个 flu assay 不在 benchmark 内）；`input/structures/`：**22 个 PDB**
- shipped 25 个 assay 合计 **376,446** 行

### 3.1 突变位点 → 结构残基的映射（**最关键的实现决策**）

实测 25 个 assay 后确认：

| 候选做法 | 结论 |
|---|---|
| 用 `mutant`（per-chain 1-based 序列位点）去索引结构残基 | ❌ **不可靠**。7 条链的结构有缺失残基（gap），序列长度 ≠ 结构残基数：`1HE8_hm` A 749 vs 941、`8BE4_hm` S 440 vs 475 / R 165 vs 168、`3KZ0_hm` A 143 vs 150、`1PQ1_hm` A 147 vs 196、`5WER_hm` A 265 vs 274 / C 317 vs 370、`7URV_hm` C 218 vs 255 / D 227 vs 242、`1N8Z_hm` C 581 vs 607、`4ZFG` H 215 vs 219、`4ZFF_CHL` H 211 vs 219 |
| **用 `mutant_pdb`（PDB residue numbering）→ 匹配结构的 `(chain, resseq, icode)`** | ✅ **采用**。逐 assay 抽样验证：每个位点的 wild-type 字母都与结构残基一致 |

⚠️ **必须自建 seq_map**：repo 的 `parse_biopython_structure` 返回的 `seq_map` 键是 `(chain_id, resseq)`，**丢掉了 icode**（`common_utils/protein/parsers.py:184`）。而抗体类 assay（`4ZFF_CHL`、`4ZFG`）用 Kabat 编号，`mutant_pdb` 里出现 `P52AL`（resseq 52 + icode `A`）这类位点，会与 resseq 52 冲突。故 BindingGYM loader **自建 `(chain, resseq, icode)` 三元键的 map**。

### 3.1b ✅ 全量审计结果（`audit_bindinggym.py`，逐行核查全部 376,446 行）

| 检查项 | 结果 |
|---|---|
| 总行数 | **376,446** —— 与 shipped 25 assay 的预期完全一致 |
| 突变位点在结构中缺失（unresolved） | **0** —— `mutant_pdb` + `(chain, resseq, icode)` 映射 100% 命中 |
| `mutant_pdb` 的 wt 字母与结构残基不符 | **0** |
| wild-type 行（`mutant` 为空） | **22**（3 个 assay 无 WT 行：Z-domain ZSPA-1 LL1/LL2、HLA-A2）|
| 突变落在 side0 only / side1 only / **两侧** | 322,487 / 266 / **53,671** |

**结论**：(a) §3.1 的映射方案零失败，无需任何丢弃/跳过逻辑；(b) 那 53,671 行全部来自 4 个 Z-domain assay，**证实 §3.2 的 A\|B 侧划分是必需的** —— 若按「被突变链 = side0」，这 5 万多行的 thermodynamic cycle 会退化成 0；(c) 22 个 WT 行按 §5.2-7 的规则处理。

### 3.2 链分组（thermodynamic cycle 的语义）

H3-DDG 的 `DDGPredictor.calc_thermodynamic_cycle` 计算 `complex_energy − Σ(被突变一侧单独的 energy)`。在 SKEMPI 里，「一侧」= `parse_biopython_structure(antibody_chain_id=…, antigen_chain_id=…)` 给出的 `chain_nb ∈ {0,1}`，即**结合的两方**（不是单条 PDB 链）。

BindingGYM 需要同样的两方定义。实测每个 assay 的「被突变链」与链长：

- **23/25 个 assay**：被突变的链恰好构成 DMS_id 里**第一个蛋白**（如 `5A12_VEGF` 突变 H+L = 抗体 5A12；`4D5_HER2` 突变 A+B = 抗体 4D5；`KRAS_SOS1` 突变 R = KRAS）→ side 0 = 被突变蛋白的链，side 1 = partner
- **4 个 Z-domain assay 例外**：`Z-domain_ZpA963_HL1/HL2_2M5A`、`Z-domain_ZSPA-1_LL1/LL2_1LP1` 是 synthetic-coevolution 研究，**两个 partner 同时被突变**（A、B 都出现突变）。若沿用「被突变链 = side 0」，side 1 会为空 → cycle 退化成 0。

**决策：显式写死一张 25 行的 `assay → (side0_chains, side1_chains)` 表**（按 DMS_id 的蛋白命名 + 链长核对得到，可审计），而不是从「哪条链被突变」自动推断。Z-domain 的 A|B 各为一侧。所有结构文件的链集合都与 `chain_id` 完全一致（无旁观链），已实测。

### 3.3 标签方向

BindingGYM 的 `DMS_score` 统一为「越大 = 结合越强」（全库 25 个 assay 已核实）。H3-DDG 预测的是 ΔΔG（越小 = 结合越强）。BindingGYM 官方 `training/dataset.py:73` 的 fallback 写作 `-self.df.loc[idx,'ddg']`，即 `DMS_score ≈ −ddg`。

**决策：`ddG_true = −DMS_score`**，与 BindingGYM 官方的符号约定一致。这样报告的 Pearson/Spearman 为正，与论文一致。
- AUROC 定义：论文 A.3 说「按 ΔΔG 的符号分类」，即 `ddG_true > 0`（= `DMS_score < 0`，结合变弱）为正类。⚠️ BindingGYM 各 assay 的 0 点参照系并不统一（19 个 assay WT 在 79–100 分位，3 个 gain-of-function assay WT 在 0–35 分位），这是本指标的**已知口径不确定性**，会在结果中标注。

## 4. Split：inter-assay（已精确复现）

严格按 BindingGYM `training/main.py:348` 的实现（官方 repo 里 split **只以代码形式存在**，没有落盘的 fold 文件；唯一落盘的 split 输入是 `training/cache/BindingGYM_cluster.tsv`，28 行的 MMseqs2 assay→cluster 表）：

```python
cluster_df  = pd.read_csv('cache/BindingGYM_cluster.tsv', sep='\t', header=None, names=['cluster','DMS_id'])
clusters    = [cluster_map[DMS_id] for DMS_id in BindingGYM.csv['DMS_id']]   # 长度 25
split       = list(GroupKFold(n_splits=5).split(clusters, groups=clusters))  # assay 级别
```

### 4.0 🔴 复现陷阱：fold 成员依赖 **sklearn 版本**

`GroupKFold` 不接受随机种子，但它按 group size 降序贪心分配，**这里 25 个 assay 落在 14 个 cluster、size 大量并列（6,4,2,2,2,1×9）**，因此结果完全由 tie-breaking 决定 —— 而 tie-breaking 在 sklearn 1.6 的重构中变了。实测：

| sklearn | fold 成员 |
|---|---|
| **1.2.1**（H3-DDG README 指定）| **A** |
| **1.3.2**（BindingGYM.yml 指定）| **A**（与 1.2.1 完全一致）|
| 1.7.2（本机 base env）| **B** —— CD19↔CXCR4 在 fold2/3 之间互换、SARS2-RBD↔HLA-A2 在 fold3/4 之间互换 |

→ **本项目一律锁定 sklearn 1.2.1**（H3-DDG 与 BindingGYM 的指定版本在此点上等价）。env 已按此固定。

### 4.1 正确的 fold 组成（sklearn 1.2.1）

| fold | assays | test 行数 | 其中 **≥2** | 其中 **≥3** | %≥3 | test assays |
|---|---|---|---|---|---|---|
| 0 | 6 | 114,341 | 100,129 | **0** | 0.0% | KRAS ×6 |
| 1 | 5 | 55,081 | 54,819 | **54,324** | 98.6% | Z-domain ×4 + BH3_Bcl-xL |
| 2 | 5 | 29,332 | 18,118 | **11,091** | 37.8% | ACE2_SARS2-RBD, CXCR4, PSD95_CRIPT, PSD95_Tm2F, hYAP65 |
| 3 | 5 | 142,905 | 137,077 | **35,775** | 25.0% | 4D5_HER2, CD19, GB1_1FCC, GB1_1FCC_2016, SARS2-RBD_ACE2 |
| 4 | 4 | 34,787 | 31,165 | **30,822** | 88.6% | 5A12_Ang2, 5A12_VEGF, BH3_Mcl-1, HLA-A2 |

### 4.2 ✅ 已决：跑满 5 个 fold（= BindingGYM 官方协议），fold 之争作废

**用户决定（2026-08-17）**：不去猜论文挑了哪个 fold，而是**按 BindingGYM 官方协议跑满 5 个 fold** —— 每个 fold 用其余 4 个 cluster-group 训练、自己做 test，产出 25 个 assay 的完整 **OOF（out-of-fold）预测**，再按 ALL / <3 / ≥3 切片、per-DMS 算指标后对 assay 等权平均。

这个口径由官方 `calc_metric.ipynb` 的 `get_finetune_inter_metric_df` 写死：

```python
df = pd.read_csv(f'{path}/train_on_BindingGYM_inter_cluster_{model_type}_seed{seed}/{DMS_id}_oof.csv')
assert train.shape[0] == df.shape[0]                                    # 每个 assay 每一行都有 OOF 预测
zero_shot_metric[DMS_id] = calc_zero_shot_metric(df, ...)               # ALL
oneORtwo_df = df[df['mutant'].apply(lambda x: len(x.split(':')) < 3)]   # <3，>=100 行才计入
multi_df    = df[df['mutant'].apply(lambda x: len(x.split(':')) >= 3)]  # >=3，>=100 行才计入
```
→ 三个切片分别有 **25 / 22 / 13** 个 assay，与 `results/*_finetune_inter_cluster_metric*.csv` 的行数逐一吻合。

**好处**：(a) 结果可直接对标 BindingGYM paper Table 5（同协议同指标）；(b) 每个 fold 的分项单列后，也能逐个对照 H3-DDG Table 2，无需事先赌论文用了哪个 fold；(c) 覆盖全部 25 个 assay，比单 fold 更完备。

下面 §4.3 保留原先关于「论文究竟指哪个 fold」的分析，作为解读 Table 2 时的参考，**不再影响实验设计**。

### 4.3 （存档）论文那句话指的是哪个 fold？

论文只写了一句：「focusing on the **fold with the most multi-point mutations** for testing」（§4.1 + Appendix A.2），没给 fold 编号或 assay 列表。**"multi-point" 的阈值本身就是歧义源**：

- H3-DDG 在 **SKEMPI** 表里 "multiple" = **≥2**（相对 single）；
- 在 **BindingGYM** 表里切片写作 `<3` / `≥3`，而 BindingGYM 官方的结果文件正是把 ≥3 命名为 `_multi`、<3 命名为 `_oneORtwo` → 在 BindingGYM 语境里 "multi" = **≥3**。

两种读法给出**不同的 fold**：

| 判据 | argmax |
|---|---|
| ≥2 行数最多 | **fold3**（137,077）|
| ≥3 行数最多 | **fold1**（54,324）|
| ≥3 占比最高 | fold1（98.6%）|

**独立的数值可行性检验（决定性证据）**：per-DMS RMSE 是「按 assay 算完再等权平均」。给定论文报的 (ALL, <3, ≥3) = (1.1294, 1.0758, **2.4976**)，检查每个 fold 在「允许每个 assay 的 RMSE 最大到该切片自身 label std 的 5 倍」这一**极宽松**上界下，能否达到 ≥3 那一栏的 2.4976：

| fold | ≥3 切片的 assay | 该切片可达的最大 mean RMSE (K=5) | 结论 |
|---|---|---|---|
| 0 | —（无 assay 有 ≥100 条 ≥3）| — | **不可能产出 ≥3 那一列** |
| 1 | 5 个（Z-domain 的 label std 只有 0.140–0.772）| **2.201** | **REFUTED**（< 2.4976）|
| 2 | 1 个（hYAP65）| 5.769 | 可行 |
| 3 | 4 个（含 CD19 std 2.35、SARS2 std 1.84）| 11.296 | 可行（宽裕）|
| 4 | 3 个 | 2.555 | 勉强（需所有 assay 都恰好 ≈5×std）|

→ **fold1 被算术排除**：它的 test assay label 尺度太小（Z-domain LL1 的 std 仅 0.140），要凑出 2.4976 的 per-DMS 平均 RMSE，模型得错到 label 自身标准差的 6–18 倍。fold4 同理近乎不可能。

**两条独立线索汇聚到 fold3**：(a) "multi-point" 取 ≥2 时 argmax 就是 fold3；(b) RMSE 可行性检验里 fold3 最宽裕。fold1（≥3 读法的 argmax）被算术排除。

> 以上仅用于**解读**论文 Table 2；实验设计已按 §4.2 改为跑满 5 个 fold。

## 4.4 🔑 heterogeneity：BindingGYM 是「设计掉」而非「解决」——这决定我们报哪些指标

BindingGYM 的 25 个 assay 的 `DMS_score` 尺度差 63 倍（`5A12_Ang2` std **0.079** ↔ `CD19` std **4.97**），且**从不做任何 label 归一化**。它靠三层设计让 scale 完全失效：

**训练端**：loss = **ListMLE**（listwise ranking，`training/loss.py`），只看序；`dataset.py:77-82` 每个 batch **先抽 assay 再抽 mutant**，保证 batch 内同尺度。

**推断/评测端（关键，也最容易被忽略）**：官方报告的每一个指标都**只依赖预测值的序，不依赖其数值**，且都**在单个 assay 内部算完再跨 assay 平均**：

| 指标 | scale-invariant 的原因 |
|---|---|
| Spearman | 秩相关 |
| AUC | `roc_auc_score(label_bin, pred)`，只按 `pred` 排序扫阈值 |
| NDCG | `pred` 只用于定序 |
| AP | 按 `pred` 序扫 precision-recall |
| **MCC** | 唯一需阈值的指标，而阈值是 **`pred` 自身在该 assay 内的 90 分位**（`pred_bin = pred > np.percentile(pred, 90)`），不是固定常数 |

正类定义同样是 assay 相对的：`label_bin = DMS_score > np.percentile(DMS_score, 90)`。
→ 模型输出**从不需要跨 assay 可比，只需在每个 assay 内部排对序**。

**官方从头到尾没有 RMSE、也没有 Pearson**：`results/*.csv` 的列就是 `DMS_id, Spearman, AUC, MCC, NDCG, AP`。`main.py` 的 `Metric()` 里那个 `rmse` 只进 log，model selection 用 `spearman`（`main.py:579`），paper Table 5 也只报这五个。

**H3-DDG Table 2 报的恰是 BindingGYM 刻意排除的两个指标**：
- per-DMS **Pearson 站得住**（Pearson 对预测的仿射变换不变，assay 内部算即可）；
- per-DMS **RMSE 站不住**（对任何变换都不不变，要求输出落在该 assay 的 `DMS_score` 标度上；单个回归头不可能同时匹配 20 个尺度差 63 倍的 assay）。这正是论文 §4.3 那段辩解的由来。

**→ 本复现的决策**：忠实照做 H3-DDG 的裸 `F.mse_loss`（`ddg_predictor.py:72`，不加任何 heterogeneity 处理，因为这是复现而非改进），但**同时报两套指标**：
1. **H3-DDG 口径**：per-DMS Pearson / Spearman / AUROC / RMSE（对标 Table 2）
2. **BindingGYM 官方口径**：per-DMS Spearman / AUC / MCC / NDCG / AP，用 `calc_zero_shot_metric` 的原样实现（对标 BindingGYM Table 5 的 ProteinMPNN=0.42 等）

两边都有参照系；一旦 RMSE 对不上，可立刻判断是口径问题还是实现问题。

## 5. Design & decision points

### 5.1 要新写的代码（本 branch 内）
| 文件 | 内容 |
|---|---|
| `bindinggym.py` | `BindingGYMDataset`：读 25 个 assay 的 csv + 22 个结构；`mutant_pdb → (chain,resseq,icode)` 映射；per-assay 两侧链分组表；产出与 `SkempiDataset.__getitem__` **完全同构**的 dict（`aa/aa_mut/mut_flag/chain_nb/res_nb/pos_atoms/ddG/complex/num_muts/id/mutstr`），从而可直接复用 repo 原有的 `MPNNPaddingCollate` 与 `DDGPredictor` |
| `bindinggym_dataset.py` | `BindingGYMDatasetManager`：**读固化的 `data_splits/inter_assay_folds.tsv`**（不在训练时现算 split）、train/val loader（沿用 `inf_iterator`） |
| `train_bindinggym.py` | 训练/评测入口，仿 `train_skempi.py`；`--test_fold F` 指定本次留出的 fold |
| `bindinggym_metrics.py` | 两套 per-DMS 指标：① H3-DDG 口径 Pearson/Spearman/AUROC/RMSE；② BindingGYM 官方口径 Spearman/AUC/MCC/NDCG/AP（照抄 `calc_zero_shot_metric`）。均按 ALL / <3 / ≥3 切片、≥100 行过滤、assay 等权平均 |
| `config/train_h3-ddg_bindinggym.json` | 论文 Appendix A.4 的超参 |
| `make_inter_assay_folds.py` ✅已完成 | 一次性固化 fold 成员到 `data_splits/inter_assay_folds.tsv`，内置 sklearn 版本 guard（非 1.2.1/1.3.2 直接拒绝运行，已实测 base env 会被挡下） |

### 5.2 KEY DECISIONS

1. **超参：按论文 Appendix A.4**（用户决定）
   - `lr=4e-4`（Adam）、`batch_size=1`、`max_iter=20000`、`num_tri_heads=4`、`hidden_dim=128`、`num_layers=3`（ProteinMPNN 默认 3 层）
   - `hyper_ratio` / `num_edges_ratio`：论文写「从 {L/10, L/6, L/4} 和 {1N, 2N, 3N} 中选」但**没说 BindingGYM 上选了哪个**。→ 采用 repo config 的 `hyper_ratio=4`(L/4) + `num_edges_ratio=3.0`(3N)，即论文 §B.5 报告的最优组合。
   - `num_cvfolds=1`（inter-assay 只训一个 fold），`seed=42`
2. **模型初始化**：与 SKEMPI 一致，从 ProteinMPNN `v_48_020.pt` 加载 inverse-folding 权重（`load_mpnn_state_dict`, `strict=False`），**不**从 SKEMPI 训练好的 checkpoint 继续训练（论文未提及跨数据集迁移）。
3. **训练集**：每个 fold 用其余 4 组的 20 个左右 assay（约 23–35 万行）；`batch_size=1` + `max_iter=20000` ⇒ 每次训练只会看到约 2 万条样本（<10%）。这是论文自己的设定，忠实照做；采样方式用 shuffle 的 `inf_iterator`（与 SKEMPI 路径一致），**不**采用 BindingGYM 官方那种「每个 batch 先抽 assay 再抽 mutant」的 listwise 采样（H3-DDG 用的是 MSE 回归而非 ListMLE，无需同 assay 成组）。
4. **实验次数与评测**：**5 次独立训练**（fold0…fold4），每次评测其留出 fold 的全部行；5 次的预测拼成 25 个 assay 的完整 OOF（合计 376,446 行），再算指标。评测输出按 assay 存成 `{DMS_id}_oof.csv`，与官方目录结构一致，便于直接套用官方 `calc_metric` 逻辑对账。
5. **结构缺失残基的处理**：审计（§3.1b）显示 **unresolved = 0**，此分支实际不会触发；代码仍保留计数与告警。
6. **GPU**：a100 ×1（`sinfo` 显示充足）。同一批实验统一 GPU 型号。
7. **wild-type 行（22 行）的处理**：这些行 `mutant` 为空 → `aa_mut == aa` → `mut_flag` 全 False → `num_mut_chains = 0`，会让 collate 出的 batch 结构失效。规则：**在 side0 上标记单个位置**，使 batch 仍带一个 complex + 一个 isolated-side 行；因 `aa_mut == aa`，`wt_scores == mut_scores` ⇒ **`ddG_pred` 恒为 0**，正是 WT 的物理正确答案。这样既保住 batch 结构，又不引入偏差，且保留了官方 `assert train.shape[0] == df.shape[0]` 所要求的完整行数。
8. **⚠️ RMSE 的定义歧义（实现上两种都算）**：H3-DDG 自己的 `utils.overall_rmse_mae`（`utils.py:499`）**不是裸 RMSE** —— 它在评测数据上现拟合一个 `LinearRegression(pred → true)`，报残差 RMSE，等价于 `std(true)·sqrt(1−r²)`，对预测的任何仿射变换不变。
   - 但 Table 2 的 RMSE **不可能**是这个函数：若是，各方法的 RMSE 只能通过 `sqrt(1−r²)` 相差，而 r∈[0.0998, 0.3057] 只给出 **4%** 的差异；实际 ProteinMPNN 3.4974 vs H3-DDG 1.1294 差 **3 倍**。
   - → 我们**同时报 `rmse_raw` 与 `rmse_calib`**（后者直接复用作者的实现），不做外部猜测。
9. **在训练中的验证节奏**：BindingGYM 的 held-out fold 最大 142,905 行（fold3），全量在线验证不可行，而论文也未给 BindingGYM 的 online-validation 协议。规则：训练中每 `val_freq` 步只在一个**固定的、确定性的 per-assay 子样本**（默认每 assay 200 条，等距抽取）上评测，**仅用于监控、绝不用于 model selection**；训练结束后在 held-out fold 上做**一次全量**评测并输出 OOF。（对照：SKEMPI 那条线的 `validate_all.py` 是在评测集上选模型，见前置验证文档 §2.2-6；BindingGYM 这条线我们**不做任何基于评测集的选择**，只取最终 checkpoint。）

## 5.3 Run config（5-fold job array）

- sbatch：`ibex-records/bindingGYM-reproduce/sh/bindinggym_interassay_5fold_20260817-092000.sh`
- SLURM：`--array=0-4`、`--gres=gpu:a100:1`、`--cpus-per-task=12`、`--mem=128G`、`--time=36:00:00`
- env：`h3ddg-reproduce`（sbatch 内含断言：sklearn 必须是 1.2.1/1.3.2，否则拒跑）
- cache：`data/BindingGYM_cache/{entries,structures}.pkl` **必须在提交 array 之前就位** —— 5 个 array task 同时启动会竞争写同一个 pickle。已在本地构建后 rsync 到 Ibex，双边 **md5 一致**（entries `46c76e45…`、structures `b5deced6…`）。
- **提交时机（用户决定 2026-08-17）**：**等 SKEMPI 前置验证 job 50613272 跑完出结果后再提交本 array**，严格保持「前置验证 → 主实验」的串行顺序。
- job id(s)：_(待填)_

### 5.4 性能实测与 walltime 依据

本地 A4500 实测 eval **2.28 s/batch(bs=4) = 1.76 rows/s**。瓶颈是 3-body hypergraph triplet attention：它在 (K, K) 的 hyperedge 对上做 attention，K = L/4，故成本 ∝ **K³** —— **结构尺寸比突变深度更主导**（4-body attention 虽逐突变位点循环，但只是 20-node 邻域，量级小得多）。

**⚠️ 纯 K³ 外推过于激进，已用 a100 实测校正。** Ibex 上 fold1（K≈28，结构最小）实测 **13.3 rows/s**（240 行 / 18 s，eval_batch_size=8），而非 K³ 外推预测的量级。说明存在很大的**固定开销**（per-batch 的 Python 循环、ProteinMPNN encoder、dataloader），小结构上它才是主导。

两点实测拟合 `t_per_row = a + b·K³`（fold1 on a100 K=28 → 0.0752 s/row；fold4 dominant K≈130 → 约 0.29 s/row）：
**a ≈ 0.073 s/row（固定开销）**，**b ≈ 9.7e-8 s per K³**。据此逐 assay 累加：

| fold | test rows | mean depth | 主导结构（K=L/4，成本最高者）| eval 估算 (a100) |
|---|---|---|---|---|
| 0 | 114,341 | 1.88 | **1HE8 K=228 → 6.5 h 单项**、8BE4 K=151 → 2.2 h | **~10.9 h** |
| 1 | 55,081 | 6.46 | 2M5A K=29、1LP1 K=27（全被固定开销主导）| ~1.2 h ✅实测锚点 |
| 2 | 29,332 | 2.09 | 6M17 K=232 | ~1.4 h |
| 3 | 142,905 | 2.44 | 6M0J K=197 → 4.9 h、1N8Z K=253 | **~9.3 h** |
| 4 | 34,787 | 5.56 | 4ZFF K=130、4ZFG K=162 | ~2.9 h |

训练侧：全库平均 forward ≈ 0.245 s/row，反向+优化器约 3× ⇒ ~0.73 s/iter × 20,000 = **~4.1 h**。
最坏 fold0 ≈ 10.9 + 4.1 ≈ **15 h**，fold3 ≈ 13 h，其余 < 8 h ⇒ walltime **36 h** 余量充足。
另注：所有结构的 K 都 < `max_num_hyperedges=420`，故 3-body attention 对每个 assay 都实际生效（该阈值一旦被超过，代码会**静默跳过**这一层）。
**若某个 fold 仍超时的 fallback**（尚未启用，需要时再评估）：`log_probs` 只依赖 (structure, mut_flag)，同一位点的 19 种替换可共享一次前向 —— 可按 (assay, 突变位点集合) 缓存，但需先确认 `corrupt_chi_angle` 的随机性不影响结果。

## 6. Change log

- **2026-08-17 09:20**：写下 plan；数据下载、inter-assay split 复现、突变映射验证完成。
- **2026-08-17 10:5x**：🔴 发现并修正 **sklearn 版本导致 fold 成员错位**的问题（见 §4.0）。最初的 split 分析误用 base env（sklearn 1.7.2），fold2/3/4 的 assay 归属是错的；已改用 pinned env（1.2.1）重算，并新增 `make_inter_assay_folds.py` 把 fold 成员**一次性固化**到 `data_splits/inter_assay_folds.tsv`（含版本 guard，非 1.2.1/1.3.2 拒绝运行）。后续所有代码读该文件，不再现算。
- **2026-08-17 11:0x**：读通官方 `calc_metric.ipynb`，确认 (a) 官方协议是**跑满 5 fold 收 OOF 再切片**，(b) 官方指标**全部 scale-invariant 且无 RMSE/Pearson**（§4.4）。据此：
  - **用户决定：改为跑满 5 个 fold**（§4.2），fold 选择的歧义作废；
  - 结果**同时报 H3-DDG 口径与 BindingGYM 官方口径两套指标**。
- 本地 SKEMPI smoke 已通过（3 folds 全程无报错，5 iter 后 all-mode Pearson 0.49–0.59，ProteinMPNN 权重加载生效），模型代码 + env 验证完毕。
- **2026-08-17 11:4x**：全量数据审计通过（§3.1b：376,446 行、0 unresolved、0 wt 不符）。
- **2026-08-17 11:5x**：BindingGYM pipeline 实现完成并本地 smoke 通过（fold4，训练 + batch>1 评测索引对齐 + 两套 per-DMS 指标全部正常产出，含 per-assay OOF csv）。发现并记录 §5.2-8 的 RMSE 定义歧义。
- **2026-08-17 12:0x**：cache 构建 + rsync 到 Ibex（md5 双边一致）；**用户决定：等 SKEMPI job 50613272 完成后再提交 5-fold array**。
- **2026-08-17 13:13**：Ibex 侧脚本验证 srun（job 50614840，a100，约 30 s 计算）通过 —— env 断言、cache 读取、fold1 划分（**train 321,365 行 / 20 assays，val 55,081 行 / 5 assays，held-out = Z-domain ×4 + BH3_Bcl-xL**，与固化的 tsv 完全一致）、训练、评测、两套指标、OOF 输出全部正常。同时拿到 a100 真实吞吐并据此校正 §5.4 的 walltime 依据（原先的纯 K³ 外推低估了固定开销）。

## 7. Results

_(待所有 job 完成后填写)_
