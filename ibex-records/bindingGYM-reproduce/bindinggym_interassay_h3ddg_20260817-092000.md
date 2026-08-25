# bindingGYM-reproduce — 主实验：H3-DDG 在 BindingGYM inter-assay split 上的复现 (Table 2)

> created 2026-08-17 09:20 ｜ **status: RUNNING**（array 50674363 已提交；前置 SKEMPI 验证已通过）
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

### 4.0 🔴 复现陷阱：fold 成员依赖 **numpy 版本**（⚠️ 归因已更正）

`GroupKFold` 不接受随机种子，按 group size 降序贪心分配（LPT 装箱）。**25 个 assay 落在 14 个 cluster，size 为 `[1,2,1,1,1,1,1,2,1,6,2,1,4,1]` —— 14 个里有 12 个并列**（九个 1、三个 2），所以分配结果完全由"并列元素的排序次序"决定。

**⚠️ 2026-08-21 更正**：此处先前写作「tie-breaking 在 sklearn 1.6 的重构中变了」，**该归因是错的**。逐行对比 `GroupKFold._iter_test_indices` 的源码证明：**sklearn 1.2.1 与 1.7.2 的非-shuffle 分支完全相同**（1.6 只是新增了一个独立的 `if self.shuffle` 分支）。真正起决定作用的是其中这一行里的 numpy：

```python
indices = np.argsort(n_samples_per_group)[::-1]   # 默认 kind='quicksort'，不稳定排序
```

不稳定排序对并列元素不保证次序，而该次序随 numpy 版本变化。在本数据的权重向量上实测：

| 环境 | numpy / sklearn | `argsort(w)[::-1]` | fold 分配 | 指纹 |
|---|---|---|---|---|
| **H3-DDG README 指定** | 1.22.4 / 1.2.1 | `[9,12,10,7,1,13,11,8,6,5,4,3,2,0]` | **A** | `d23e15f9…` |
| **BindingGYM.yml 指定** | 1.24.4 / 1.3.2 | 同上 | **A** | `d23e15f9…` |
| base env（对照）| 2.3.5 / 1.7.2 | `[9,12,10,7,1,13,`**`8,11,5,6`**`,4,3,2,0]` | **B** | `d217e2bf…` |

B 相对 A 的差异：`CD19↔CXCR4` 在 fold2/3 互换、`SARS2-RBD↔HLA-A2` 在 fold3/4 互换（4 个 assay 换组）。

**✅ 对本项目的实际影响：无。** 两篇论文各自声明的环境给出**完全相同**的划分 A，而 `data_splits/inter_assay_folds.tsv` 的指纹正是 `d23e15f9…` —— 与二者逐位一致。base env 那次误用只发生在最初的探查阶段，当天即发现并改正（commit `1a38c0f`），从未进入任何实验。

**为什么归因错了、结果却对**：当初锁 sklearn 1.2.1 时，实际上是把整个 pinned env（含 numpy 1.22.4）一起锁住了 —— 歪打正着；更关键的是**把结果固化成 committed 文件**这一步对根因是不可知论的，无论真凶是谁都能挡住。

**加固后的三道防线**（`make_inter_assay_folds.py`，2026-08-21）：
1. **numpy 版本 guard**（`1.22.4` / `1.24.4`）—— 补上了真正的风险点，此前只锁 sklearn
2. sklearn 版本 guard（`1.2.1` / `1.3.2`）—— 保留，用于断言整个 pinned env
3. **指纹校验**：生成后与 `EXPECTED_FINGERPRINT = d23e15f9…` 比对，不符则拒绝写盘。**这道防线不依赖归因是否正确** —— 即使出现未见过的版本组合，也不可能静默产出不同划分

三种情形均已实测：正确环境通过；base env 被版本 guard 拦下；**用 `--allow_any_version` 绕过版本 guard 后，指纹防线仍然拦住**。

> 残余不确定性（诚实记录）：BindingGYM 未发布 fold 成员清单，只发布 cluster 表 + 代码。故"正确"的最强含义是「在两篇论文各自声明的环境下由官方代码确定性重算，且两者一致」。若作者实机所用 numpy 与其 yml 声明不符，仍可能存在差异 —— 但这一点无法从公开材料证伪。

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
8. **✅ RMSE 的两种口径（已证实，非推测）**：H3-DDG 自己的 `utils.overall_rmse_mae`（`utils.py:499`）**不是裸 RMSE** —— 它在**正被评测的这批数据上**现拟合一个 `LinearRegression(pred → true)`，把 pred 仿射校正到 gt 的标度后再报残差 RMSE。闭式解为

   $$\text{RMSE}=\text{std}(\Delta\Delta G_\text{true})\cdot\sqrt{1-\text{Pearson}^2}$$

   因此它 (a) 对预测的任何仿射变换不变（完全不惩罚尺度错位），(b) **不是独立于 Pearson 的信息**，只是把 Pearson 换算成 label 单位重新表达，(c) 在测试集上拟了 2 个参数。

   **SKEMPI Table 1 = 这个校正 RMSE（已确证）**。用上式反推每行隐含的 `std(ddG_true)`，与数据实测（all 2.0667 / single 1.7392 / multiple 2.6982）对照：

   | 方法 | mode | Pearson | RMSE | 反推 std | 实测 std |
   |---|---|---|---|---|---|
   | H3-DDG | all | 0.7501 | 1.3665 | 2.0663 | 2.0667 |
   | BA-DDG | all | 0.7118 | 1.4516 | 2.0667 | 2.0667 |
   | Prompt-DDG | all | 0.6772 | 1.5207 | 2.0667 | 2.0667 |
   | RDE-Network | all | 0.6447 | 1.5799 | 2.0668 | 2.0667 |
   | H3-DDG | single | 0.7471 | 1.1560 | 1.7391 | 1.7392 |
   | H3-DDG | multiple | 0.7341 | 1.8320 | 2.6979 | 2.6982 |

   吻合到 4–5 位有效数字、跨方法跨子集自洽。（`Rosetta` 反推 1.7019、`FoldX` 2.0082、`ProMIM` 2.0471、`DiffAffinity` 2.0454 —— 这几行是从别的论文抄的，未用此函数重算。）

   **BindingGYM Table 2 ≠ 这个校正 RMSE（算术排除）**：若是，各方法 RMSE 只能通过 `sqrt(1−r²)` 相差，r∈[0.0998, 0.3057] ⇒ 最多差 **4.5%**；而实际 ProteinMPNN 3.4974 vs H3-DDG 1.1294 差 **3.1 倍**。两张表口径不同。

   → **对标规则**：SKEMPI Table 1 用 `rmse_calib`，BindingGYM Table 2 用 `rmse_raw`。`bindinggym_metrics.py` 两者都算并并列输出。
9. **在训练中的验证节奏**：BindingGYM 的 held-out fold 最大 142,905 行（fold3），全量在线验证不可行，而论文也未给 BindingGYM 的 online-validation 协议。规则：训练中每 `val_freq` 步只在一个**固定的、确定性的 per-assay 子样本**（默认每 assay 200 条，等距抽取）上评测，**仅用于监控、绝不用于 model selection**；训练结束后在 held-out fold 上做**一次全量**评测并输出 OOF。（对照：SKEMPI 那条线的 `validate_all.py` 是在评测集上选模型，见前置验证文档 §2.2-6；BindingGYM 这条线我们**不做任何基于评测集的选择**，只取最终 checkpoint。）

## 5.3 Run config（5-fold job array）

- sbatch：`ibex-records/bindingGYM-reproduce/sh/bindinggym_interassay_5fold_20260817-092000.sh`
- SLURM：`--array=0-4`、`--gres=gpu:a100:1`、`--cpus-per-task=12`、`--mem=128G`、`--time=36:00:00`
- env：`h3ddg-reproduce`（sbatch 内含断言：sklearn 必须是 1.2.1/1.3.2，否则拒跑）
- cache：`data/BindingGYM_cache/{entries,structures}.pkl` **必须在提交 array 之前就位** —— 5 个 array task 同时启动会竞争写同一个 pickle。已在本地构建后 rsync 到 Ibex，双边 **md5 一致**（entries `46c76e45…`、structures `b5deced6…`）。
- **提交时机（用户决定 2026-08-17）**：**等 SKEMPI 前置验证 job 50613272 跑完出结果后再提交本 array**，严格保持「前置验证 → 主实验」的串行顺序。
- ~~array **`50674363_[0-4]`**（5 × 23 h，2026-08-18 22:5x 提交）~~ → 排队过久，2026-08-19 scancel，改为 5 个独立 job（见 §5.5）
- **job id(s)**（2026-08-19 提交，全部 `gpu,gpu24`）：

| fold | job | walltime |
|---|---|---|
| 0 | `50680591` | 18 h |
| 1 | `50680592` | 8 h |
| 2 | `50680593` | 8 h |
| 3 | `50680595` | 16 h |
| 4 | `50680596` | 10 h |

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

## 5.5 🔄 排队优化：array → 5 个独立 job + 差异化 walltime + 定期 checkpoint（用户决定 2026-08-19）

**起因**：`50674363_[0-4]`（5 × 23 h）排了 22 轮（约 11 h）始终 PENDING，`squeue --start` 预估一路在 `08-20` ～ `08-23` 之间漂移，最坏时约 4.5 天。同期集群 a100 **268 卡仅 11 张空闲（96% 占用）**，排队等 a100 的 job 从两天前的 133 个涨到 **333 个**。统一的 23 h walltime + 5-task array 形状让 backfill 很难插空。

### (1) 拆成 5 个独立 job

- job 脚本：`sh/bindinggym_perfold_20260819.sh`（fold 由位置参数 `$1` 传入）
- 提交器：`sh/submit_bindinggym_perfold.sh`（内含 walltime 表，可审计）
- 依据：SKEMPI 那次从「单 job 48 h」拆成「array 3 × 14 h」后，partition 从 `gpu,gpu72`（35 节点）变为 `gpu,gpu24`（49 节点），预估排队从 3 天降到 1.2 天，最终实际约 13 h 开跑。

### (2) 按 fold 差异化压缩 walltime

原先 23 h 是按最坏的 fold0 定的，但各 fold 成本差一个数量级（§5.4）。改为逐 fold 按自身实测/估算给：

| fold | eval 估算 | + 训练 ~4 h | walltime | 主导结构 |
|---|---|---|---|---|
| 0 | ~10.9 h | ~15 h | **18 h** | 1HE8 K=228 |
| 1 | ~1.2 h（a100 实测锚点）| ~5.3 h | **8 h** | 2M5A/1LP1 K≈28 |
| 2 | ~1.4 h | ~5.5 h | **8 h** | 6M17 K=232 |
| 3 | ~9.3 h | ~13.4 h | **16 h** | 6M0J K=197 |
| 4 | ~2.9 h | ~7 h | **10 h** | 4ZFF K=130 |

全部 ≤ 24 h，保住最大的 a100 池 `gpu24`（32 节点）。最坏 GPU 预算由 5×23 = 115 GPU-h 降到 **60 GPU-h**。

### (3) 新增训练中的 model-weights checkpoint + resume

**动机**：SKEMPI 的 f1 在 97.4% 处被 walltime 杀掉，权重全丢 —— 因为作者代码的定期保存被 `if args.num_cvfolds == 1:` 门控住了，只在训练循环**结束后**才存一次。有了差异化（更紧的）walltime，这个风险必须堵住。

`train_bindinggym.py` 的改动：

- **每 `ckpt_freq` 步存一个 resume point**（config 默认 `5000`，即 20,000 iter 存 3 次 + 收尾 1 次 —— 按用户要求"不用太密"）
- 存的是 `cv_mgr.state_dict()`，**含 model + optimizer + scheduler**，故可真正续训而非只留权重
- **原子写入**：先写 `.tmp` 再 `os.replace()`，被 kill 在写盘中途也不会毁掉唯一副本
- 新增 `--resume`：从 `checkpoint/resume_fold{F}.pt` 读回并从 `iteration+1` 继续；找不到就从头开始（幂等，可安全放进 sbatch 常开）
- 新增 `--save_dir`：resume 必须有**固定**目录（默认目录带时间戳，重投会指向一个全新空目录）。该路径**刻意不用 `check_dir()`** —— 它的 `overwrite=True` 会 `shutil.rmtree`，正好会删掉 resume 所需的 checkpoint
- ⚠️ **已知局限**：resume 不恢复 DataLoader 的抽样位置，续训后的样本序列与不间断跑不完全一致（与 §2.3 里 SKEMPI 那条同类的"随机抽样差异"，不影响数据划分）

⇒ walltime 到点不再是灾难：最多损失一个 ckpt 区间（≤5,000 iter），重投带 `--resume` 即可接上。

**本地实测（2026-08-19，A4500）**：
1. 第一次运行 `max_iter=7, ckpt_freq=3` → 在 iter 3、6 各存一次 resume point（27.8 MB，单文件覆盖写）
2. 第二次带 `--resume` 且 `max_iter=12` → 日志打出 `[resume] loaded ... @ iteration 6; continuing at 7/12`，续训后继续正常存点并跑到 `DONE`

✅ 机制验证通过后才提交 Ibex（§1b local-first）。

## 5.6 运行日志（per-fold jobs）

| fold | job | walltime | 开跑时间 | 状态 |
|---|---|---|---|---|
| 1 | `50680592` | 8 h | **2026-08-20 06:30** | RUNNING |
| 2 | `50680593` | 8 h | 预估 `08-21 06:40` | PENDING |
| 0 | `50680591` | 18 h | 预估 `08-22 15:52` | PENDING |
| 3 | `50680595` | 16 h | 预估 `08-22 21:08` | PENDING |
| 4 | `50680596` | 10 h | 预估 `08-22 23:00` | PENDING |

**排队实况**：5 个 job 于 2026-08-19 提交，f1 排队约 **20 h** 后开跑。期间 `squeue --start` 的预估在 `08-21`～`08-25` 之间反复漂移（详见 §5.5 的分析：Ibex 用 `sched/backfill`，`bf_interval=120` 每 2 分钟重算，预估是"假设所有 job 跑满 walltime"的保守上界）。同期集群 a100 **占用 96%（268 卡仅 11–20 张空闲）**、等 a100 的 pending job **330+**，而我们的 `sprio` 优先级 573 分中 **fairshare 占 572、AGE 贡献 0** —— 即排队时长不加分，只能等 backfill 找到空隙。

**拆分策略得到验证**：f1 是 5 个里 walltime 最短的（8 h），也是第一个被 backfill 插进空隙的。此前它曾三次拿到较早的预估窗口又被抢走，第四次坐实。

### f1 启动检查（全部通过）

```
torch 1.13.1+cu117 | cuda 11.7 | gpu NVIDIA A100-SXM4-80GB
numpy 1.22.4 | pandas 1.5.3 | sklearn 1.2.1 | biopython 1.81   ← 版本 guard 通过
[fold 1] train: 321365 rows / 20 assays | val: 55081 rows / 5 assays
[fold 1] held-out assays: ['BH3_Bcl-xL_normed_1PQ1', 'Z-domain_ZSPA-1_LL1_fitness_1LP1',
                           'Z-domain_ZSPA-1_LL2_fitness_1LP1', 'Z-domain_ZpA963_HL1_fitness_2M5A',
                           'Z-domain_ZpA963_HL2_fitness_2M5A']
[resume] no checkpoint at ./results/bindinggym_interassay_fold1/checkpoint/resume_fold1.pt; starting from scratch
```

- 划分与固化的 `data_splits/inter_assay_folds.tsv` **完全一致**（§4.1 的 fold1 行）
- **`--resume` 的幂等分支按设计工作**：首次运行无 checkpoint 时正常从头开始而非报错，故可永久留在 sbatch 里

## 5.7 ⚠️ 训练中观察：per-DMS 相关性随训练**下降**，f1 甚至转负

监控子集（每 assay 200 行，共 1,000 行）上的 per-DMS 指标随 iteration 的变化：

| | iter 1（未训练，纯 ProteinMPNN init）| iter 5001 | iter 10001 |
|---|---|---|---|
| **f1** Pearson | **+0.3008** | −0.0639 | **−0.2642** |
| **f1** Spearman | +0.3240 | −0.1306 | **−0.3785** |
| **f1** rmse_raw | 6.7240 | 0.9425 | 0.9811 |
| **f2** Pearson | +0.0934 | −0.0014 | +0.0677 |
| **f2** Spearman | +0.1004 | −0.0350 | +0.0369 |
| **f2** rmse_raw | 1.3750 | 1.1758 | 1.1741 |

**现象**：f1 的相关性从 +0.30 单调跌到 −0.26（符号翻转，远超噪声）；f2 在 0 附近震荡、无改善。**与此同时 RMSE 大幅下降**（f1 6.72 → 0.94）。

**解读（与 §4.4 的分析一致）**：模型确实在优化 MSE 目标 —— 它学会了预测值的**数值尺度**，但这与"在每个 assay 内部把突变体排对序"是两个不同目标。训练集把 20 个 assay 混在一起做裸 `F.mse_loss`，而这些 assay 的 label 尺度差 **63 倍**（`5A12_Ang2` std 0.079 ↔ `CD19` std 4.97），pooled MSE 的最优解是拟合全局尺度，可能与 within-assay ranking 冲突。BindingGYM 官方正是用 **ListMLE + 同 assay 内 batch** 规避此问题；H3-DDG 论文对此只字未提，其发布代码用的就是裸 MSE（`ddg_predictor.py:72`），我们忠实照做（§5.2-3）。

**尚不能下结论的三点**：
1. 这是 **1,000 行子样本**的监控值，非最终指标；须等训练结束后 held-out fold 的**全量**评测（f1 = 55,081 行）。
2. f1 的 5 个 held-out assay 最特殊（4 个 Z-domain 是双方同时突变的 coevolution 研究，见 §3.1b 的 53,671 行"两侧突变"）；f2 平缓得多，未必是全局现象。
3. ⭐ **f1 未训练时的 +0.3008 与论文 Table 2 的 ALL Pearson 0.3057 几乎相同** —— 可能是巧合，但"论文数值是否接近未训练基线"值得在拿到全量 OOF 后专门核对。

**处置**：不做任何改动，按计划跑完 5 个 fold、拿到完整 OOF 再统一分析。若全量结果确认此趋势，候选诊断方向为：(a) 按 assay 标准化 label 后重训；(b) 改用 ListMLE 复现 BindingGYM 官方口径作为对照。两者都属于**偏离论文**的改动，需先与用户确认。

## 5.8 全量 OOF 结果（陆续填入，5 个 fold 齐了才能算最终口径）

### f1（`50680592`，COMPLETED 02:47:18，exit 0:0）
held-out：Z-domain ×4 + BH3_Bcl-xL，55,081 行

| slice | Pearson | Spearman | AUROC | rmse_raw | rmse_calib | n_assays | n_rows |
|---|---|---|---|---|---|---|---|
| ALL | **0.1108** | 0.0904 | 0.6098 | 0.9723 | 0.4245 | 5 | 55,081 |
| <3 | 0.1334 | 0.1149 | 0.6060 | 0.9589 | 0.5611 | 3 | 685 |
| ≥3 | 0.1243 | 0.1193 | 0.6461 | 0.9942 | 0.3572 | 5 | 54,324 |

BindingGYM 官方口径：ALL Spearman 0.0904 / AUC 0.5718 / MCC 0.0294 / NDCG 0.5390 / AP 0.1327

### f2（`50680593`，COMPLETED 02:31:51，exit 0:0）
held-out：ACE2, CXCR4, PSD95_CRIPT, PSD95_Tm2F, hYAP65，29,332 行

| slice | Pearson | Spearman | AUROC | rmse_raw | rmse_calib | n_assays | n_rows |
|---|---|---|---|---|---|---|---|
| ALL | **0.0279** | 0.0158 | 0.5100 | 1.1457 | 0.8931 | 5 | 29,332 |
| <3 | 0.0270 | 0.0179 | 0.5100 | 1.0111 | 0.8364 | 5 | 18,241 |
| ≥3 | 0.0230 | 0.0384 | NaN | 2.2428 | 1.1236 | 1 | 11,091 |

BindingGYM 官方口径：ALL Spearman 0.0158 / AUC 0.4937 / MCC −0.0065 / NDCG 0.4837 / AP 0.0992

### f4（`50680596`，COMPLETED 04:05:46，exit 0:0）
held-out：5A12_Ang2, 5A12_VEGF, BH3_Mcl-1, HLA-A2，34,787 行

| slice | Pearson | Spearman | AUROC | rmse_raw | rmse_calib | n_assays | n_rows |
|---|---|---|---|---|---|---|---|
| ALL | **−0.1103** | −0.1508 | 0.4275 | 0.8297 | 0.6427 | 4 | 34,787 |
| <3 | −0.0952 | −0.0683 | 0.4643 | 0.9418 | 0.6108 | 4 | 3,965 |
| ≥3 | **−0.1705** | −0.2575 | 0.3697 | 0.7540 | 0.4849 | 3 | 30,822 |

BindingGYM 官方口径：ALL Spearman −0.1508 / AUC 0.4329 / MCC −0.0115 / NDCG 0.4369 / AP 0.0934

**这是首个在全量数据上转为负相关的 fold**，AUROC 0.4275 也低于随机。

### 三个已完成 fold 的一致规律

| fold | 未训练基线（监控子集）| 训练后（全量）| Δ |
|---|---|---|---|
| f1 | +0.3008 | +0.1108 | −0.19 |
| f2 | +0.0934 | +0.0279 | −0.07 |
| f4 | +0.1230 | **−0.1103** | −0.23 |

**三个 fold 全部下降，无一例外**；同期 rmse_raw 全部大幅下降（f1 6.72→0.97、f2 1.38→1.15、f4 2.21→0.83）。这把 §5.7 的机制解读从"单 fold 观察"升级为**跨 fold 的一致现象**：pooled MSE 让模型学到 label 的数值尺度（RMSE 改善），却系统性破坏 within-assay ranking（相关性下降）。

⚠️ 仍缺 f0（KRAS ×6，114,341 行）与 f3（4D5/CD19/GB1×2/SARS2-RBD，142,905 行）—— 这两个占全部 376,446 行的 **68%**，且是最主流的 assay。最终对标必须等它们完成后拼成 25-assay 的完整 OOF。

### 阶段性观察

**① §5.7 的趋势在全量数据上得到证实。** f1 未训练时（iter 1，1,000 行子样本）Pearson **+0.3008**，训练完在 **55,081 行全量**上只有 **0.1108**。训练确实**降低**了 per-DMS 相关性；未跌到子样本上那种负值，说明子样本（1,000 行）噪声偏大，但方向一致。

**② 与论文 Table 2 的差距（暂以单 fold 计，非最终口径）**：f1 ALL Pearson 0.1108 仅为论文 0.3057 的 **36%**；f2 的 0.0279 近乎无相关。

**③ 切片口径正确，metrics 实现无误的旁证**：f1 的 <3 只有 3 个 assay（685 行）、f2 的 ≥3 只剩 1 个（hYAP65 11,091 行），正是 BindingGYM 官方 ≥100 行过滤的预期行为；f2 ≥3 的 AUROC 为 NaN 是该切片内 `ddG>0` 只有单一类别所致（代码已按单类返回 NaN 而非报错）。

**④ 尚不能定论**：这是 5 个 fold 里最"偏"的两个（f1 全是 Z-domain 双侧突变；f2 的 ≥3 只有 hYAP65）。f0/f3/f4 覆盖 KRAS×6、GB1×2、4D5、SARS2-RBD、CD19 等主流 assay，占 376,446 行的绝大部分。**必须等 5 个 fold 齐了拼成完整 OOF、按 25 assay 等权平均**，才是可与 Table 2 直接比较的数字。

## 5.9 f0/f3 的 walltime 下调（2026-08-24，临时排队操作）

**问题**：f0/f3 按 §5.5 的 walltime 表申请了 18h/16h，从 2026-08-19 10:19 提交后**连续 PENDING 5 天 6 小时**，预估开始时间还在往后漂（08-25 16:02 → 08-26 09:06）。同期 8h 的 f1/f2 只排了约 20h、10h 的 f4 排了约 33h 就跑起来了。

**根因**：§5.5 的 walltime 是按 `t ≈ a + b·K³` 外推的，**高估了约 3 倍**。三个已完成 fold 的实测 elapsed：

| fold | 申请 | 实测 elapsed | 用掉比例 |
|---|---|---|---|
| f1 | 8h | 2:47:18 | 35% |
| f2 | 8h | 2:31:51 | 32% |
| f4 | 10h | 4:05:46 | 41% |

模型每个 fold 都跑固定 20,000 iter，训练时间几乎相同，fold 之间的差异**只来自 eval**；而上表的离散度已经基本就是 eval 的离散度。K³ 外推把 eval 的结构尺寸依赖放得太大了。

**处理**：用 `scontrol update jobid=<id> TimeLimit=07:00:00` **原地下调**，而不是 `scancel` + 重投 ——
- 用户可以下调自己 pending job 的 TimeLimit（只有 admin 能上调），**不丢已积累的 5 天排队资历**；重投会把 submit time 重置，白等 5 天。
- 下调后两个 job 的 `NODELIST(REASON)` 从 `(Priority)` 变成 `(None)`，预估开始时间提前到 08-26 05:50 / 07:44。

7h 的依据：实测上限 4:05:46 + 71% buffer。**大概率一个 job 就能跑完**。

**万一 7h 不够**：`train_bindinggym.py` 每 5,000 iter 存一次 resume point，用 `--resume` 续跑即可。为此专门加了一对**临时**脚本，与正常流程的 `sh/bindinggym_perfold_20260819.sh`（产出 f1/f2/f4 的那个）严格分开：

- `sh/TEMP_bindinggym_chunked_20260824.sh` —— 7h、可续跑，开头会打印本 chunk 从第几个 iteration 接上；`srun` 正常退出才会打印 `FULLY DONE`，所以「日志里没有 FULLY DONE」就是需要发 chunk 2 的信号。
- `sh/TEMP_submit_chunked.sh 0 3` —— 提交器。

两者**只在 f0/f3 收尾期间存在，跑完即删**。训练本身没有任何改动：同一份 config、同一份固化的 fold split、同一条代码路径，变的只有 walltime 和「可能需要两个 job」这个预期。

⚠️ **注意**：eval 阶段没有 checkpoint。如果 chunk 1 在训练完成后、eval 途中被砍，chunk 2 会 `start_it == max_iter` 直接跳过训练、重跑整个 eval —— 结果正确，只是浪费一次 eval。

## 5.10 诊断：模型塌缩，而非「学到 assay 尺度」（2026-08-24）

用户判断「本次训练有问题」。f0/f3 已 `scontrol hold` 暂停。以下诊断**全部零算力**，只用三个已完成 fold 的 OOF csv（119,200 行 / 14 assay）和日志。脚本：`diagnostics/diag_oof.py`、`diagnostics/verify_align.py`。

### ① 先排除：评测对齐是对的

`collect_results()` 里 `id` 用下标 `i`（batch 内位置）、`ddG`/`ddG_pred` 用下标 `k`（`complex_row_indices` 内位置）。这两个错位就会让所有结果按构造变成噪声。逐行对回未经修改的 `data/input/Binding_substitutions_DMS/*.csv`：

**119,200 行零错配（max|Δ| ~1e-7，float32 往返误差），14 个 assay 行数全部完全覆盖、无重复。** `eval_batch_size=8` 下的对齐验证通过，排除。

### ② §5.7/§5.8 的「pooled MSE 学到 assay 尺度」解释是**错的**

方差分解（把 `ddG_pred` 的方差拆成 between-assay / within-assay）：

| fold | 量 | between | within | between 占比 |
|---|---|---|---|---|
| 1 | ddG（真值）| 0.0382 | 0.0563 | 40.4% |
| 1 | **ddG_pred** | 0.0041 | 0.0915 | **4.3%** |
| 2 | ddG（真值）| 0.9250 | 1.3300 | 41.0% |
| 2 | **ddG_pred** | 0.0003 | 0.0673 | **0.5%** |
| 4 | ddG（真值）| 0.0100 | 0.9110 | 1.1% |
| 4 | **ddG_pred** | 0.0002 | 0.0158 | 1.4% |

真值有 40% 的方差在 assay 之间，**预测只有 0.5%–4.3%**。模型根本没在编码「这是哪个 assay」。原解释推翻。

### ③ 真实情况：预测塌缩成近似常数

| 量 | ddG（真值）| ddG_pred |
|---|---|---|
| std | 1.198 | **0.263** |
| IQR | [−0.727, 1.099] | **[−0.055, 0.163]** |
| mean | 0.252 | 0.056 |

逐 assay 的 `pred_std / true_std` 大多在 **0.11–0.25**。逐 assay Pearson **7 正 7 负，等权均值 +0.0180**，Spearman −0.0052 —— 与零不可区分。

（顺带排除第二个嫌疑：如果是 label 方向在 assay 之间不一致，应该看到清晰的双峰分布；实际是全部挤在 0 附近的散点。**符号假设不成立**。唯一例外是 `Z-domain_ZpA963_HL1` 的 +0.4752 和 `BH3_Mcl-1` 的 −0.3041。）

### ④ 塌缩发生在**前 5,000 个 iteration 之内**，之后再无变化

f1 的监控轨迹：

| iter | Pearson | rmse_raw |
|---|---|---|
| 1（未训练）| **+0.3008** | **6.7240** |
| 5001 | −0.0639 | 0.9425 |
| 10001 | −0.2642 | 0.9811 |
| 15001 | +0.1818 | 0.9329 |
| 20000（全量）| +0.1108 | 0.9723 |

`rmse_raw` 在前 5,000 步从 **6.72 掉到 0.94，之后四次读数全部锁在 0.93–0.98**；同期 Pearson 在 −0.26 ↔ +0.18 之间**震荡而非收敛**。这正是「输出塌到近似常数、之后相关性只是残余噪声」的签名。f2（0.0934→−0.0014→0.0677→0.0666→0.0279）和 f4（0.1230→−0.0841→−0.0028→0.0346→−0.1103）同型。

### ⑤ 机制：label 不是 kcal/mol，而 MSE 最省力的解是砍掉输出

未训练 `rmse_raw = 6.72` 是因为热力学循环输出的是 kcal/mol 量级的能量，而 BindingGYM 的 label 是**各 assay 自己单位的 DMS score**（逐 assay true_std 从 0.0787 到 1.4300，跨 18 倍）。MSE 下降最快的方向不是改善排序，而是**把输出整体缩小约 7 倍去贴 label 的尺度**——它做到了，然后就死了。

**这就是为什么 SKEMPI 没事**：SKEMPI 的 label 本身就是 kcal/mol，预训练输出天生在正确尺度上，梯度可以全部花在排序上。这个失效模式对 BindingGYM 是特异的。

### ⑥ 一个 10× 的超参差异 —— 待核实，可能是我的 bug

| | 作者 `config/train_h3-ddg.json`（SKEMPI）| 我的 `config/train_h3-ddg_bindinggym.json` |
|---|---|---|
| `lr` | **4e-05** | **4e-04**（10×）|
| `max_iter` | 38000 | 20000 |
| 训练集行数 | ~5–7k | 321,365（f1）|
| **等效 epoch** | **~6** | **0.06** |

`lr=4e-4` 是我按「论文 Appendix A.4」记下的（§5.1 第 224 行），但**无法复核**：OpenReview 的 PDF 被 Cloudflare 挡住，`curl` 和 WebFetch 都只拿到验证页。作者仅发布了 SKEMPI 的 config，其中 `lr` 是 4e-05。

`lr=4e-4` + `batch_size=1`，作用在一个「只需要把输出缩小 10 倍」的目标上，正是能在 5,000 步内塌缩的配置。**这是当前的头号嫌疑。**

### ⑦ 一个值得注意的巧合

**f1 未训练 = +0.3008，论文 Table 2 = 0.3057。**

由此产生一个可检验的假设：**Table 2 可能报的是 zero-shot 数值**（即不在 BindingGYM 上训练，用 SKEMPI 训练好的或 ProteinMPNN 预训练权重直接评测）。旁证：BindingGYM 官方的指标函数名字就叫 **`calc_zero_shot_metric`**（§4.4 里我逐字移植的那个），且 BindingGYM 论文本身同时有 zero-shot 和 supervised 两条赛道。

### ⑧ 下一步（按性价比排序）

1. **决定性实验：iter-0 未训练全量评测，25 个 assay。** 不训练，只跑一遍 eval。落在 ~0.30 ⇒ Table 2 是 zero-shot 口径，pipeline 无需修；落在 ~0 ⇒ pipeline 仍有 bug，而 f1 的 +0.3008 只是 1,000 行子样本的运气。**无论哪个结果都能终结「分支 A vs 分支 B」的二分。**
2. **核实论文 Appendix A.4 的 lr**（需要能打开 PDF 的浏览器）。
3. **lr 对照：4e-5 vs 4e-4**，单 fold 即可，看塌缩是否消失。

⚠️ 以上 1/3 都是**论文之外的诊断实验**，不改变已报告的复现口径；per-assay 标准化、换 ListMLE 等偏离论文的改动仍未获批准，不做。

## 5.11 读到论文原文后的三处更正与一个新发现（2026-08-24）

拿到 `repos/Sources/H3-DDG (NIPS 2025).pdf` 后逐字核对 §3.4 / §4.1 / §4.3 / Table 2 / A.2–A.4。

### ① `lr=4e-4` 不是我的 bug —— 但 A.4 与作者自己发布的 config 矛盾

A.4.1 原文：

> We used the Adam optimizer with a **learning rate of 4e-4** and a **batch size of 1, 2**, depending on
> GPU memory and graph size. The model was trained for **20,000 iterations** with 4 attention heads
> and a hidden dimension of 128.

所以 §5.10 ⑥ 里「可能是我抄错」的猜测**撤回**，`config/train_h3-ddg_bindinggym.json` 是忠实转录。

**但 A.4 是一个不分数据集的全局章节，而它与作者仓库里唯一发布的 config 直接矛盾**：

| | A.4 原文 | 作者 `config/train_h3-ddg.json`（唯一发布的）|
|---|---|---|
| `lr` | 4e-4 | **4e-05** |
| `max_iter` | 20,000 | **38,000**（commit `d03eda5` 从 50,000 改下来）|

**而我们复现出 Table 1 用的是发布的 config（4e-5 / 38k），不是 A.4（4e-4 / 20k）。** 也就是说 **A.4 的数字连它自己论文里唯一可验证的那个 run（SKEMPI）都描述不了**。这把 A.4 从「规格说明」降级为「不可靠的二手描述」，那么它对 BindingGYM 是否可靠同样无从保证。

### ② zero-shot 假设被 Table 2 自己推翻

Table 2 里 **ProteinMPNN 是一行独立的 baseline**：ALL Pearson 0.0998 / Spearman 0.2050 / AUROC 0.5341 / RMSE **3.4974**。H3-DDG 的 0.3057 与它并列出现，所以 **0.3057 是监督训练的数值，不是 zero-shot**。§5.10 ⑦ 的假设作废。

（RMSE 3.4974 这个量级本身印证了 §5.10 ⑤ 的机制：zero-shot 的 ProteinMPNN 输出在 kcal/mol 尺度上，对着 DMS score 的 label 算 RMSE 就是这么大 —— 和我们 iter-1 测到的 6.72 同源。另外我们未训练测到 +0.3008，是论文 ProteinMPNN 行 0.0998 的 3 倍、BA-Cycle 0.1320 的 2.3 倍，**这个数字反而偏高得可疑**，很可能是 1,000 行子样本的运气，这也是为什么 iter-0 全量 baseline 值得单独跑。）

### ③ 损失函数确认就是裸 MSE，全文无任何 label 归一化

§3.4 式 (17)：`L_MSE = (1/n) Σ (ΔΔG_pred − ΔΔG_true)²`。全文（含附录）没有出现 per-assay 标准化 / 归一化。

而 §4.3 结尾作者**明确承认了尺度问题**：

> While H3-DDG's RMSE under single-point mutations is slightly higher than the baseline methods,
> this can be attributed to the BindingGYM dataset spanning different DMS experiments, where the
> **absolute ΔΔG values vary significantly across experiments**. However, we focus more on the
> ranking and correlation within each DMS experiment.

即：作者知道跨 assay 的 label 尺度不可比，声明只关心 within-DMS 排序，**但用的仍是裸 MSE**。这正是 §5.10 ⑤ 那个失效模式的配方。

### ④ 论文的单 fold 是哪个 —— f0 排除，f1/f3 二选一，f3 更可能

§4.1 原文：「the hardest inter-assay split, **focusing on the fold with the most multi-point mutations** for testing」。逐 fold 统计（`diagnostics/pin_fold.py`，突变数按 `bindinggym.py` 同一套逻辑解析 `mutant_pdb` dict）：

| fold | held-out 行数 | ≥2 | ≥3 | ≥3 占比 | ≥100 行过滤后 ALL/<3/≥3 的 assay 数 |
|---|---|---|---|---|---|
| 0 | 114,341 | 100,129 | **0** | 0.0% | 6 / 6 / **0** ← 填不出 Table 2 的 ≥3 列 |
| 1 | 55,081 | 54,819 | **54,324** | **98.6%** | 5 / 3 / 5 |
| 2 | 29,332 | 18,118 | 11,091 | 37.8% | 5 / 5 / **1** |
| 3 | **142,905** | **137,077** | 35,775 | 25.0% | 5 / 4 / 4 |
| 4 | 34,787 | 31,165 | 30,822 | 88.6% | 4 / 4 / 3 |

- **f0 直接排除**：≥3 切片一个 assay 都活不下来（max_mut = 2），Table 2 的 ≥3 列无从产生。
- **「multi-point」= ≥3 ⇒ f1**（绝对数 54,324 与占比 98.6% 双第一）—— f1 **已跑完**，ALL Pearson 0.1108 vs 论文 0.3057。
- **「multi-point」= ≥2 ⇒ f3**（137,077）—— f3 正是我们刚暂停的两个之一。

尝试用 Table 2 的 RMSE 比值做第二个独立指纹（`diagnostics/rmse_sig2.py`）：所有 5 个方法都呈 `RMSE(≥3)/RMSE(ALL) ≈ 2.08–2.35`。若弱预测器的 RMSE 追随 label spread，这个比值应能反推 fold。结果 **两种口径、5 个 fold、外加 25-assay 全集，比值全在 0.75–1.52，没有一个接近 2.2 —— 此检验不成立，结论 inconclusive**，不能用来定 fold。

但同一批数字给出一个较弱的旁证：论文 ALL RMSE = 1.1294，而各候选的 per-DMS label spread 是 f0 0.4614 / f1 0.4399 / f2 0.8941 / **f3 2.0090** / f4 0.6504 / 25-assay 0.8834。**只有 f3 的 label spread 大于论文 RMSE**（即论文的模型比「逐 assay 猜均值」更好）；在 f1 上论文的 RMSE 会比猜均值差 2.6 倍。这偏向 **f3**。

### ⑤ 一个未解释的数据量差异

§4.1 说 BindingGYM 有 **508,962 curated entries**，而官方仓库 shipped 的 25 个 assay 合计 **376,446 行**（§3.1b 审计过）。差 132,516 行（26%）。可能是论文在数 curation 前的总量，也可能他们用了我们手上没有的数据。**暂记待查，尚不影响当前结论。**

## 5.12 未训练 baseline（pretrained ProteinMPNN + 未训练 H3-DDG 头）—— 已提交

### 为什么需要它

§5.10 证明训练后的模型逐 assay 相关性与零不可区分（7 正 7 负，均值 +0.0180），输出塌缩成近似常数。**要判断「pipeline 本身能不能产出信号」，就必须有一个不经训练的参照点。** 目前手上只有 iter-1 的 1,000 行监控子样本（f1 +0.3008 / f2 +0.0934 / f4 +0.1230），噪声太大，且 f1 那个 +0.3008 高得可疑 —— 是论文 ProteinMPNN 行（0.0998）的 3 倍、BA-Cycle（0.1320）的 2.3 倍。

### 为什么选 f1 和 f3

论文只跑单 fold（§5.11 ④）。f0 已排除；「multi-point」= ≥3 ⇒ f1，= ≥2 ⇒ f3。**这两个就是全部候选**，所以只跑这两个。

### 实现

`train_bindinggym.py` 新增 `--eval_only`：跳过整个训练循环与 checkpoint 保存，直接评测初始化后的模型。两处安全设计：
- 输出一律加 `_untrained` 后缀 → baseline 与正式 run 共用 `--save_dir` 也不会互相覆盖；
- `--eval_only` 与 `--resume` 互斥并直接报错 → 不会把权重悄悄加载进一个号称「未训练」的模型。

本地 A4500 smoke 通过（`--max_eval_batches 6`，`FINAL fold1_untrained` 正常产出，无训练日志）。

### 已提交

| job | fold | walltime | 依据 |
|---|---|---|---|
| `50817084` | 1 | 02:30:00 | 55,081 行，按 a100 实测 eval ~1.0h |
| `50817085` | 3 | 05:00:00 | 142,905 行，~2.6h |

脚本：`sh/bindinggym_untrained_baseline_20260824.sh` + `sh/submit_untrained_baseline.sh`。输出写到独立的 `results/bindinggym_untrained_fold{F}/`，**不触碰**已训练 fold 的产物，也不触碰 f0/f3 的 resume checkpoint。

同时在本地 A4500 跑一份**预览**（scratchpad，永不进 `results/`、永不作为报告数字，仅为提前知道方向）。

### 判读标准（先写死，避免事后找解释）

| 未训练全量结果 | 结论 |
|---|---|
| ALL Pearson ≈ 0.3 | pipeline 能产出信号，问题在训练配方；且未训练就已达论文水平，需解释论文 ProteinMPNN 行只有 0.0998 |
| ≈ 0.10–0.15 | 与论文 ProteinMPNN(0.0998) / BA-Cycle(0.1320) 一致，pipeline 正常，训练把它训坏了 |
| ≈ 0 | pipeline 仍有 bug；iter-1 的 +0.3008 是子样本运气 |

### 关于 fold 划分的确认边界（用户提问）

- **已验证**：fold 成员固化于 `data_splits/inter_assay_folds.tsv`，md5 `d23e15f9f54e6b339e833600c12ff673` 与 guard 期望值一致；两篇论文各自声明的 env（numpy 1.22.4 / 1.24.4）都产出同一份划分；生成逻辑逐字对应 BindingGYM `training/main.py:348`；自 `383b0e8` 未改动。
- **无法验证**：H3-DDG 作者是否用的就是这一份 —— 他们**没有发布任何 BindingGYM 代码**。可声称的最强命题是「这是 BindingGYM 官方代码在两篇论文 pinned env 下的产物」。
- **⚠ 不在验证范围内**：`data_splits/assay_chain_sides.tsv` 是**我自己整理**的 25 行表，定义热力学循环的两个 side。BindingGYM 官方仓库没有它（他们的模型是序列模型，不做热力学循环）。**这是整套 setup 里最大的一块没有外部交叉验证的判断**，尤其是 4 个 Z-domain assay 两侧都有突变的情形 —— 而 f1 恰好全是 Z-domain。

## 5.13 `assay_chain_sides.tsv` 验证通过（2026-08-24，`diagnostics/verify_sides.py`）

§5.12 里我把这张表标为「整套 setup 里唯一没有外部交叉验证的部分」。现在验完了 —— **25 个 assay 全部通过四项检查**。

四项检查各自证伪什么：
1. **链存在性 + 覆盖** —— 声明的链都在 PDB 里，且 PDB 的链没有一条落在两个 side 之外（否则 `parse_biopython_structure` 会静默丢链）。
2. **对齐 BindingGYM 自己的 `chain_id` 列** —— 这是唯一可用的外部参照。`side0 ∪ side1` 必须等于它。
3. **划分是否切在正确的缝上** —— 打印全部两两重原子接触数（5 Å 内的残基对），若**任何 side 内部的链对接触数少于跨 side 界面**，说明刀切错了位置。
4. **突变落点** —— 每个突变位点都必须在声明的链上；同时统计突变在两个 side 的分布（这决定热力学循环需要几次 isolated-side forward）。

### 最可疑的三个 Fab 案例：判定明确

| assay | side 内接触 | 跨 side 界面 | 比值 |
|---|---|---|---|
| 4D5_HER2 | **A-B 117**（Fab 轻-重链）| A-C 22 + B-C 24 = 46 | **2.5×** |
| 5A12_Ang2 | **H-L 117** | H-A 12 + L-A 36 = 48 | **2.4×** |
| 5A12_VEGF | **H-L 122** | H-C 21 + L-C 2 = 23 | **5.3×** |

三例中 Fab 两条链彼此的接触都远强于 Fab-抗原界面，所以「A+B 合为一个 side」是对的 —— 另外两种划分（A|BC 或 B|AC）都会从一条 117–122 接触的缝上切开。这是把它们当一个 side 的结构性证据，不是命名约定的推断。

### 4 个 Z-domain：双侧突变确认

| assay | A-B 接触 | side0 突变数 | side1 突变数 |
|---|---|---|---|
| ZpA963_HL1 | 51 | 6,141 | 6,538 |
| ZpA963_HL2 | 51 | 843 | 756 |
| ZSPA-1_LL1 | 50 | **145,101** | **162,314** |
| ZSPA-1_LL2 | 50 | 18,392 | 14,619 |

两侧确实都被突变（这也是当初必须建这张表的原因），链长与 `wildtype_sequence` 完全一致（55/55、54/54、58/58），且都是单链蛋白 —— 只存在一种划分，按构造正确。

### 顺带发现（非 bug，但影响可解释性）

**5A12_VEGF 的轻链 L 与抗原 C 只有 2 个接触，却承载了 57,839 个突变。** 也就是说该 assay 在 L 上的大部分突变信号根本不在界面上，是通过稳定性/远程效应间接影响结合的。对一个基于界面结构的模型来说这是本质困难，不是实现缺陷 —— 但它解释了为什么这类 assay 天然难做。

另有若干大链的建模残基数少于序列长度（如 KRAS_PICK3CG 的 A 链 749/941、HLA-A2 的 C 链 317/370、CD19 的 C 链 218/255），是晶体结构未解析区域，正常；§3.1b 的审计已确认**突变位点**的未解析数为 0。

### 这一项通过后，数据管线的验证覆盖情况

| 环节 | 验证方式 | 结果 |
|---|---|---|
| fold 成员 | md5 fingerprint + 双 env 交叉 | ✅ |
| 突变映射/解析 | §3.1b 全量审计 | ✅ 0 unresolved / 0 wt 不符 |
| 评测对齐 | 119,200 行逐行对回源 csv | ✅ 0 错配 |
| **side 定义** | **本节四项检查** | **✅ 25/25** |

**数据侧已无已知疑点。** 这把 §5.10 的结论进一步收紧：问题在训练配置，不在管线。

## 5.14 根因定位：训练策略，且有官方发布结果作为量化靶子（2026-08-24）

### ① label 方向没有问题（三重独立验证）

| 证据 | 内容 |
|---|---|
| curation 层 | BindingGYM `utils/data_utils.py:25`：`DMS_score = raw_phenotype * DMS_directionality` —— 方向在建库时就归一化了 |
| 官方代码层 | `training/dataset.py`：`reg_label = -ddg if 'DMS_score' not in columns else DMS_score` —— 官方把 `-ddg` 与 `DMS_score` 当同一个量，即 **`ddg = -DMS_score`**，正是 `bindinggym.py` 的 `label_sign = -1` |
| 数据层（我独立验）| 22 个有 WT 行的 assay 中 **18 个** WT 分位 ≥50%（多在 87–99%）；25 个中 21 个 `ρ(突变数, score) ≤ 0`；**flipped-sign 候选 0 个**（`diagnostics/verify_direction.py`）|

**(1) 结论：无符号问题，`ddG_true = -DMS_score` 正确且已外部确认。**

### ② 但查出第三种失效模式：4 个 gain-of-function assay

| assay | WT 分位 | ρ(突变数, score) | 所在 fold | 占该 fold |
|---|---|---|---|---|
| `5A12_VEGF_fitness_4ZFF` | **0.02%**（29,981 行库里最差）| **+0.417** | 4 | **86%** |
| `CD19_FMC63_Fitness_7URV` | 29.7% | **+0.735** | 3 | 3% |
| `hYAP65_peptide` | 35.4% | **+0.495** | 2 | **63%** |
| `ACE2_SARS2-RBD_enrich_6M17` | 44.3% | — | 2 | 7% |

H3-DDG 的预测由 ProteinMPNN 的负对数似然构成（论文式 16），而似然奖励的是「这个残基在此结构语境下像天然的」。**在 gain-of-function 库里野生型恰恰是最差的结合体**，所以「像天然」与结合强度反相关 —— **zero-shot 的 inverse-folding 打分在这些 assay 上按构造是反的**。

对得上我们的结果：f4 是唯一为负的 fold（−0.1103），而 `5A12_VEGF` 单独就是 −0.1617 且占该 fold 86%。

⚠️ 但**这条只适用于 zero-shot**：官方微调后在 `5A12_VEGF` 上拿到 Spearman **+0.4460**（见④），说明 ranking 损失能翻转这个先验。

### ③ (2) 我的训练策略 vs 官方 —— 三个机制全缺

| | BindingGYM 官方 inter | **我当前（A.4 口径）** | H3-DDG 论文 A.4 |
|---|---|---|---|
| batch 组成 | **同一 assay 的 8 条** | 20 assay 混采的 1 条 | batch size 1, 2 |
| 采样权重 | **assay 均匀**（GB1 92,891 行与 BH3 518 行等权）| **按行数比例**（KRAS/GB1 拿走几乎全部梯度）| — |
| loss | **listMLE**（listwise ranking，尺度无关）| **裸 MSE on raw ddG** | MSE（式 17）|
| optimizer | AdamW 1e-3, betas (0.9,0.99), wd 0.05, OneCycleLR | Adam 4e-4, wd 0 | Adam 4e-4 |
| 训练量 | 256 步/epoch × ≤100 epoch, patience 3 | 20,000 步一口气 | 20,000 iter |
| 模型选择 | **valid per-DMS Spearman** | **无**（取最后一步）| — |
| backbone | ProteinMPNN `v_48_020`, augment_eps 0.2 | 同一份权重, backbone_noise 0 | 同一份 |

来源：`/home/guoj0f/repos/BindingGYM/training/{main.py:346-403,579, dataset.py:77-84, loss.py, run.sh}`。

**一个纯逻辑上的关键点：`batch_size=1` 时 listMLE 恒等于 0** —— 单元素列表没有排序可学。所以 **A.4 声明的「MSE + batch_size 1」在结构上根本无法表达官方策略**。（本地 smoke 里内存压力把 batch 压到 1，日志正好打出 `listMLE 0.000000`，而模型仍在退化 —— 因为 AdamW 的 `weight_decay=0.05` 与梯度无关地衰减权重。已给探测加下界 2。）

**另记一笔官方协议自身的问题**：`main.py` 的 `fold_valid = split[fold][1]` 就是**测试 fold**，而第 579 行按 `valid_metrics['spearman']` 选 epoch —— **官方是在测试集上做模型选择**，其发布数值含 early-stopping oracle。我们复现其协议时照做，但必须标注。

### ④ 量化靶子：官方发布了同 backbone、同 split、同策略的结果

`BindingGYM/results/ProteinMPNN_finetune_inter_cluster_metric*.csv` —— **pretrained ProteinMPNN 在 inter-assay cluster split 下按其策略微调**的逐 assay 结果：

| slice | 官方 ProteinMPNN 微调 | H3-DDG 论文 Table 2 | Table 2 的 ProteinMPNN 行（**zero-shot**）|
|---|---|---|---|
| ALL Spearman | **0.4217** | 0.2725 | 0.2050 |
| <3 | **0.4254** | 0.3031 | 0.2439 |
| ≥3 | 0.3043 | 0.2755 | 0.1614 |

🔴 **BindingGYM 官方的「朴素 ProteinMPNN + 他们的策略」（0.4217）比 H3-DDG 论文报告的 0.2725 高 55%。** 而 Table 2 里那行 ProteinMPNN（0.2050）是 **zero-shot** 版本，不是这个微调结果 —— 即 H3-DDG 没有与 BindingGYM 自己发布的、更强的同 backbone 监督基线对比。

**逐 fold 靶子**（由该文件按我们固化的 fold 划分聚合）：

| fold | 官方 ALL Spearman |
|---|---|
| 0 | **0.5542** |
| 1 | 0.2719 |
| 2 | 0.3035 |
| 3 | **0.5550** |
| 4 | 0.3916 |

**14 个重叠 assay 逐个对比：官方等权 Spearman 0.3174，我们 −0.0052，差 0.3225。** 差距最大的是 `BH3_Mcl-1`（+1.08）、`BH3_Bcl-xL`（+0.80）、`5A12_VEGF`（+0.61）。

**我们反超官方的有两个**，都是 Z-domain 双侧突变 assay：`ZpA963_HL1`（我们 0.4542 vs 官方 0.1721）、`ZSPA-1_LL1`（0.2127 vs 0.0121）。官方是纯序列/结构打分，不做热力学循环；这两个 assay 两侧都突变，正是循环该发挥作用的地方 —— 这是 H3-DDG 路线一个真实（虽窄）的优势。

### ⑤ (3) 已实现并提交

`train_bindinggym_official.py` + `config/train_h3-ddg_bindinggym_official.json`（独立文件，`train_bindinggym.py` 未动）。实现要点与两个新发现见该文件头注释与 commit `0a7d472`：

- **`batch_size 8` 在 a100 上也可能放不下** —— H3-DDG 在 BindingGYM 的朴素 ProteinMPNN 之上多了 O(K²) 的 3-body triplet attention。这**反过来解释了 A.4 那句 "batch size of 1, 2, depending on GPU memory and graph size"**。改为开跑前对**最大结构的 assay** 探测一次并固定，而不是训练中动态切分（切分已 collate 的 batch 涉及 per-item / per-row 下标，出错会污染 label 而不只是崩溃）。
- 探测下界 2（见③的 listMLE 退化）。

**提交的 fold：`50818834`(f0) 与 `50818835`(f2)**，各 7h。
- **f2 是实测最快的 fold**（29,332 行，2:31:51）—— 用户要求的那个。但它 63% 是 `hYAP65`（gain-of-function），官方靶子只有 0.3035。
- **f0 官方靶子最高（0.5542）**，是信噪比最好的判据：跑出 ~0.55 则策略是解、我们的实现正确；跑出 ~0 则另有问题。代价只是 final eval 多 1.6h。

两个都跑，正好把「策略是否是解」与「该 fold 是否本身对抗」分开。

## 5.15 归属划分修正：只换策略，不换优化器（2026-08-24）

### 用户指正

「**数据 loading 和模型训练策略**采用 BindingGYM 官方；**模型结构、模型训练参数**与 H3-DDG 拉齐。」

§5.14 提交的 `50818834`/`50818835` 违反了后半条 —— 我把 BindingGYM 的优化器一起搬了过来。**已 `scancel`**（提交 20 min，未开始）。

### 三个 config 的逐项审计

**模型结构：三者完全一致 ✅**（逐字段比对，`diagnostics/` 无差异）
`hidden_dim 128 / num_layers 3 / num_tri_heads 4 / num_edges 48 / use_hypergraph / hyper_ratio 4 / max_num_hyperedges 420 / num_mut_subgraph_nodes 20 / num_edges_ratio 3.0 / edges_selection dynamic / patch_size 128 / ca_only false / backbone_noise 0.0 / loss_weight_boltzmann 1.0 / seed 42 / v_48_020.pt`

**模型训练参数：原提交全被换成 BindingGYM 的 ❌**

| | H3-DDG | 被撤销的 official | **修正后 strategy** |
|---|---|---|---|
| optimizer | Adam | AdamW + OneCycleLR | **Adam** |
| `lr` | 4e-4 | 1e-3 | **4e-4** |
| `weight_decay` | 0.0 | 0.05 | **0.0** |
| `batch_size` | A.4: "1, 2" | 8 | **2** |
| 总步数 | 20,000 | 256×40 | **256×78 = 19,968** |

**为什么这个混淆是致命的**：若 f0 跑出 0.55，无法区分是 within-assay batch + listMLE 起作用，还是换成 AdamW 1e-3 + OneCycleLR 起作用。

### `batch_size = 2` 不是随手挑的

A.4 原文是 "a batch size of **1, 2**"，而 **listMLE 至少需要 2**（单元素列表无排序）。2 是唯一同时满足两边的值，无需仲裁。

### 显存实测，以及它对 A.4 的印证

在 1016 残基的 `4D5_HER2`（fold 0 与 fold 2 的最大训练结构）上实测训练峰值：

| slate | 前向行数 | 峰值 |
|---|---|---|
| 1 | 2 | **11.00 GiB** |
| 2 | 4 | **OOM @ 18.46 GiB**（19.57 GiB 卡上），失败的单次分配 990 MB → 实需约 **22 GiB** |

- 40 GB a100：22 GiB，**1.8× 余量，稳过**。20 GB 的 A4500 过不去，所以这一条无法本地验证。
- 40 GB 上 slate 3 ≈ 33 GiB 已很紧 → **`batch_size 2` 既是 A.4 的声明值也接近硬件上限**。
- 🔎 **反过来印证 A.4 那句话是真的**：作者用 24 GB 的 RTX 4090（A.4.2），slate 1 = 11 GiB 宽松、slate 2 ≈ 22 GiB 正好卡在上限 —— 完全对应 "batch size of 1, 2, depending on **GPU memory and graph size**"。我们的显存画像与他们一致。

据此**不再发探测作业**（`--probe_only` 保留在脚本里备用）。

### 数据 loading / 训练策略：哪些采用了官方，哪些不可能采用

| 项 | 来源 | 说明 |
|---|---|---|
| batch 全部来自同一 assay | **官方** ✅ | 复现 `seed = index // batch_size` 的效果 |
| assay 均匀采样（非按行数）| **官方** ✅ | |
| 有放回抽样 | **官方** ✅ | 官方用 `np.random.randint` |
| 256 步/epoch | **官方** ✅ | 其 `__len__ = batch_size*256` |
| 每 epoch 重设种子 | **官方** ✅ | 其 `seed_bias = epoch` |
| listMLE | **官方** ✅ | 逐字移植 `loss.py` |
| patience 3 + 按 valid per-DMS Spearman 选 epoch | **官方** ✅ | |
| **突变→结构的特征化** | **不可能采用** ⚠️ | 官方 `TaskDataset` 用 `parse_PDB` + `tied_featurize` 并把突变位点 mask 成 `'X'`，产出的是**单个序列**；H3-DDG 需要热力学循环的输入（complex + 各 isolated side）。必须用我们的 `bindinggym.py`。**这是硬约束，不是选择。** |
| 每 epoch 的选择集 | **偏离**（成本）| 官方每 epoch 评测整个 held-out fold（此处约 0.5h/epoch）；我们用固定等距的 300 行/assay 子集选 epoch，选出的权重再在全量上评一次 |
| 在**测试 fold** 上选 epoch | **照做 + 标注** | 官方 `fold_valid = split[fold][1]` 就是测试 fold，`main.py:579` 按其 Spearman 选。**官方发布的 0.4217 含 early-stopping oracle。** 照做以保持可比，但必须标注 |

### 已提交

| job | 评测 fold | held-out | 官方靶子 |
|---|---|---|---|
| `50820222` | **f0** | KRAS ×6（114,341 行）| **0.5542** |
| `50820223` | **f2** | ACE2/CXCR4/PSD95×2/hYAP65（29,332 行）| 0.3035 |

`full_recipe` 那一臂（连优化器一起换）保留为 `submit_official_strategy.sh full_recipe <fold>`，作为「官方全配方能到 0.4217」的上界参照，待 strategy 臂出结果后再决定是否需要。

### 本地任务已全部终止

用户要求不耽误主线，本地 A4500 上的 untrained baseline 预览（f1，已跑 1h01m）已 kill，GPU 释放。未训练 baseline 的正式数字改由 Ibex 的 `50817084`/`50817085`（仍 hold）提供。

## 5.16 三问核实：split 归属、官方靶子的确切定义、fold 划分的版本无关性（2026-08-24）

### ① 所有 Ibex 作业都是 inter-assay split

| job | 脚本 | config | 训练 | 评测 |
|---|---|---|---|---|
| `50820222` / `50820223` | `train_bindinggym_official.py` | `..._strategy.json` | 4 个 cluster 的 20 assay | held-out cluster（f0 / f2）|
| `50680591` / `50680595`（hold）| `train_bindinggym.py` | `..._bindinggym.json`（A.4）| 同上 | f0 / f3 |
| `50817084` / `50817085`（hold）| `train_bindinggym.py --eval_only` | 同上 | **不训练** | f1 / f3 |

**全部**通过 `BindingGYMDataset` 读同一份 `data_splits/inter_assay_folds.tsv`，且 `bindinggym.py:152` 有 assay 级泄漏断言 `assert not (train_ids & val_ids)`。**从未提交过 intra-assay 作业。** 两个 untrained baseline 不训练，所以「inter-assay」只作用在评测侧 —— 但 held-out 集合用的是同一套 fold 定义。

### ② 「官方靶子」的确切定义

`BindingGYM/results/ProteinMPNN_finetune_inter_cluster_metric*.csv`，即：

- **模型**：`ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.2, k_neighbors=48)`（`main.py:363`）
- **权重**：`--use_weight pretrained` → `cache/v_48_020.pt` —— **与 H3-DDG backbone 同一份权重**
- **是** pretrained ProteinMPNN，**监督微调**（非 zero-shot），用 listMLE
- **split**：`--mode inter --split cluster` → 同一个 GroupKFold cluster 5-fold
- **文件后缀**（从 `calc_metric.ipynb` 核实）：无后缀 = **ALL**；`_oneORtwo` = `len(mutant.split(':')) < 3` 即 **<3**；`_multi` = **≥3**；每切片 **≥100 行**过滤 —— 与 `bindinggym_metrics.py` 的实现一致

**`-R` 后缀 = 随机初始化**（`--use_weight native`）。notebook 把 `ProteinMPNN-R` 的 zero-shot 硬编码成 Spearman 0 / AUC 0.5，因为随机初始化无 zero-shot 能力。

| 模型（全 25 assay 等权 per-DMS Spearman）| ALL |
|---|---|
| **ProteinMPNN（pretrained）+ 官方策略微调** | **0.4217** ← 我们的靶子 |
| ESM2（pretrained）+ 官方策略微调 | 0.3024 |
| **ProteinMPNN-R（随机初始化）+ 同样微调** | **0.1585** |
| ESM2-R（随机初始化）| 0.0946 |
| H3-DDG 论文 Table 2 | 0.2725 |
| Table 2 的 ProteinMPNN 行（**zero-shot**）| 0.2050 |

顺带一个有用的量：**预训练权重本身贡献 0.4217 − 0.1585 = 0.263**，是这个任务上最大的单一因素。而 H3-DDG 报的 0.2725 介于「随机初始化微调」(0.1585) 与「预训练微调」(0.4217) 之间。

### ③ fold 划分与 BindingGYM 官方**同时**拉齐 —— 不存在二选一

这一条我之前只在 docstring 里断言过，现在**实测**：建了一个 BindingGYM 官方 pin 的 venv（`numpy==1.24.4 / scikit-learn==1.3.2 / pandas==2.0.3 / scipy==1.10.1`，取自 `BindingGYM.yml`），在其中重跑 `make_inter_assay_folds.py`：

```
fingerprint d23e15f9f54e6b339e833600c12ff673 matches the committed split
wrote ... (numpy 1.24.4 / sklearn 1.3.2)
```
与已固化的 tsv `diff` → **0 行差异**。

决定 tie 的那一步（`np.argsort(n_samples_per_group)[::-1]`，权重向量 `[1,2,1,1,1,1,1,2,1,6,2,1,4,1]`，14 个里 12 个并列）的实测输出：

| env | argsort 结果 |
|---|---|
| numpy 1.22.4 / sklearn 1.2.1（**H3-DDG README pin**，我们在用）| `[9,12,10,7,1,13,11,8,6,5,4,3,2,0]` |
| numpy 1.24.4 / sklearn 1.3.2（**BindingGYM.yml pin**）| `[9,12,10,7,1,13,11,8,6,5,4,3,2,0]` —— **完全相同** |
| numpy 2.3.5 / sklearn 1.7.2（base env）| `[9,12,10,7,1,13,`**`8,11,5,6`**`,4,3,2,0]` —— 第 6–9 位不同 |

**结论：划分本身就是 BindingGYM 官方的**（逐字调用其 `main.py:348` 的 `GroupKFold(n_splits=5).split(clusters, groups=clusters)`，cluster 表用官方 shipped 的 `BindingGYM_cluster.tsv`），而它在**两篇论文各自声明的 env 下产出同一份结果**。所以不是「我们选了 H3-DDG 的版本」—— **没有可选的东西**，两边一致。只有 numpy ≥2 才会偏离，而那不是任何一篇论文声明的环境，且已被 `make_inter_assay_folds.py` 的 fingerprint guard 拦住。

venv 保留在 scratchpad（`bgym_pin/`），后续若要在官方 pin 下复核任何数据侧结论可直接复用。

## 5.17 更正一处无效比较，并把主线收敛到 f1（2026-08-24）

### ① 更正：§5.14 ④ 的「官方比论文高 55%」不成立

我曾写「BindingGYM 官方的 0.4217 比 H3-DDG 报告的 0.2725 高 55%」。**这是拿合并值比单 fold 值，不成立，作废。**

两篇论文的口径已核实清楚：

| | H3-DDG Table 2 | BindingGYM Table 5 |
|---|---|---|
| 评测范围 | **单个 fold** —— §4.1「the hardest inter-assay split, focusing on the fold with the most multi-point mutations」| **五个 fold 全跑、合并覆盖全部数据** —— §4.4「Using a five-fold inter-assay split allows us to generate predictions for **all data**」|
| ALL Spearman | 0.2725（H3-DDG）| **0.42**（pretrained ProteinMPNN）/ 0.16（ProteinMPNN-R，随机初始化）|

BindingGYM 论文 Table 5 的原文数字（ProteinMPNN）：ALL `0.42 / 0.70 / 0.16 / 0.72 / 0.23`，<3 `0.43/0.70/0.16/0.72/0.22`，≥3 `0.30/0.70/0.17/0.69/0.25`。仓库 `results/ProteinMPNN_finetune_inter_cluster_metric*.csv` 逐 assay 平均得 `0.4217/0.6995/0.1641/0.7182/0.2276` —— **逐位吻合，csv 就是 Table 5 的原始数据**。

`main.py:355,624-625` 确认机制：`all_valid = []` 在 fold 循环外，循环内 `all_valid.append(fold_valid)`，最后写 `oof.csv` + 逐 assay `{DMS_id}_oof.csv`；`calc_metric.ipynb` 从这些 oof 文件算 25 行指标。**每个 assay 恰好 held-out 一次**（已核：25 行、出现次数均为 1、5 fold 全覆盖）。

⚠️ **两篇论文没有交集**：BindingGYM Table 5 里没有 H3-DDG；H3-DDG Table 2 里那行 ProteinMPNN 是 **zero-shot**（0.2050），不是 Table 5 的微调结果。所以「H3-DDG 相对 BindingGYM 官方基线如何」两篇论文都没回答，只能我们自己在同一 fold 上跑。

### ② 数据来源归属（§5.14 ④ 那张逐 fold 表）

| 量 | 来源 |
|---|---|
| 逐 assay 的 Spearman/AUC/MCC/NDCG/AP | **BindingGYM 官方发布**（Table 5 的原始 csv）|
| 「per-assay 等权平均」聚合口径 | **BindingGYM 官方**（`calc_metric.ipynb`）|
| **逐 fold 分组与分项均值** | **本记录重构** —— 官方 csv 只有 `DMS_id,Spearman,AUC,MCC,NDCG,AP` 六列、**无 fold 列**，两篇论文均**未发布逐 fold 数字** |

重构合法性依据：(a) 是 OOF，每 assay 恰好 held-out 一次；(b) 我们的 fold 划分与官方完全一致（§5.16 在其 `BindingGYM.yml` pin 下重算，`diff` 0 行）。

### ③ 再更正：指向 f1 的证据不是「三条独立」

我曾称有「三条互相独立的证据」指向 f1。**不对。**

- **独立的只有一条**：f1 的 ≥3 突变数 54,324 行、占比 98.6%，绝对数与占比双第一 —— 来自原始 DMS 数据。
- 「f1 是官方跑得最差的 fold（0.2719，5 个里最低）」与「官方 f1 = 0.2719 ≈ 论文 0.2725」**同出于②的那一个重构**，是同一派生量的两个侧面，不是两条证据。

且数值线索本身有弱点：f1 的 ≥3 切片对不上（重构值 0.1497 vs 论文 0.2755，差 0.126），只有 ALL 切片对得极准（差 0.0006）。若真是同一 fold 且两模型水平相近，三个切片应当都接近。

**要硬判定，只能我们自己在 f1 上跑，用同一模型同一代码的三个切片去比论文的三个切片** —— 比拿 ProteinMPNN 当代理硬得多。

### ④ 主线收敛到 f1

用户决定：取消全部在排任务，只保留 f1。

- 取消：`50820222`(strategy f0)、`50820223`(strategy f2)
- 此前已 `scancel`：`50680591`/`50680595`（A.4 臂 f0/f3）、`50817084`/`50817085`（untrained baseline f1/f3）
- **新提交：`50829137`，strategy 臂，评测 fold = f1，7h**

对标基准（f1，ALL / <3 / ≥3 的 Spearman）：

| | ALL | <3 | ≥3 |
|---|---|---|---|
| H3-DDG 论文 Table 2 | **0.2725** | 0.3031 | 0.2755 |
| 官方 ProteinMPNN 微调（本记录重构的 f1 分项）| 0.2719 | 0.3422 | 0.1497 |
| 我们 A.4 臂在 f1 的实测 | **0.0904** | 0.1149 | 0.1193 |

## 5.18 🎯 f1 上的 strategy 臂结果：训练策略确认为根因（2026-08-25）

`50829137`，COMPLETED，**08-25 08:56 启动，耗时 1:37:47**（4h walltime 用掉 41%；SLURM 预估的是 08-26 07:30，实际早了约 23 小时）。GPU 是 **A100-SXM4-80GB**，`batch_size 2` 与 `eval_batch_size 8` 探测均通过。

### 结果（f1 held-out，55,081 行，5 个 assay）

| slice | Pearson | Spearman | AUROC | rmse_raw | n_assays | n_rows |
|---|---|---|---|---|---|---|
| ALL | 0.1903 | **0.2311** | **0.6283** | 0.8553 | 5 | 55,081 |
| <3 | 0.0610 | 0.0489 | 0.4460 | 0.9062 | 3 | 685 |
| ≥3 | 0.2001 | 0.2048 | **0.6864** | 0.8240 | 5 | 54,324 |

BindingGYM 官方口径：ALL Spearman 0.2311 / AUC 0.6249 / MCC 0.0263 / NDCG 0.5839 / AP 0.1380

### 与 A.4 臂、论文、官方基线的对比（同一个 fold = f1）

| | ALL Pearson | ALL Spearman | ALL AUROC |
|---|---|---|---|
| **A.4 臂**（裸 MSE / batch 1 / 全 assay 混采）| 0.1108 | **0.0904** | 0.6098 |
| **strategy 臂**（同 assay batch + listMLE + assay 均匀采样）| **0.1903** | **0.2311** | **0.6283** |
| 论文 Table 2 | 0.3057 | 0.2725 | 0.5703 |
| 官方 ProteinMPNN 微调（f1 分项，本记录重构）| — | 0.2719 | — |

**只换 data loading 与损失、模型结构与训练参数一字未动，Spearman 从 0.0904 → 0.2311（2.56×），达到论文 0.2725 的 85%、官方 f1 分项 0.2719 的 85%。**

- **AUROC 反超论文**：ALL 0.6283 vs 0.5703，≥3 0.6864 vs 0.6734。
- **Pearson 落后更多**（0.1903，论文的 62%），符合预期：listMLE 只优化排序不优化尺度，而 Pearson 对尺度与线性性敏感。
- **塌缩消失了**：A.4 臂的 `rmse_raw` 是「预测坍成常数」下的 0.9723，这里是 0.8553 且相关性同时上升 —— 不再是「RMSE 变好、排序变坏」那个病态组合。

**§5.10 的诊断得到确证：问题在训练配方，不在数据管线，也不在模型。**

### 但这次运行被过早掐断了

| epoch | listMLE | 选择集 Spearman | |
|---|---|---|---|
| 0 | 0.398179 | 0.1495 | new best |
| 1 | 0.356878 | 0.1321 | 1/3 |
| 2 | 0.357461 | 0.1440 | 2/3 |
| 3 | 0.352436 | **0.2371** | **new best** |
| 4 | 0.358079 | 0.2103 | 1/3 |
| 5 | 0.344305 | 0.0439 | 2/3 |
| 6 | 0.347059 | −0.0030 | 3/3 → **早停** |

**只用了 7 × 256 = 1,792 步，占 19,968 预算的 9%。** 最优权重来自 epoch 3。

选择集指标在相邻 epoch 间摆动达 ±0.2（0.2371 → 0.2103 → 0.0439 → −0.0030），**patience 3 在这种噪声下几乎必然误杀**。根源是 §5.15 记录的那处偏离：官方每个 epoch 在**整个 held-out fold** 上评测（此处 55,081 行），我为省算力改成 300 行/assay 的子集（f1 共 1,500 行）—— **子集小到让 patience 判据失效**。这是我引入的偏离，现在看到了它的代价。

`listMLE` 本身几乎没降（0.398 → 0.344），也印证训练远未收敛。

### 下一步的候选（待用户定）

1. **放大选择集**（如 2,000 行/assay，或直接用官方口径的全量）+ 保持 patience 3 —— 最贴近官方协议，代价是每 epoch 的评测变慢。
2. **保持子集但放宽 patience**（如 8–10）—— 便宜，但偏离官方更多。
3. 上述任一 + `--resume` 从现有 `best_fold1.pt` 继续，而不是从头重训。

⚠️ 三个选项都只调「早停/选择」这一处偏离，**不涉及 per-assay 标准化或改 listMLE**。

## 5.19 两篇论文 + 两份代码的训练策略逐项溯源（2026-08-25）

四个独立出处逐项核对。**H 论**=H3-DDG 论文；**H 码**=H3-DDG 发布代码（仅 SKEMPI）；**B 论**=BindingGYM 论文；**B 码**=BindingGYM `training/`。

| 项 | H3-DDG 论文 | H3-DDG 发布代码 | BindingGYM 论文 | BindingGYM 代码 | 我们当前 |
|---|---|---|---|---|---|
| 训练量 | **20,000 iterations**（A.4.1）| `max_iter 38000`（SKEMPI config；commit `d03eda5` 从 50000 改下来）| **100 epochs**（A.3.2）| `epochs = 100`（main.py:234）| 78 epoch × 256 = **19,968 步** |
| 一个 epoch 多大 | 未提（全文只按 iteration 计）| 无 epoch 概念 | **未定义** | `__len__ = min(batch_size*256, n)` → **256 个 batch** | 256 步/epoch |
| **早停** | **完全未提** | **无早停**，跑满 `max_iter` | **有，patience 3**（A.3.2）| `patience = 3`（main.py:236）；`if not_improve_epochs >= patience: break` | patience 3 |
| 早停/选择的指标 | 未提 | `validate_all.py`：按各 fold 的 **Spearman** 排序取最优 | 未提 | `valid_metrics['spearman']`（main.py:579，`obj_max=1`）| per-DMS Spearman |
| **早停/选择用哪个 split** | 未提 | **held-out fold 本身**（`online_validate(fold)` 用 `get_val_loader(fold)`）| 未提 | **`fold_valid = split[fold][1]` = 测试 fold** | 测试 fold 的**子集** |
| **是否存在独立 validation split** | **没有** | **没有**（3-fold CV，held-out 即 val 即 test）| **没有** —— A.2 原文「Data from one group are used **exclusively for testing**, while data from the remaining four groups are used for training」（两分） | **没有**，`split[fold]` 只有 (train, valid)；`test = None` 在 inter 模式 | 没有 |
| 模型选择时机 | 未提 | **事后**：跑完后 `validate_all.py` 扫全部 checkpoint | 未提 | **训练中**：`best_model = deepcopy(...)` 每次 Spearman 改进时 | 训练中 |
| optimizer | Adam, lr 4e-4（A.4.1）| **Adam, lr 4e-05**（config）| **AdamW, lr 1e-3, wd 0.05, eps 1e-5**（A.3.2）| 同（main.py:402）+ OneCycleLR | **Adam 4e-4, wd 0**（H3-DDG 侧）|
| loss | **MSE**（式 17）| MSE（`ddg_predictor.py:72`）| **ListMLE**（A.3.2）| `loss_tr = listMLE` | **listMLE**（B 侧）|
| batch 组成 | 未提 | 全数据 shuffle | 「every batch drawn ...」（§4 句子跨页被截断）| **同一 assay**（`seed = index // batch_size`）| 同一 assay |
| batch size | **"1, 2, depending on GPU memory and graph size"** | 1 | 未给数值 | 48（argparse）→ **8**（run.sh 覆盖）| **2** |
| 突变位点处理 | 热力学循环（式 16）| 同 | **全部 mask 成 X**，预测 = Σmt_logit − Σwt_logit（A.3.2）| `mseq[pos-1]='X'` | 热力学循环（H3-DDG 侧）|
| 骨架噪声 | `backbone_noise` 未提 | `0.0`（config）| 未提 | **`augment_eps=0.2`** | 0.0（H3-DDG 侧）|
| 检验频率 | 未提 | 每 `val_freq=500` iter | 每 epoch | 每 epoch | 每 epoch |
| 硬件 | RTX 4090（A.4.2）| — | 一张 A100（A.3.3）| — | A100-80GB |

### 由此得出的四条对复现有直接影响的结论

**① 两个数据集都没有 validation split。** 两篇论文都是二分（train / test），两份代码也都是。**早停与模型选择用的都是测试 fold 本身** —— 这是两边共有的设计，不是 BindingGYM 独有的问题。所以我们照做是对的，但两边发布的数值都含 early-stopping / checkpoint-selection oracle，必须标注。

**② 「100 epochs」远没有听起来那么多。** BindingGYM 的一个 epoch 是 `min(batch_size*256, n)` 条样本 = **256 个 batch**，不是一遍数据。batch 8 时 = 2,048 样本，占 f1 训练集 321,365 行的 **0.64%** —— **100 epoch 也只有 0.64 个真实 pass。**

**③ 两个训练预算其实是相容的：**

| | optimizer 步数 | 样本数 |
|---|---|---|
| BindingGYM 100 epoch | 25,600 | 204,800（batch 8）|
| H3-DDG A.4 20,000 iter | 20,000 | 40,000（batch 2）|
| 我们 78 epoch | 19,968 | 39,936 |

步数上二者只差 28%，所以「用 H3-DDG 的 20,000 步预算 + BindingGYM 的 epoch 结构」不是折衷，而是两边本来就接近。**但样本数上我们只有官方的 1/5**（batch 2 vs 8，而 batch 8 因 O(K²) 的 triplet attention 装不下，见 §5.15）。

**④ 「每 epoch 评全量」对我们比对官方贵得多。** f1 上每 epoch：训练 256 步 × batch 2 = 512 次 item 前向+反向；全量评测 55,081 行 × 2 次前向（complex + isolated side）= **110,162 次前向**，比例约 **1 : 215**。官方是纯 ProteinMPNN，每行 1 次前向且无 triplet attention，他们这个比例小一个量级以上。**所以「照抄官方的每-epoch 全量评测」在我们的模型上不是等价操作，而是把算力结构整体倒置。** §5.18 里 patience 3 误杀的根源就在这个取舍上。

## 5.20 bgymfull 臂：全面采用 BindingGYM 官方配置（2026-08-25）

### 用户的判断与新的归属划分

「H3-DDG 论文里描述的 experimental-config 应该都属于是在 SKEMPIv2 上的实验」—— 据此，在 BindingGYM 上复现应**全面采用 BindingGYM 官方配置**，只保留 H3-DDG 的 (1) 热力学循环、(2) `backbone_noise = 0.0`。

这个判断有独立证据支持（§5.19）：A.4 的 `lr 4e-4 / 20,000 iter` 连作者自己发布的 SKEMPI config（`4e-05 / 38000`）都描述不了，而我们复现出 Table 1 用的是后者。A.4 作为「BindingGYM 实验的规格」本就无从保证。

| 项 | 来源 | 值 |
|---|---|---|
| optimizer | **BindingGYM** | AdamW, lr **1e-3**, betas (0.9,0.99), wd **0.05**, eps 1e-5 |
| scheduler | **BindingGYM** | **OneCycleLR**(max_lr=1e-3, 256 步/epoch, epochs=100) |
| loss | **BindingGYM** | **listMLE** |
| batch 组成 | **BindingGYM** | 同一 assay |
| assay 采样 | **BindingGYM** | 均匀（非按行数）|
| epoch 结构 | **BindingGYM** | 256 步/epoch × **100 epoch** |
| 早停 | **BindingGYM** | **patience 3** on per-DMS Spearman |
| **早停/选择集** | **BindingGYM** | **整个 held-out fold**（`select_per_assay: 0`，新增支持）|
| batch_size | **BindingGYM** | **8**（探测会下调，见下）|
| **readout** | **H3-DDG（保留）** | 热力学循环（式 16），非官方的「mask + Σlogit 差」|
| **骨架噪声** | **H3-DDG（保留）** | **0.0**，非官方的 `augment_eps 0.2` |
| 架构 | H3-DDG | 与 `config/train_h3-ddg.json` **逐字段一致**（已校验，0 差异）|

### 两处必须记录的现实约束

**① `batch_size 8` 几乎肯定装不下。** 实测（§5.15）1016 残基结构上 slate 1 峰值 11.00 GiB → slate 8 需约 88 GiB。启动时的探测会按 8→4→2 下调并把实际值打进日志，所以**运行记录里会有它真正用了多少**，不会静默偏离。

**② full-fold 选择集的算力结构与官方不同。** f1 上每 epoch：训练 256 步 ≈ 3 min，全量选择集 55,081 行 ≈ **1.0h**（§5.19 ④ 的 1:215）。所以 **7h walltime 约覆盖 6 个 epoch**，靠 `--resume` 跨多轮续跑（resume 保真度已验证等同从头重跑，§5.17 的 RNG 修复）。这是「照抄官方」在我们模型上的真实代价，不是我又做了取舍 —— 官方的 ProteinMPNN 每行 1 次前向且无 triplet attention，他们这一步很便宜。

另注：`OneCycleLR(epochs=100)` 配 patience 3 意味着若在第 15 个 epoch 早停，学习率只走完 15% 的周期、仍在爬升段。**这正是官方的行为**（他们的 config 完全一样），照做。

### 用户修订：去掉早停，固定 50 epoch，每 10 epoch 评一次全量

「考虑到 H3-DDG 过于 time-consuming，我们就先不要 early-stop 了，设定最多跑 50 个 epoch，然后每隔 10 个 epochs 在 f1 上 testing 一下，最后汇报 5 个不同模型 checkpoints 的结果。」

**这个改法反而大幅省算力**：官方的「每 epoch 评全量」在 f1 上是 50 × 1.0h = **50h**；改成 5 次全量评测只要 **5.0h**。

| | 官方 patience 3 | 本臂修订后 |
|---|---|---|
| 停止规则 | 3 个 epoch 无改进即停 | **无**，固定 50 epoch |
| 全量评测次数 | 最多 100 次（每 epoch）| **5 次**（epoch 10/20/30/40/50）|
| 产出 | 1 个「最优」权重 | **5 个 checkpoint + 整条轨迹** |
| f1 上评测总耗时 | ~50h（若跑满 50 epoch）| **~5.0h** |

实现为 `early_stop: false` + `eval_every: 10`。`early_stop=false` 时**完全不构建 selection loader**（epoch 之间不做任何评测），只在调度点评全量。每个调度点各写自己的 `oof_fold1_ep{N}.csv` 与两套指标 csv，**所以跨作业 resume 不会重做已完成的评测**，收尾的轨迹表从磁盘重建而非依赖内存状态 —— 这是必须的，因为 50 epoch + 5 次评测装不进一个 7h 作业。

**这是对 BindingGYM 的 patience 3 的一处刻意偏离，其余字段仍全部是官方的**，两个 H3-DDG 例外（热力学循环、`backbone_noise 0.0`）不变。

### 预算与轮次

| 探测到的 batch | 训练 50 epoch | 5 次全量评测 | 合计 |
|---|---|---|---|
| 2 | 2.6h | 5.0h | **~7.6h** |
| 4 | 5.2h | 5.0h | **~10.2h** |

7h walltime **需要 2 轮**（`--resume` 续跑，保真度已验证等同从头重跑）。第一轮预计跑到 epoch 30–35（拿到 3 个评测点），第二轮收尾。

### 已提交

**`50844967`**，arm = `bgymfull`，评测 fold = **f1**，7h walltime。config 在作业启动时读取，所以修订后的代码与配置已同步到 Ibex，该作业**无需重投即会按新设定运行**（队列位置保留）。

本地 smoke 全通过：`no early stopping; full held-out fold (29332 rows) evaluated at epochs [2, 4] -> 2 checkpoints`、逐调度点的 `CHECKPOINT EVAL` + ckpt 落盘、收尾的 `TRAJECTORY` 表。

三条 smoke 验证通过（本地，缩小超边数以适配 20GB）：`selection set: the FULL held-out fold`、`optimizer AdamW lr 0.001 weight_decay 0.05 scheduler OneCycleLR`、选择集规模 = 该 fold 全量。

### 三个臂的对照关系

| 臂 | optimizer | loss | batch 组成 | 选择集 | f1 ALL Spearman |
|---|---|---|---|---|---|
| A.4（`bindinggym_perfold`）| Adam 4e-4 wd 0 | 裸 MSE | 全 assay 混采, bs 1 | 无选择 | **0.0904** |
| strategy | Adam 4e-4 wd 0（H3-DDG）| listMLE | 同 assay, bs 2 | 300 行/assay | **0.2311** |
| **bgymfull**（本次）| **AdamW 1e-3 wd 0.05 + OneCycle** | listMLE | 同 assay, bs≤8 | **全量 fold** | 待测 |

参照：论文 Table 2 = 0.2725；官方 ProteinMPNN 微调的 f1 分项（本记录重构）= 0.2719。

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
- **2026-08-24 12:5x**：f0/f3 walltime 18h/16h → **7h**（`scontrol update`，原地下调保住 5 天排队资历）；根因是 §5.5 的 K³ 外推高估约 3 倍，实测三个 fold 全部 ≤ 4:05:46。同时加入临时 chunked 脚本作为 7h 不够时的续跑兜底（§5.9）。
- **2026-08-24 13:2x**：f0/f3 `scontrol hold` 暂停。零算力诊断（§5.10）：评测对齐经 119,200 行逐行验证无误；**推翻** §5.7/§5.8 的「学到 assay 尺度」解释（预测的 between-assay 方差仅 0.5–4.3%）；真实病因是**输出在前 5,000 iter 内塌缩成近似常数**；头号嫌疑是 `lr=4e-4`（作者 SKEMPI config 是 4e-5，10×），且 f1 未训练的 +0.3008 ≈ 论文 0.3057 提示 Table 2 可能是 zero-shot 口径。
- **2026-08-24 13:4x**：读到论文原文（§5.11）。**撤回**「lr 抄错」猜测 —— A.4 原文确实是 4e-4/20k iter，但它与作者发布的 SKEMPI config（4e-5/38k）矛盾，而我们复现 Table 1 用的是后者。**撤回** zero-shot 假设 —— Table 2 里 ProteinMPNN 是独立 baseline（0.0998）。确认损失就是裸 MSE 且全文无 label 归一化，而 §4.3 作者明确承认跨 assay 尺度不可比。论文单 fold：f0 排除，≥3 口径指向 f1（已跑完，0.1108）、≥2 口径指向 f3（已暂停），RMSE 旁证偏向 f3。
- **2026-08-24 14:3x**：新增 `--eval_only`，提交未训练 baseline `50817084`(f1, 2:30h) / `50817085`(f3, 5:00h)（§5.12），判读标准已先行写死。同时确认 fold 划分的验证边界：成员固化且 fingerprint 通过，但 `assay_chain_sides.tsv` 是自建、无外部交叉验证。
- **2026-08-24 15:0x**：`assay_chain_sides.tsv` 验证通过（§5.13）—— 25/25 过四项检查；三个 Fab 案例的 side 内接触是跨 side 界面的 2.4–5.3 倍，证明「轻+重链合为一个 side」切在正确的缝上；4 个 Z-domain 的双侧突变确认。**数据侧至此无已知疑点**，问题收紧到训练配置。
- **2026-08-24 15:2x**：根因定位到训练策略（§5.14）。label 方向三重验证无误；查出 4 个 gain-of-function assay（`5A12_VEGF` WT 在 0.02 分位、占 f4 的 86%），解释 f4 为负。官方 inter-assay 策略 = 同 assay batch + listMLE + assay 均匀采样，三个机制我全缺；`batch_size=1` 使 listMLE 恒为 0，故 A.4 口径结构上无法表达该策略。🔴 **官方发布的「朴素 ProteinMPNN + 其策略」ALL Spearman 0.4217，比 H3-DDG 论文的 0.2725 高 55%**，且 Table 2 的 ProteinMPNN 行是 zero-shot 版。已实现 `train_bindinggym_official.py` 并提交 `50818834`(f0)/`50818835`(f2)。
- **2026-08-24 16:0x**：按用户指正修正归属划分（§5.15）。撤销 `50818834`/`50818835`（把 BindingGYM 优化器一起搬了，混淆了「策略」与「优化器」）。新 config `train_h3-ddg_bindinggym_strategy.json`：模型结构与训练参数全部 H3-DDG（Adam/4e-4/wd 0/19,968 步/batch 2），只有 data loading 与训练策略取自 BindingGYM。实测 slate 1 = 11.00 GiB、slate 2 ≈ 22 GiB，40 GB a100 有 1.8× 余量，且**反证 A.4 的 "batch size 1, 2" 在 24 GB RTX 4090 上正好卡这个界**。提交 `50820222`(f0) / `50820223`(f2)。本地预览已 kill。
- **2026-08-24 16:3x**：三问核实（§5.16）。(a) 6 个作业全部 inter-assay，共用固化的 fold tsv + assay 级泄漏断言，从未提交 intra-assay。(b) 「官方靶子 0.4217」= **pretrained ProteinMPNN（同一份 v_48_020）+ 官方 listMLE 策略微调**；`-R` 后缀是随机初始化（0.1585），故预训练权重单独贡献 0.263。(c) **实测**在 BindingGYM 官方 pin（numpy 1.24.4 / sklearn 1.3.2）下重算 fold，fingerprint 与 committed tsv `diff` 0 行 —— 划分同时与两篇论文拉齐，**不存在二选一**；只有 numpy ≥2 才偏离且已被 guard 拦住。
- **2026-08-24 17:1x**：**更正两处**（§5.17）：(a) 「官方 0.4217 比论文 0.2725 高 55%」作废 —— H3-DDG Table 2 是单 fold，BindingGYM Table 5 是五 fold 合并覆盖全部数据，两者不可直接比；(b) 指向 f1 的证据只有一条独立（≥3 突变数双第一），另两条同出于同一重构。核实 BindingGYM 论文 Table 5 原文为 ProteinMPNN ALL 0.42、ProteinMPNN-R 0.16，与仓库 csv 逐位吻合。用户决定收敛主线：取消全部在排任务，**只提交 `50829137`（strategy 臂，f1，7h）**。
- **2026-08-25 10:5x**：🎯 **f1 strategy 臂完成**（`50829137`，1:37:47，A100-80GB）。**只换 data loading 与损失，Spearman 从 A.4 臂的 0.0904 升到 0.2311（2.56×），达论文 0.2725 的 85%；AUROC 0.6283 反超论文 0.5703。塌缩消失，§5.10 的诊断确证 —— 根因是训练配方。** 但选择集指标 epoch 间摆动 ±0.2，patience 3 在 epoch 6 误杀，只用掉 19,968 步预算的 9%，最优权重来自 epoch 3；根源是我为省算力把每-epoch 选择集从官方的全量 fold 缩到 300 行/assay（§5.18）。
- **2026-08-25 11:2x**：四出处逐项溯源训练策略（§5.19）。**关键：两个数据集都没有 validation split，早停与模型选择用的都是测试 fold 本身**（H3-DDG 是事后 `validate_all.py` 扫 checkpoint、BindingGYM 是训练中 `best_model` + patience 3）；H3-DDG **完全没有早停**。BindingGYM 的「100 epochs」= 256 batch/epoch，batch 8 时仅 0.64 个真实 pass；其 25,600 步与 H3-DDG 的 20,000 步只差 28%，两预算相容。但我们每 epoch 的训练:评测算力比是 1:215（官方低一个量级以上），故「照抄每-epoch 全量评测」在我们模型上不等价。
- **2026-08-25 12:1x**：新增 **bgymfull 臂**（§5.20）—— 按用户判断（A.4 描述的是 SKEMPI 实验，不该作为 BindingGYM 的规格）全面采用 BindingGYM 官方配置：AdamW 1e-3/wd 0.05/OneCycleLR、listMLE、100 epoch、patience 3、**全量 held-out fold 做选择集**（新增 `select_per_assay: 0` 支持）、batch_size 8（探测会下调）；只保留 H3-DDG 的热力学循环与 `backbone_noise 0.0`。提交 `50844967`（f1，7h）。full-fold 选择集使每 epoch 约 1.0h，7h 约覆盖 6 epoch，靠 `--resume` 跨轮续跑。
