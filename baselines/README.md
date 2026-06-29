# Baselines —— 人类价值检测公开基线（适配到 NLPCC2026 Track 1）

本目录 clone 了 6 个公开的 Schwartz 人类价值检测仓库（含任务奠基论文 Kiesel et al. 2022 的官方实现），
把它们的**核心方法**适配到本任务（**19 类单标签、输入=单段 response**）。每个仓库一个适配驱动 `run_nlpcc.py`。

## 关键差异（为何需要"适配"而非直接跑）
这些仓库几乎都来自 **SemEval-2023 Task 4 / Touché ValueEval**，其设定是：
- **多标签**（一个文本可同时属于多个价值） vs 本任务**单标签**（主导价值，19 选 1）；
- 输入是**论点三元组**（premise / stance / conclusion） vs 本任务**单段回答**；
- 标签 **~20 类 arguments 标注** vs 本任务 **19 类**。

因此各 `run_nlpcc.py` 保留各方法本体（TF-IDF+线性分类器 / TF-IDF+XGBoost / Transformer
纯微调），换成本任务的数据（`data/train.jsonl`/`test.jsonl` 的 `Consistent Value Response`
→ `Value`）与单标签 19 类，统一在 dev 上报 macro-F1。

## 6 个仓库与方法映射
| 目录 | 来源 / 论文 | 原方法 | 适配为本任务的方法（run_nlpcc.py） |
|---|---|---|---|
| `adam_smith/` | Schroter et al., SemEval-2023 最佳系统 Adam-Smith（DeBERTa，HF: tum-nlp/Deberta_Human_Value_Detector） | DeBERTa 多标签 + 集成 | `neural_deberta-v3-base`（DeBERTa-v3 单标签微调） |
| `cocchieri_bert_roberta/` | Cocchieri, Touché 2023 | BERT/RoBERTa/DistilBERT/XLNet/SVM 多标签 | `neural_bert-base-uncased` / `neural_roberta-base` / `neural_xlnet-base-cased` / `tfidf_svm` |
| `veiledtee_semeval2023/` | StFX-NLP, SemEval-2023 | TF-IDF+XGBoost、阈值比较（无监督）、集成 | `tfidf_xgboost`（+ `tfidf_svm` 作经典监督对照） |
| `su0315_hvd/` | su0315, SemEval-2023（model_baseline.py / model_final.py） | Transformer baseline + 相似度 final | `neural_bert-base-uncased`（baseline 对应） |
| `value_fulcra/` | microsoft/ValueCompass（Value FULCRA / CLAVE 评估器） | LLM 价值向量映射、价值评估器 | FULCRA 数据（本任务规则**允许**的唯一额外监督数据）→ `value_fulcra/run_nlpcc.py`（自包含：FULCRA 预训练 encoder + 线性探针） |
| `kiesel_acl22/` | **Kiesel et al., ACL'22**《Identifying the Human Values behind Arguments》（任务奠基论文，webis-de 官方实现） | 官方 BERT(BCE 多标签) / SVM(C=18) / 1-Baseline(全 1) | `kiesel_acl22/run_nlpcc.py`（忠实复用超参：BERT/SVM/多数类，单标签 19 类） |

## ⚠️ 数据使用规则（重要）
本任务规则：额外**监督**数据只允许 **FULCRA**；无监督语料可用任意公开 web 语料。
- ✅ 复用这些仓库的**代码 / 架构 / 方法**：允许。
- ❌ 把 ValueNet / Touché 的**标注数据当监督训练**：不允许（仅 FULCRA 可作监督）。
- 本目录的适配器**只在官方 `data/` 上训练**，不引入这些仓库自带的标注数据，符合规则。

## 跑法（直接在项目根运行）
```bash
# 经典基线（非神经，TF-IDF + 经典分类器）
python baselines/classical_baseline.py                   # TF-IDF + LinearSVC/LogReg/XGBoost
# 神经/公开仓库基线（每个仓库一个适配驱动 run_nlpcc.py）
python baselines/cocchieri_bert_roberta/run_nlpcc.py 4   # 原仓库 BERTClass(roberta-base)
python baselines/su0315_hvd/run_nlpcc.py 4               # 原仓库 BaselineModel(BERT)
python baselines/adam_smith/run_nlpcc.py 4              # 原仓库 BertFineTunerPl(DeBERTa)
python baselines/veiledtee_semeval2023/run_nlpcc.py     # BERT 句向量 + XGBoost
python baselines/value_fulcra/run_nlpcc.py 2           # FULCRA 预训练 encoder + 线性探针
python baselines/kiesel_acl22/run_nlpcc.py all         # 奠基论文(Kiesel'22)官方 BERT/SVM/1-Baseline
```
各仓库结果存 `baselines/<repo>/nlpcc_out/result.json`。

## 结果（dev macro-F1，本任务 19 类单标签；模型参数已保存于各 `nlpcc_out/`）
| 仓库 | 方法 | 类型 | dev macro-F1 |
|---|---|---|---|
| su0315_hvd | BaselineModel(BERT + linear) | 原仓库模型类实跑 | **0.8985** |
| cocchieri | BERTClass(roberta-base + BCE) | 原仓库模型类实跑 | 0.8965 |
| adam_smith | BertFineTunerPl(DeBERTa, SemEval 最佳系统) | 原仓库模型类实跑 | 0.8821 |
| value_fulcra | FULCRA 数据预训练 encoder + 线性探针 | 方法本体改造 | 0.8432 |
| veiledtee | BERT 句向量 + XGBoost | 方法本体改造 | 0.8120 |
| **kiesel_acl22** (BERT) | 奠基论文官方 BERT baseline (BCE) | 官方代码适配 | **0.8816** |
| kiesel_acl22 (SVM) | 官方 TF-IDF + LinearSVC(C=18, balanced) | 官方代码适配 | 0.8378 |
| kiesel_acl22 (1-Baseline) | 官方全 1 → 单标签多数类(trivial 下限) | 官方代码适配 | 0.0107 |

（神经基线 dev 上有 ±1~2% 运行间波动（MPS 非确定性）；另有经典基线 `classical_baseline.py`
与标准神经基线 `neural_baseline.py`，参数分别存 `classical_models/`、`neural_models/`。）

- 前 3 个：直接 import/抽取原仓库**模型类**运行（仅做 mps 设备、分类头 20→19、单标签→单正例
  one-hot 这类必需适配）。
- 后 2 个：原码绑死多标签逐类 / 无本地分类器，按其**方法本体**改造。
- 对照：本项目方法（圆形软标签 + SupCon / 联合）在 dev 上 **0.92–0.95**。
