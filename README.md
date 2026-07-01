# CIRCLE: A Circumplex-Aware Contrastive Learning Framework

**CIRCLE: A Circumplex-Aware Contrastive Learning Framework for Fine-grained Human
Value Identification in LLM-Generated Content** (NLPCC 2026 Shared Task 2, Track 1).

This is the **lightweight code + data** package. It contains the **data, model code,
and experiment code** needed to reproduce every result and figure in the paper. The
**trained model weights and learned representations are distributed separately** in
`CIRCLE_experiment_weights.zip` (~12 GB, organized one folder per experiment, on
[Google Drive](https://drive.google.com/file/d/1GwT3hQiwCtcvDwc9ONyLAJqaH0ysbFZZ/view?usp=sharing))
because of their size — see §5 for how to download and place them. A single DeBERTa-v3-large model
trained with CIRCLE reaches a development macro-F1 of **0.9471**, the best among all
evaluated systems.

---

## 1. What this code is

The task is **fine-grained value identification**: given one response text, predict the
single Schwartz basic value (1 of 19) it expresses. CIRCLE treats the 19 values not as
independent labels but as points on Schwartz's **circular motivational continuum
(circumplex)**, and injects this structure into the model from two ends:

- **Circular soft labels** (`src/values.py: circular_soft_labels`) — the training target
  is a Gaussian over *ring distance* instead of one-hot, so predicting a neighbour is
  penalised less than predicting an opposing value.
- **Supervised contrastive learning** (`src/model.py: supcon_loss`) — pulls same-value
  responses together and pushes different ones apart, using each instance's
  `Contrastive Response` as a hard negative.
- **Class-balanced weighting** (`src/train.py: effective_number_weights`, β=0.999) — for
  the long-tailed value distribution.

These three terms are combined either in a single **joint** objective or in a
**two-stage** schedule (contrastive pre-shaping → classification fine-tuning).

## 2. Model architecture

All models share one backbone (`src/model.py: ValueClassifier`):

```
response text
   │
   ▼  DeBERTa-v3 (base or large) encoder
   ▼  mean pooling over final-layer tokens  →  h ∈ R^d
   ├─► classification head (Linear d→19)        →  logits   →  circular-soft-label CE + class balance
   └─► projection head (Linear→ReLU→Linear, L2)  →  z        →  supervised contrastive loss (τ=0.1)
```

At inference only the **classification head** is used; the predicted value is `argmax`
of the logits. Hyperparameters: peak α=0.85 and σ=1 (soft labels), τ=0.1 (contrastive),
β=0.999 (class balance), λ=0.3 (contrastive weight in the joint loss); AdamW, 6% linear
warmup, weight decay 0.01, max length 96; lr 2e-5 / batch 32 for base, lr 1e-5 / batch
16 for large.

## 3. Main results (development set, 514 instances)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| Two-stage, base  | 0.9416 | 0.9285 | 0.9339 | 0.9301 |
| Two-stage, large | 0.9572 | 0.9420 | 0.9567 | 0.9465 |
| Joint, base      | 0.9475 | 0.9324 | 0.9427 | 0.9355 |
| **Joint, large** | **0.9533** | **0.9499** | **0.9486** | **0.9471** |

Strongest baseline: fine-tuned RoBERTa at 0.9065 (CIRCLE is ~4 points higher).
Ablation (joint base): removing circular soft labels is the largest drop
(0.9355 → 0.9126), confirming it is the single most important component.

---

## 4. Setup

```bash
pip install -r requirements.txt
# All commands below are run from this package root, with src/ on the path:
export PYTHONPATH=src
```
The device (CUDA / Apple MPS / CPU) is auto-detected. **Note:** on Apple MPS, training
is non-deterministic, so re-training reproduces results only up to ±1–2% run-to-run
noise; **loading the provided checkpoints is fully deterministic.**

## 5. Reproduce the paper from the saved checkpoints (recommended, exact)

The model weights and representations are distributed **separately** (because of their
size, ~12 GB) as `CIRCLE_experiment_weights.zip`, organized one folder per experiment.

**Download (Google Drive):** https://drive.google.com/file/d/1GwT3hQiwCtcvDwc9ONyLAJqaH0ysbFZZ/view?usp=sharing

```bash
# From this package root. Download + unzip the weights (any of the three ways):
pip install gdown
gdown 1GwT3hQiwCtcvDwc9ONyLAJqaH0ysbFZZ -O CIRCLE_experiment_weights.zip
#   or: gdown "https://drive.google.com/uc?id=1GwT3hQiwCtcvDwc9ONyLAJqaH0ysbFZZ"
#   or: download manually from the link above, then place the .zip here.
unzip -q CIRCLE_experiment_weights.zip          # -> CIRCLE_experiment_weights/exp1.../ ... exp5.../

# Place the weights: merge every experiment folder into this package root, so each file
# lands at the path the scripts expect (outputs/... and baselines/...):
for e in CIRCLE_experiment_weights/exp*/; do cp -R "$e"/. .; done
```

To reproduce **one** experiment instead, copy only that experiment's folder, e.g.
`cp -R CIRCLE_experiment_weights/exp1_training_mode/. .` (see `HOW_TO_PLACE.md` inside
the zip). Loading checkpoints is **deterministic** — you get exactly the paper's numbers.

```bash
export PYTHONPATH=src     # all commands are run from this package root
```

### Experiment 1 — Training-mode (joint vs. two-stage), Table 1
- **Files needed** (place under `outputs/`):
  `joint_base/joint_model.pt`, `joint_large/joint_model.pt`,
  `twostage_base/model.pt`, `twostage_large/model.pt`  → from `exp1_training_mode/`
- **Command** (Accuracy / Macro-P / Macro-R / Macro-F1 per model):
```bash
python src/eval_ckpt.py --model microsoft/deberta-v3-base  --ckpt outputs/joint_base/joint_model.pt      # 0.9355
python src/eval_ckpt.py --model microsoft/deberta-v3-large --ckpt outputs/joint_large/joint_model.pt     # 0.9471
python src/eval_ckpt.py --model microsoft/deberta-v3-base  --ckpt outputs/twostage_base/model.pt         # 0.9301
python src/eval_ckpt.py --model microsoft/deberta-v3-large --ckpt outputs/twostage_large/model.pt        # 0.9465
```

### Experiment 2 — Identification accuracy vs. baselines, Table 2
- **Files needed**: the four CIRCLE models from `exp1_training_mode/` **and** all baseline
  weights from `exp2_baselines/` (place under `baselines/`):
  `classical_models/*.joblib`, `neural_models/*.pt`, and each `<repo>/nlpcc_out/...`.
- **Command** (one table with all four metrics for CIRCLE + every baseline):
```bash
python src/eval_all_metrics.py            # -> outputs/all_metrics.{json,csv}
# baseline rows only (macro-F1): python baselines/eval_saved.py
```

### Experiment 3 — Component ablation, Table 3
- **Files needed** (place under `outputs/ablation/`):
  `no_soft/best.pt`, `no_cb/best.pt`, `no_contrastive/best.pt`, `no_supcon/best.pt`
  → from `exp3_ablation/`
- **Command**:
```bash
for v in no_soft no_cb no_contrastive no_supcon; do
  python src/eval_ckpt.py --model microsoft/deberta-v3-base --ckpt outputs/ablation/$v/best.pt
done   # -> 0.9126 / 0.9169 / 0.9181 / 0.9187
```

### Experiment 4 — Error analysis, Tables 4–5
- **Files needed** (place under `outputs/`):
  `joint_base/joint_model.pt`, `onehot_simple/best.pt`  → from `exp4_error_analysis/`
- **Command** (error totals, within/cross-group, arc-distance distribution):
```bash
python src/error_analysis.py --model microsoft/deberta-v3-base --ckpt outputs/joint_base/joint_model.pt    # 27 errors, 70% in-group
python src/error_analysis.py --model microsoft/deberta-v3-base --ckpt outputs/onehot_simple/best.pt        # 41 errors, 56% in-group
```

### Experiment 5 — Representation analysis (silhouette + t-SNE), Table 6 & figures
- **Files needed** (place under `outputs/`): the representation `.npz` files (no `.pt`):
  `joint_base/joint_embs.npz`, `joint_large/joint_embs.npz`,
  `twostage_base/embs.npz`, `twostage_large/embs.npz`  → from `exp5_representation/`
- **Command** (prints silhouette coefficients and writes the t-SNE figures):
```bash
python src/make_best_tsne.py     # two-stage: silhouette + outputs/best_tsne_{4group,19class}.png
python src/make_joint_tsne.py    # joint:     silhouette + outputs/joint_tsne_{4group,19class}.png
python src/make_class_freq.py    # class-frequency figure
```

> **One-shot:** merge all five `expN/` folders into this package root, then run
> `eval_all_metrics.py`, the two `error_analysis.py` commands, and the two `make_*_tsne.py`
> scripts to regenerate every table and figure in one pass.

## 6. Train from scratch

Re-training reproduces the results up to seeds noise (see §4). Saved checkpoints are
written to the given `--out` directory.

```bash
export PYTHONPATH=src

# --- CIRCLE, joint objective ---
python src/train_joint.py --model microsoft/deberta-v3-large --out outputs/joint_large \
    --epochs 30 --bs 16 --lr 1e-5 --supcon 0.3 --temp 0.10 --seed 1
python src/train_joint.py --model microsoft/deberta-v3-base  --out outputs/joint_base \
    --epochs 22 --bs 32 --lr 2e-5 --supcon 0.3 --temp 0.10 --seed 42

# --- CIRCLE, two-stage schedule (SupCon stage -> classification stage) ---
python src/train_twostage.py --model microsoft/deberta-v3-base  --out outputs/twostage_base  --s1_epochs 10 --s2_epochs 12
python src/train_twostage.py --model microsoft/deberta-v3-large --out outputs/twostage_large --s1_epochs 20 --s2_epochs 10 --bs 16

# --- Ablation (leave-one-out on joint base) ---
python src/train_joint_ablation.py --model microsoft/deberta-v3-base --out outputs/ablation/no_soft        --no_soft_label
python src/train_joint_ablation.py --model microsoft/deberta-v3-base --out outputs/ablation/no_cb          --no_class_balance
python src/train_joint_ablation.py --model microsoft/deberta-v3-base --out outputs/ablation/no_contrastive --no_contrastive
python src/train_joint_ablation.py --model microsoft/deberta-v3-base --out outputs/ablation/no_supcon      --supcon 0

# --- Baselines ---
python baselines/classical_baseline.py                  # TF-IDF + LinearSVC / LogReg / XGBoost
python baselines/neural_baseline.py                     # BERT-base / RoBERTa-base fine-tuning
python baselines/kiesel_acl22/run_nlpcc.py all          # Kiesel official BERT / SVM
# The five participant-system baselines below are fully bundled (the complete cloned
# repositories are included under baselines/<repo>/, so no extra download is needed):
python baselines/su0315_hvd/run_nlpcc.py 4
python baselines/adam_smith/run_nlpcc.py 4
python baselines/cocchieri_bert_roberta/run_nlpcc.py 4
python baselines/veiledtee_semeval2023/run_nlpcc.py
python baselines/value_fulcra/run_nlpcc.py 2
```

## 7. (Optional) Generate the official submission file

All paper results are reproduced from `data/test.jsonl` above. The **unlabeled official
test set** (`track1.jsonl`) is **not bundled** in this package, since it is only needed
to regenerate the leaderboard submission, not to reproduce any reported number. If you
have it, point `predict.py` at it:

```bash
export PYTHONPATH=src
python src/predict.py --test /path/to/track1.jsonl --ckpt outputs/joint_large/joint_model.pt \
    --out preds/track1.pred.jsonl     # uses the DeBERTa-v3-large joint model
```

---

## 8. Directory layout

```
CIRCLE_reproduction/
├── README.md / requirements.txt
├── data/
│   └── train.jsonl (3,520, labeled)   test.jsonl (514, labeled = evaluation set)
├── src/                         # CIRCLE code
│   ├── values.py                # 19 values, ring distance, circular soft labels
│   ├── model.py                 # ValueClassifier + soft_label_ce + supcon_loss
│   ├── data.py                  # dataset (consistent / contrastive responses)
│   ├── train.py                 # device, effective-number weights, evaluate()
│   ├── train_joint.py / train_twostage.py / train_joint_ablation.py
│   ├── eval_ckpt.py / eval_all_metrics.py / error_analysis.py / predict.py
│   └── tsne_plot.py / make_*_tsne.py / make_*_figs.py
├── baselines/                   # baseline code + saved baseline weights
│   ├── classical_baseline.py / neural_baseline.py / eval_saved.py / README.md
│   ├── classical_models/ neural_models/        (saved baseline weights)
│   └── <repo>/                                  (FULL cloned repo + run_nlpcc.py + nlpcc_out/ weights)
└── outputs/                     # trained CIRCLE weights + representations (npz)
    ├── joint_{base,large}/      joint_model.pt + joint_embs.npz + results
    ├── twostage_{base,large}/   model.pt + stage1.pt + embs.npz + result
    ├── ablation/{no_soft,no_cb,no_contrastive,no_supcon}/best.pt
    └── onehot_simple/best.pt    (one-hot variant for the error analysis)
```

## 9. Data fields and rules

Each training instance has `Scenario`, `Question`, `Value` (label), `Consistent Value
Response`, and `Contrastive Response`. **The classifier input is always the single
`Consistent Value Response`** (the only field available at test time). `Contrastive
Response` is used only as a training-time hard negative; `Scenario`/`Question` are not
used. This matches the shared-task constraint that extra supervised data is restricted.
