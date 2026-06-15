# Email Ticket Classifier — MLOps Project Documentation

> A production-grade, fully automated email classification and routing system powered by a fine-tuned DistilBERT Transformer, deployed across a multi-cloud MLOps stack with continuous learning, human-in-the-loop validation, and live email automation via Resend.

| Field | Value |
|---|---|
| **Architecture** | DistilBERT Transformer + Sklearn Baseline |
| **Transformer** | `distilbert-base-uncased` (66M params, 6 layers) |
| **Training Strategy** | Continuous Learning Pipeline (CLP) with GitHub Actions |
| **Model Registry** | Hugging Face Hub (`ouel/bert-ticket-classifier`) |
| **Serving** | FastAPI (Railway) + Streamlit Cloud Dashboard |
| **Database** | PostgreSQL (Railway) — shared between all services |
| **Email Automation** | Resend Inbound Webhooks → Auto-routing |
| **Status** | ✅ Production |

---

## 01 — Project Overview

The system automatically classifies inbound support emails into 5 structured categories, instantly routes them to the correct department via Resend, and continuously improves through human validation feedback. Every human correction is automatically committed to GitHub as a training event, triggering a model retrain through GitHub Actions when enough corrections accumulate.

**Key capabilities:**

- ⚡ **Real-time email ingestion** — Resend webhooks fire on every inbound email, triggering classification within milliseconds
- 🧠 **DistilBERT Transformer** — fine-tuned sequence classifier with 66M parameters and 97% of BERT's accuracy at 60% of the compute cost
- 🔁 **Continuous Learning Pipeline (CLP)** — GitHub Actions retrains the live model on accumulated human corrections automatically
- 🔒 **Quality gate** — no model ships to production below a configurable F1 threshold
- 🗂️ **Universal validation queue** — every ticket (auto-routed or flagged) appears in the human queue for ground-truth labeling
- 🌍 **Multi-cloud architecture** — Streamlit Cloud (dashboard), Railway (API + PostgreSQL), Hugging Face Hub (model registry)

---

## 02 — Transformer Architecture (Core Model)

This is the most critical component of the system. The production model is a **DistilBERT fine-tuned sequence classifier** trained on labeled support email text.

### 2.1 — What is DistilBERT?

DistilBERT is a distilled (compressed) version of BERT (Bidirectional Encoder Representations from Transformers), developed by Hugging Face in 2019. It is trained using **knowledge distillation** — a technique where a smaller "student" model is trained to mimic the behavior of a larger "teacher" model (BERT-base).

| Property | BERT-base | DistilBERT |
|---|---|---|
| Parameters | 110M | **66M** |
| Layers (Transformer blocks) | 12 | **6** |
| Hidden dimension | 768 | **768** |
| Attention heads | 12 | **12** |
| Inference speed | 1× baseline | **1.6× faster** |
| Accuracy retention | 100% | **97% of BERT** |
| Memory footprint | ~440 MB | **~265 MB** |

### 2.2 — Transformer Block (Detailed)

Each of the **6 Transformer encoder blocks** in DistilBERT is composed of the following sub-layers:

```
Input Tokens (WordPiece tokenized)
        │
        ▼
┌──────────────────────────────────────────────┐
│           Token Embeddings (768-dim)          │
│  = WordPiece Embedding                        │
│  + Position Embedding                         │
│  (Note: No segment/token-type embedding       │
│   unlike BERT — this is the key difference)   │
└──────────────────────────────────────────────┘
        │
        ▼ × 6 blocks
┌──────────────────────────────────────────────┐
│         Multi-Head Self-Attention             │
│                                              │
│  Q = W_q · x,  K = W_k · x,  V = W_v · x   │
│                                              │
│         QK^T                                 │
│  Attn = ─────── · V   (scaled dot-product)   │
│          √d_k                                │
│                                              │
│  12 attention heads, each with dim = 64      │
│  (12 × 64 = 768 total)                       │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│     Add & LayerNorm  (residual connection)    │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│        Feed-Forward Network (FFN)             │
│                                              │
│  FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂      │
│  Intermediate dim: 3072  (4 × hidden)        │
│  Activation: GELU (not ReLU like BERT)        │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│     Add & LayerNorm  (residual connection)    │
└──────────────────────────────────────────────┘
        │
        ▼ (after all 6 blocks)
┌──────────────────────────────────────────────┐
│     [CLS] token representation (768-dim)      │
│     (used as the sentence embedding)          │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│        Classification Head                    │
│                                              │
│  Linear(768 → 768) + GELU + Dropout(0.2)     │
│  Linear(768 → num_labels)                    │
│  Softmax → probability over 5 classes        │
└──────────────────────────────────────────────┘
```

### 2.3 — Key Architectural Differences vs. BERT

| Feature | BERT | DistilBERT | Impact |
|---|---|---|---|
| Segment embeddings | ✅ Yes | ❌ Removed | Smaller embedding table |
| Pooler layer | ✅ Yes | ❌ Removed | Less post-processing |
| Number of layers | 12 | **6** | 2× faster forward pass |
| NSP (Next Sentence Prediction) | ✅ Trained on | ❌ Removed | Simplified pre-training |
| Distillation loss | ❌ No | ✅ Cosine + CE + MLM | Better compression |

### 2.4 — Tokenization

The model uses **WordPiece tokenization** (vocabulary of 30,522 tokens):

```
Input:  "I cannot access my account since yesterday"
Tokens: [CLS] I cannot access my account since yesterday [SEP]
IDs:    [101, 1045, 3685, 3229, 2026, 4070, 2144, 7483, 102]
```

- Maximum sequence length: **512 tokens**
- Longer texts are truncated to 512 tokens during inference
- `[CLS]` token embedding is used as the document representation for classification

### 2.5 — Fine-tuning Configuration

The pre-trained `distilbert-base-uncased` weights are fine-tuned end-to-end with the following hyperparameters (stored in `params.yaml`):

```yaml
model:
  name: distilbert-base-uncased
  num_labels: 5
  max_length: 512

training:
  learning_rate: 2.0e-5          # standard BERT fine-tuning LR
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  warmup_ratio: 0.1              # 10% of steps used for LR warm-up
  weight_decay: 0.01             # L2 regularisation
  evaluation_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: eval_f1

optimizer: AdamW                 # with decoupled weight decay
scheduler: linear_with_warmup
```

### 2.6 — Classification Labels (id2label mapping)

```python
id2label = {
    0: "account_access",    # Login issues, password resets, locked accounts
    1: "billing",           # Charges, invoices, subscription problems
    2: "bug_report",        # Software crashes, unexpected behavior, errors
    3: "refund_request",    # Refund demands, chargeback requests
    4: "shipping_delivery", # Delivery status, lost packages, delays
}
```

---

## 03 — Continuous Learning Pipeline (CLP)

The CLP is the ML automation backbone. It runs as a GitHub Actions workflow triggered either automatically via `repository_dispatch` from the Streamlit dashboard or manually.

### 3.1 — Pipeline Flow

```
Human validates ticket in Streamlit
        │
        ▼
save_feedback() pushes fb_<uuid>.json to GitHub
  data/feedback/YYYY/MM/DD/fb_<uuid>.json
        │
        ▼
GitHub Actions: repository_dispatch fires ("retrain" event)
        │
        ▼
train_clp.py — Continuous Learning Pipeline
        │
        ├── 1. Scan data/feedback/**/*.json
        │       Count unprocessed events
        │       Check MIN_BATCH_SIZE gate (default: 10)
        │
        ├── 2. Build training dataset
        │       Load all feedback JSON files
        │       Extract: text = subject + body
        │       Extract: label = corrected_label (human ground truth)
        │       Train/eval split: 80/20
        │
        ├── 3. Load current production model from Hugging Face Hub
        │       Pull: ouel/bert-ticket-classifier (main branch)
        │       Preserves all weights from previous training
        │
        ├── 4. Fine-tune on new feedback data
        │       HuggingFace Trainer with above config
        │       Continues from existing weights (not from scratch)
        │
        ├── 5. Evaluate on held-out split
        │       Compute: accuracy, F1, precision, recall
        │       Compare against QUALITY_THRESHOLD
        │
        ├── 6. Quality gate check
        │   If F1 >= threshold:
        │       Push new model weights to HF Hub (main branch)
        │       Update model_pointers.json with new version tag
        │       Mark feedback files as processed
        │   Else:
        │       Skip — model not promoted
        │
        └── 7. Mark feedback events as processed
                Write processed_ids.json to prevent double-training
```

### 3.2 — Version Tagging

Each trained model version is tagged with a timestamp: `v{YYYYMMDDHHMMSS}`.

`model_pointers.json` tracks which version is live:
```json
{
  "active": "v20260615171643",
  "stable": "main"
}
```

The Streamlit dashboard reads this file every 5 minutes and displays the active model version in the sidebar.

---

## 04 — Email Automation Architecture

### 4.1 — End-to-End Flow (Production)

```
Customer sends email to any@neurodynamics.tech
        │
        ▼
Resend MX Servers (inbound-smtp.eu-west-1.amazonaws.com)
        │  DNS MX record routes all email to Resend
        ▼
Resend Webhook → POST /webhook/resend
        │  URL: email-ticket-classifier-production.up.railway.app
        ▼
FastAPI Webhook Receiver (Railway — src/api/webhook.py)
        │
        ├── Parse: subject, body, from_address
        │
        ├── Classify: HuggingFace pipeline(text[:512])
        │       Returns: label + confidence score
        │
        ├── Lookup routing: PostgreSQL → routing_settings table
        │       WHERE label = predicted_label
        │       Returns: destination_email
        │
        ├── Routing decision:
        │   If confidence >= 0.85 AND destination_email configured:
        │       status = "auto_routed"
        │       Resend SDK → send forwarded email to destination
        │   Else:
        │       status = "pending_review"
        │
        └── INSERT INTO tickets (PostgreSQL)
                → Instantly visible in Streamlit queue
```

### 4.2 — Multi-Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT CLOUD                           │
│                                                             │
│  streamlit_app.py                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Test         │  │ Human Queue  │  │ Settings         │  │
│  │ Classifier   │  │ (validate    │  │ (routing email   │  │
│  │              │  │  all tickets)│  │  per category)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                 │                   │              │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │                 │                   │
          └─────────────────┼───────────────────┘
                            │ PostgreSQL (SSL)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAILWAY                                 │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  Webhook API         │    │  PostgreSQL Database      │  │
│  │  (FastAPI/Uvicorn)   │◄──►│                          │  │
│  │                      │    │  tables:                 │  │
│  │  POST /webhook/resend│    │  - tickets               │  │
│  │                      │    │  - routing_settings      │  │
│  └──────────────────────┘    └──────────────────────────┘  │
│           ▲ (internal network, no SSL)                       │
└───────────┼─────────────────────────────────────────────────┘
            │
            │ Resend Webhook
┌───────────┼─────────────────────────────────────────────────┐
│  RESEND   │                                                  │
│           │                                                  │
│  Inbound email received at neurodynamics.tech               │
│  MX: inbound-smtp.eu-west-1.amazonaws.com                   │
│  Fires POST to Railway webhook URL                          │
└─────────────────────────────────────────────────────────────┘
            │
            │ GitHub API (feedback events)
┌───────────▼─────────────────────────────────────────────────┐
│  GITHUB + GITHUB ACTIONS                                     │
│                                                             │
│  data/feedback/YYYY/MM/DD/fb_<uuid>.json                    │
│  → triggers retrain.yml workflow                            │
│  → trains DistilBERT on corrections                         │
│  → pushes new model to Hugging Face Hub                     │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  HUGGING FACE HUB                                            │
│                                                             │
│  ouel/bert-ticket-classifier                                │
│  ├── model.safetensors  (268 MB — DistilBERT weights)       │
│  ├── config.json        (num_labels=5, id2label mapping)    │
│  ├── tokenizer.json     (WordPiece vocab, 30522 tokens)     │
│  └── model_pointers.json (active version tag)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 05 — Classification Categories

| Label | Description | Threshold | Example |
|---|---|---|---|
| `account_access` | Login failures, locked accounts, password resets | `>= 0.85` | *"I can't log in, my account is locked"* |
| `billing` | Charges, invoices, subscription issues | `>= 0.85` | *"I was charged twice this month"* |
| `bug_report` | Software crashes, errors, unexpected behavior | `>= 0.85` | *"The app crashes when I click submit"* |
| `refund_request` | Refund demands, cancellations, chargebacks | `>= 0.85` | *"I want my money back immediately"* |
| `shipping_delivery` | Delivery status, lost packages, delays | `>= 0.85` | *"My order hasn't arrived after 2 weeks"* |

> **Pending Review:** Any prediction with confidence below `0.85` is queued for human review instead of being auto-routed. Every human correction generates a labeled training event.

---

## 06 — Technology Stack

| Layer | Tool | Role |
|---|---|---|
| **Transformer Model** | `distilbert-base-uncased` (HuggingFace) | Core NLP classifier — 6-layer, 66M param BERT distillation |
| **Training Framework** | HuggingFace `Trainer` + `datasets` | Fine-tuning loop, evaluation, checkpointing |
| **Model Registry** | Hugging Face Hub | Version-controlled model storage (`model_pointers.json`) |
| **Baseline Model** | scikit-learn TF-IDF + Logistic Regression | Fast fallback when HF model unavailable |
| **Webhook API** | FastAPI + Uvicorn | Receives Resend emails, classifies, forwards, inserts to DB |
| **Dashboard** | Streamlit Cloud | Human validation queue, routing settings, CLP status |
| **Database** | PostgreSQL (Railway) | Shared tickets + routing_settings tables |
| **Email Automation** | Resend (inbound + outbound) | Receives raw emails, forwards classified emails |
| **CI/CD + CLP** | GitHub Actions (`retrain.yml`) | Auto-retrains DistilBERT on human corrections |
| **Feedback Store** | GitHub repository (`data/feedback/`) | Immutable JSON event log for CLP training data |
| **Experiment Tracking** | MLflow | Metrics logging per training run |
| **Data Versioning** | DVC | Reproducible pipeline and dataset snapshots |
| **Containerization** | Docker | Local development and testing |

---

## 07 — Project Structure

```
email-ticket-classifier/
│
├── src/
│   ├── api/
│   │   └── webhook.py          # FastAPI: Resend inbound → classify → forward → DB
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine (PostgreSQL or SQLite fallback)
│   │   └── models.py           # Ticket + RoutingSettings ORM models
│   ├── pipeline/
│   │   └── train_clp.py        # Continuous Learning Pipeline (CLP)
│   └── serving/
│       ├── model_loader.py     # Load DistilBERT from HuggingFace Hub
│       └── predict.py          # Inference wrapper
│
├── data/
│   └── feedback/               # CLP training events (JSON, committed to Git)
│       └── YYYY/MM/DD/
│           └── fb_<uuid>.json  # {text, label, corrected_label, confidence}
│
├── .github/workflows/
│   └── retrain.yml             # GitHub Actions CLP: triggered by repository_dispatch
│
├── streamlit_app.py            # Dashboard: Test / Queue / History / Settings tabs
├── main.py                     # Railway entry point: uvicorn src.api.webhook:app
├── Procfile                    # Heroku/Render: web: uvicorn ...
├── railway.json                # Railway Nixpacks builder config
├── params.yaml                 # All hyperparameters (single source of truth)
├── requirements.txt            # Runtime dependencies (pip)
└── pyproject.toml              # Package metadata + optional dev deps
```

---

## 08 — MLOps Feedback Loop

```
Customer Email Arrives
        │
        ▼
Resend → Railway Webhook
        │  classify in < 500ms
        ▼
High confidence? ──YES──► Forward to department email (Resend SDK)
        │                  Status: auto_routed
        │ NO
        ▼
Queue for review
Status: pending_review
        │
        ▼
Human opens Streamlit Queue
        │  sees ALL tickets (auto-routed + pending)
        ▼
Human validates label
        │  "Validate & Resolve" button
        ▼
save_feedback() → push fb_<uuid>.json to GitHub
        │
        ▼
GitHub Actions detects new feedback file
        │  repository_dispatch("retrain")
        ▼
MIN_BATCH_SIZE gate (default: 10 corrections)
        │
        ▼
train_clp.py downloads current model from HF Hub
        │  continues training from existing weights
        ▼
Fine-tune on all new corrections
        │
        ▼
Evaluate F1 on held-out split
        │
        ▼
F1 >= threshold? ──YES──► Push new model to HF Hub
        │                  Update model_pointers.json
        │                  Streamlit loads new version (TTL 5min)
        │ NO
        ▼
Model not promoted (current champion stays)
```

> Every human validation is a **free labeled training example**. The model that serves in month 3 will be substantially more accurate than the one deployed at launch — without any manual ML engineering effort.

---

## 09 — CI/CD Pipeline (GitHub Actions)

### `retrain.yml` — Continuous Learning Workflow

Triggered by: `repository_dispatch` (event type: `retrain`) OR `workflow_dispatch` (manual)

```yaml
Steps:
  1. Checkout repository
  2. Set up Python 3.11
  3. Install dependencies (requirements.txt)
  4. Run: python src/pipeline/train_clp.py
        ├── Scan data/feedback/ for unprocessed events
        ├── Gate: MIN_BATCH_SIZE check
        ├── Load DistilBERT from HF Hub
        ├── Fine-tune on feedback data
        ├── Evaluate F1
        └── Push to HF Hub if quality gate passes
  5. Commit processed_ids.json back to repo
```

### Quality Gate

| Metric | Threshold | Action if below |
|---|---|---|
| Weighted F1 | Configurable in `params.yaml` | Model not promoted — current champion stays |
| Eval Accuracy | Logged to MLflow | Informational only |

---

## 10 — Database Schema

### `tickets` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ticket ID |
| `subject` | STRING | Email subject line |
| `body` | STRING | Email body text |
| `predicted_label` | STRING | Model's predicted category |
| `confidence` | FLOAT | Softmax probability (0.0–1.0) |
| `human_label` | STRING | Human-corrected label (nullable) |
| `status` | STRING | `pending_review` / `auto_routed` / `resolved` |
| `response_email` | STRING | (legacy, nullable) |
| `created_at` | DATETIME | UTC timestamp |

### `routing_settings` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `label` | STRING UNIQUE | Ticket category (e.g. `billing`) |
| `destination_email` | STRING | Where to forward this category |
| `updated_at` | DATETIME | Last modified timestamp |

---

*Email Ticket Classifier — Production MLOps Documentation*
*DistilBERT Transformer + Continuous Learning Pipeline + Resend Automation*
*Built: June 2026*
