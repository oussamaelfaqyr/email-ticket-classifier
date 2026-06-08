# Email Ticket Classifier — MLOps Project Documentation

> A two-stage NLP classification system trained on a Kaggle database and fine-tuned on real-world email corpora, deployed as a production MLOps pipeline with automated retraining and live email integration.

| Field | Value |
|---|---|
| Architecture | Two-stage NLP + MLOps |
| Models | DistilBERT / TF-IDF + LR |
| Serving | FastAPI + Docker |
| Tracking | MLflow + DVC |
| Status | Design Specification |

---

## 01 — Project Overview

The system automatically classifies inbound support emails into structured categories, routes them to the appropriate team or ticketing system, and continuously improves through human feedback. It follows the same two-stage training strategy used in production AI systems: first learning on a clean Kaggle database, then adapting to messy real-world language.

**Key capabilities**

- Automated inbox triage — no manual sorting required
- Two-stage NLP training on a Kaggle database then real-world data
- Continuous retraining triggered by drift detection
- Quality gates enforced in CI/CD — no model ships below F1 0.85
- Drift monitoring with Evidently AI in production
- Human-in-the-loop feedback that feeds the next retrain cycle

> **Target metric:** F1-score of 0.85 or above across all four categories is enforced as a CI quality gate. No model is promoted to production below this threshold.

---

## 02 — Training Pipeline

Training is split into two sequential stages. The first stage builds a reliable baseline using a clean, controlled Kaggle database. The second stage adapts that baseline to real-world noise, abbreviations, and domain-specific language using the Enron corpus and live support ticket archives.

### Stage 01 — Kaggle database training

- Kaggle dataset supplies labeled email samples per category
- Balanced class distribution enforced across all labels
- TF-IDF + Logistic Regression trained as fast baseline model
- Every run logged to MLflow with full parameter set
- Validates the pipeline end-to-end before touching real data

### Stage 02 — Real-world fine-tuning

- Enron corpus + support ticket archive as source data
- Header stripping, deduplication, tokenization applied
- DistilBERT fine-tuned with low learning rate (domain adaptation)
- Optuna runs hyperparameter optimization with pruning
- Champion vs challenger comparison before any promotion

### Data flow

```
Raw emails  →  Preprocess  →  Features  →  Train  →  Evaluate  →  Registry
(Kaggle         (clean +       (TF-IDF /    (stage     (F1 quality   (MLflow
 + Enron)        tokenize)      embeddings)  1 → 2)     gate)          promote)
```

All stages are defined as DVC pipeline nodes in `dvc.yaml`. Running `dvc repro` recomputes only the stages whose inputs have changed, making reruns fast and reproducible.

---

## 03 — Classification Categories

The model outputs one of four labels per email, along with a confidence score between 0 and 1. Routing behavior is conditioned on both the label and the confidence threshold.

| Label | Description | Auto-route destination | Threshold |
|---|---|---|---|
| `bug_report` | Crashes, unexpected behavior, data errors | Jira / Linear — engineering queue | `>= 0.85` |
| `billing` | Charges, refunds, subscription questions | Zendesk — billing team | `>= 0.85` |
| `feature_request` | Product suggestions, improvement ideas | Slack — #product channel | `>= 0.85` |
| `general` | Account help, how-to, other queries | Zendesk — default support queue | `>= 0.80` |

> **Human review queue:** Any prediction with confidence below the threshold is routed to human review rather than auto-routed. Agent labels on these cases feed back into the next retraining cycle.

---

## 04 — Email Integration Architecture

Inbound emails arrive at `support@yourapp.com` and are converted into structured webhook events by the email provider. A FastAPI receiver parses each event, extracts subject and body, strips signatures and quoted history, then pushes to a Redis queue for async classification.

### End-to-end flow

```
Inbox
  |
  v
Email provider  (SendGrid Inbound Parse / Mailgun / Gmail API / IMAP poll)
  |
  v
POST /inbound   (FastAPI webhook receiver — parse, strip, enqueue)
  |
  v
Redis queue     (async buffer — decouples receiving from classifying)
  |
  v
POST /classify  (ML classifier — label + confidence score)
  |
  v
Routing logic   (confidence threshold check)
  |
  +---> >= 0.85 high conf  -->  auto-route to destination
  +---> 0.60–0.85 medium   -->  route with review flag
  +---> < 0.60 low conf    -->  human review queue
```

### Email provider options

**SendGrid Inbound Parse** — recommended starting point. Configure an MX record and SendGrid POSTs every inbound email as multipart form data to your webhook URL. Free tier covers most projects. 5-minute setup.

**Mailgun Routes** — same webhook model, slightly more flexible pre-filtering rules before the webhook fires. Good if you need routing rules at the provider level.

**Gmail API + Pub/Sub** — if the inbox is already a Gmail account. Subscribe to push notifications via Google Cloud Pub/Sub. No MX record changes required.

**IMAP polling** — simplest of all. `imaplib` or `aioimaplib` polling every 30 seconds. Works with any mailbox. Not real-time but sufficient for low volume.

### Routing destinations

| Destination | Label | Integration |
|---|---|---|
| Jira / Linear | `bug_report` | REST API — creates ticket with assignee, priority, label |
| Zendesk | `billing` | REST API — lands in billing group inbox with category tag |
| Slack `#product` | `feature_request` | Slack API — posts summary card with sender and subject |
| Zendesk | `general` | REST API — default support queue |
| Human review | any low confidence | Internal queue — agent re-labels, correction written back |
| Auto-reply | all | Sends acknowledgment email with category and expected SLA |

### Confidence-gated routing (Python sketch)

```python
def route(label: str, confidence: float, email: dict):
    if confidence >= 0.85:
        auto_route(label, email)           # straight to destination
    elif confidence >= 0.60:
        route_with_flag(label, email)      # routed but flagged for spot-check
    else:
        send_to_human_review(email)        # agent labels → training data
```

---

## 05 — Technology Stack

| Component | Tool | Purpose |
|---|---|---|
| Data versioning | DVC | Reproducible pipeline, dataset and model artifact versioning |
| Experiment tracking | MLflow | Log every run, compare metrics, promote to registry |
| NLP model | HuggingFace Transformers | DistilBERT fine-tuning, tokenizer, inference |
| Baseline model | scikit-learn | TF-IDF vectorizer + logistic regression stage 1 |
| Hyperparameter search | Optuna | Automated HPO with pruning and study persistence |
| Serving | FastAPI + Uvicorn | REST inference endpoint, webhook receiver, health check |
| Containerization | Docker + Compose | App + MLflow + Prometheus in one compose stack |
| Monitoring | Prometheus + Grafana | Latency, throughput, prediction distribution dashboards |
| Drift detection | Evidently AI | Data and concept drift alerts, auto-retrain trigger |
| CI/CD | GitHub Actions | Lint, test, quality gate, build Docker, push, deploy |
| Email ingestion | SendGrid Inbound Parse | Convert inbound email to webhook POST events |
| Task queue | Redis + Celery | Async email processing, retrain job scheduling |

---

## 06 — Project Structure

```
email-ticket-classifier/
├── .dvc/                       # DVC config and remote settings
├── data/                       # versioned by DVC, never committed raw
│   ├── raw/kaggle/             # Kaggle database labeled emails
│   ├── raw/enron/              # Enron email corpus
│   └── processed/              # cleaned, split, tokenized
├── src/
│   ├── data/
│   │   ├── ingest_kaggle.py        # Kaggle dataset downloader
│   │   ├── ingest_enron.py         # download + parse Enron
│   │   └── preprocess.py           # clean, tokenize, split
│   ├── features/
│   │   └── build_features.py       # TF-IDF, embeddings, DVC stage
│   ├── training/
│   │   ├── train_baseline.py       # TF-IDF + LR, stage 1
│   │   ├── train_bert.py           # DistilBERT fine-tune, stage 2
│   │   ├── hpo.py                  # Optuna HPO search
│   │   └── evaluate.py             # F1, precision, recall, CM
│   ├── serving/
│   │   ├── app.py                  # FastAPI app (predict + inbound)
│   │   ├── model_loader.py         # load champion from MLflow registry
│   │   └── schemas.py              # Pydantic request/response models
│   └── monitoring/
│       ├── drift_detector.py       # Evidently AI drift reports
│       └── metrics_exporter.py     # Prometheus metrics
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_model_quality.py       # F1 quality gate (threshold 0.85)
│   └── test_api.py                 # FastAPI endpoint tests
├── .github/workflows/
│   ├── ci.yml                  # lint, test, quality gate on PR
│   └── cd.yml                  # build Docker, push, deploy on merge
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml      # app + MLflow + Prometheus
├── dvc.yaml                    # pipeline DAG definition
├── params.yaml                 # all hyperparameters (single source of truth)
├── pyproject.toml              # deps, ruff, mypy, pytest config
└── Makefile                    # make train / test / serve / retrain
```

**Key conventions:**

- `params.yaml` is the single source of truth for every hyperparameter — no magic numbers in code
- `dvc.yaml` defines the full pipeline DAG — `dvc repro` reruns only what changed
- `data/` is never committed to git — only `.dvc` pointer files are tracked
- The `tests/test_model_quality.py` quality gate runs in CI on every pull request

---

## 07 — MLOps Feedback Loop

The system is designed as a closed loop. Production predictions that are corrected by human agents automatically become training data. Evidently AI monitors the incoming email distribution for drift. When drift or performance degradation is detected, a retraining job is scheduled via Celery, the model is evaluated against the quality gate, and — if it passes — promoted to production without manual intervention.

### Continuous improvement cycle

```
Production predictions
        |
        v
Human review queue  (agents re-label low-confidence predictions)
        |
        v
Training store      (corrections written back automatically)
        |
        v
Drift trigger       (Evidently AI detects distribution shift)
        |
        v
Retrain job         (Celery schedules DVC repro run)
        |
        v
Quality gate        (F1 >= 0.85 required to proceed)
        |
        v
Promote to production  (new champion replaces old in MLflow registry)
        |
        +---> back to Production predictions  (loop continues)
```

### What this means in practice

The model that serves predictions in month three will be meaningfully better than the one deployed at launch — without any manual intervention beyond reviewing low-confidence tickets, which support agents already do as part of their normal workflow. Every human correction is an unlabeled training example being labeled for free.

---

## 08 — CI/CD Pipeline

Every pull request triggers the CI workflow. Every merge to `main` triggers the CD workflow.

### CI workflow (`ci.yml`)

1. Install dependencies from `pyproject.toml`
2. Run `ruff` linting and `mypy` type checking
3. Run `pytest tests/` — unit and integration tests
4. Run `test_model_quality.py` — F1 quality gate against held-out test set
5. Fail the PR if any step fails — no merging below threshold

### CD workflow (`cd.yml`)

1. Build Docker image from `docker/Dockerfile`
2. Push image to container registry (GCP Artifact Registry or AWS ECR)
3. Deploy to cloud run / ECS using updated image tag
4. Run smoke test against live `/health` and `/classify` endpoints
5. Roll back automatically if smoke test fails

---

*Email Ticket Classifier — MLOps Project Specification*
*Two-stage NLP + Production MLOps Pipeline*
