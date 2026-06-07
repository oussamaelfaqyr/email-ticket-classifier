# Email Ticket Classifier — Full Project Report

> A production-grade, two-stage NLP classification system for inbound support emails, built using a complete MLOps pipeline with automated training, serving, monitoring, and CI/CD.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Repository Structure](#3-repository-structure)
4. [Dataset](#4-dataset)
5. [Configuration System](#5-configuration-system)
6. [Data Pipeline](#6-data-pipeline)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Training](#8-model-training)
   - [Stage 1 — Baseline (TF-IDF + Logistic Regression)](#stage-1--baseline-tf-idf--logistic-regression)
   - [Stage 2 — Advanced (DistilBERT Fine-Tuning)](#stage-2--advanced-distilbert-fine-tuning)
   - [Hyperparameter Optimization (Optuna)](#hyperparameter-optimization-optuna)
9. [Experiment Tracking — MLflow](#9-experiment-tracking--mlflow)
10. [DVC Pipeline](#10-dvc-pipeline)
11. [Model Serving — FastAPI](#11-model-serving--fastapi)
12. [Monitoring](#12-monitoring)
13. [Containerization — Docker](#13-containerization--docker)
14. [CI/CD — GitHub Actions](#14-cicd--github-actions)
15. [MLOps Feedback Loop](#15-mlops-feedback-loop)
16. [How to Run](#16-how-to-run)
17. [Tech Stack Summary](#17-tech-stack-summary)

---

## 1. Project Overview

The **Email Ticket Classifier** automatically categorizes inbound support emails into structured labels and routes them to the correct team or ticketing system. It eliminates manual inbox triage by combining fast classical ML with state-of-the-art transformer models.

### Core Capabilities

| Capability | Description |
|---|---|
| Automated triage | No manual sorting; every email gets a label and routing destination |
| Two-stage NLP training | Fast TF-IDF baseline first, then DistilBERT fine-tuning |
| MLflow experiment tracking | Every run logged with full params, metrics, and model artifacts |
| DVC data versioning | Reproducible pipeline; only re-runs changed stages |
| Confidence-gated routing | High confidence → auto-route; low confidence → human review |
| FastAPI inference endpoint | REST API for real-time classification |
| Docker packaging | Entire stack runnable with a single `docker-compose up` |
| GitHub Actions CI/CD | Automated linting, testing, quality gate, build, and deploy |

### Target Metric

> **F1-score ≥ 0.85** is the enforced quality gate across all categories. No model is promoted to production below this threshold.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                           │
│                                                                     │
│  data/raw/data.csv                                                  │
│       │                                                             │
│       ▼                                                             │
│  src/data/ingest.py       ← verifies / copies raw data             │
│       │                                                             │
│       ▼                                                             │
│  src/data/preprocess.py   ← cleans, combines subject+text, splits  │
│       │                                                             │
│       ├──────────────────────────────────────────────┐             │
│       ▼                                              ▼             │
│  src/features/            src/training/bert/         │             │
│  build_features.py        train.py                   │             │
│  (TF-IDF vectorizer)      (DistilBERT fine-tune)     │             │
│       │                          │                   │             │
│       ▼                          ▼                   │             │
│  src/training/baseline/   models/bert_model/         │             │
│  train.py                 (saved HF model)           │             │
│  (Logistic Regression)           │                   │             │
│       │                          │                   │             │
│       ▼                          ▼                   │             │
│  models/baseline.pkl      MLflow Experiment          │             │
│       │                   Tracking (mlflow.db)       │             │
│       └──────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         SERVING PIPELINE                            │
│                                                                     │
│  Inbound Email (subject + body)                                     │
│       │                                                             │
│       ▼                                                             │
│  POST /classify  (FastAPI)                                          │
│       │                                                             │
│       ▼                                                             │
│  src/serving/model_loader.py  ← loads model + vectorizer           │
│       │                                                             │
│       ▼                                                             │
│  src/serving/predict.py       ← runs inference, returns confidence  │
│       │                                                             │
│       ▼                                                             │
│  Confidence Routing Logic                                           │
│       ├── ≥ 0.85  →  auto-route to label queue                     │
│       ├── ≥ 0.60  →  route with review flag                        │
│       └── < 0.60  →  human review queue                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
email-ticket-classifier/
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI: lint → test → quality gate on PRs
│       └── cd.yml               # CD: build Docker → push → deploy on merge
│
├── configs/
│   ├── config.yaml              # Runtime paths, MLflow URI, app host/port
│   └── logging.yaml             # Python logging format configuration
│
├── data/
│   ├── raw/
│   │   └── data.csv             # Source dataset (DVC tracked)
│   └── processed/
│       ├── train.csv            # 80% stratified split
│       ├── test.csv             # 20% stratified split
│       ├── vectorizer.pkl       # Fitted TF-IDF vectorizer
│       ├── train_features.pkl   # Transformed training matrix
│       └── test_features.pkl    # Transformed test matrix
│
├── docker/
│   ├── Dockerfile               # Production image (python:3.10-slim)
│   └── docker-compose.yml       # API + MLflow server stack
│
├── models/
│   ├── baseline.pkl             # Trained Logistic Regression model
│   └── bert_model/              # Fine-tuned DistilBERT (config, weights, tokenizer)
│
├── artifacts/                   # Reserved for experiment artifacts
│
├── src/
│   ├── data/
│   │   ├── ingest.py            # Data ingestion / copy to raw/
│   │   └── preprocess.py        # Clean, combine columns, train/test split
│   │
│   ├── features/
│   │   └── build_features.py    # TF-IDF vectorizer + feature matrix
│   │
│   ├── training/
│   │   ├── baseline/
│   │   │   └── train.py         # Logistic Regression + MLflow logging
│   │   ├── bert/
│   │   │   └── train.py         # DistilBERT fine-tuning + MLflow logging
│   │   └── hpo.py               # Optuna HPO study for DistilBERT
│   │
│   ├── evaluation/              # Reserved for metrics.py, report.py
│   │
│   ├── serving/
│   │   ├── app.py               # FastAPI routes (/classify, /health)
│   │   ├── model_loader.py      # ModelLoader class (loads pkl + vectorizer)
│   │   ├── predict.py           # Predictor class (pure inference logic)
│   │   └── schemas.py           # Pydantic request/response models
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py    # Evidently AI drift detection stub
│   │   └── metrics_exporter.py  # Prometheus metrics stub
│   │
│   └── pipeline/
│       └── run_pipeline.py      # Main orchestrator (--model baseline|bert)
│
├── .dockerignore                # Excludes .venv, mlruns, etc. from Docker builds
├── .gitignore                   # Excludes data, models, .venv, __pycache__
├── dvc.yaml                     # DVC pipeline DAG definition
├── Makefile                     # Developer shortcuts (make train/serve/lint)
├── params.yaml                  # Single source of truth for all hyperparameters
├── pyproject.toml               # Dependencies, Python 3.10, ruff, mypy config
├── requirements.txt             # Flat requirements list for pip
└── README.md                    # Project quickstart guide
```

---

## 4. Dataset

**Source:** [Kaggle — Support Ticket Classification by devtry3d](https://www.kaggle.com/datasets/devtry3d/support-ticket-classification)

### Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique ticket identifier |
| `label` | string | **Target variable** — the ticket category |
| `subject` | string | One-line summary of the ticket |
| `text` | string | Full body of the support ticket |

### Preprocessing Applied

1. `subject` and `text` are **concatenated** into a single `text` field to give the model maximum context per ticket.
2. Combined text is **lowercased** and **stripped** of leading/trailing whitespace.
3. Rows with missing `label` or `text` are **dropped**.
4. An 80/20 **stratified train/test split** is applied using `random_state: 42` to ensure reproducibility.

### Split Statistics

| Split | Rows | Proportion |
|---|---|---|
| Train | 1,600 | 80% |
| Test | 400 | 20% |

---

## 5. Configuration System

The project uses a **two-file configuration system** that completely eliminates magic numbers from code.

### `params.yaml` — Hyperparameters

```yaml
base:
  random_state: 42          # Seed for all reproducible operations
  log_level: INFO

data:
  test_size: 0.2            # 20% of data reserved for evaluation
  val_size: 0.1             # Reserved for future validation split

features:
  max_features: 5000        # TF-IDF vocabulary size cap
  ngram_range: [1, 2]       # Unigrams and bigrams

training:
  baseline:
    C: 1.0                  # Logistic Regression regularization strength
    max_iter: 1000           # Maximum solver iterations
  bert:
    model_name: "distilbert-base-uncased"
    max_length: 128          # Token sequence length cap
    learning_rate: 2.0e-5   # Fine-tuning learning rate (float, NOT string)
    batch_size: 16
    epochs: 3

serving:
  confidence_threshold_high: 0.85    # Auto-route threshold
  confidence_threshold_medium: 0.60  # Route with review flag threshold
```

> [!IMPORTANT]
> The `learning_rate` must be written as `2.0e-5` (without quotes) in YAML. Quoting it as `"2e-5"` causes YAML to parse it as a string, which crashes the PyTorch optimizer with a `TypeError`.

### `configs/config.yaml` — Runtime Settings

```yaml
paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  models: "models/"
  artifacts: "artifacts/"

mlflow:
  experiment_name: "email-classifier-baseline"
  tracking_uri: "sqlite:///mlflow.db"   # Local SQLite DB

app:
  host: "0.0.0.0"
  port: 8000
```

---

## 6. Data Pipeline

### `src/data/ingest.py`

Handles data ingestion. For the current project, the Kaggle CSV is placed manually in `data/raw/data.csv` and this script verifies its presence.

```python
def ingest_data(input_path: str, output_path: str):
    if input_path != output_path and os.path.exists(input_path):
        shutil.copy2(input_path, output_path)
    elif os.path.exists(output_path):
        print(f"Data already present at {output_path}")
```

In a production system, this would call the Kaggle API or pull from cloud storage (S3/GCS).

---

### `src/data/preprocess.py`

Cleans the raw data and produces reproducible train/test splits.

```python
def preprocess(input_path, train_out, test_out):
    df = pd.read_csv(input_path)

    # Combine subject + body for maximum NLP context
    if "subject" in df.columns:
        df["text"] = df["subject"].fillna("") + " " + df["text"].fillna("")

    df["text"] = df["text"].str.lower().str.strip()
    df = df.dropna(subset=["label", "text"])

    # Stratified split — preserves class proportions in both splits
    train_df, test_df = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["base"]["random_state"],
        stratify=df["label"]
    )
```

**Key design decisions:**
- `stratify=df["label"]` ensures every class appears proportionally in both train and test, preventing skewed evaluation.
- `random_state: 42` from `params.yaml` makes every split fully reproducible.

---

## 7. Feature Engineering

### `src/features/build_features.py`

Converts raw text into a numerical feature matrix using **TF-IDF** (Term Frequency–Inverse Document Frequency).

```python
vectorizer = TfidfVectorizer(
    max_features=5000,    # Vocabulary capped at top 5000 terms by frequency
    ngram_range=(1, 2)    # Include single words AND two-word phrases
)

X_train = vectorizer.fit_transform(train_df["text"])   # Fit on train ONLY
X_test  = vectorizer.transform(test_df["text"])        # Transform test (no leakage)
```

**What TF-IDF does:**
- **TF (Term Frequency):** How often a word appears in a document.
- **IDF (Inverse Document Frequency):** Down-weights words that appear in many documents (e.g., "the", "is") and up-weights rare, informative words.
- **Bigrams `(1,2)`:** Captures phrases like "account locked" or "billing charge" that carry more meaning than individual words.

**Saved outputs:**
- `vectorizer.pkl` — the fitted vectorizer (used at serving time)
- `train_features.pkl` — sparse matrix + labels
- `test_features.pkl` — sparse matrix + labels

> [!NOTE]
> The vectorizer is **only fitted on training data**. Applying it to test data without re-fitting prevents data leakage — a critical correctness requirement.

---

## 8. Model Training

### Stage 1 — Baseline (TF-IDF + Logistic Regression)

**File:** `src/training/baseline/train.py`

The baseline model is a **Logistic Regression** classifier trained on TF-IDF features.

```python
clf = LogisticRegression(
    C=1.0,          # Regularization (from params.yaml)
    max_iter=1000,
    random_state=42
)
clf.fit(X_train, y_train)

f1  = f1_score(y_test, preds, average="weighted")
acc = accuracy_score(y_test, preds)
```

**Why start here:**
- Trains in seconds vs. minutes for a transformer.
- Provides a strong performance baseline to compare against.
- Validates the entire pipeline end-to-end before spending compute on BERT.
- Interpretable — you can inspect which TF-IDF features drive each class.

**MLflow logging:**
```python
mlflow.log_params({"C": C, "max_iter": max_iter})
mlflow.log_metrics({"f1_score": f1, "accuracy": acc})
mlflow.sklearn.log_model(clf, "model")
```

---

### Stage 2 — Advanced (DistilBERT Fine-Tuning)

**File:** `src/training/bert/train.py`

**DistilBERT** is a smaller, faster version of BERT that retains ~97% of BERT's language understanding while being 40% smaller and 60% faster. We fine-tune the pretrained `distilbert-base-uncased` model from HuggingFace on our support ticket labels.

#### Step-by-Step Process

**Step 1 — Label Encoding**
```python
unique_labels = sorted(train_df["label"].unique())
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label  = {i: l for l, i in label2id.items()}
```
Maps string labels to integers (required by PyTorch) and preserves the reverse mapping for human-readable predictions.

**Step 2 — Tokenization**
```python
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
```
Converts raw text into token IDs and attention masks. `max_length=128` caps sequences to 128 tokens to balance context vs. memory usage.

**Step 3 — Model Initialization**
```python
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
```
The pre-trained weights are loaded and a new **classification head** (two linear layers) is randomly initialized. The `MISSING` keys in the load report (`classifier.weight`, `pre_classifier.weight`) are expected — these are the new layers that fine-tuning will train.

**Step 4 — Training Arguments**
```python
training_args = TrainingArguments(
    output_dir="./models/bert_checkpoints",
    learning_rate=2.0e-5,    # Low LR → gradual adaptation, avoids catastrophic forgetting
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,       # L2 regularization
    eval_strategy="epoch",   # Evaluate after every epoch
    save_strategy="epoch",   # Save checkpoint after every epoch
    load_best_model_at_end=True,  # Keep the best checkpoint
)
```

> [!IMPORTANT]
> The `eval_strategy` parameter replaced `evaluation_strategy` in Transformers 5.x. Always use `eval_strategy` with newer versions.

**Step 5 — Training**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Step 6 — Saving**
```python
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")
mlflow.transformers.log_model(...)
```
Both model weights and tokenizer are saved together so the serving layer can load them as a complete unit.

---

### Hyperparameter Optimization (Optuna)

**File:** `src/training/hpo.py`

Runs an automated Optuna study to find the best DistilBERT hyperparameters.

```python
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16])
    epochs        = trial.suggest_int("epochs", 2, 4)
    # ... train and evaluate ...
    return f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
```

Each trial is logged as a **nested MLflow run** under the parent HPO run, making it easy to compare all trials in the MLflow UI.

**Run it:**
```bash
python -m src.training.hpo
```

---

## 9. Experiment Tracking — MLflow

Every training run is tracked in a local SQLite database (`mlflow.db`).

### What is Logged

| Category | Examples |
|---|---|
| **Parameters** | `C`, `max_iter`, `learning_rate`, `batch_size`, `epochs` |
| **Metrics** | `f1_score`, `accuracy`, `eval_loss` |
| **Artifacts** | Serialized model files, tokenizer config |
| **Tags** | Experiment name, run name |

### Viewing the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Navigate to `http://localhost:5000` to compare baseline vs. BERT runs side-by-side.

### Experiments

| Experiment | Model |
|---|---|
| `email-classifier-baseline` | TF-IDF + Logistic Regression |
| `email-classifier-baseline-bert` | DistilBERT fine-tuning |
| `email-classifier-baseline-bert-hpo` | Optuna HPO runs |

---

## 10. DVC Pipeline

**File:** `dvc.yaml`

DVC defines the pipeline as a **directed acyclic graph (DAG)** of stages. Each stage declares its inputs (`deps`), parameters (`params`), and outputs (`outs`). DVC tracks file hashes and only re-runs a stage if its inputs or parameters have changed.

```yaml
stages:
  ingest:
    cmd: python -m src.data.ingest
    outs: [data/raw/data.csv]

  preprocess:
    cmd: python -m src.data.preprocess
    deps: [data/raw/data.csv, src/data/preprocess.py]
    params: [base.random_state, data.test_size]
    outs: [data/processed/train.csv, data/processed/test.csv]

  build_features:
    cmd: python -m src.features.build_features
    deps: [data/processed/train.csv, data/processed/test.csv]
    params: [features.max_features, features.ngram_range]
    outs: [data/processed/vectorizer.pkl, ...]

  train_baseline:
    cmd: python -m src.training.baseline.train
    deps: [data/processed/train_features.pkl, ...]
    params: [training.baseline.C, training.baseline.max_iter]
    outs: [models/baseline.pkl]
```

**Key commands:**
```bash
dvc repro          # Re-run only changed stages
dvc dag            # Visualize the pipeline graph
dvc status         # Check which stages are stale
```

**Responsibility split:**

| Tool | Tracks |
|---|---|
| Git | Source code, config files |
| DVC | Raw data, processed data, feature matrices |
| MLflow | Model weights, experiment metrics |

---

## 11. Model Serving — FastAPI

### Architecture

The serving layer is split into three focused classes to keep each component single-responsibility:

```
EmailRequest (Pydantic)
      │
      ▼
  app.py (FastAPI routes)
      │
      ▼
  predict.py (Predictor)
      │
      ▼
  model_loader.py (ModelLoader)
      │
      ▼
  models/baseline.pkl + vectorizer.pkl
```

### `src/serving/model_loader.py`

```python
class ModelLoader:
    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
```

Loads the serialized sklearn model and TF-IDF vectorizer from disk. Called once at API startup.

---

### `src/serving/predict.py`

```python
class Predictor:
    def predict(self, text: str) -> Tuple[str, float]:
        X = self.loader.vectorizer.transform([text])
        probabilities = self.loader.model.predict_proba(X)[0]
        label = self.loader.model.classes_[probabilities.argmax()]
        confidence = probabilities.max()
        return label, float(confidence)
```

Pure inference logic completely decoupled from the HTTP layer. Can be imported and used outside FastAPI (e.g., in batch jobs or tests).

---

### `src/serving/schemas.py`

```python
class EmailRequest(BaseModel):
    subject: str
    body: str

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    routed_to: str
```

Pydantic models enforce strict input/output validation and auto-generate the OpenAPI docs at `/docs`.

---

### `src/serving/app.py` — API Endpoints

#### `POST /classify`

```json
// Request
{
  "subject": "I was charged twice",
  "body": "My credit card shows two charges for this month."
}

// Response
{
  "label": "billing",
  "confidence": 0.93,
  "routed_to": "auto-route to billing queue"
}
```

#### Confidence-Gated Routing Logic

```python
if confidence >= 0.85:
    return f"auto-route to {label} queue"
elif confidence >= 0.60:
    return f"route to {label} queue (flagged for review)"
else:
    return "human review queue"
```

| Confidence | Action |
|---|---|
| ≥ 0.85 | Auto-routed to the correct team |
| 0.60–0.85 | Routed but flagged for spot-check |
| < 0.60 | Sent to human review queue |

#### `GET /health`

Returns `"healthy"` if the model is loaded or `"degraded - model not loaded"` if startup failed. Used by Docker and cloud health checks.

---

### Running the API

```bash
uvicorn src.serving.app:app --reload
```

Interactive docs: `http://localhost:8000/docs`

---

## 12. Monitoring

### `src/monitoring/drift_detector.py`

Stub for integrating **Evidently AI** to detect data and concept drift in production.

```python
class DriftDetector:
    def check_data_drift(self, current_data: pd.DataFrame) -> bool:
        # Compares incoming distribution to training reference
        # Triggers retraining if drift score exceeds threshold
        ...
```

When activated, drift detection runs on a schedule and if detected, publishes a message to the Celery task queue to trigger `dvc repro` for retraining.

---

### `src/monitoring/metrics_exporter.py`

Stub for **Prometheus** metrics collection.

```python
class MetricsExporter:
    def record_prediction(self, label: str):
        # self.prediction_counter.labels(label=label).inc()
        ...

    def record_latency(self, latency_seconds: float):
        # self.latency_histogram.observe(latency_seconds)
        ...
```

Once wired, these metrics power **Grafana dashboards** showing:
- Predictions per second by label
- API latency histogram (p50, p95, p99)
- Prediction class distribution drift over time

---

## 13. Containerization — Docker

### `docker/Dockerfile`

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Design decisions:**
- `python:3.10-slim` — minimal base image, reduces attack surface and image size.
- `pip install -e .` — installs only the production dependencies from `pyproject.toml`.
- `.dockerignore` excludes `.venv`, `mlruns/`, `mlflow.db`, and `__pycache__` to keep the build context small and fast.

---

### `docker/docker-compose.yml`

Spins up two services:

| Service | Port | Description |
|---|---|---|
| `classifier-api` | 8000 | FastAPI inference server |
| `mlflow` | 5000 | MLflow tracking server with SQLite backend |

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 14. CI/CD — GitHub Actions

### CI Workflow (`.github/workflows/ci.yml`)

Triggers on every **Pull Request to `main`**.

| Step | Tool | What it does |
|---|---|---|
| 1 | `pip install -e .[dev]` | Install all dependencies |
| 2 | `ruff check .` | Lint for code style violations |
| 3 | `mypy .` | Static type checking |
| 4 | `pytest tests/` | Unit and integration tests |
| 5 | Quality Gate | F1 ≥ 0.85 check on held-out test set |

If any step fails, the PR is blocked from merging.

---

### CD Workflow (`.github/workflows/cd.yml`)

Triggers on every **merge to `main`**.

| Step | What it does |
|---|---|
| 1 | Build Docker image from `docker/Dockerfile` |
| 2 | Push image to container registry (ECR / Artifact Registry) |
| 3 | Deploy to cloud (Cloud Run / ECS) |
| 4 | Smoke test `/health` and `/classify` endpoints |
| 5 | Auto-rollback if smoke test fails |

---

## 15. MLOps Feedback Loop

The system is designed as a **closed loop** that continuously improves itself without manual intervention.

```
Production Predictions
        │
        ▼
Human Review Queue       ← agents relabel low-confidence tickets
        │
        ▼
Training Store           ← corrections written back automatically
        │
        ▼
Drift Detection          ← Evidently AI detects distribution shift
        │
        ▼
Retrain Trigger          ← Celery schedules dvc repro
        │
        ▼
Quality Gate             ← F1 ≥ 0.85 required to promote
        │
        ▼
New Champion             ← replaces old model in MLflow registry
        │
        └──────────────► back to Production Predictions
```

**In practice:** Every human correction on a low-confidence ticket is free training data for the next cycle. The model that runs in month three will be meaningfully better than the one deployed at launch, automatically.

---

## 16. How to Run

### Option A — Local Development

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -e .[transformers]

# 3. Train Baseline (TF-IDF + Logistic Regression)
python -m src.pipeline.run_pipeline

# 4. Train DistilBERT
python -m src.pipeline.run_pipeline --model bert

# 5. Run Optuna HPO (optional)
python -m src.training.hpo

# 6. Start the API
uvicorn src.serving.app:app --reload

# 7. View MLflow UI (in a separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Option B — Docker (Full Stack)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Option C — Makefile Shortcuts

```bash
make setup    # Create venv and install deps
make train    # Run the baseline pipeline
make serve    # Start the FastAPI server
make lint     # Run ruff + mypy
make clean    # Wipe data/processed and models/
```

---

## 17. Tech Stack Summary

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| NLP Baseline | scikit-learn | ≥ 1.3 |
| NLP Advanced | HuggingFace Transformers | ≥ 4.31 |
| Transformer Model | DistilBERT | `distilbert-base-uncased` |
| Training Backend | PyTorch | ≥ 2.0 |
| Accelerator | HuggingFace Accelerate | ≥ 1.1 |
| Dataset Loading | HuggingFace Datasets | ≥ 2.14 |
| HPO | Optuna | ≥ 3.2 |
| Experiment Tracking | MLflow | ≥ 2.5 |
| Data Versioning | DVC | ≥ 3.0 |
| API Framework | FastAPI + Uvicorn | ≥ 0.100 |
| Validation | Pydantic | ≥ 2.0 |
| Config | PyYAML | ≥ 6.0 |
| Containerization | Docker + Compose | latest |
| CI/CD | GitHub Actions | — |
| Linting | Ruff | ≥ 0.0.280 |
| Type Checking | Mypy | ≥ 1.5 |
| Testing | Pytest | ≥ 7.4 |
| Drift Detection | Evidently AI | (stub — ready to wire) |
| Monitoring | Prometheus + Grafana | (stub — ready to wire) |
| Task Queue | Celery + Redis | (stub — ready to wire) |

---

*Generated for the Email Ticket Classifier MLOps Project — Python 3.10 · Two-Stage NLP · Full MLOps Pipeline*
