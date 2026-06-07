# Email Ticket Classifier

A two-stage NLP classification system built for categorizing inbound support emails. The project features a fast TF-IDF baseline and a high-accuracy DistilBERT transformer, all managed within a production-ready MLOps pipeline.

## Features
- **Dual Pipeline:** TF-IDF + Logistic Regression (baseline) and DistilBERT (advanced).
- **Optuna HPO:** Hyperparameter optimization for DistilBERT.
- **MLflow Tracking:** Full experiment tracking and model registry capabilities.
- **DVC Integration:** Data versioning for the raw and processed datasets.
- **FastAPI Serving:** Decoupled inference and API layer ready for production.
- **Dockerized:** Packaged with `docker-compose` to run the API and MLflow server seamlessly.

## Quickstart

### Option 1: Run via Docker (Recommended)
```bash
docker-compose -f docker/docker-compose.yml up --build
```
- **API:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000

### Option 2: Local Development Setup
1. **Initialize the environment:**
```bash
make setup
```
2. **Train the Baseline Model:**
```bash
make train
# OR: python -m src.pipeline.run_pipeline
```
3. **Train the DistilBERT Model (Optional):**
```bash
python -m src.pipeline.run_pipeline --model bert
```
4. **Serve the API:**
```bash
make serve
```

## Project Structure
- `configs/`: YAML configs for logging and runtime parameters.
- `data/`: Raw and processed datasets (versioned by DVC).
- `src/`: Core Python modules (data, features, training, evaluation, serving, pipeline).
- `docker/`: Dockerfile and docker-compose configurations.
- `.github/`: CI/CD workflows for GitHub Actions.
