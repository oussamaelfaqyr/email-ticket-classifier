import os
import json
import yaml
import pickle
import pandas as pd
import torch
import mlflow
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_configs():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_baseline(test_path, vectorizer_path, model_path, out_metrics):
    print("Evaluating Baseline Model...")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return

    test_df = pd.read_csv(test_path)
    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    preds = clf.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
    }

    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Baseline Metrics saved to {out_metrics}")

    # Log to MLflow
    configs = load_configs()
    mlflow.set_tracking_uri(configs.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(configs.get("mlflow", {}).get("experiment_name", "email-ticket-classifier") + "-eval")
    with mlflow.start_run(run_name="evaluate_baseline"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "baseline")

def evaluate_bert(test_path, model_path, out_metrics):
    print("Evaluating BERT Model...")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return

    test_df = pd.read_csv(test_path)
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    preds_output = pipe(test_df["text"].tolist(), truncation=True, max_length=512)
    preds = [p["label"] for p in preds_output]
    y_test = test_df["label"].tolist()

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
    }

    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"BERT Metrics saved to {out_metrics}")

    # Log to MLflow
    configs = load_configs()
    mlflow.set_tracking_uri(configs.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(configs.get("mlflow", {}).get("experiment_name", "email-ticket-classifier") + "-eval")
    with mlflow.start_run(run_name="evaluate_bert"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "bert")

if __name__ == "__main__":
    evaluate_baseline(
        test_path="data/processed/test.csv",
        vectorizer_path="data/processed/vectorizer.pkl",
        model_path="models/baseline.pkl",
        out_metrics="reports/metrics_baseline.json"
    )
    evaluate_bert(
        test_path="data/processed/test.csv",
        model_path="models/bert_model",
        out_metrics="reports/metrics_bert.json"
    )
