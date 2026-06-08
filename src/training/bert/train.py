import os
import yaml
import mlflow
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_configs():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}

def train_bert(train_path: str, test_path: str, model_out: str):
    params = load_params()
    configs = load_configs()
    
    mlflow.set_tracking_uri(configs["mlflow"]["tracking_uri"])
    mlflow.set_experiment(configs["mlflow"]["experiment_name"] + "-bert")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Map labels to integers
    unique_labels = sorted(train_df["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    train_df["label_id"] = train_df["label"].map(label2id)
    test_df["label_id"] = test_df["label"].map(label2id)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    model_name = params["training"]["bert"]["model_name"]
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=params["training"]["bert"]["max_length"])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./models/bert_checkpoints",
        learning_rate=params["training"]["bert"]["learning_rate"],
        per_device_train_batch_size=params["training"]["bert"]["batch_size"],
        per_device_eval_batch_size=params["training"]["bert"]["batch_size"],
        num_train_epochs=params["training"]["bert"]["epochs"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():
        mlflow.log_params(params["training"]["bert"])
        trainer.train()
        
        eval_results = trainer.evaluate()
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_results.items()})
        
        os.makedirs(model_out, exist_ok=True)
        model.save_pretrained(model_out)
        tokenizer.save_pretrained(model_out)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            pip_requirements=[
                f"transformers=={__import__('transformers').__version__}",
                f"torch=={__import__('torch').__version__}",
                f"accelerate=={__import__('accelerate').__version__}",
            ]
        )
        print(f"DistilBERT trained and saved to {model_out}")

if __name__ == "__main__":
    train_bert("data/processed/train.csv", "data/processed/test.csv", "models/bert_model")
