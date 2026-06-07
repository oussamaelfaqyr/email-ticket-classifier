import os
import yaml
import mlflow
import optuna
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
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
    return {"f1": f1}

def run_hpo(train_path: str, test_path: str, n_trials: int = 5):
    params = load_params()
    configs = load_configs()
    
    mlflow.set_tracking_uri(configs["mlflow"]["tracking_uri"])
    mlflow.set_experiment(configs["mlflow"]["experiment_name"] + "-bert-hpo")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
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

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        epochs = trial.suggest_int("epochs", 2, 4)

        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=f"./models/hpo_{trial.number}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        with mlflow.start_run(nested=True):
            mlflow.log_params({"learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs})
            trainer.train()
            eval_results = trainer.evaluate()
            f1 = eval_results["eval_f1"]
            mlflow.log_metric("f1", f1)
            
        return f1

    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    run_hpo("data/processed/train.csv", "data/processed/test.csv")
