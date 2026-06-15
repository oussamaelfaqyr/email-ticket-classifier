import os
import sys
import glob
import json
import uuid
import shutil
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, ValidationError
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from huggingface_hub import HfApi, hf_hub_download

class FeedbackEvent(BaseModel):
    id: str
    text: str
    label: str
    corrected_label: str
    timestamp: str

def get_processed_files():
    """Read all manifests to determine which files are already processed."""
    processed = set()
    manifest_dir = "data/manifests"
    if os.path.exists(manifest_dir):
        for manifest_path in glob.glob(f"{manifest_dir}/*.json"):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    processed.update(manifest.get("files", []))
            except Exception as e:
                print(f"Warning: failed to read manifest {manifest_path}: {e}")
    return processed

def run_pipeline():
    min_batch = int(os.environ.get("MIN_BATCH_SIZE", "50"))
    print(f"=== Continuous Learning Pipeline Started (MIN_BATCH_SIZE={min_batch}) ===")
    
    # 1. Find all feedback files
    all_files = set(glob.glob("data/feedback/**/*.json", recursive=True))
    
    # Exclude quarantine, gitkeep and state files
    all_files = {
        f for f in all_files
        if "quarantine" not in f
        and not f.endswith("_state.json")
        and not f.endswith(".gitkeep")
    }
    
    processed_files = get_processed_files()
    unprocessed_files = list(all_files - processed_files)
    
    print(f"[CLP] Feedback files found  : {len(all_files)}")
    print(f"[CLP] Already processed     : {len(processed_files)}")
    print(f"[CLP] New unprocessed events: {len(unprocessed_files)}")
    print(f"[CLP] Batch gate            : {min_batch}")
    
    # 2. Batch Check
    if len(unprocessed_files) < min_batch:
        print(f"[CLP] SKIP — only {len(unprocessed_files)} new sample(s), need {min_batch}. Exiting cleanly.")
        print(f"[CLP] TIP: Trigger workflow manually with min_batch_size=1 to force a test run.")
        sys.exit(0)
        
    print("Batch size met. Processing events...")
    
    # 3. Validate with Pydantic and Quarantine
    valid_data = []
    valid_files = []
    
    quarantine_dir = "data/quarantine"
    os.makedirs(quarantine_dir, exist_ok=True)
    
    for file_path in unprocessed_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Schema validation
            event = FeedbackEvent(**data)
            valid_data.append({
                "text": event.text,
                "label": event.corrected_label # using human corrected label
            })
            valid_files.append(file_path)
        except ValidationError as e:
            print(f"Validation error for {file_path}. Moving to quarantine.")
            shutil.move(file_path, os.path.join(quarantine_dir, os.path.basename(file_path)))
        except Exception as e:
            print(f"Error reading {file_path}. Moving to quarantine.")
            shutil.move(file_path, os.path.join(quarantine_dir, os.path.basename(file_path)))
            
    if len(valid_files) < min_batch:
        print(f"After validation, less than {min_batch} samples remain. Skipping training.")
        sys.exit(0)
        
    # 4. Build Training Dataset
    print(f"Building dataset with {len(valid_data)} new samples...")
    new_df = pd.DataFrame(valid_data)
    
    base_data_path = "data/raw/data.csv"
    if os.path.exists(base_data_path):
        base_df = pd.read_csv(base_data_path)
        full_df = pd.concat([base_df, new_df], ignore_index=True)
    else:
        full_df = new_df
        
    # 5. Train BERT
    print("Training BERT model...")
    # Map labels to integers
    unique_labels = sorted(full_df["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    full_df["label_id"] = full_df["label"].map(label2id)
    
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # We will start from standard distilbert to train
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1_macro}
        
    training_args = TrainingArguments(
        output_dir="./models/clp_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
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
    
    trainer.train()
    
    # 6. Evaluation Gate
    eval_results = trainer.evaluate()
    f1_macro = eval_results.get("eval_f1_macro", 0.0)
    print(f"Evaluation F1-Macro: {f1_macro:.4f}")
    
    if f1_macro < 0.85:
        print("F1-Macro is below 0.85. Failing pipeline to prevent bad model deployment.")
        sys.exit(1)
        
    print("Model passed evaluation gate!")
    
    # 7. Push to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID", "dummy/bert-ticket-classifier")
    
    if not hf_token:
        print("Warning: HF_TOKEN not set. Skipping Hugging Face upload.")
        # Local test fallback
        sys.exit(0)
        
    print("Pushing model to Hugging Face Hub...")
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version_tag = f"v{run_id}"
    
    # We push using a specific revision/branch
    model.push_to_hub(hf_repo_id, token=hf_token, commit_message=f"CLP run {version_tag}", branch=version_tag)
    tokenizer.push_to_hub(hf_repo_id, token=hf_token, commit_message=f"CLP run {version_tag}", branch=version_tag)
    
    # 8. Update model_pointers.json
    api = HfApi(token=hf_token)
    try:
        pointers_path = hf_hub_download(repo_id=hf_repo_id, filename="model_pointers.json", token=hf_token)
        with open(pointers_path, "r") as f:
            pointers = json.load(f)
    except:
        pointers = {"active": "main", "stable": "main"}
        
    pointers["stable"] = pointers.get("active", "main")
    pointers["active"] = version_tag
    
    with open("model_pointers.json", "w") as f:
        json.dump(pointers, f, indent=2)
        
    api.upload_file(
        path_or_fileobj="model_pointers.json",
        path_in_repo="model_pointers.json",
        repo_id=hf_repo_id,
        commit_message=f"Update pointers to {version_tag}"
    )
    
    # 9. Write Segmented Manifest
    print("Writing event log manifest...")
    os.makedirs("data/manifests", exist_ok=True)
    manifest_data = {
        "event_type": "processed_batch",
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "model_version": version_tag,
        "f1_macro": f1_macro,
        "files": valid_files
    }
    
    manifest_path = f"data/manifests/batch_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"=== CLP Completed Successfully (Version {version_tag}) ===")

if __name__ == "__main__":
    run_pipeline()
