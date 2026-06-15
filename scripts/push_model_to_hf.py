"""
One-time script: Push the local trained bert_model to Hugging Face Hub
and create model_pointers.json so Streamlit can find it.

Run with:
    python scripts/push_model_to_hf.py
"""
import json
import os
import sys

# Try to load from .env file if it exists
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip().strip('"\'')

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "ouel/bert-ticket-classifier")
MODEL_PATH = "models/bert_model"

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN env var before running this script.")
    print("  Example: $env:HF_TOKEN='hf_xxxx'; python scripts/push_model_to_hf.py")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print("ERROR: Install required packages first:")
    print("  pip install transformers huggingface_hub")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH} ...")
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Pushing to HF Hub: {HF_REPO_ID} (branch: main) ...")
model.push_to_hub(HF_REPO_ID, token=HF_TOKEN,
                  commit_message="Initial trained DistilBERT model (v_baseline)")
tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN,
                      commit_message="Initial tokenizer")

print("Creating model_pointers.json ...")
api      = HfApi(token=HF_TOKEN)
pointers = {"active": "main", "stable": "main"}

with open("model_pointers.json", "w") as f:
    json.dump(pointers, f, indent=2)

api.upload_file(
    path_or_fileobj="model_pointers.json",
    path_in_repo="model_pointers.json",
    repo_id=HF_REPO_ID,
    commit_message="Add model_pointers.json (active=main)",
    token=HF_TOKEN,
)

print()
print("=" * 60)
print("SUCCESS! Model is now live on Hugging Face Hub.")
print(f"  https://huggingface.co/{HF_REPO_ID}")
print("  active  -> main")
print("  stable  -> main")
print()
print("Streamlit will pick it up automatically on next load.")
print("=" * 60)
