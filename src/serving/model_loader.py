import os
import json
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline

# Module-level singleton — safe for both FastAPI and Streamlit contexts
_pipeline_cache = {}

def get_model_pointers(repo_id: str):
    """Fetch the model_pointers.json from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        filepath = hf_hub_download(repo_id=repo_id, filename="model_pointers.json")
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to fetch model_pointers.json: {e}")
        return None

def load_hf_model(repo_id: str, revision: str):
    """Load model and tokenizer from HF Hub (cached in module-level dict)."""
    cache_key = f"{repo_id}@{revision}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    pipe = hf_pipeline("text-classification", model=model, tokenizer=tokenizer)
    _pipeline_cache[cache_key] = pipe
    return pipe

def load_local_fallback():
    """Load local persistent fallback if HF fails."""
    fallback_path = "models/stable/"
    if os.path.exists(fallback_path):
        model = AutoModelForSequenceClassification.from_pretrained(fallback_path)
        tokenizer = AutoTokenizer.from_pretrained(fallback_path)
        return hf_pipeline("text-classification", model=model, tokenizer=tokenizer)
    return None

def get_inference_pipeline(repo_id: str = None):
    repo_id = repo_id or os.environ.get("HF_REPO_ID", "")

    if repo_id:
        pointers = get_model_pointers(repo_id)
        if pointers:
            try:
                return load_hf_model(repo_id, revision=pointers.get("active"))
            except Exception as e:
                print(f"Failed to load active revision: {e}")
            try:
                if pointers.get("stable"):
                    return load_hf_model(repo_id, revision=pointers.get("stable"))
            except Exception as e:
                print(f"Failed to load stable revision: {e}")

    print("Falling back to local model.")
    fallback = load_local_fallback()
    if fallback:
        return fallback

    # Return a dummy pipeline so server doesn't crash if no model exists yet
    return lambda text: [{"label": "unknown", "score": 0.0}]
