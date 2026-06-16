import os
import json

# All heavy ML imports are LAZY (inside functions) so this module
# can always be imported safely without torch/transformers installed.

# Module-level singleton cache — works in both FastAPI and Streamlit contexts
_pipeline_cache = {}


def get_model_pointers(repo_id: str):
    """Fetch model_pointers.json from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        filepath = hf_hub_download(repo_id=repo_id, filename="model_pointers.json")
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to fetch model_pointers.json: {e}")
        return None


def load_hf_model(repo_id: str, revision: str):
    """Load model and tokenizer from HF Hub, cached by repo+revision."""
    cache_key = f"{repo_id}@{revision}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    # Lazy import — only runs when actually loading a model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )

    model = AutoModelForSequenceClassification.from_pretrained(repo_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    pipe = hf_pipeline("text-classification", model=model, tokenizer=tokenizer)
    _pipeline_cache[cache_key] = pipe
    return pipe


def load_local_fallback():
    """Load local DistilBERT model from models/stable/ if present."""
    fallback_path = "models/stable/"
    if not os.path.exists(fallback_path):
        return None

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
    model = AutoModelForSequenceClassification.from_pretrained(fallback_path)
    tokenizer = AutoTokenizer.from_pretrained(fallback_path)
    return hf_pipeline("text-classification", model=model, tokenizer=tokenizer)


def get_inference_pipeline(repo_id: str = None):
    """
    Return a callable inference pipeline.
    Priority: HF Hub active -> HF Hub stable -> local fallback -> dummy.
    """
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

    return lambda text: [{"label": "unknown", "score": 0.0}]
