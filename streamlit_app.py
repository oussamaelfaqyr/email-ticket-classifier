import streamlit as st
import time
import os
import sys
import uuid
import json
import base64
import requests
from datetime import datetime

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.serving.model_loader import get_inference_pipeline

st.set_page_config(
    page_title="Ticket Classifier CLP",
    page_icon=":material/support_agent:",
    layout="wide",
)

@st.cache_resource
def load_system():
    return get_inference_pipeline()

classifier = load_system()

def push_file_to_github(filepath: str, content: str):
    """Creates a new file in the GitHub repository using the GitHub REST API."""
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO", "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        # Fallback to local save if no GitHub token is provided (for local testing)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return

    url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    data = {
        "message": f"Add feedback {filepath}",
        "content": encoded_content
    }
    
    # Retry logic
    for _ in range(3):
        try:
            resp = requests.put(url, headers=headers, json=data)
            if resp.status_code in [201, 200]:
                break
        except Exception:
            time.sleep(2)

def trigger_github_action():
    """Triggers the retrain pipeline via repository_dispatch."""
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO", "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "event_type": "retrain",
        "client_payload": {
            "source": "streamlit"
        }
    }
    
    for _ in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code == 204:
                break
        except Exception:
            time.sleep(2)

def save_feedback(text: str, predicted_label: str, human_label: str):
    """Save corrected label as an immutable JSON file via GitHub API."""
    feedback_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    feedback_data = {
        "id": feedback_id,
        "text": text,
        "label": predicted_label,
        "corrected_label": human_label,
        "timestamp": now.isoformat() + "Z"
    }
    
    # Partitioned path: data/feedback/YYYY/MM/DD/fb_uuid.json
    filepath = f"data/feedback/{now.strftime('%Y/%m/%d')}/fb_{feedback_id}.json"
    json_content = json.dumps(feedback_data, indent=2)
    
    # 1. Push immutable record
    push_file_to_github(filepath, json_content)
    
    # 2. Trigger async pipeline
    trigger_github_action()
    
    st.toast("Feedback saved! Model will be retrained in the background.", icon="🚀")


st.title(":material/support_agent: Email Ticket Classifier CLP")

st.header("Test Classification")
with st.form("test_form"):
    subject = st.text_input("Subject")
    body = st.text_area("Email Body")
    submit = st.form_submit_button("Classify", icon=":material/send:")

    if submit and (subject or body):
        text_input = f"{subject} {body}".strip()
        try:
            # Inference
            res = classifier(text_input)
            if isinstance(res, list) and len(res) > 0:
                prediction = res[0]
            else:
                prediction = {"label": "unknown", "score": 0.0}
            
            st.session_state.current_text = text_input
            st.session_state.predicted_label = prediction["label"]
            st.session_state.confidence = prediction["score"]
        except Exception as e:
            st.error(f"Error during inference: {e}")

if "predicted_label" in st.session_state:
    st.success(f"Predicted Label: **{st.session_state.predicted_label}** (Confidence: {st.session_state.confidence:.2f})")
    
    with st.expander("Is this prediction incorrect?", expanded=True):
        st.write("Provide feedback to continuously improve the model.")
        with st.form("feedback_form"):
            # Dummy labels since we don't have model classes explicitly loaded here
            options = ["billing_issue", "technical_support", "account_access", "general_inquiry", "other"]
            default_idx = options.index(st.session_state.predicted_label) if st.session_state.predicted_label in options else 0
            
            corrected_label = st.selectbox("Correct Label", options=options, index=default_idx)
            if st.form_submit_button("Submit Feedback", type="primary"):
                if corrected_label == st.session_state.predicted_label:
                    st.warning("The corrected label is the same as the predicted label.")
                else:
                    save_feedback(
                        text=st.session_state.current_text, 
                        predicted_label=st.session_state.predicted_label, 
                        human_label=corrected_label
                    )
                    del st.session_state.predicted_label
                    time.sleep(1)
                    st.rerun()
