import streamlit as st
import time
import os
import sys
import uuid
import json
import base64
import requests
import yaml
import datetime

# Ensure repo root is importable when run from repo root by Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.database import SessionLocal, engine, Base
from src.db.models import Ticket

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Ticket Classifier",
    page_icon=":material/support_agent:",
    layout="wide",
)

# ── Load params ────────────────────────────────────────────────────────────────
@st.cache_data
def load_params():
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"serving": {"confidence_threshold_high": 0.85, "confidence_threshold_medium": 0.60}}

params = load_params()
HIGH_THRESHOLD   = params["serving"]["confidence_threshold_high"]
MEDIUM_THRESHOLD = params["serving"]["confidence_threshold_medium"]

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_db():
    Base.metadata.create_all(bind=engine)
    return SessionLocal()

# ── Inference ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_classifier():
    """
    Try the new HF pipeline first (CLP model).
    Fall back to the original sklearn baseline if HF is not configured.
    """
    hf_repo_id = (
        st.secrets.get("HF_REPO_ID")
        or os.environ.get("HF_REPO_ID")
        or ""
    )

    if hf_repo_id:
        try:
            from src.serving.model_loader import get_inference_pipeline
            pipe = get_inference_pipeline(hf_repo_id)
            # Return a unified (label, confidence) callable
            def hf_predict(text):
                res = pipe(text)
                if isinstance(res, list) and res:
                    return res[0]["label"], float(res[0]["score"])
                return "unknown", 0.0
            return hf_predict, None   # (predict_fn, label_list)
        except Exception as e:
            st.warning(f"HF model unavailable ({e}). Using baseline sklearn model.")

    # Fallback: original sklearn baseline
    try:
        from src.serving.model_loader import ModelLoader
        from src.serving.predict import Predictor
        loader = ModelLoader("models/baseline.pkl", "data/processed/vectorizer.pkl")
        loader.load()
        predictor = Predictor(loader)
        label_list = loader.model.classes_.tolist() if loader.is_loaded() else []

        def sk_predict(text):
            return predictor.predict(text)

        return sk_predict, label_list
    except Exception as e:
        st.error(f"Could not load any model: {e}")
        return lambda t: ("unknown", 0.0), []

predict_fn, _label_list = load_classifier()

# Derive label list (for dropdowns). Works for both HF and sklearn.
LABEL_OPTIONS = _label_list or [
    # Real classes from models/baseline.pkl — keep in sync with training data
    "account_access", "billing", "bug_report", "refund_request", "shipping_delivery"
]

def determine_routing(label: str, confidence: float) -> str:
    if confidence >= HIGH_THRESHOLD:
        return f"Auto-routed to **{label}** queue"
    elif confidence >= MEDIUM_THRESHOLD:
        return f"Routed to **{label}** queue — flagged for review"
    else:
        return "Sent to **Human Review Queue**"

# ── CLP Event Emitter (GitHub API) ────────────────────────────────────────────
def _push_file_to_github(filepath: str, content: str):
    """Create an immutable feedback file in the GitHub repo via REST API."""
    token = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo  = st.secrets.get("GITHUB_REPO")  or os.environ.get("GITHUB_REPO",  "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        # Local-testing fallback: write directly
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return True

    url     = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    encoded = base64.b64encode(content.encode()).decode()
    data    = {"message": f"feat: Add feedback event {filepath}", "content": encoded}

    for _ in range(3):
        try:
            r = requests.put(url, headers=headers, json=data, timeout=10)
            if r.status_code in (200, 201):
                return True
        except Exception:
            time.sleep(2)
    return False

def _trigger_github_dispatch():
    """Fire repository_dispatch to kick off the retraining workflow."""
    token = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo  = st.secrets.get("GITHUB_REPO")  or os.environ.get("GITHUB_REPO",  "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        return

    url     = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    payload = {"event_type": "retrain", "client_payload": {"source": "streamlit"}}

    for _ in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            if r.status_code == 204:
                return
        except Exception:
            time.sleep(2)

def save_feedback(ticket: Ticket, human_label: str, response_email: str = ""):
    """
    1. Persist corrected label as an immutable CLP event (GitHub API).
    2. Update the local SQLite ticket to 'resolved'.
    3. Trigger GitHub Actions retraining dispatch.
    """
    feedback_id  = str(uuid.uuid4())
    now          = datetime.datetime.utcnow()
    feedback_obj = {
        "id":              feedback_id,
        "ticket_db_id":    ticket.id,
        "text":            f"{ticket.subject} {ticket.body}".strip(),
        "label":           ticket.predicted_label,
        "corrected_label": human_label,
        "confidence":      ticket.confidence,
        "timestamp":       now.isoformat() + "Z",
    }

    # Partitioned path: data/feedback/YYYY/MM/DD/fb_<uuid>.json
    filepath = f"data/feedback/{now.strftime('%Y/%m/%d')}/fb_{feedback_id}.json"
    _push_file_to_github(filepath, json.dumps(feedback_obj, indent=2))
    _trigger_github_dispatch()

    # Update SQLite record
    db = get_db()
    try:
        t = db.query(Ticket).filter(Ticket.id == ticket.id).first()
        if t:
            t.human_label    = human_label
            t.response_email = response_email
            t.status         = "resolved"
            db.commit()
    finally:
        db.close()


# ── Training Status helpers ────────────────────────────────────────────────────
@st.cache_data(ttl=30)   # refresh every 30 s automatically
def get_latest_workflow_run():
    """Query GitHub Actions API for the latest CLP workflow run."""
    token = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo  = st.secrets.get("GITHUB_REPO")  or os.environ.get("GITHUB_REPO",  "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        return None
    try:
        url = f"https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/runs?per_page=1"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            return runs[0] if runs else None
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)  # refresh every 5 min
def get_current_model_version():
    """Fetch model_pointers.json from HF Hub to get the active model version."""
    hf_repo_id = st.secrets.get("HF_REPO_ID") or os.environ.get("HF_REPO_ID", "")
    if not hf_repo_id:
        return None, None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=hf_repo_id, filename="model_pointers.json")
        with open(path) as f:
            p = json.load(f)
        return p.get("active", "main"), p.get("stable", "main")
    except Exception:
        return None, None

def count_feedback_files():
    """Count total feedback events committed to the repo."""
    return len([f for f in glob.glob("data/feedback/**/*.json", recursive=True)
                if not f.endswith(".gitkeep")])

try:
    import glob as _glob_mod
    import glob
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CLP Status Panel
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🤖 CLP Status")

    # — Model version —
    active_ver, stable_ver = get_current_model_version()
    if active_ver:
        st.markdown(f"**Active model:** `{active_ver}`")
        st.markdown(f"**Stable model:** `{stable_ver}`")
    else:
        st.markdown("**Model source:** Baseline (sklearn)")
        st.caption("Set HF_REPO_ID secret to enable HF tracking.")

    st.divider()

    # — GitHub Actions latest run —
    run = get_latest_workflow_run()
    if run:
        status     = run.get("status", "unknown")      # queued / in_progress / completed
        conclusion = run.get("conclusion") or ""        # success / failure / cancelled / None
        run_number = run.get("run_number", "?")
        updated_at = run.get("updated_at", "")[:16].replace("T", " ")
        html_url   = run.get("html_url", "#")

        if status == "in_progress" or status == "queued":
            st.markdown("### 🟡 Training in progress…")
            st.caption(f"Run #{run_number} · started {updated_at} UTC")
            st.info("A new model is being trained. The page will refresh automatically.", icon="⏳")
        elif conclusion == "success":
            st.markdown("### ✅ Last run: Success")
            st.caption(f"Run #{run_number} · finished {updated_at} UTC")
            st.success("Model trained and deployed to Hugging Face Hub!", icon="🚀")
        elif conclusion == "failure":
            st.markdown("### ❌ Last run: Failed")
            st.caption(f"Run #{run_number} · {updated_at} UTC")
            st.error("Pipeline failed — check GitHub Actions for details.", icon="🚨")
        elif conclusion == "skipped" or (conclusion == "" and status == "completed"):
            st.markdown("### ⏭️ Last run: Skipped")
            st.caption(f"Run #{run_number} · {updated_at} UTC")
            st.warning("Not enough feedback yet to trigger retraining.", icon="📭")
        else:
            st.markdown(f"### ℹ️ Last run: `{status}`")
            st.caption(f"Run #{run_number} · {updated_at} UTC")

        st.markdown(f"[View in GitHub Actions ↗]({html_url})")
    else:
        st.markdown("### ⚪ No runs yet")
        st.caption("Submit feedback to trigger the first training run.")

    st.divider()

    # — Feedback stats —
    try:
        import glob
        n_feedback = len([f for f in glob.glob("data/feedback/**/*.json", recursive=True)
                          if not f.endswith(".gitkeep")])
        st.metric("Feedback events", n_feedback, help="Immutable events stored in data/feedback/")
        st.caption(f"Need 50 to retrain (or trigger manually with min_batch_size=1)")
    except Exception:
        pass

    st.divider()

    # — Manual refresh button —
    if st.button("🔄 Refresh model & status", use_container_width=True):
        get_latest_workflow_run.clear()
        get_current_model_version.clear()
        load_classifier.clear()
        st.rerun()

    # — Non-blocking auto-refresh while training —
    # Uses an invisible HTML meta-refresh so the browser reloads itself
    # without freezing the Python/Streamlit process at all.
    if run and run.get("status") in ("in_progress", "queued"):
        st.caption("⏱️ Page auto-refreshes every 30 s while training…")
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title(":material/support_agent: Email Ticket Classifier Dashboard")

tab1, tab2, tab3 = st.tabs([
    ":material/science: Test Classifier",
    ":material/pending_actions: Human Review Queue",
    ":material/mark_email_read: Processed & Sent",
])

# ── Tab 1 — Test Classifier ────────────────────────────────────────────────────
with tab1:
    st.header("Test Classification")
    with st.form("test_form"):
        subject = st.text_input("Subject")
        body    = st.text_area("Email Body")
        submit  = st.form_submit_button("Classify", icon=":material/send:")

        if submit and (subject or body):
            text_input = f"{subject} {body}".strip()
            try:
                label, confidence = predict_fn(text_input)
                routing           = determine_routing(label, confidence)

                # Persist to DB
                status = (
                    "auto_routed" if confidence >= HIGH_THRESHOLD
                    else "pending_review"
                )
                db = get_db()
                try:
                    ticket = Ticket(
                        subject=subject,
                        body=body,
                        predicted_label=label,
                        confidence=confidence,
                        status=status,
                    )
                    db.add(ticket)
                    db.commit()
                finally:
                    db.close()

                st.success(f"Predicted Label: **{label}** (Confidence: {confidence:.2f})")
                st.info(routing)

                if status == "pending_review":
                    st.warning(
                        "Low confidence — ticket added to the **Human Review Queue**.",
                        icon=":material/warning:",
                    )
            except Exception as e:
                st.error(f"Error during classification: {e}")

# ── Tab 2 — Human Review Queue ─────────────────────────────────────────────────
with tab2:
    st.header("Pending Review Queue")
    st.write("Tickets with low confidence that require a human decision.")

    db = get_db()
    try:
        tickets = (
            db.query(Ticket)
            .filter(Ticket.status == "pending_review")
            .order_by(Ticket.created_at.desc())
            .all()
        )
    except Exception as e:
        tickets = []
        st.error(f"Could not load queue: {e}")
    finally:
        db.close()

    if not tickets:
        st.success("Queue is empty! All caught up. ✅", icon=":material/done_all:")
    else:
        st.info(f"{len(tickets)} ticket(s) awaiting review.")

    for t in tickets:
        with st.expander(f"Ticket #{t.id} — {t.subject}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Body:** {t.body}")
            with col2:
                st.metric("Predicted", t.predicted_label)
                st.metric("Confidence", f"{t.confidence:.0%}")

            with st.form(f"resolve_{t.id}"):
                st.write("### Resolution")
                default_idx = (
                    LABEL_OPTIONS.index(t.predicted_label)
                    if t.predicted_label in LABEL_OPTIONS else 0
                )
                new_label      = st.selectbox(
                    "Correct Label", options=LABEL_OPTIONS, index=default_idx,
                    help="Select the correct category. This trains the next model version.",
                )
                response_email = st.text_area(
                    "Email Response Draft",
                    placeholder="Type the response to send to the customer…",
                )
                if st.form_submit_button(
                    "Send Email & Resolve",
                    type="primary",
                    icon=":material/send_and_archive:",
                ):
                    save_feedback(t, new_label, response_email)
                    st.success(
                        "✅ Resolved! Label saved — model retraining triggered in background.",
                    )
                    time.sleep(1)
                    st.rerun()

# ── Tab 3 — History ────────────────────────────────────────────────────────────
with tab3:
    st.header("Processed & Sent History")

    db = get_db()
    try:
        resolved = (
            db.query(Ticket)
            .filter(Ticket.status.in_(["resolved", "auto_routed"]))
            .order_by(Ticket.created_at.desc())
            .all()
        )
    except Exception as e:
        resolved = []
        st.error(f"Could not load history: {e}")
    finally:
        db.close()

    if not resolved:
        st.info("No resolved tickets yet.", icon=":material/info:")
    else:
        st.write(f"**{len(resolved)}** total resolved tickets.")

    for t in resolved:
        status_icon = "✅" if t.status == "resolved" else "⚡"
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.write(f"**Ticket #{t.id}:** {t.subject}")
                if t.response_email:
                    st.write(f"> {t.response_email}")
            with col2:
                st.write(f"**Predicted:** `{t.predicted_label}`")
                if t.human_label:
                    st.write(f"**Corrected:** `{t.human_label}`")
            with col3:
                st.write(f"{status_icon} `{t.status}`")
                if t.created_at:
                    st.caption(t.created_at.strftime("%Y-%m-%d %H:%M"))
