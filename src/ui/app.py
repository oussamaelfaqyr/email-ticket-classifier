import streamlit as st
import time
import os
import pandas as pd
import sys

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.serving.app import classify_email, get_tickets, resolve_ticket, load_model
from src.serving.app import loader  # ModelLoader instance for class names
from src.serving.schemas import EmailRequest, ResolveRequest
from src.db.database import SessionLocal

FEEDBACK_PATH = os.path.join(
    os.path.dirname(__file__), "data", "human_feedback.csv"
)

@st.cache_resource
def init_system():
    load_model()
    return True

init_system()

def get_db_session():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

def save_feedback(ticket, human_label: str):
    """Append a corrected label row to data/human_feedback.csv for retraining."""
    row = pd.DataFrame([{
        "subject": ticket.subject,
        "body": ticket.body,
        "predicted_label": ticket.predicted_label,
        "human_label": human_label,
        "confidence": ticket.confidence,
        "timestamp": pd.Timestamp.now(),
    }])
    if os.path.exists(FEEDBACK_PATH):
        row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        row.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)

# ── Page layout ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon=":material/support_agent:",
    layout="wide",
)
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
        body = st.text_area("Email Body")
        submit = st.form_submit_button("Classify", icon=":material/send:")

        if submit and (subject or body):
            db = get_db_session()
            try:
                req = EmailRequest(subject=subject, body=body)
                res = classify_email(req, db)
                st.success(f"Predicted Label: **{res.label}** (Confidence: {res.confidence:.2f})")
                st.info(f"Action: {res.routed_to}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                db.close()

# ── Tab 2 — Human Review Queue ─────────────────────────────────────────────────
with tab2:
    st.header("Pending Review Queue")
    st.write("Tickets with low confidence requiring human validation.")

    # Resolve label options from the loaded model
    label_options = loader.model.classes_.tolist() if loader.is_loaded() else []

    db = get_db_session()
    try:
        tickets = get_tickets("pending_review", db)
        if not tickets:
            st.success("Queue is empty! All caught up.", icon=":material/done_all:")
        for t in tickets:
            with st.expander(f"Ticket #{t.id} — {t.subject}"):
                st.write(f"**Predicted:** `{t.predicted_label}` ({t.confidence:.2f})")
                st.write(f"**Body:** {t.body}")

                with st.form(f"resolve_{t.id}"):
                    st.write("### Resolution")
                    default_idx = (
                        label_options.index(t.predicted_label)
                        if t.predicted_label in label_options else 0
                    )
                    new_label = st.selectbox(
                        "Correct Label",
                        options=label_options,
                        index=default_idx,
                        help="Select the correct category. This will be used to retrain the model.",
                    )
                    response_email = st.text_area("Email Response Draft")

                    if st.form_submit_button(
                        "Send Email & Resolve",
                        type="primary",
                        icon=":material/send_and_archive:",
                    ):
                        # 1. Persist corrected label for retraining
                        save_feedback(t, new_label)
                        # 2. Mark ticket resolved in DB
                        req = ResolveRequest(human_label=new_label, response_email=response_email)
                        resolve_res = resolve_ticket(t.id, req, db)
                        if resolve_res:
                            st.success("Resolved! Label saved for model retraining.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to resolve ticket.")
    except Exception as e:
        st.error(f"Could not load queue: {e}")
    finally:
        db.close()

# ── Tab 3 — History ────────────────────────────────────────────────────────────
with tab3:
    st.header("Processed & Sent History")

    db = get_db_session()
    try:
        tickets = get_tickets("resolved", db)
        if not tickets:
            st.info("No resolved tickets yet.", icon=":material/info:")
        for t in tickets:
            with st.container(border=True):
                st.write(f"**Ticket #{t.id}:** {t.subject}")
                st.write(
                    f"**Original Prediction:** `{t.predicted_label}` "
                    f"| **Final Label:** `{t.human_label}`"
                )
                st.write(f"**Sent Email:**\n> {t.response_email}")
    except Exception as e:
        st.error(f"Could not load history: {e}")
    finally:
        db.close()
