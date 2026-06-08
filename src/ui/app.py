import streamlit as st
import time

from src.serving.app import classify_email, get_tickets, resolve_ticket, load_model
from src.serving.schemas import EmailRequest, ResolveRequest
from src.db.database import SessionLocal

@st.cache_resource
def init_system():
    load_model()
    return True

# Initialize model and db once
init_system()

def get_db_session():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

st.set_page_config(page_title="Ticket Classifier", page_icon=":material/support_agent:", layout="wide")

st.title(":material/support_agent: Email Ticket Classifier Dashboard")

tab1, tab2, tab3 = st.tabs([
    ":material/science: Test Classifier", 
    ":material/pending_actions: Human Review Queue", 
    ":material/mark_email_read: Processed & Sent"
])

with tab1:
    st.header("Test Classification API")
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

with tab2:
    st.header("Pending Review Queue")
    st.write("Tickets with low confidence requiring human validation.")
    
    db = get_db_session()
    try:
        tickets = get_tickets("pending_review", db)
        if not tickets:
            st.success("Queue is empty! All caught up.", icon=":material/done_all:")
        for t in tickets:
            with st.expander(f"Ticket #{t.id} - {t.subject}"):
                st.write(f"**Predicted:** `{t.predicted_label}` ({t.confidence:.2f})")
                st.write(f"**Body:** {t.body}")
                
                with st.form(f"resolve_{t.id}"):
                    st.write("### Resolution")
                    new_label = st.text_input("Correct Label", value=t.predicted_label)
                    response_email = st.text_area("Email Response Draft")
                    if st.form_submit_button("Send Email & Resolve", type="primary", icon=":material/send_and_archive:"):
                        req = ResolveRequest(human_label=new_label, response_email=response_email)
                        resolve_res = resolve_ticket(t.id, req, db)
                        if resolve_res:
                            st.success("Resolved successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to resolve.")
    except Exception as e:
        st.error(f"Could not load queue: {e}")
    finally:
        db.close()

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
                st.write(f"**Original Prediction:** `{t.predicted_label}` | **Final Label:** `{t.human_label}`")
                st.write(f"**Sent Email:**\n> {t.response_email}")
    except Exception as e:
        st.error(f"Could not load history: {e}")
    finally:
        db.close()
