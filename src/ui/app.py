import streamlit as st
import requests
import subprocess
import time
import socket

API_URL = "http://127.0.0.1:8000"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_api():
    if not is_port_in_use(8000):
        st.toast("Starting FastAPI server...", icon=":material/rocket_launch:")
        # Start uvicorn in the background
        subprocess.Popen(["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"])
        time.sleep(3) # Give it a moment to boot up
    
st.set_page_config(page_title="Ticket Classifier", page_icon=":material/support_agent:", layout="wide")

st.title(":material/support_agent: Email Ticket Classifier Dashboard")

start_api()

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
            try:
                res = requests.post(f"{API_URL}/classify", json={"subject": subject, "body": body})
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Predicted Label: **{data['label']}** (Confidence: {data['confidence']:.2f})")
                    st.info(f"Action: {data['routed_to']}")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.header("Pending Review Queue")
    st.write("Tickets with low confidence requiring human validation.")
    
    try:
        res = requests.get(f"{API_URL}/tickets?status=pending_review")
        if res.status_code == 200:
            tickets = res.json()
            if not tickets:
                st.success("Queue is empty! All caught up.", icon=":material/done_all:")
            for t in tickets:
                with st.expander(f"Ticket #{t['id']} - {t['subject']}"):
                    st.write(f"**Predicted:** `{t['predicted_label']}` ({t['confidence']:.2f})")
                    st.write(f"**Body:** {t['body']}")
                    
                    with st.form(f"resolve_{t['id']}"):
                        st.write("### Resolution")
                        new_label = st.text_input("Correct Label", value=t['predicted_label'])
                        response_email = st.text_area("Email Response Draft")
                        if st.form_submit_button("Send Email & Resolve", type="primary", icon=":material/send_and_archive:"):
                            resolve_res = requests.post(
                                f"{API_URL}/tickets/{t['id']}/resolve",
                                json={"human_label": new_label, "response_email": response_email}
                            )
                            if resolve_res.status_code == 200:
                                st.success("Resolved successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to resolve.")
    except Exception as e:
        st.error(f"Could not load queue: {e}. Is the API running?")

with tab3:
    st.header("Processed & Sent History")
    
    try:
        res = requests.get(f"{API_URL}/tickets?status=resolved")
        if res.status_code == 200:
            tickets = res.json()
            if not tickets:
                st.info("No resolved tickets yet.", icon=":material/info:")
            for t in tickets:
                with st.container(border=True):
                    st.write(f"**Ticket #{t['id']}:** {t['subject']}")
                    st.write(f"**Original Prediction:** `{t['predicted_label']}` | **Final Label:** `{t['human_label']}`")
                    st.write(f"**Sent Email:**\n> {t['response_email']}")
    except Exception as e:
        st.error(f"Could not load history: {e}")
