import os
import json
import resend
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

from src.serving.model_loader import get_inference_pipeline
from src.db.database import get_db, Base, engine
from src.db.models import Ticket

resend.api_key = os.environ.get("RESEND_API_KEY", "")

# Load pipeline at startup
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    Base.metadata.create_all(bind=engine)
    print("Loading inference pipeline...")
    pipeline = get_inference_pipeline()
    yield

app = FastAPI(title="Email Webhook Receiver", lifespan=lifespan)

HIGH_THRESHOLD = 0.85

def get_routing_address(label: str) -> str:
    try:
        with open("configs/routing.json", "r") as f:
            settings = json.load(f)
            return settings.get(label, "")
    except Exception:
        return ""

@app.post("/webhook/resend")
async def receive_email(request: Request):
    """
    Webhook receiver for Resend inbound emails.
    Expected payload follows Resend inbound webhook structure.
    """
    payload = await request.json()
    
    # Resend webhook wraps the email in 'data'
    email_data = payload.get("data", {})
    if not email_data:
        raise HTTPException(status_code=400, detail="Invalid payload")
        
    subject = email_data.get("subject", "No Subject")
    text_body = email_data.get("text", "")
    from_address = email_data.get("from", "Unknown")
    
    if not text_body:
        return {"status": "ignored", "reason": "No text body"}
        
    # Classify
    text_input = f"{subject} {text_body}".strip()
    result = pipeline(text_input[:512])[0]
    label = result["label"]
    confidence = result["score"]
    
    target_email = get_routing_address(label)
    
    # Decide routing status
    if confidence >= HIGH_THRESHOLD and target_email:
        status = "auto_routed"
        # Actually forward the email
        if resend.api_key:
            try:
                resend.Emails.send({
                    "from": "support@neurodynamics.tech",
                    "to": target_email,
                    "subject": f"[{label.upper()}] FW: {subject}",
                    "text": f"Original Sender: {from_address}\nConfidence: {confidence:.0%}\n\n{text_body}"
                })
            except Exception as e:
                print(f"Failed to forward email via Resend: {e}")
                status = "pending_review" # Downgrade if sending fails
    else:
        status = "pending_review"
        
    # Save to SQLite so it appears in the Streamlit human queue
    db = next(get_db())
    try:
        ticket = Ticket(
            subject=subject,
            body=text_body,
            predicted_label=label,
            confidence=confidence,
            status=status
        )
        db.add(ticket)
        db.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        db.close()
        
    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "forwarded_to": target_email if status == "auto_routed" else None
    }
