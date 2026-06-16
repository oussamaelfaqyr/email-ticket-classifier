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
        from src.db.database import SessionLocal, Base, engine
        from src.db.models import RoutingSettings
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        row = db.query(RoutingSettings).filter(RoutingSettings.label == label).first()
        db.close()
        return row.destination_email if row and row.destination_email else ""
    except Exception as e:
        print(f"Could not read routing settings: {e}")
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
        
    subject = email_data.get("subject", "")
    text_body = email_data.get("text", "")
    html_body = email_data.get("html", "")
    from_address = email_data.get("from", "Unknown")
    email_id = email_data.get("email_id", "")

    # If Resend didn't include body in the webhook (it only sends metadata),
    # fetch the full email content from the Resend API using the email_id
    if not text_body and not html_body and email_id and resend.api_key:
        try:
            fetched = resend.Emails.get(email_id)
            if isinstance(fetched, dict):
                text_body = fetched.get("text", "") or ""
                html_body = fetched.get("html", "") or ""
            else:
                # Some SDK versions return an object
                text_body = getattr(fetched, "text", "") or ""
                html_body = getattr(fetched, "html", "") or ""
                
            if not text_body and not html_body:
                text_body = f"[DEBUG - RESEND RESPONSE]: {str(fetched)}"
                
            print(f"Fetched body for {email_id}. Text length: {len(text_body)}, HTML length: {len(html_body)}")
        except Exception as e:
            print(f"Could not fetch email body from Resend API for {email_id}: {e}")
            text_body = f"[DEBUG - RESEND API ERROR]: {str(e)}"

    # Fallback to HTML if text is still missing
    if not text_body and html_body:
        import re
        text_body = re.sub('<[^<]+>', ' ', html_body)

    text_input = f"{subject} {text_body}".strip()

    if not text_input:
        return {"status": "ignored", "reason": "Email is completely empty (no subject, no body)"}

    # Classify
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
                final_body = text_body if text_body else "[No body content provided by original sender]"
                resend.Emails.send({
                    "from": "support@neurodynamics.tech",
                    "to": target_email,
                    "subject": f"[{label.upper()}] FW: {subject}",
                    "text": f"Original Sender: {from_address}\nConfidence: {confidence:.0%}\n\n{final_body}"
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
