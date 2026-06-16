import os
import json
import base64
import requests
import uuid
import datetime
import time
import resend
import yaml
from fastapi import FastAPI, Request, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.serving.model_loader import get_inference_pipeline
from src.db.database import get_db, Base, engine, SessionLocal
from src.db.models import Ticket, RoutingSettings

# Load api key
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

app = FastAPI(title="Email Ticket Classifier API", lifespan=lifespan)

# Add CORS Middleware to support requests from Vercel deployments and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load thresholds from params.yaml
try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    HIGH_THRESHOLD = params["serving"]["confidence_threshold_high"]
    MEDIUM_THRESHOLD = params["serving"]["confidence_threshold_medium"]
except Exception:
    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.60

# Auth dependency
def verify_admin_password(authorization: str = Header(None)):
    admin_password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not admin_password:
        return
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use 'Bearer <password>'"
        )
    if parts[1] != admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )

# Helper functions for GitHub commits and repository dispatches
def _push_file_to_github(filepath: str, content: str):
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return True

    url     = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    encoded = base64.b64encode(content.encode()).decode()
    data = {"message": f"feat: Add feedback event {filepath}", "content": encoded}

    for _ in range(3):
        try:
            r = requests.put(url, headers=headers, json=data, timeout=10)
            if r.status_code in (200, 201):
                return True
        except Exception:
            time.sleep(2)
    return False

def _trigger_github_dispatch():
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "OWNER/REPO")
    if not token or repo == "OWNER/REPO":
        return

    url     = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    payload = {"event_type": "retrain", "client_payload": {"source": "dashboard-api"}}

    for _ in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            if r.status_code == 204:
                return
        except Exception:
            time.sleep(2)

def get_routing_address(label: str) -> str:
    try:
        db = SessionLocal()
        row = db.query(RoutingSettings).filter(RoutingSettings.label == label).first()
        db.close()
        return row.destination_email if row and row.destination_email else ""
    except Exception as e:
        print(f"Could not read routing settings: {e}")
        return ""

def get_label_options():
    global pipeline
    if pipeline and hasattr(pipeline, "model") and hasattr(pipeline.model, "config") and hasattr(pipeline.model.config, "id2label"):
        labels = list(pipeline.model.config.id2label.values())
        return [l for l in labels if l != "unknown"]
    return ["account_access", "billing", "bug_report", "refund_request", "shipping_delivery"]

# Webhook Endpoint (No Auth required for Resend server inbound events)
@app.post("/webhook/resend")
async def receive_email(request: Request):
    payload = await request.json()
    email_data = payload.get("data", {})
    if not email_data:
        raise HTTPException(status_code=400, detail="Invalid payload")
        
    subject = email_data.get("subject", "")
    text_body = email_data.get("text", "")
    html_body = email_data.get("html", "")
    from_address = email_data.get("from", "Unknown")
    email_id = email_data.get("email_id", "")

    if not text_body and not html_body and email_id and resend.api_key:
        try:
            fetched = resend.Emails.Receiving.get(email_id)
            if isinstance(fetched, dict):
                text_body = fetched.get("text", "") or ""
                html_body = fetched.get("html", "") or ""
            else:
                text_body = getattr(fetched, "text", "") or ""
                html_body = getattr(fetched, "html", "") or ""
            print(f"Fetched body for {email_id}. Text length: {len(text_body)}")
        except Exception as e:
            print(f"Could not fetch email body from Resend API for {email_id}: {e}")

    if not text_body and html_body:
        import re
        text_body = re.sub('<[^<]+>', ' ', html_body)

    text_input = f"{subject} {text_body}".strip()
    if not text_input:
        return {"status": "ignored", "reason": "Email is completely empty"}

    # Classify
    global pipeline
    if not pipeline:
        pipeline = get_inference_pipeline()
    result = pipeline(text_input[:512])[0]
    label = result["label"]
    confidence = result["score"]
    
    target_email = get_routing_address(label)
    
    if target_email and resend.api_key:
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
            
    status = "pending_review"
    
    # Save to DB
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
        "forwarded_to": target_email
    }


# Dashboard API Endpoints (Secured via verify_admin_password dependency)
@app.get("/api/status", dependencies=[Depends(verify_admin_password)])
async def get_status():
    hf_repo_id = os.environ.get("HF_REPO_ID", "")
    active_ver, stable_ver = None, None
    if hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=hf_repo_id, filename="model_pointers.json")
            with open(path) as f:
                p = json.load(f)
            active_ver = p.get("active", "main")
            stable_ver = p.get("stable", "main")
        except Exception as e:
            print(f"Error reading model version: {e}")

    # GitHub Actions workflow runs info
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "OWNER/REPO")
    latest_run = None
    if token and repo != "OWNER/REPO":
        try:
            url = f"https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/runs?per_page=1"
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                runs = r.json().get("workflow_runs", [])
                if runs:
                    run = runs[0]
                    latest_run = {
                        "status": run.get("status"),
                        "conclusion": run.get("conclusion"),
                        "run_number": run.get("run_number"),
                        "updated_at": run.get("updated_at"),
                        "html_url": run.get("html_url")
                    }
        except Exception as e:
            print(f"Error fetching workflow run: {e}")

    # Local feedback events count
    import glob
    feedback_count = len([f for f in glob.glob("data/feedback/**/*.json", recursive=True)
                          if not f.endswith(".gitkeep")])

    return {
        "active_model": active_ver or "baseline (sklearn)",
        "stable_model": stable_ver,
        "latest_run": latest_run,
        "feedback_count": feedback_count,
        "hf_repo_id": hf_repo_id
    }

@app.post("/api/refresh", dependencies=[Depends(verify_admin_password)])
async def refresh_cache():
    global pipeline
    print("Reloading pipeline and clearing caches...")
    pipeline = get_inference_pipeline()
    return {"status": "success", "message": "Pipeline reloaded"}

@app.get("/api/labels", dependencies=[Depends(verify_admin_password)])
async def get_labels():
    return {"labels": get_label_options()}

@app.post("/api/classify", dependencies=[Depends(verify_admin_password)])
async def classify_test(request: Request):
    payload = await request.json()
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    text_input = f"{subject} {body}".strip()
    if not text_input:
        raise HTTPException(status_code=400, detail="Subject and body cannot both be empty")

    global pipeline
    if not pipeline:
        pipeline = get_inference_pipeline()
    result = pipeline(text_input[:512])[0]
    label = result["label"]
    confidence = result["score"]

    status_val = "auto_routed" if confidence >= HIGH_THRESHOLD else "pending_review"
    
    # Save to DB
    db = next(get_db())
    ticket_id = None
    try:
        ticket = Ticket(
            subject=subject,
            body=body,
            predicted_label=label,
            confidence=confidence,
            status=status_val
        )
        db.add(ticket)
        db.commit()
        ticket_id = ticket.id
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        db.close()

    routing = "Sent to Human Review Queue"
    if status_val == "auto_routed":
        routing = f"Auto-routed to {label} queue"
    elif confidence >= MEDIUM_THRESHOLD:
        routing = f"Routed to {label} queue — flagged for review"

    return {
        "id": ticket_id,
        "label": label,
        "confidence": confidence,
        "status": status_val,
        "routing": routing
    }

@app.get("/api/tickets", dependencies=[Depends(verify_admin_password)])
async def get_tickets(status: str = None):
    db = next(get_db())
    try:
        query = db.query(Ticket)
        if status:
            statuses = status.split(",")
            query = query.filter(Ticket.status.in_(statuses))
        tickets = query.order_by(Ticket.status.desc(), Ticket.created_at.desc()).all()
        
        result = []
        for t in tickets:
            result.append({
                "id": t.id,
                "subject": t.subject,
                "body": t.body,
                "predicted_label": t.predicted_label,
                "confidence": t.confidence,
                "human_label": t.human_label,
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/api/tickets/{ticket_id}/resolve", dependencies=[Depends(verify_admin_password)])
async def resolve_ticket(ticket_id: int, request: Request):
    payload = await request.json()
    corrected_label = payload.get("corrected_label")
    if not corrected_label:
        raise HTTPException(status_code=400, detail="corrected_label is required")

    db = next(get_db())
    try:
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        # 1. Save feedback for retraining
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        feedback_obj = {
            "id": feedback_id,
            "ticket_db_id": ticket.id,
            "text": f"{ticket.subject} {ticket.body}".strip(),
            "label": ticket.predicted_label,
            "corrected_label": corrected_label,
            "confidence": ticket.confidence,
            "timestamp": now.isoformat() + "Z",
        }
        filepath = f"data/feedback/{now.strftime('%Y/%m/%d')}/fb_{feedback_id}.json"
        _push_file_to_github(filepath, json.dumps(feedback_obj, indent=2))
        
        # 2. Trigger GHA retraining dispatch
        _trigger_github_dispatch()

        # 3. Resend forwarding (only if corrected label differs from prediction)
        if ticket.predicted_label != "unknown" and corrected_label != ticket.predicted_label:
            dest_email = get_routing_address(corrected_label)
            if dest_email and resend.api_key:
                try:
                    resend.Emails.send({
                        "from": "support@neurodynamics.tech",
                        "to": dest_email,
                        "subject": f"[{corrected_label.upper()}] CORRECTION FW: {ticket.subject}",
                        "text": f"Original Sender: Unknown\nStatus: Manually Corrected\n\n{ticket.body}"
                    })
                except Exception as e:
                    print(f"Failed to forward via Resend in API: {e}")

        # 4. Update ticket status
        ticket.human_label = corrected_label
        ticket.status = "resolved"
        db.commit()
        
        return {"status": "success", "ticket_id": ticket_id, "resolved_label": corrected_label}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/settings", dependencies=[Depends(verify_admin_password)])
async def get_settings():
    db = next(get_db())
    try:
        rows = db.query(RoutingSettings).all()
        return {r.label: (r.destination_email or "") for r in rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/api/settings", dependencies=[Depends(verify_admin_password)])
async def update_settings(request: Request):
    payload = await request.json()
    db = next(get_db())
    try:
        for label, email in payload.items():
            row = db.query(RoutingSettings).filter(RoutingSettings.label == label).first()
            if row:
                row.destination_email = email
            else:
                db.add(RoutingSettings(label=label, destination_email=email))
        db.commit()
        return {"status": "success", "message": "Settings updated"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
