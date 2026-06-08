import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, HTTPException, Depends
import yaml
from sqlalchemy.orm import Session
from typing import List
from src.serving.schemas import EmailRequest, ClassificationResponse, TicketResponse, ResolveRequest
from src.serving.model_loader import ModelLoader
from src.serving.predict import Predictor
from src.db.database import SessionLocal, engine, Base, get_db
from src.db.models import Ticket

app = FastAPI(title="Email Ticket Classifier", version="0.1.0")

# Load configs
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Initialize components
loader = ModelLoader("models/baseline.pkl", "data/processed/vectorizer.pkl")
predictor = Predictor(loader)

def determine_routing(label: str, confidence: float) -> str:
    high_threshold = params["serving"]["confidence_threshold_high"]
    medium_threshold = params["serving"]["confidence_threshold_medium"]
    
    if confidence >= high_threshold:
        return f"auto-route to {label} queue"
    elif confidence >= medium_threshold:
        return f"route to {label} queue (flagged for review)"
    else:
        return "human review queue"

@app.on_event("startup")
def load_model():
    Base.metadata.create_all(bind=engine)
    try:
        loader.load()
    except Exception as e:
        print(f"Warning: could not load model on startup: {e}")

@app.post("/classify", response_model=ClassificationResponse)
def classify_email(request: EmailRequest, db: Session = Depends(get_db)):
    text_input = f"{request.subject} {request.body}".strip()
    if not text_input:
        raise HTTPException(status_code=400, detail="Empty email content")
        
    try:
        label, confidence = predictor.predict(text_input)
        routed_to = determine_routing(label, confidence)
        
        # Save to DB
        status = "auto_routed" if confidence >= params["serving"]["confidence_threshold_high"] else "pending_review"
        ticket = Ticket(
            subject=request.subject,
            body=request.body,
            predicted_label=label,
            confidence=confidence,
            status=status
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        
        return ClassificationResponse(
            label=label, 
            confidence=confidence, 
            routed_to=routed_to,
            ticket_id=ticket.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tickets", response_model=List[TicketResponse])
def get_tickets(status: str = "pending_review", db: Session = Depends(get_db)):
    return db.query(Ticket).filter(Ticket.status == status).order_by(Ticket.created_at.desc()).all()

@app.post("/tickets/{ticket_id}/resolve", response_model=TicketResponse)
def resolve_ticket(ticket_id: int, request: ResolveRequest, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    ticket.human_label = request.human_label
    ticket.response_email = request.response_email
    ticket.status = "resolved"
    db.commit()
    db.refresh(ticket)
    return ticket

@app.get("/health")
def health_check():
    status = "healthy" if loader.is_loaded() else "degraded - model not loaded"
    return {"status": status}
