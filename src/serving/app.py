from fastapi import FastAPI, HTTPException
import yaml
from src.serving.schemas import EmailRequest, ClassificationResponse
from src.serving.model_loader import ModelLoader
from src.serving.predict import Predictor

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
    try:
        loader.load()
    except Exception as e:
        print(f"Warning: could not load model on startup: {e}")

@app.post("/classify", response_model=ClassificationResponse)
def classify_email(request: EmailRequest):
    text_input = f"{request.subject} {request.body}".strip()
    if not text_input:
        raise HTTPException(status_code=400, detail="Empty email content")
        
    try:
        label, confidence = predictor.predict(text_input)
        routed_to = determine_routing(label, confidence)
        return ClassificationResponse(label=label, confidence=confidence, routed_to=routed_to)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    status = "healthy" if loader.is_loaded() else "degraded - model not loaded"
    return {"status": status}
