from pydantic import BaseModel, Field

class EmailRequest(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")

class ClassificationResponse(BaseModel):
    label: str = Field(..., description="Predicted category label")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    routed_to: str = Field(..., description="Where the ticket was routed")
