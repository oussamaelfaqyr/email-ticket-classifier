from pydantic import BaseModel, Field

class EmailRequest(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    routed_to: str
    ticket_id: int | None = None

class TicketResponse(BaseModel):
    id: int
    subject: str
    body: str
    predicted_label: str
    confidence: float
    status: str
    human_label: str | None = None
    response_email: str | None = None

    class Config:
        from_attributes = True

class ResolveRequest(BaseModel):
    human_label: str
    response_email: str
