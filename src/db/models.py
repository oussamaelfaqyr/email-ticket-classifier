from sqlalchemy import Column, Integer, String, Float, DateTime
from src.db.database import Base
import datetime

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, index=True)
    body = Column(String)
    predicted_label = Column(String, index=True)
    confidence = Column(Float)
    human_label = Column(String, nullable=True)
    status = Column(String, default="pending_review", index=True) # auto_routed, pending_review, resolved
    response_email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RoutingSettings(Base):
    __tablename__ = "routing_settings"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, unique=True, index=True)  # e.g. "billing"
    destination_email = Column(String, nullable=True)  # e.g. "billing@company.com"
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
