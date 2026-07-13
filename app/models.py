from pydantic import BaseModel

class VerificationRequest(BaseModel):
    claim: str
    context: str

class VerificationResponse(BaseModel):
    claim: str
    context: str
    is_faithful: bool
    confidence_score: float
    reasoning: str