from pydantic import BaseModel
from typing import List

class VerificationRequest(BaseModel):
    claim: str
    context: str

class VerificationResponse(BaseModel):
    claim: str
    context: str
    is_faithful: bool
    confidence_score: float
    reasoning: str


class BatchVerificationRequest(BaseModel):
    claims: List[str]
    context: str