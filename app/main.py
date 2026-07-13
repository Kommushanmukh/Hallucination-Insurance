from fastapi import FastAPI, HTTPException
from app.models import VerificationRequest, VerificationResponse
from app.services.verifier import verify_claim

app = FastAPI(title="Hallucination Insurance API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify", response_model=VerificationResponse)
def verify(request: VerificationRequest):
    try:
        result = verify_claim(request.claim, request.context)
        return VerificationResponse(
            claim=request.claim,
            context=request.context,
            is_faithful=result['is_faithful'],
            confidence_score=result['confidence_score'],
            reasoning=result['reasoning']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))