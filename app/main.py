from fastapi import FastAPI, HTTPException
from app.models import VerificationRequest, VerificationResponse
from app.services.verifier import verify_claim
from typing import List
from app.models import VerificationRequest, VerificationResponse, BatchVerificationRequest
from app.models import VerificationRequest, VerificationResponse, BatchVerificationRequest, ClaimExtractionRequest
from app.services.verifier import verify_claim, extract_claims
from app.models import VerificationRequest, VerificationResponse, BatchVerificationRequest, ClaimExtractionRequest
from app.services.verifier import verify_claim, extract_claims


app = FastAPI(title="Hallucination Insurance API", version="1.0.0")

@app.post("/verify/batch")
def verify_batch(request: BatchVerificationRequest):
    results = []
    for claim in request.claims:
        result = verify_claim(claim, request.context)
        results.append({
            "claim": claim,
            "is_faithful": result['is_faithful'],
            "confidence_score": result['confidence_score'],
            "reasoning": result['reasoning']
        })
    return {"results": results}

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


@app.post("/verify/extract")
def verify_extracted_claims(request: ClaimExtractionRequest):
    # Extract claims from text
    claims = extract_claims(request.text)
    
    if not claims:
        return {"claims": [], "message": "No claims found"}
    
    # Verify each claim
    results = []
    for claim in claims:
        result = verify_claim(claim, request.context)
        results.append({
            "claim": claim,
            "is_faithful": result['is_faithful'],
            "confidence_score": result['confidence_score'],
            "reasoning": result['reasoning']
        })
    
    return {
        "total_claims": len(claims),
        "faithful_count": sum(1 for r in results if r['is_faithful']),
        "hallucination_count": sum(1 for r in results if not r['is_faithful']),
        "results": results
    }