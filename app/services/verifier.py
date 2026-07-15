from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import uuid
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB client
chroma_client = chromadb.Client()

def chunk_text(text: str) -> list[str]:
    """Split text into sentences using NLTK"""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def extract_claims(text: str) -> list[str]:
    """Use Ollama to extract individual factual claims from text"""
    prompt = f"""Extract all individual factual claims from this text. 
Return ONLY a numbered list of claims, one per line, nothing else.

Text: "{text}"

Example format:
1. The Eiffel Tower is in Paris
2. It was built in 1889
3. It stands 330 meters tall"""

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    raw = response['message']['content']
    
    # Parse numbered list into clean claims
    claims = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove "1. " prefix
            claim = line.split('.', 1)[-1].strip()
            if claim:
                claims.append(claim)
    
    return claims


def store_context(context: str, collection_name: str) -> chromadb.Collection:
    """Store context chunks as vectors in ChromaDB"""
    collection = chroma_client.get_or_create_collection(collection_name)
    
    chunks = chunk_text(context)
    embeddings = model.encode(chunks).tolist()
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in chunks]
    )
    
    return collection

def verify_claim(claim: str, context: str) -> dict:
    # Store context in ChromaDB
    collection_name = f"ctx_{str(uuid.uuid4())[:8]}"
    collection = store_context(context, collection_name)
    
    # Encode the claim
    claim_embedding = model.encode([claim]).tolist()
    
    # Find most similar chunks in context
    results = collection.query(
        query_embeddings=claim_embedding,
        n_results=min(3, collection.count())
    )
    
    # Get similarity scores
    distances = results['distances'][0]
    similarities = [1 - d for d in distances]
    best_similarity = max(similarities) if similarities else 0
    
    # Get most relevant context chunks
    relevant_chunks = results['documents'][0]
    relevant_context = '. '.join(relevant_chunks)
    
    # Default verdict based on similarity
    is_faithful = best_similarity >= 0.6
    reasoning = f"{'Claim is supported' if is_faithful else 'Claim is not supported'} by context. Similarity score: {best_similarity:.2f}"
    
    # Only call Mistral for ambiguous cases (between 0.4 and 0.7)
    if 0.4 <= best_similarity <= 0.7:
        prompt = f"""You are a fact-checking assistant.

Claim: "{claim}"
Context: "{relevant_context}"

Is this claim faithful to the context? Answer ONLY in this exact format:
VERDICT: FAITHFUL or VERDICT: HALLUCINATION
REASONING: One sentence explanation"""

        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}]
        )
        reasoning_text = response['message']['content']
        is_faithful = 'HALLUCINATION' not in reasoning_text
        reasoning = reasoning_text
    
    # Clean up collection
    chroma_client.delete_collection(collection_name)
    
    return {
        'is_faithful': is_faithful,
        'confidence_score': round(best_similarity, 2),
        'reasoning': reasoning
    }