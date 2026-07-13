from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import uuid

# Load model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB client
chroma_client = chromadb.Client()

def chunk_text(text: str) -> list[str]:
    """Split text into sentences"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

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
    """
    Verify if a claim is faithful to the context.
    Returns faithfulness score and reasoning.
    """
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
    
    # Get similarity scores (ChromaDB returns distances, convert to similarity)
    distances = results['distances'][0]
    similarities = [1 - d for d in distances]
    best_similarity = max(similarities) if similarities else 0
    
    # Get most relevant context chunks
    relevant_chunks = results['documents'][0]
    relevant_context = '. '.join(relevant_chunks)
    
    # Use Ollama to generate reasoning
    prompt = f"""You are a fact-checking assistant. 

Claim: "{claim}"

Relevant context: "{relevant_context}"

Similarity score: {best_similarity:.2f}

Is this claim faithful to the context? Answer in this exact format:
VERDICT: [FAITHFUL or HALLUCINATION]
REASONING: [One sentence explanation]"""

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    reasoning_text = response['message']['content']
    
    # Parse verdict
    is_faithful = 'FAITHFUL' in reasoning_text and 'HALLUCINATION' not in reasoning_text.split('FAITHFUL')[0]
    
    # Clean up collection
    chroma_client.delete_collection(collection_name)
    
    return {
        'is_faithful': is_faithful,
        'confidence_score': round(best_similarity, 2),
        'reasoning': reasoning_text
    }