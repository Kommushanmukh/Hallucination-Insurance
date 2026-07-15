# 📘 Beginner's Guide to the Codebase

This guide explains the project **file by file** and **function by function**,
assuming no prior AI engineering background. If you know Python but have never
touched embeddings, vector databases, or local LLMs, start here.

See [README.md](README.md) for a quick-start setup guide.

## 0. The Big Idea in Plain English

Large language models (LLMs) sometimes state things confidently that aren't
actually true, or that aren't backed up by the source document you gave them.
This is called a **hallucination**.

This project answers one question: *"Given a source document (the context) and
a sentence someone claims is true (the claim), is that claim actually supported
by the document?"*

It does this **without calling any paid API** — everything runs on your own
machine using:

| Concept | What it means here | Tool used |
|---|---|---|
| Embeddings | Turning a sentence into a list of numbers (a vector) that captures its *meaning* | `sentence-transformers` |
| Vector search | Finding which sentences in the document are numerically "closest" in meaning to the claim | `ChromaDB` |
| LLM reasoning | Asking a local language model to make a judgment call on ambiguous cases | `Ollama` running `Mistral` |

Two sentences with similar meaning end up as two vectors that are close together
in space. That "closeness" is a number between 0 and 1 called **cosine
similarity**, and it's the backbone of this whole project.

## 1. File-Level Guide

The project is intentionally small — three files under `app/`, one entry-point
UI file, and one dependency list.

```
hallucination-insurance/
├── app/
│   ├── main.py              # FastAPI app — defines the HTTP endpoints (the "front door")
│   ├── models.py            # Pydantic schemas — defines the shape of requests/responses
│   └── services/
│       └── verifier.py       # The actual verification logic (embeddings, ChromaDB, Ollama)
├── ui.py                     # Streamlit web UI — a human-friendly front end that calls the API
├── requirements.txt          # Python packages this project depends on
└── README.md                 # Quick-start setup guide
```

**Why split it this way?** This follows a common backend pattern:
- `models.py` = **data contracts** (what shape data must be in)
- `services/verifier.py` = **business logic** (what the app actually *does*)
- `main.py` = **routing/glue** (what URL triggers what logic)
- `ui.py` = a completely separate process that just talks to the API like any
  other client would (it doesn't import any backend code directly — it uses
  HTTP requests, the same way a mobile app or another company's server would)

This separation means you could delete `ui.py` entirely and the API would
still work — or build a totally different frontend (mobile app, CLI, Slack
bot) without touching `verifier.py` at all.

### `requirements.txt`

```
fastapi              # Web framework — builds the API
uvicorn               # Web server that actually runs the FastAPI app
pydantic              # Data validation library (used automatically by FastAPI)
sentence-transformers # Turns text into embeddings (vectors)
chromadb               # Local vector database — stores & searches embeddings
ollama                 # Python client for talking to a local Ollama LLM server
torch                  # Deep learning library that sentence-transformers runs on
nltk                    # Natural Language Toolkit — used here just for sentence splitting
```

### `app/models.py` — the data contracts

Defines every request/response shape as a Pydantic `BaseModel`. FastAPI uses
these classes to automatically validate incoming JSON and generate API docs.

| Class | Used by | Fields |
|---|---|---|
| `VerificationRequest` | `POST /verify` | `claim: str`, `context: str` |
| `VerificationResponse` | `POST /verify` | `claim`, `context`, `is_faithful: bool`, `confidence_score: float`, `reasoning: str` |
| `BatchVerificationRequest` | `POST /verify/batch` | `claims: List[str]`, `context: str` |
| `ClaimExtractionRequest` | `POST /verify/extract` | `text: str`, `context: str` |

If a client sends a request missing `claim` or with `confidence_score` as a
string instead of a number, FastAPI rejects it automatically with a `422`
error — you never have to write that validation code by hand.

### `app/services/verifier.py` — the core engine

This is where all the "AI" actually happens. Covered function-by-function
in [Section 2](#2-function-level-guide) below.

### `app/main.py` — the API layer

Defines the FastAPI `app` object and three HTTP endpoints. It contains almost
no logic itself — it just validates input (via `models.py`), calls functions
in `verifier.py`, and shapes the output. Covered endpoint-by-endpoint in
[Section 3](#3-api-level-guide) below.

### `ui.py` — the human-facing demo app

A [Streamlit](https://streamlit.io) script — Streamlit turns a plain Python
script into a web page, re-running the whole script top-to-bottom every time
you click a button. It never imports `verifier.py`; it only sends HTTP
requests to `http://127.0.0.1:8000` (your locally running FastAPI server),
exactly like a separate client application would. It has three sections
matching the three API endpoints: single verification, batch verification,
and auto claim extraction.

## 2. Function-Level Guide

All functions below live in [`app/services/verifier.py`](app/services/verifier.py).

### Module-level setup (runs once, at import time)

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
```

- `SentenceTransformer('all-MiniLM-L6-v2')` downloads (on first run) and loads
  a small, fast neural network that converts any sentence into a 384-number
  vector. This happens **once** when the server starts — not on every
  request — because loading the model is slow (~seconds) while using it is
  fast (~milliseconds).
- `chromadb.Client()` creates an **in-memory** vector database client. "In
  memory" means it's not saved to disk — every time you restart the API, all
  stored vectors disappear. That's fine here because each verification
  request builds a fresh, temporary collection and deletes it right after.

### `chunk_text(text: str) -> list[str]`

```python
sentences = sent_tokenize(text)
return [s.strip() for s in sentences if s.strip()]
```

Splits a block of text into individual sentences using NLTK's sentence
tokenizer (smarter than just splitting on `.`, since it knows "Dr. Smith" and
"U.S.A." aren't sentence boundaries). Blank strings are filtered out.

**Why sentence-level chunks?** Comparing a claim against a whole paragraph
dilutes the similarity signal. Comparing it against individual sentences lets
the system pinpoint exactly which sentence (if any) supports the claim.

### `extract_claims(text: str) -> list[str]`

Takes a blob of AI-generated text (e.g., a full LLM answer) and asks the
local Mistral model, via Ollama, to break it into a numbered list of
individual factual claims.

1. Builds a prompt instructing Mistral to return **only** a numbered list.
2. Sends it via `ollama.chat(model='mistral', messages=[...])` — this makes a
   request to your local Ollama server (must be running separately) and
   blocks until Mistral responds.
3. Parses the raw text response line-by-line: keeps only lines that start
   with a digit, strips the `"1. "` prefix, and collects the rest as a claim
   string.

**Why regex-free manual parsing?** LLM output isn't guaranteed to be
perfectly structured JSON, so the code defensively parses loosely-formatted
text rather than assuming a strict format that might break the whole request
if the model deviates slightly.

### `store_context(context: str, collection_name: str) -> chromadb.Collection`

```python
collection = chroma_client.get_or_create_collection(collection_name)
chunks = chunk_text(context)
embeddings = model.encode(chunks).tolist()
collection.add(documents=chunks, embeddings=embeddings, ids=[...])
return collection
```

Takes the raw source context and stores it in ChromaDB as searchable vectors:

1. Creates a new, uniquely-named ChromaDB **collection** (think of it like a
   temporary table/index).
2. Splits the context into sentences via `chunk_text`.
3. Converts every sentence into an embedding vector via `model.encode(...)`
   — this is the sentence-transformers model doing its job. `.tolist()`
   converts the NumPy array output into plain Python lists, which ChromaDB
   requires.
4. Adds each sentence + its embedding + a random UUID (as a required unique
   ID) into the collection.

### `verify_claim(claim: str, context: str) -> dict`

The main entry point — everything else in this file supports this function.

**Step 1 — Store context.**
```python
collection_name = f"ctx_{str(uuid.uuid4())[:8]}"
collection = store_context(context, collection_name)
```
A fresh, uniquely named collection is created *per request* so concurrent
requests never mix up each other's context sentences.

**Step 2 — Embed the claim.**
```python
claim_embedding = model.encode([claim]).tolist()
```
Same embedding model, applied to the single claim sentence.

**Step 3 — Vector search.**
```python
results = collection.query(query_embeddings=claim_embedding, n_results=min(3, collection.count()))
```
Asks ChromaDB: "of all the context sentences you're holding, which 3 are
numerically closest in meaning to this claim?" `min(3, collection.count())`
guards against asking for more results than sentences actually exist (e.g. a
1-sentence context).

**Step 4 — Compute similarity.**
```python
distances = results['distances'][0]
similarities = [1 - d for d in distances]
best_similarity = max(similarities) if similarities else 0
```
ChromaDB returns *distances* (lower = more similar). The code flips this into
a more intuitive *similarity* score (higher = more similar) via `1 - distance`,
then takes the single best (highest) similarity found among the top matches.

**Step 5 — Cheap verdict via threshold.**
```python
is_faithful = best_similarity >= 0.6
reasoning = f"..."
```
If the best-matching context sentence is at least 60% similar to the claim,
it's provisionally marked faithful. This is fast and free — no LLM call
needed for clear-cut cases.

**Step 6 — Expensive verdict for ambiguous cases only.**
```python
if 0.4 <= best_similarity <= 0.7:
    # ask Mistral to make the judgment call
```
Similarity scores between 0.4 and 0.7 are "too close to call" — not clearly
matching, not clearly unrelated. Only in this narrow band does the code pay
the cost of calling the local LLM to reason about it in natural language,
overriding the threshold-based verdict with Mistral's judgment
(`is_faithful = 'HALLUCINATION' not in reasoning_text`).

**Why this two-tier design?** Embedding similarity is fast (milliseconds,
free) but crude — it can't always tell "the tower is in Paris" from "the
tower is in London" apart if the surrounding words are similar enough. LLM
reasoning is slow (seconds) but nuanced. Blending them — using the LLM only
where the cheap method is unsure — keeps the system fast for the 90% of
clear-cut cases while staying accurate for edge cases.

**Step 7 — Cleanup.**
```python
chroma_client.delete_collection(collection_name)
```
Deletes the temporary per-request collection so memory doesn't grow
unbounded across many API calls.

**Return value:**
```python
{
    'is_faithful': bool,          # the final verdict
    'confidence_score': float,    # rounded similarity score, 0.0–1.0
    'reasoning': str              # human-readable explanation
}
```

## 3. API-Level Guide

All endpoints are defined in [`app/main.py`](app/main.py) and served by
`uvicorn app.main:app`. Once running, interactive docs are auto-generated by
FastAPI at **http://127.0.0.1:8000/docs** — open that in a browser to try
every endpoint without writing any code.

### `GET /health`

Purpose: liveness check — "is the server up?"

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok"}
```

No dependencies, no model calls — useful for monitoring/deployment scripts.

### `POST /verify`

Purpose: verify a **single** claim against a context.

**Request body** (validated against `VerificationRequest`):
```json
{
  "claim": "The Eiffel Tower is in London",
  "context": "The Eiffel Tower is located in Paris, France."
}
```

**What happens internally:** `main.py` calls `verify_claim(request.claim, request.context)`
from `verifier.py`, then wraps the result dict into a `VerificationResponse`
Pydantic object (which also echoes back the original `claim`/`context`). If
`verify_claim` throws an exception (e.g. Ollama isn't running), it's caught
and converted into an HTTP `500` error with the exception message.

**Response:**
```json
{
  "claim": "The Eiffel Tower is in London",
  "context": "The Eiffel Tower is located in Paris, France.",
  "is_faithful": false,
  "confidence_score": 0.49,
  "reasoning": "Claim is not supported by context. Similarity score: 0.49"
}
```

### `POST /verify/batch`

Purpose: verify **multiple** claims against the same context in one call.

**Request body** (`BatchVerificationRequest`):
```json
{
  "claims": ["The Eiffel Tower is in Paris", "It is 500 meters tall"],
  "context": "The Eiffel Tower is located in Paris, France. It is 330 meters tall."
}
```

**What happens internally:** loops over `request.claims`, calling
`verify_claim(claim, request.context)` once per claim (each call re-embeds
the context from scratch — see [Notes & Limitations](#4-notes--limitations-worth-knowing)
below), and collects the results into a list.

**Response:**
```json
{
  "results": [
    {"claim": "The Eiffel Tower is in Paris", "is_faithful": true, "confidence_score": 0.7, "reasoning": "..."},
    {"claim": "It is 500 meters tall", "is_faithful": false, "confidence_score": 0.35, "reasoning": "..."}
  ]
}
```

### `POST /verify/extract`

Purpose: given a raw chunk of AI-generated text (not pre-split into claims),
automatically extract individual factual claims **and** verify each one —
the "just paste an LLM's answer" endpoint.

**Request body** (`ClaimExtractionRequest`):
```json
{
  "text": "The Eiffel Tower is in Paris. It was built in 1889 and is 330 meters tall.",
  "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889."
}
```

**What happens internally:**
1. Calls `extract_claims(request.text)` — this makes an Ollama/Mistral call
   to split the text into discrete claims.
2. If no claims were found, returns early with an empty result and a message.
3. Otherwise, loops over each extracted claim and calls `verify_claim(claim, request.context)`,
   same as the batch endpoint.
4. Adds summary counts on top of the per-claim results.

**Response:**
```json
{
  "total_claims": 3,
  "faithful_count": 2,
  "hallucination_count": 1,
  "results": [
    {"claim": "The Eiffel Tower is in Paris", "is_faithful": true, "confidence_score": 0.7, "reasoning": "..."},
    {"claim": "It was built in 1889", "is_faithful": true, "confidence_score": 0.65, "reasoning": "..."},
    {"claim": "It is 330 meters tall", "is_faithful": false, "confidence_score": 0.4, "reasoning": "..."}
  ]
}
```

### Endpoint summary table

| Method & Path | Input | Calls into `verifier.py` | Use case |
|---|---|---|---|
| `GET /health` | none | — | Uptime check |
| `POST /verify` | 1 claim + context | `verify_claim` | Verify one claim |
| `POST /verify/batch` | N claims + context | `verify_claim` × N | Verify several known claims |
| `POST /verify/extract` | free-text + context | `extract_claims` then `verify_claim` × N | Verify an entire LLM response with unknown claims |

## 4. Notes & Limitations Worth Knowing

These are honest observations about the current implementation — useful to
know before relying on this in production or extending it:

- **Ollama must be running separately.** This project only talks to Ollama's
  local server (`ollama serve`, started automatically by the Ollama desktop
  app) with the `mistral` model pulled. If Ollama isn't running, `/verify`
  calls will throw and return a `500` error.
- **ChromaDB storage is in-memory and per-request.** Every single call to
  `verify_claim` re-embeds the *entire* context from scratch and throws it
  away afterward. This is simple and avoids stale-data bugs, but is
  inefficient for `/verify/batch` and `/verify/extract`, which re-embed the
  same context once per claim instead of once per request.
- **Thresholds (0.6 faithful cutoff, 0.4–0.7 ambiguous band) are hardcoded**
  in `verify_claim`. They were likely tuned by hand on a few examples, not
  learned or configurable — worth revisiting if you see too many
  false positives/negatives.
- **No persistence.** Nothing is saved to a database — every verification is
  stateless and forgotten immediately after the response is sent.
