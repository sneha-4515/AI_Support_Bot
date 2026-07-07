"""
server.py
─────────
This file runs the BACKEND SERVER using FastAPI.

It has 3 simple API endpoints (URLs the frontend can call):

  POST /chat        → Send a message, get a reply from the bot
  GET  /history     → Get the full conversation history
  POST /clear       → Clear the conversation and start fresh

HOW TO RUN:
  uvicorn server:app --reload --port 8000

Then open: http://localhost:8000/docs  ← to see all endpoints in a nice UI
"""

import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import load_knowledge_base, get_answer

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="TechStore AI Support Bot")

# CORS allows the frontend (HTML file) to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load knowledge base ONCE when the server starts ───────────────────────────
print("[INFO] Loading knowledge base...")
KB_CHUNKS = load_knowledge_base("knowledge_base.txt")
print(f"[OK] Loaded {len(KB_CHUNKS)} chunks from knowledge base.")

# ── In-memory storage for conversations ───────────────────────────────────────
# Key = session_id (unique ID per browser tab)
# Value = list of messages [{role: "user", content: "..."}, ...]
conversations = {}


# ── Request / Response models (what data looks like going in and out) ──────────

class ChatRequest(BaseModel):
    session_id: str   # which conversation is this?
    message: str      # what did the user type?

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    message_count: int   # how many messages in this session


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def home():
    """Simple health check — tells you the server is running."""
    return {"status": "running", "bot": "TechStore AI Support Bot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    1. Get (or create) the conversation history for this session
    2. Get an answer from the RAG engine
    3. Save the new messages to history
    4. Return the answer
    """
    # Get existing history for this session (or start fresh)
    if request.session_id not in conversations:
        conversations[request.session_id] = []

    history = conversations[request.session_id]

    # Get answer from the RAG engine (the brain in rag_engine.py)
    answer = get_answer(
        question=request.message,
        conversation_history=history,
        chunks=KB_CHUNKS,
    )

    # Save this turn to memory
    conversations[request.session_id].append({"role": "user",      "content": request.message})
    conversations[request.session_id].append({"role": "assistant", "content": answer})

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        message_count=len(conversations[request.session_id]),
    )


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return the full conversation history for a session."""
    history = conversations.get(session_id, [])
    return {"session_id": session_id, "messages": history}


@app.post("/clear/{session_id}")
def clear_history(session_id: str):
    """Clear the conversation — start fresh."""
    conversations.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/stats")
def stats():
    """Quick stats — useful for debugging."""
    return {
        "knowledge_base_chunks": len(KB_CHUNKS),
        "active_sessions": len(conversations),
        "total_messages": sum(len(v) for v in conversations.values()),
    }
