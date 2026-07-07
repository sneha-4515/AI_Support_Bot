"""
rag_engine.py
─────────────
This file is the BRAIN of the bot. It does 3 things:

  1. LOAD   → Read the knowledge base text file
  2. SEARCH → Find the most relevant sections for a user's question
  3. ANSWER → Send the question + relevant sections to GPT-4o for an answer

HOW RAG WORKS (simple explanation):
  - RAG = Retrieval Augmented Generation
  - Instead of the AI guessing answers, we FIRST search our documents
    for relevant info, THEN give that info to the AI along with the question.
  - This means the AI answers from YOUR data — no hallucinations!
"""

import os
from openai import OpenAI

# ── Connect to OpenAI ──────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── STEP 1: Load & chunk the knowledge base ────────────────────────────────────

def load_knowledge_base(filepath="knowledge_base.txt"):
    """
    Read the text file and split it into small chunks.
    Each chunk = one paragraph / section of the FAQ.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by blank lines → each paragraph becomes one chunk
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


# ── STEP 2: Find relevant chunks using embeddings ─────────────────────────────

def get_embedding(text):
    """
    Convert text into a list of numbers (embedding vector).
    Similar texts will have similar number lists.
    This is how we do semantic search — matching MEANING not just keywords.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.replace("\n", " ")
    )
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two vectors are.
    Score of 1.0 = identical meaning, 0.0 = completely different.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
    magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def find_relevant_chunks(question, chunks, top_k=3):
    """
    Given a question, find the TOP 3 most relevant chunks from our knowledge base.

    How it works:
      1. Embed the question into a vector
      2. Embed every chunk into a vector
      3. Calculate similarity between question and each chunk
      4. Return the top_k most similar chunks
    """
    question_embedding = get_embedding(question)

    # Score every chunk
    scored_chunks = []
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        score = cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append((score, chunk))

    # Sort by score, highest first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return only the text of top_k chunks
    return [chunk for score, chunk in scored_chunks[:top_k]]


# ── STEP 3: Generate answer using GPT-4o ──────────────────────────────────────

def get_answer(question, conversation_history, chunks):
    """
    Use GPT-4o to answer the question using:
      - The retrieved context (relevant chunks from knowledge base)
      - The conversation history (so the bot remembers previous messages)

    Args:
        question:             The user's current question
        conversation_history: List of previous messages [{role, content}, ...]
        chunks:               The knowledge base file split into chunks

    Returns:
        The bot's answer as a string
    """
    # Find relevant sections from the knowledge base
    relevant_chunks = find_relevant_chunks(question, chunks)
    context = "\n\n---\n\n".join(relevant_chunks)

    # System prompt — tells the AI how to behave and gives it the context
    system_prompt = f"""You are a friendly and helpful customer support assistant for TechStore.

Use ONLY the information below to answer the customer's question.
If the answer is not in the information below, say:
"I don't have that information. Please contact support@techstore.com or call 1-800-TECH-HELP."

Do NOT make up any information. Keep your answer clear and concise.

=== RELEVANT KNOWLEDGE BASE INFORMATION ===
{context}
===========================================
"""

    # Build the messages array: system prompt + conversation history + new question
    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history
    messages.append({"role": "user", "content": question})

    # Call GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,   # low = more factual, consistent answers
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()
