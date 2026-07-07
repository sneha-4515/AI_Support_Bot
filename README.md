# 🤖 AI Customer Support Bot with RAG

A simple, beginner-friendly AI chatbot that answers customer questions
from a knowledge base — powered by OpenAI GPT-4o.

---

## 📁 Project Files (only 5!)

```
📂 project/
 ├── knowledge_base.txt   ← Your FAQ / company info (edit this!)
 ├── rag_engine.py        ← The brain: search + generate answers
 ├── server.py            ← The backend API (FastAPI)
 ├── index.html           ← The frontend chat UI (open in browser)
 └── requirements.txt     ← Python packages to install
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install Python packages
```
pip install -r requirements.txt
```

### Step 2 — Add your OpenAI API key
Create a file called `.env` in the project folder and paste:
```
OPENAI_API_KEY=sk-your-key-here
```

### Step 3 — Start the server
```
uvicorn server:app --reload --port 8000
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

### Step 4 — Open the chat UI
Just open `index.html` in your browser (double-click the file).

That's it! Start chatting 🎉

---

## 💡 How RAG Works (Simple Explanation)

```
User asks a question
        ↓
Search knowledge_base.txt for relevant sections
        ↓
Send: question + relevant sections → GPT-4o
        ↓
GPT-4o answers using ONLY those sections
        ↓
Answer shown to user ✅
```

**Why RAG?** Without RAG, the AI guesses and can make things up (hallucinate).
With RAG, the AI only answers from YOUR data — so answers are always accurate.

---

## 🛠️ How to Customise

**Change the company name:** Edit `knowledge_base.txt` and the company name in `server.py`

**Add more FAQs:** Just add more Q&A sections to `knowledge_base.txt`

**Change the bot name:** Edit the header in `index.html`

---

## 📡 API Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/chat` | POST | Send a message, get a reply |
| `/history/{session_id}` | GET | See conversation history |
| `/clear/{session_id}` | POST | Clear conversation |
| `/stats` | GET | See knowledge base stats |

Test them at: http://localhost:8000/docs
