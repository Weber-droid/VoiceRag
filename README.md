# VoiceRAG

Upload a PDF. Ask a question out loud. Get a spoken answer back.

VoiceRAG is a multimodal document Q&A app that chains speech-to-text, retrieval-augmented generation, and text-to-speech into a single browser interaction — no typing required.

---

## How it works

```
Browser mic → WebM audio
      ↓
Groq Whisper  →  transcript
      ↓
ChromaDB + ONNX embeddings  →  relevant chunks
      ↓
Groq LLaMA 3.3 70B  →  conversational answer
      ↓
gTTS  →  MP3 audio
      ↓
Browser  →  plays audio + shows transcript
```

1. Upload a PDF — it gets chunked into 500-character overlapping segments and stored in a local ChromaDB vector store using ONNX MiniLM-L6-v2 embeddings.
2. Hold the mic button and ask a question — your audio is sent to Groq Whisper for transcription.
3. The transcript is used to retrieve the top 4 most relevant chunks from the vector store.
4. Groq LLaMA 3.3 70B generates a concise, conversational answer (no markdown, no bullet points — optimised for speech).
5. gTTS converts the answer to audio, which plays automatically in the browser.

---

## Stack

| Layer          | Tool                                       |
| -------------- | ------------------------------------------ |
| Speech-to-text | Groq Whisper (`whisper-large-v3`)          |
| LLM            | Groq LLaMA 3.3 (`llama-3.3-70b-versatile`) |
| Embeddings     | ONNX MiniLM-L6-v2 (local, no API key)      |
| Vector store   | ChromaDB (persistent, local)               |
| Text-to-speech | gTTS                                       |
| Backend        | FastAPI                                    |
| Frontend       | Vanilla HTML / CSS / JS                    |

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com), then:

```bash
export GROQ_API_KEY=your_key_here
```

### 3. Run

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## API

| Method | Endpoint     | Description                         |
| ------ | ------------ | ----------------------------------- |
| `GET`  | `/`          | Serves the frontend                 |
| `POST` | `/upload`    | Upload and ingest a PDF (max 10 MB) |
| `POST` | `/ask/voice` | Send audio, receive spoken answer   |
| `POST` | `/ask/text`  | Text-based Q&A (testing / fallback) |
| `GET`  | `/health`    | Health check                        |

### Rate limits

| Endpoint                   | Limit              |
| -------------------------- | ------------------ |
| `/upload`                  | 5 requests / hour  |
| `/ask/voice` & `/ask/text` | 30 requests / hour |

---

## Deployment

### Digital Ocean App Platform

1. Add a `Procfile` to the project root:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
2. Push to GitHub and connect the repo in App Platform.
3. Add `GROQ_API_KEY` as an environment variable.

> **Note:** App Platform containers are ephemeral. The ChromaDB vector store and uploaded PDFs live on local disk and will be wiped on redeploy or container restart. For persistent storage, replace ChromaDB with Supabase pgvector and store PDFs in Supabase Storage.

---

## Project structure

```
.
├── main.py          # FastAPI app and API endpoints
├── rag.py           # PDF ingestion and RAG query logic
├── voice.py         # Whisper transcription and gTTS synthesis
├── index.html       # Frontend (single-page, no framework)
├── requirements.txt
├── chroma_db/       # Local vector store (auto-created)
└── uploads/         # Uploaded PDFs (auto-created)
```
