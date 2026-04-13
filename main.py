import os
import uuid
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag import ingest_pdf, query_rag
from voice import transcribe_audio, text_to_speech

MAX_PDF_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # 5 MB

# Per-endpoint rate limits: (max_requests, window_seconds)
RATE_LIMITS = {
    "upload": (5, 3600),   # 5 uploads per hour
    "ask":    (30, 3600),  # 30 questions per hour
}
_rate_store: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(request: Request, endpoint: str):
    max_requests, window = RATE_LIMITS[endpoint]
    key = f"{endpoint}:{request.client.host}"
    now = time.time()
    _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
    if len(_rate_store[key]) >= max_requests:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    _rate_store[key].append(now)

app = FastAPI(title="VoiceRAG - Week 8 AI Sprint")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return FileResponse(str(Path(__file__).parent / "index.html"))


@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a PDF into the vector store."""
    check_rate_limit(request, "upload")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF too large. Maximum size is 10 MB.")
        f.write(content)

    chunk_count = ingest_pdf(str(file_path), file.filename)

    return {
        "message": f"Document '{file.filename}' ingested successfully",
        "chunks": chunk_count,
        "filename": file.filename,
    }


@app.post("/ask/voice")
async def ask_voice(request: Request, audio: UploadFile = File(...)):
    """Receive audio, transcribe it, run RAG, return audio answer."""
    check_rate_limit(request, "ask")
    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large. Maximum size is 5 MB.")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # STT: Groq Whisper
        question = transcribe_audio(tmp_path)
        if not question:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # RAG: retrieve + generate
        answer, sources = query_rag(question)

        # TTS: gTTS
        audio_path = text_to_speech(answer)

        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            headers={
                "X-Question": question,
                "X-Answer": answer[:300],
                "X-Sources": ", ".join(sources[:3]),
            },
        )
    finally:
        os.unlink(tmp_path)


class TextQuestion(BaseModel):
    question: str


@app.post("/ask/text")
async def ask_text(request: Request, body: TextQuestion):
    """Text-based Q&A (fallback / testing)."""
    check_rate_limit(request, "ask")
    answer, sources = query_rag(body.question)
    return {"question": body.question, "answer": answer, "sources": sources}


@app.get("/health")
async def health():
    return {"status": "ok"}
