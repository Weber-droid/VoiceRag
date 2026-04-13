import os
import tempfile
from pathlib import Path
from groq import Groq
from gtts import gTTS

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Groq Whisper."""
    with open(audio_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text",
        )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert text to speech using gTTS and save to a temp file."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name
