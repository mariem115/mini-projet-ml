"""Étape 2 : texte, OCR et transcription vers le même classifieur."""

import io
import os
from pathlib import Path

import joblib
import pytesseract
import speech_recognition as sr
from PIL import Image

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
VECTORIZER_PATH = ROOT / "models" / "vectorizer.pkl"
MODEL_PATH = ROOT / "models" / "model.pkl"

LABELS = ("positif", "négatif", "neutre")


class NoTextError(Exception):
    """OCR ou transcription n'a rien renvoyé."""


class SpeechServiceError(Exception):
    """Le service de reconnaissance vocale est injoignable."""


def configure_tesseract() -> None:
    env_path = os.environ.get("TESSERACT_CMD")
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            break

    # L'installateur Windows ne fournit souvent que l'anglais, et Program Files
    # n'est pas toujours modifiable. Le pack fra du projet est alors utilisé.
    # Cette version de Tesseract attend le dossier tessdata lui-même.
    local_fra = ROOT / "tessdata" / "fra.traineddata"
    if local_fra.exists():
        os.environ["TESSDATA_PREFIX"] = str(local_fra.parent)


def _load_models():
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèles introuvables. Lancez d'abord : python train.py"
        )
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


configure_tesseract()
vectorizer, model = _load_models()


def predict_text(text: str) -> tuple[str, str, float]:
    """Retourne (texte d'origine nettoyé des bords, sentiment, confiance)."""
    original = text.strip()
    cleaned = clean_text(original)
    if not cleaned:
        raise NoTextError("no text detected")
    probabilities = model.predict_proba(vectorizer.transform([cleaned]))[0]
    index = int(probabilities.argmax())
    sentiment = str(model.classes_[index])
    if sentiment not in LABELS:
        raise ValueError(f"Label inattendu : {sentiment}")
    confidence = float(probabilities[index])
    confidence = min(max(confidence, 0.0), 1.0)
    return original, sentiment, confidence


def predict_image(image: bytes | str | Path) -> tuple[str, str, float]:
    if isinstance(image, (str, Path)):
        opened = Image.open(image)
    else:
        opened = Image.open(io.BytesIO(image))
    with opened:
        gray = opened.convert("L")
        extracted = pytesseract.image_to_string(gray, lang="fra").strip()
    if not extracted:
        raise NoTextError("no text detected")
    _, sentiment, confidence = predict_text(extracted)
    return extracted, sentiment, confidence


def transcribe_wav(source: bytes | str | Path) -> str:
    recognizer = sr.Recognizer()
    if isinstance(source, (str, Path)):
        audio_file = sr.AudioFile(str(source))
    else:
        audio_file = sr.AudioFile(io.BytesIO(source))
    with audio_file as handle:
        audio = recognizer.record(handle)
    try:
        transcript = recognizer.recognize_google(audio, language="fr-FR")
    except sr.UnknownValueError as exc:
        raise NoTextError("no text detected") from exc
    except sr.RequestError as exc:
        raise SpeechServiceError("Speech service unreachable") from exc
    transcript = (transcript or "").strip()
    if not transcript:
        raise NoTextError("no text detected")
    return transcript


def predict_audio(audio: bytes | str | Path) -> tuple[str, str, float]:
    transcript = transcribe_wav(audio)
    return predict_text(transcript)
