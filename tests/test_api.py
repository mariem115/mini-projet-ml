"""Tests API et quelques tests unitaires.

Les tests audio remplacent la transcription Google par une phrase fixe
(monkeypatch), donc pytest n'a pas besoin d'internet.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from pipelines import predict_text
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "dataset_sentiment_ecommerce" / "images" / "I01.png"
AUDIO_PATH = ROOT / "dataset_sentiment_ecommerce" / "audios" / "A01.wav"
LABELS = {"positif", "négatif", "neutre"}
KEYS = {"source", "texte_extrait", "sentiment_predit", "confiance"}

client = TestClient(app)

FIXED_TRANSCRIPT = (
    "Bonjour, je suis vraiment ravi de ma commande, le produit est magnifique."
)


def _assert_prediction(payload: dict, source: str) -> None:
    assert set(payload) == KEYS
    assert payload["source"] == source
    assert isinstance(payload["texte_extrait"], str) and payload["texte_extrait"].strip()
    assert payload["sentiment_predit"] in LABELS
    assert 0 <= payload["confiance"] <= 1


@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setattr("pipelines.transcribe_wav", lambda source: FIXED_TRANSCRIPT)


def test_text_ok():
    response = client.post(
        "/predict/text",
        json={"texte": "Livraison super rapide, je recommande vivement ce produit."},
    )
    assert response.status_code == 200
    _assert_prediction(response.json(), "texte")


def test_image_ok():
    with IMAGE_PATH.open("rb") as handle:
        response = client.post(
            "/predict/image",
            files={"file": ("I01.png", handle, "image/png")},
        )
    assert response.status_code == 200
    _assert_prediction(response.json(), "image")


def test_audio_ok(fake_asr):
    with AUDIO_PATH.open("rb") as handle:
        response = client.post(
            "/predict/audio",
            files={"file": ("A01.wav", handle, "audio/wav")},
        )
    assert response.status_code == 200
    payload = response.json()
    _assert_prediction(payload, "audio")
    assert payload["texte_extrait"] == FIXED_TRANSCRIPT


def test_auto_text():
    response = client.post(
        "/predict",
        data={"texte": "Livraison super rapide, je recommande vivement."},
    )
    assert response.status_code == 200
    _assert_prediction(response.json(), "texte")


def test_auto_image():
    with IMAGE_PATH.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": ("I01.png", handle, "image/png")},
        )
    assert response.status_code == 200
    _assert_prediction(response.json(), "image")


def test_auto_audio(fake_asr):
    with AUDIO_PATH.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": ("A01.wav", handle, "audio/wav")},
        )
    assert response.status_code == 200
    _assert_prediction(response.json(), "audio")


@pytest.mark.parametrize("payload", [{"texte": ""}, {"texte": "   "}])
def test_text_empty(payload):
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [{}, {"text": "bonjour tout va bien"}])
def test_text_missing_field(payload):
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 422


def test_text_wrong_type():
    response = client.post("/predict/text", json={"texte": 123})
    assert response.status_code == 422


def test_image_wrong_mime():
    response = client.post(
        "/predict/image",
        files={"file": ("note.txt", b"ceci n'est pas une image", "text/plain")},
    )
    assert response.status_code == 422


def test_image_too_big():
    response = client.post(
        "/predict/image",
        files={"file": ("big.png", b"a" * (5 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 422


def test_image_no_file():
    response = client.post("/predict/image")
    assert response.status_code == 422


def test_audio_wrong_mime():
    with IMAGE_PATH.open("rb") as handle:
        response = client.post(
            "/predict/audio",
            files={"file": ("I01.png", handle, "image/png")},
        )
    assert response.status_code == 422


def test_auto_nothing():
    response = client.post("/predict")
    assert response.status_code == 422


def test_auto_both(fake_asr):
    with AUDIO_PATH.open("rb") as handle:
        response = client.post(
            "/predict",
            data={"texte": "Livraison rapide"},
            files={"file": ("A01.wav", handle, "audio/wav")},
        )
    assert response.status_code == 422


def test_clean_text_lowercase_and_keeps_pas():
    cleaned = clean_text("Ne fonctionne PAS du tout !!!")
    assert cleaned == cleaned.lower()
    assert "pas" in cleaned.split()
    assert "ne" in cleaned.split()


def test_predict_text_label_and_confidence():
    _text, sentiment, confidence = predict_text(
        "Livraison super rapide, je recommande vivement."
    )
    assert sentiment in LABELS
    assert 0 <= confidence <= 1
