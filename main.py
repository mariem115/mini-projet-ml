"""API FastAPI : un classifieur de sentiment, trois sources."""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from pipelines import (
    NoTextError,
    SpeechServiceError,
    predict_audio,
    predict_image,
    predict_text,
)
from schemas import AudioFile, ImageFile, PredictionResponse, TextRequest

app = FastAPI(
    title="Classifieur de sentiment multi-source",
    description=(
        "Un classifieur texte (positif / négatif / neutre) alimenté par un avis "
        "saisi, une image via OCR, ou un audio via transcription."
    ),
)


def _validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(status_code=422, detail=exc.errors())


def _response(source: str, extracted: str, sentiment: str, confidence: float) -> PredictionResponse:
    return PredictionResponse(
        source=source,
        texte_extrait=extracted,
        sentiment_predit=sentiment,
        confiance=confidence,
    )


def _run_text(texte: str) -> PredictionResponse:
    try:
        extracted, sentiment, confidence = predict_text(texte)
    except NoTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _response("texte", extracted, sentiment, confidence)


def _run_image(content_type: str | None, data: bytes) -> PredictionResponse:
    try:
        ImageFile(content_type=content_type, size=len(data))
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    try:
        extracted, sentiment, confidence = predict_image(data)
    except NoTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _response("image", extracted, sentiment, confidence)


def _run_audio(content_type: str | None, data: bytes) -> PredictionResponse:
    try:
        AudioFile(content_type=content_type, size=len(data))
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    try:
        extracted, sentiment, confidence = predict_audio(data)
    except NoTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _response("audio", extracted, sentiment, confidence)


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text_route(body: TextRequest) -> PredictionResponse:
    """Classifie un avis envoyé en JSON."""
    return _run_text(body.texte)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image_route(file: UploadFile = File(...)) -> PredictionResponse:
    """Extrait le texte d'une image (OCR) puis classifie le sentiment."""
    data = await file.read()
    return _run_image(file.content_type, data)


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio_route(file: UploadFile = File(...)) -> PredictionResponse:
    """Transcrit un fichier WAV puis classifie le sentiment."""
    data = await file.read()
    return _run_audio(file.content_type, data)


def _text_given(texte: str | None) -> bool:
    return texte is not None and texte.strip() != ""


def _file_given(file: UploadFile | None) -> bool:
    return file is not None and bool(file.filename)


@app.post("/predict", response_model=PredictionResponse)
async def predict_route(
    texte: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> PredictionResponse:
    """Détecte la source : champ texte, image, ou audio."""
    has_text = _text_given(texte)
    has_file = _file_given(file)
    if has_text and has_file:
        raise HTTPException(
            status_code=422,
            detail="Provide either texte or file, not both",
        )
    if has_text:
        try:
            body = TextRequest(texte=texte)
        except ValidationError as exc:
            raise _validation_error(exc) from exc
        return _run_text(body.texte)
    if not has_file:
        raise HTTPException(status_code=422, detail="Provide texte or file")

    data = await file.read()
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type.startswith("image/"):
        return _run_image(content_type, data)
    if content_type.startswith("audio/"):
        return _run_audio(content_type, data)
    raise HTTPException(status_code=422, detail="Unsupported file type")
