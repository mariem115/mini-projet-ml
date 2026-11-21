"""Schémas Pydantic : requête texte, fichiers, et réponse unique."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class TextRequest(BaseModel):
    texte: str = Field(min_length=1, max_length=2000)

    @field_validator("texte", mode="before")
    @classmethod
    def strip_whitespace(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ImageFile(BaseModel):
    content_type: Literal["image/png", "image/jpeg"]
    size: int = Field(gt=0, le=MAX_IMAGE_BYTES)


class AudioFile(BaseModel):
    content_type: Literal["audio/wav", "audio/x-wav", "audio/wave"]
    size: int = Field(gt=0, le=MAX_AUDIO_BYTES)


class PredictionResponse(BaseModel):
    source: Literal["texte", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Literal["positif", "négatif", "neutre"]
    confiance: float = Field(ge=0, le=1)
