"""Métriques OCR / ASR et sentiment sur les 9 images et 9 audios."""

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from pipelines import NoTextError, SpeechServiceError, predict_audio, predict_image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_sentiment_ecommerce"
RESULTS_DIR = ROOT / "results"


def _normalize(label: str) -> str:
    return label.replace("négatif", "negatif")


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _evaluate_row(path: Path, reference: str, sentiment_ref: str, kind: str) -> dict:
    try:
        if kind == "image":
            extracted, predicted, _confidence = predict_image(path)
        else:
            extracted, predicted, _confidence = predict_audio(path)
        error = ""
    except (NoTextError, SpeechServiceError, OSError, FileNotFoundError) as exc:
        extracted, predicted, error = "", "", str(exc)
    return {
        "fichier": path.name,
        "source": kind,
        "texte_extrait": extracted,
        "texte_reference": reference,
        "similarite": round(_similarity(extracted, reference), 3) if extracted else 0.0,
        "sentiment_predit": predicted,
        "sentiment_ref": sentiment_ref,
        "sentiment_ok": bool(predicted) and _normalize(predicted) == _normalize(sentiment_ref),
        "erreur": error,
    }


def _print_accuracy(frame: pd.DataFrame, source: str) -> None:
    subset = frame[frame["source"] == source]
    correct = int(subset["sentiment_ok"].sum())
    total = len(subset)
    mean_similarity = float(subset["similarite"].mean()) if total else 0.0
    print(
        f"{source}: sentiment {correct}/{total} = {correct / total:.1%}  "
        f"similarité moyenne = {mean_similarity:.3f}"
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    images = pd.read_csv(DATA_DIR / "images.csv", encoding="utf-8")
    for _, row in images.iterrows():
        path = DATA_DIR / "images" / row["fichier"]
        rows.append(
            _evaluate_row(path, str(row["texte_attendu_ocr"]), str(row["sentiment_ref"]), "image")
        )

    audios = pd.read_csv(DATA_DIR / "audios.csv", encoding="utf-8")
    for _, row in audios.iterrows():
        path = DATA_DIR / "audios" / row["fichier"]
        rows.append(
            _evaluate_row(
                path,
                str(row["transcription_reference"]),
                str(row["sentiment_ref"]),
                "audio",
            )
        )

    frame = pd.DataFrame(rows)
    output = RESULTS_DIR / "pipeline_evaluation.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame.drop(columns=["texte_reference"]).to_string(index=False))
    print()
    _print_accuracy(frame, "image")
    _print_accuracy(frame, "audio")
    print(f"Tableau : {output}")


if __name__ == "__main__":
    main()
