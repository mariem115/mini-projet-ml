"""Étape 1 : nettoyer, comparer LogReg et SVM, entraîner, évaluer, sauvegarder."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_sentiment_ecommerce"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

LABELS = ["positif", "négatif", "neutre"]
LABEL_MAP = {"positif": "positif", "negatif": "négatif", "neutre": "neutre"}
RANDOM_STATE = 42


def load_labeled_texts(path: Path, text_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    unknown = set(frame["sentiment_ref"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Labels inconnus dans {path.name} : {sorted(unknown)}")
    return pd.DataFrame(
        {
            "texte": frame[text_column].astype(str),
            "label": frame["sentiment_ref"].map(LABEL_MAP),
        }
    )


def save_confusion(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=LABELS,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_cv(name: str, pipeline: Pipeline, texts, labels, cv) -> dict:
    predicted = cross_val_predict(pipeline, texts, labels, cv=cv)
    accuracy = accuracy_score(labels, predicted)
    macro_f1 = f1_score(labels, predicted, average="macro", labels=LABELS)
    report = classification_report(
        labels,
        predicted,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
    save_confusion(
        labels,
        predicted,
        RESULTS_DIR / f"cv_{name}_confusion.png",
        f"Validation croisée — {name}",
    )
    report_path = RESULTS_DIR / f"cv_{name}_report.txt"
    report_path.write_text(
        f"accuracy={accuracy:.3f}\nmacro_f1={macro_f1:.3f}\n\n{report}",
        encoding="utf-8",
    )
    print(f"\n=== {name} — CV 4 folds ===")
    print(f"accuracy={accuracy:.3f}  macro_f1={macro_f1:.3f}")
    print(report)
    return {
        "name": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
    }


def main() -> None:
    nltk.download("stopwords", quiet=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_df = load_labeled_texts(DATA_DIR / "textes.csv", "texte")
    texts = train_df["texte"].map(clean_text)
    labels = train_df["label"]
    if texts.eq("").any():
        raise ValueError("Certains textes d'entraînement sont vides après nettoyage.")

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    logreg = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("model", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )

    logreg_scores = evaluate_cv("logreg", logreg, texts, labels, cv)
    svm_scores = evaluate_cv("linearsvc", svm, texts, labels, cv)

    # Régression logistique : scores proches du SVM, et predict_proba fournit « confiance ».
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(features, labels)
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(model, MODELS_DIR / "model.pkl")
    print("\nModèle retenu : LogisticRegression")
    print(f"Sauvegardé : {MODELS_DIR / 'vectorizer.pkl'}")
    print(f"Sauvegardé : {MODELS_DIR / 'model.pkl'}")

    images = load_labeled_texts(DATA_DIR / "images.csv", "texte_attendu_ocr")
    audios = load_labeled_texts(DATA_DIR / "audios.csv", "transcription_reference")
    external = pd.concat([images, audios], ignore_index=True)
    external_clean = external["texte"].map(clean_text)
    external_pred = model.predict(vectorizer.transform(external_clean))
    external_acc = accuracy_score(external["label"], external_pred)
    external_f1 = f1_score(
        external["label"], external_pred, average="macro", labels=LABELS
    )
    external_report = classification_report(
        external["label"],
        external_pred,
        labels=LABELS,
        digits=3,
        zero_division=0,
    )
    save_confusion(
        external["label"],
        external_pred,
        RESULTS_DIR / "external_confusion.png",
        "Test externe — 18 textes image/audio",
    )
    (RESULTS_DIR / "external_report.txt").write_text(
        f"n={len(external)}\naccuracy={external_acc:.3f}\nmacro_f1={external_f1:.3f}\n\n{external_report}",
        encoding="utf-8",
    )
    print("\n=== Test externe (18 textes de référence, jamais vus) ===")
    print(f"accuracy={external_acc:.3f}  macro_f1={external_f1:.3f}")
    print(external_report)

    comparison = (
        "Comparaison — StratifiedKFold(n_splits=4, shuffle=True, random_state=42)\n"
        "Vectorisation : TfidfVectorizer(ngram_range=(1, 2)), dans un Pipeline sklearn.\n"
        f"LogisticRegression  accuracy={logreg_scores['accuracy']:.3f}  "
        f"macro_f1={logreg_scores['macro_f1']:.3f}\n"
        f"LinearSVC           accuracy={svm_scores['accuracy']:.3f}  "
        f"macro_f1={svm_scores['macro_f1']:.3f}\n"
        "\n"
        "Modèle retenu : LogisticRegression.\n"
        "Raison : les deux modèles sont comparables sur 24 textes, mais seule la\n"
        "régression logistique fournit predict_proba, exigé par le champ confiance.\n"
        "\n"
        "Le modèle sauvegardé est réentraîné sur les 24 textes. Le test externe\n"
        "utilise les 18 transcriptions de référence (images.csv + audios.csv).\n"
        f"Test externe  accuracy={external_acc:.3f}  macro_f1={external_f1:.3f}\n"
    )
    (RESULTS_DIR / "cv_comparison.txt").write_text(comparison, encoding="utf-8")
    print(comparison)


if __name__ == "__main__":
    main()
