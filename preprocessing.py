"""Nettoyage de texte partagé par l'entraînement et l'API."""

import re

import nltk
from nltk.corpus import stopwords

# NLTK retire « ne » et « pas ». Les enlever transforme « ne fonctionne pas »
# en « fonctionne », donc on les conserve, ainsi que les autres négations.
NEGATION_WORDS = frozenset(
    {"ne", "pas", "ni", "plus", "sans", "jamais", "rien", "aucun"}
)

_STOPWORDS: set[str] | None = None


def _french_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            words = stopwords.words("french")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("french")
        _STOPWORDS = set(words) - NEGATION_WORDS
    return _STOPWORDS


def clean_text(text: str) -> str:
    """Minuscules, sans ponctuation ni chiffres, sans stop-words français.

    Les mots de négation sont conservés.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\d_]+", " ", text)
    tokens = [token for token in text.split() if token not in _french_stopwords()]
    return " ".join(tokens)
