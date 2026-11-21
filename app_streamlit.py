"""Mini application de test : trois onglets qui appellent l'API."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Sentiment multi-source")
st.title("Classifieur de sentiment multi-source")
st.caption(f"API : {API_URL}")


def _show_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(detail)


def _show_result(payload: dict) -> None:
    st.subheader("Texte extrait")
    st.info(payload["texte_extrait"])
    sentiment = payload["sentiment_predit"]
    st.subheader("Sentiment")
    if sentiment == "positif":
        st.success(sentiment)
    elif sentiment == "négatif":
        st.error(sentiment)
    else:
        st.info(sentiment)
    confidence = float(payload["confiance"])
    st.subheader("Confiance")
    st.progress(confidence)
    st.write(f"{confidence:.0%}")


def _post(path: str, **kwargs) -> None:
    try:
        response = requests.post(f"{API_URL}{path}", timeout=90, **kwargs)
    except requests.ConnectionError:
        st.error(f"API injoignable sur {API_URL}. Lancez : uvicorn main:app --reload")
        return
    if response.ok:
        _show_result(response.json())
    else:
        _show_error(response)


tab_texte, tab_image, tab_audio = st.tabs(["Texte", "Image", "Audio"])

with tab_texte:
    with st.form("form_texte"):
        texte = st.text_area("Avis client", height=160)
        submitted_texte = st.form_submit_button("Analyser le texte", type="primary")
    if submitted_texte:
        if not texte.strip():
            st.error("Saisissez un texte.")
        else:
            _post("/predict/text", json={"texte": texte})

with tab_image:
    image = st.file_uploader("Image PNG ou JPEG", type=["png", "jpg", "jpeg"])
    if image is not None:
        st.image(image.getvalue(), caption=image.name)
        if st.button("Analyser l'image", type="primary"):
            mime = image.type if image.type in {"image/png", "image/jpeg"} else (
                "image/png" if image.name.lower().endswith(".png") else "image/jpeg"
            )
            _post(
                "/predict/image",
                files={"file": (image.name, image.getvalue(), mime)},
            )

with tab_audio:
    audio = st.file_uploader("Audio WAV", type=["wav"])
    if audio is not None:
        st.audio(audio.getvalue())
        if st.button("Analyser l'audio", type="primary"):
            mime = audio.type if audio.type in {"audio/wav", "audio/x-wav", "audio/wave"} else "audio/wav"
            _post(
                "/predict/audio",
                files={"file": (audio.name, audio.getvalue(), mime)},
            )
