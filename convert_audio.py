"""Conversion unique : audios/*.mp3 → audios/*.wav (mono, 16 kHz)."""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "dataset_sentiment_ecommerce" / "audios"


def configure_ffmpeg() -> None:
    env_path = os.environ.get("FFMPEG_BINARY")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    winget_links = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    winget_packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    candidates.extend(
        [
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            winget_links,
        ]
    )
    if winget_packages.exists():
        candidates.extend(winget_packages.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        ffprobe = candidate.with_name("ffprobe.exe")
        if not ffprobe.exists():
            continue
        # pydub cherche ffmpeg et ffprobe via PATH, pas via AudioSegment.converter.
        os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
        return
    raise SystemExit(
        "ffmpeg est introuvable. Installez-le, ou définissez FFMPEG_BINARY "
        "vers ffmpeg.exe, puis relancez python convert_audio.py."
    )


def convert_all() -> None:
    configure_ffmpeg()
    from pydub import AudioSegment

    mp3_files = sorted(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        raise SystemExit(f"Aucun fichier mp3 dans {AUDIO_DIR}")
    for mp3_path in mp3_files:
        wav_path = mp3_path.with_suffix(".wav")
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(wav_path, format="wav")
        print(f"{mp3_path.name} -> {wav_path.name}")


if __name__ == "__main__":
    convert_all()
