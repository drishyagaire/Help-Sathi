import numpy as np
import librosa

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".webm",
    ".mp4",
}

DEFAULT_TRIM_TOP_DB = 25
DEFAULT_MIN_CLEAN_DURATION_SECONDS = 0.35


def remove_silence_sections(audio, top_db=DEFAULT_TRIM_TOP_DB):
    """Remove silent sections from the full clip, not just the edges."""
    if audio.size == 0:
        return audio

    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return np.array([], dtype=np.float32)

    chunks = [audio[start:end] for start, end in intervals if end > start]
    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32, copy=False)


def inspect_audio_file(
    audio_path,
    sample_rate,
    top_db=DEFAULT_TRIM_TOP_DB,
):
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    cleaned_audio = remove_silence_sections(audio, top_db=top_db)

    return {
        "audio": cleaned_audio,
        "raw_duration_seconds": len(audio) / sample_rate if len(audio) else 0.0,
        "clean_duration_seconds": (
            len(cleaned_audio) / sample_rate if len(cleaned_audio) else 0.0
        ),
    }


def load_and_prepare_audio(
    audio_path,
    sample_rate,
    target_duration_seconds,
    max_raw_duration_seconds=None,
    min_clean_duration_seconds=DEFAULT_MIN_CLEAN_DURATION_SECONDS,
    top_db=DEFAULT_TRIM_TOP_DB,
    normalize=True,
):
    info = inspect_audio_file(
        audio_path=audio_path,
        sample_rate=sample_rate,
        top_db=top_db,
    )

    if (
        max_raw_duration_seconds is not None
        and info["raw_duration_seconds"] > max_raw_duration_seconds
    ):
        raise ValueError(
            f"raw duration {info['raw_duration_seconds']:.2f}s exceeds "
            f"{max_raw_duration_seconds:.2f}s"
        )

    audio = info["audio"]
    if info["clean_duration_seconds"] < min_clean_duration_seconds:
        raise ValueError(
            f"clean duration {info['clean_duration_seconds']:.2f}s is below "
            f"{min_clean_duration_seconds:.2f}s"
        )

    target_samples = int(sample_rate * target_duration_seconds)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    return audio.astype(np.float32, copy=False), info
