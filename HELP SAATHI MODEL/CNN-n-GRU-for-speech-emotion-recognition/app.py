"""
Help Sathi
Transcript-gated distress screening.

Flow:
1. Browser records a short clip when the user presses the button.
2. This app forwards the clip to a remote STT server.
3. Only if danger keywords are found in the transcript do we run emotion analysis.
4. SOS is shown when both transcript risk and emotion risk are high.
"""

import os
import base64
import socket
import subprocess
import sys
import tempfile
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import urlparse

import librosa
import numpy as np
import requests
import torch
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


# Settings
MODEL_PATH = os.environ.get(
    "HELP_SATHI_MODEL_PATH",
    "/Users/nischalthapa/Desktop/CNN-n-GRU-for-speech-emotion-recognition/experiments/nepali/cnn18gru/best_model.pth",
)
STT_SERVER_URL = os.environ.get("HELP_SATHI_STT_SERVER_URL", "http://192.168.1.65:5000/transcribe")
SAMPLE_RATE = 16000
EMOTION_WINDOW_SECONDS = 3
CHUNK_SECONDS = int(os.environ.get("HELP_SATHI_CHUNK_SECONDS", "5"))
CONFIDENCE_THRESHOLD = float(os.environ.get("HELP_SATHI_CONFIDENCE_THRESHOLD", "80"))
TRANSCRIBE_TIMEOUT_SECONDS = int(os.environ.get("HELP_SATHI_TRANSCRIBE_TIMEOUT_SECONDS", "30"))
KEYWORD_SIMILARITY_THRESHOLD = float(os.environ.get("HELP_SATHI_KEYWORD_SIMILARITY_THRESHOLD", "0.70"))
RNNOISE_MODEL_PATH = os.environ.get("HELP_SATHI_RNNOISE_MODEL_PATH", "").strip()
NOISE_FILTER_CHAIN = os.environ.get(
    "HELP_SATHI_NOISE_FILTER_CHAIN",
    "highpass=f=120,lowpass=f=3800,afftdn=nf=-25",
)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_CLASSES = list(DEFAULT_EMOTION_CLASSES)
DISTRESS_EMOTIONS = {"angry", "fear", "disgust", "sad"}

DEFAULT_DANGER_KEYWORDS = [
    "सुरक्षित छैन",
    "सुरक्षित छैन",
    "मलाई सहयोग चाहिएको छ",
    "मलाई सहयोग चाहियो",
    "मलाई बचाउनुहोस्",
    "बचाउनुहोस्",
    "कृपया सहयोग गर्नुहोस्",
    "पुलिस बोलाउनुहोस्",
    "म डराएको छु",
    "म डराएकी छु",
    "छोड्नुहोस्",
    "टाढा जानुहोस्",
    "बचाओ",
    "बाचाओ",
    "बचाऊ",
    "help",
    "हेल्प"
]

custom_keywords = os.environ.get("HELP_SATHI_KEYWORDS", "").strip()
DANGER_KEYWORDS = [k.strip() for k in custom_keywords.split(",") if k.strip()] or DEFAULT_DANGER_KEYWORDS

model = None


def get_effective_noise_filter_chain():
    if RNNOISE_MODEL_PATH and os.path.exists(RNNOISE_MODEL_PATH):
        return f"highpass=f=120,lowpass=f=3800,arnndn=model='{RNNOISE_MODEL_PATH}':mix=1"
    return NOISE_FILTER_CHAIN


def normalize_text(text):
    return unicodedata.normalize("NFKC", text).strip().lower()


def similarity_score(left, right):
    return SequenceMatcher(None, left, right).ratio()


def find_matched_keywords(text):
    normalized = normalize_text(text)
    transcript_tokens = normalized.split()
    matches = []

    for keyword in DANGER_KEYWORDS:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue

        if normalized_keyword in normalized:
            matches.append({"keyword": keyword, "score": 1.0})
            continue

        keyword_tokens = normalized_keyword.split()
        candidate_scores = [similarity_score(normalized_keyword, normalized)]

        if transcript_tokens:
            if len(keyword_tokens) <= 1:
                candidate_scores.extend(
                    similarity_score(normalized_keyword, token)
                    for token in transcript_tokens
                )
            else:
                for window_size in range(
                    max(1, len(keyword_tokens) - 1),
                    len(keyword_tokens) + 2,
                ):
                    if len(transcript_tokens) < window_size:
                        continue
                    for index in range(len(transcript_tokens) - window_size + 1):
                        candidate = " ".join(transcript_tokens[index:index + window_size])
                        candidate_scores.append(
                            similarity_score(normalized_keyword, candidate)
                        )

        best_score = max(candidate_scores) if candidate_scores else 0.0
        if best_score >= KEYWORD_SIMILARITY_THRESHOLD:
            matches.append({"keyword": keyword, "score": round(best_score, 3)})

    return matches


def load_model():
    global model, EMOTION_CLASSES
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.cnn_n_gru import CNN18GRU

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        checkpoint_emotions = None
        if isinstance(checkpoint, dict):
            checkpoint_emotions = checkpoint.get("emotions")

        if checkpoint_emotions:
            EMOTION_CLASSES = list(checkpoint_emotions)
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            fc2_bias = checkpoint["model_state_dict"].get("fc2.bias")
            if fc2_bias is not None and len(fc2_bias) != len(EMOTION_CLASSES):
                EMOTION_CLASSES = [f"class_{i}" for i in range(len(fc2_bias))]

        loaded_model = CNN18GRU(
            n_input=1,
            hidden_dim=64,
            n_layers=1,
            n_output=len(EMOTION_CLASSES),
            stride=4,
            n_channel=18,
            dropout=0.0,
        ).to(DEVICE)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                loaded_model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                loaded_model.load_state_dict(checkpoint["state_dict"])
            else:
                loaded_model.load_state_dict(checkpoint)
        else:
            loaded_model.load_state_dict(checkpoint)

        loaded_model.eval()
        model = loaded_model
        print(f"Model loaded on {DEVICE}")
    except Exception as exc:
        print(f"Model load error: {exc}")
        model = None


def get_stt_server_status():
    parsed = urlparse(STT_SERVER_URL)
    host = parsed.hostname
    port = parsed.port

    if not host:
        return {"reachable": False, "error": "Invalid STT server URL"}

    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        with socket.create_connection((host, port), timeout=2):
            return {"reachable": True, "error": None}
    except OSError as exc:
        return {"reachable": False, "error": str(exc)}


def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    target_length = SAMPLE_RATE * EMOTION_WINDOW_SECONDS
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))

    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return waveform.to(DEVICE)


def convert_audio_to_wav(audio_path, filter_chain=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
        output_path = wav_tmp.name

    command = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-i",
        audio_path,
    ]

    if filter_chain:
        command.extend(["-af", filter_chain])

    command.extend([
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        output_path,
    ])

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"ffmpeg conversion failed: {completed.stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("ffmpeg conversion produced an empty wav file")

    return output_path


def build_audio_preview(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("ascii")
    return f"data:audio/wav;base64,{encoded_audio}"


def predict_emotion(audio_path):
    if model is None:
        return {"error": "Emotion model not loaded."}

    try:
        waveform = preprocess_audio(audio_path)
        hidden = model.init_hidden(batch_size=1, device=DEVICE)

        with torch.no_grad():
            output, _ = model(waveform, hidden)
            probs = torch.exp(output)[0]
            idx = torch.argmax(probs).item()
            confidence = probs[idx].item() * 100

        emotion = EMOTION_CLASSES[idx]
        is_distress = emotion in DISTRESS_EMOTIONS
        sos_trigger = is_distress and confidence >= CONFIDENCE_THRESHOLD

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "is_distress": is_distress,
            "sos_trigger": sos_trigger,
            "all_probs": {
                label: round(probs[i].item() * 100, 2)
                for i, label in enumerate(EMOTION_CLASSES)
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


def transcribe_with_remote_server(audio_path, original_filename):
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            STT_SERVER_URL,
            files={"audio": (original_filename, audio_file)},
            timeout=TRANSCRIBE_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


@app.route("/")
def index():
    return render_template(
        "index.html",
        device=str(DEVICE),
        chunk_seconds=CHUNK_SECONDS,
        stt_server_url=STT_SERVER_URL,
    )


@app.route("/status")
def status():
    stt_status = get_stt_server_status()
    return jsonify(
        {
            "loaded": model is not None,
            "device": str(DEVICE),
            "stt_server_url": STT_SERVER_URL,
            "chunk_seconds": CHUNK_SECONDS,
            "danger_keywords": DANGER_KEYWORDS,
            "keyword_similarity_threshold": KEYWORD_SIMILARITY_THRESHOLD,
            "noise_filter_chain": get_effective_noise_filter_chain(),
            "rnnoise_model_path": RNNOISE_MODEL_PATH if RNNOISE_MODEL_PATH else None,
            "rnnoise_enabled": bool(RNNOISE_MODEL_PATH and os.path.exists(RNNOISE_MODEL_PATH)),
            "stt_reachable": stt_status["reachable"],
            "stt_error": stt_status["error"],
        }
    )


@app.route("/screen_audio", methods=["POST"])
def screen_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename or "chunk.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    original_wav_path = None
    filtered_wav_path = None
    try:
        effective_filter_chain = get_effective_noise_filter_chain()
        original_wav_path = convert_audio_to_wav(tmp_path)
        filtered_wav_path = convert_audio_to_wav(original_wav_path, filter_chain=effective_filter_chain)
        transcription = transcribe_with_remote_server(filtered_wav_path, "chunk.wav")
        transcript = transcription.get("text", "").strip()
        matched_keywords = find_matched_keywords(transcript)
        keyword_hit = len(matched_keywords) > 0

        emotion_result = None
        if keyword_hit:
            emotion_result = predict_emotion(filtered_wav_path)

        sos_trigger = bool(
            keyword_hit
            and emotion_result
            and not emotion_result.get("error")
            and emotion_result.get("is_distress")
        )

        return jsonify(
            {
                "transcript": transcript,
                "detected_language": transcription.get("detected_language"),
                "language_probability": transcription.get("language_probability"),
                "matched_keywords": [item["keyword"] for item in matched_keywords],
                "matched_keyword_scores": matched_keywords,
                "keyword_hit": keyword_hit,
                "emotion_result": emotion_result,
                "original_audio_preview": build_audio_preview(original_wav_path),
                "filtered_audio_preview": build_audio_preview(filtered_wav_path),
                "noise_filter_chain": effective_filter_chain,
                "rnnoise_enabled": bool(RNNOISE_MODEL_PATH and os.path.exists(RNNOISE_MODEL_PATH)),
                "danger_emotion_detected": bool(
                    emotion_result
                    and not emotion_result.get("error")
                    and emotion_result.get("is_distress")
                ),
                "sos_trigger": sos_trigger,
            }
        )
    except requests.RequestException as exc:
        print(f"STT request failed: {exc}")
        return jsonify({"error": f"STT server request failed: {exc}"}), 502
    except Exception as exc:
        print(f"screen_audio failed: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if original_wav_path and os.path.exists(original_wav_path):
            os.unlink(original_wav_path)
        if filtered_wav_path and os.path.exists(filtered_wav_path):
            os.unlink(filtered_wav_path)


if __name__ == "__main__":
    load_model()
    print("=" * 60)
    print("Help Sathi running")
    print("Open: http://localhost:5001")
    print(f"STT server: {STT_SERVER_URL}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5001)
