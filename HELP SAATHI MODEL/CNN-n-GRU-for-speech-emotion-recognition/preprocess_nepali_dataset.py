import os
from pathlib import Path

import pandas as pd
import soundfile as sf

import config
from utils.audio_preprocessing import inspect_audio_file

RAW_DATASET_DIR = Path(os.environ.get("RAW_NEPALI_DATA_PATH", "dataset/nepali_dataset"))
OUTPUT_DATASET_DIR = Path(config.NEPALI_DATA_FOLDER)
OUTPUT_CSV = Path(config.NEPALI_DATA_CSV)
REJECTED_CSV = OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}_rejected.csv")

EMOTIONS = set(config.NEPALI_EMOTIONS)
SAMPLE_RATE = config.NEPALI_SAMPLE_RATE
MAX_RAW_DURATION = config.NEPALI_MAX_RAW_DURATION_SECONDS
MIN_CLEAN_DURATION = config.NEPALI_MIN_CLEAN_DURATION_SECONDS
TRIM_TOP_DB = config.NEPALI_TRIM_TOP_DB


def preprocess_dataset():
    records = []
    rejected = []

    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(RAW_DATASET_DIR):
        emotion = Path(root).name.lower().strip()
        if emotion not in EMOTIONS:
            continue

        emotion_output_dir = OUTPUT_DATASET_DIR / emotion
        emotion_output_dir.mkdir(parents=True, exist_ok=True)

        for file_name in sorted(files):
            source_path = Path(root) / file_name

            try:
                info = inspect_audio_file(
                    audio_path=source_path.as_posix(),
                    sample_rate=SAMPLE_RATE,
                    top_db=TRIM_TOP_DB,
                )
                raw_duration = info["raw_duration_seconds"]
                clean_duration = info["clean_duration_seconds"]

                if raw_duration > MAX_RAW_DURATION:
                    raise ValueError(
                        f"raw duration {raw_duration:.2f}s exceeds {MAX_RAW_DURATION:.2f}s"
                    )
                if clean_duration < MIN_CLEAN_DURATION:
                    raise ValueError(
                        f"clean duration {clean_duration:.2f}s is below {MIN_CLEAN_DURATION:.2f}s"
                    )

                output_path = emotion_output_dir / f"{source_path.stem}.wav"
                sf.write(output_path.as_posix(), info["audio"], SAMPLE_RATE)

                records.append(
                    {
                        "path": output_path.as_posix(),
                        "labels": emotion,
                        "source": "NEPALI",
                        "original_path": source_path.as_posix(),
                        "raw_duration_seconds": round(raw_duration, 4),
                        "clean_duration_seconds": round(clean_duration, 4),
                    }
                )
            except Exception as exc:
                rejected.append(
                    {
                        "path": source_path.as_posix(),
                        "labels": emotion,
                        "reason": str(exc),
                    }
                )

    df = pd.DataFrame(records)
    rejected_df = pd.DataFrame(rejected)

    df.to_csv(OUTPUT_CSV, index=False)
    rejected_df.to_csv(REJECTED_CSV, index=False)

    print(f"Saved cleaned dataset CSV -> {OUTPUT_CSV}")
    print(f"Saved rejected clip report -> {REJECTED_CSV}")
    print(f"Usable clips: {len(df)}")
    if not df.empty:
        print(df["labels"].value_counts().sort_index())
    print(f"Rejected clips: {len(rejected_df)}")
    if not rejected_df.empty:
        print(rejected_df["reason"].value_counts().head(10))


if __name__ == "__main__":
    preprocess_dataset()
