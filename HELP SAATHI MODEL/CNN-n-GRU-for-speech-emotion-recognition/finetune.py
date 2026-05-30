import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, ".")

import config
from models.cnn_n_gru import CNN18GRU
from utils.audio_preprocessing import (
    SUPPORTED_AUDIO_EXTENSIONS,
    load_and_prepare_audio,
)

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
CHECKPOINT_PATH = "experiments/tess/cnn18gru/best_model_4.pth"
NEPALI_DATA_PATH = os.environ.get("NEPALI_DATA_PATH", "dataset/nepali_dataset")
SAVE_PATH = "experiments/nepali/cnn18gru/"
os.makedirs(SAVE_PATH, exist_ok=True)

SAMPLE_RATE = 16000
TARGET_DURATION = 3
MAX_RAW_DURATION = 7.0
MIN_CLEAN_DURATION = 0.35
TRIM_TOP_DB = 25
BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
DISTRESS_EMOTIONS = ["angry", "fear", "disgust", "sad"]

print(f"Device: {DEVICE}")
print(f"Classes: {len(EMOTIONS)} -> {EMOTIONS}")


def _summarize_durations(df, column):
    values = df[column].to_numpy(dtype=float)
    if len(values) == 0:
        return "n/a"
    percentiles = np.percentile(values, [50, 90, 95, 99])
    return (
        f"min={values.min():.2f}s | p50={percentiles[0]:.2f}s | "
        f"p90={percentiles[1]:.2f}s | p95={percentiles[2]:.2f}s | "
        f"p99={percentiles[3]:.2f}s | max={values.max():.2f}s"
    )


def load_nepali_dataset(data_path):
    """
    Expected folder structure:
    dataset/nepali_dataset/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
    """
    records = []
    rejected = []

    for root, _, files in os.walk(data_path):
        emotion = os.path.basename(root).lower().strip()
        if emotion not in EMOTION_TO_IDX:
            continue

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in SUPPORTED_AUDIO_EXTENSIONS:
                continue

            full_path = os.path.join(root, file_name)

            try:
                _, info = load_and_prepare_audio(
                    audio_path=full_path,
                    sample_rate=SAMPLE_RATE,
                    target_duration_seconds=TARGET_DURATION,
                    max_raw_duration_seconds=MAX_RAW_DURATION,
                    min_clean_duration_seconds=MIN_CLEAN_DURATION,
                    top_db=TRIM_TOP_DB,
                    normalize=False,
                )
                records.append(
                    {
                        "path": full_path,
                        "emotion": emotion,
                        "label": EMOTION_TO_IDX[emotion],
                        "raw_duration_seconds": info["raw_duration_seconds"],
                        "clean_duration_seconds": info["clean_duration_seconds"],
                    }
                )
            except Exception as exc:
                rejected.append(
                    {
                        "path": full_path,
                        "emotion": emotion,
                        "reason": str(exc),
                    }
                )

    df = pd.DataFrame(records)
    rejected_df = pd.DataFrame(rejected)

    if rejected_df.empty is False:
        rejected_path = os.path.join(SAVE_PATH, "filtered_out.csv")
        rejected_df.to_csv(rejected_path, index=False)
        print(f"\nFiltered out {len(rejected_df)} clips -> {rejected_path}")
        print(rejected_df["reason"].value_counts().head(10))

    if df.empty:
        raise ValueError(f"No valid audio files found in {data_path}")

    manifest_path = os.path.join(SAVE_PATH, "nepali_manifest.csv")
    df.to_csv(manifest_path, index=False)

    print(f"\nNepali dataset loaded: {len(df)} usable files")
    print(df["emotion"].value_counts().sort_index())
    print(f"Raw duration stats:    {_summarize_durations(df, 'raw_duration_seconds')}")
    print(
        f"Clean duration stats:  {_summarize_durations(df, 'clean_duration_seconds')}"
    )
    print(f"Saved manifest -> {manifest_path}")
    return df


class NepaliEmotionDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            audio, _ = load_and_prepare_audio(
                audio_path=row["path"],
                sample_rate=SAMPLE_RATE,
                target_duration_seconds=TARGET_DURATION,
                max_raw_duration_seconds=MAX_RAW_DURATION,
                min_clean_duration_seconds=MIN_CLEAN_DURATION,
                top_db=TRIM_TOP_DB,
                normalize=True,
            )

            if self.augment:
                audio = self.augment_audio(audio)

            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(row["label"], dtype=torch.long)
            return waveform, label

        except Exception as exc:
            print(f"Error loading {row['path']}: {exc}")
            return torch.zeros(1, SAMPLE_RATE * TARGET_DURATION), torch.tensor(
                row["label"], dtype=torch.long
            )

    def augment_audio(self, audio):
        augmented = np.copy(audio)

        if np.random.random() < 0.4:
            gain = np.random.uniform(0.9, 1.1)
            augmented = augmented * gain

        if np.random.random() < 0.4:
            augmented = augmented + np.random.randn(len(augmented)) * 0.002

        peak = np.max(np.abs(augmented))
        if peak > 0:
            augmented = augmented / peak

        return augmented.astype(np.float32, copy=False)


def load_pretrained_model():
    print("\nLoading TESS checkpoint...")

    model = CNN18GRU(
        n_input=1,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.GRU_LAYERS,
        n_output=len(EMOTIONS),
        stride=4,
        n_channel=18,
        dropout=0.2,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if name in {
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "conv3.weight",
            "conv3.bias",
            "bn1.weight",
            "bn1.bias",
            "bn2.weight",
            "bn2.bias",
            "bn3.weight",
            "bn3.bias",
        }:
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
            trainable += 1

    print("Checkpoint loaded and adapted for Nepali fine-tuning")
    print(f"Frozen params: {frozen} | Trainable params: {trainable}")
    return model.to(DEVICE)


def fine_tune():
    df = load_nepali_dataset(NEPALI_DATA_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    train_ds = NepaliEmotionDataset(train_df, augment=True)
    val_ds = NepaliEmotionDataset(val_df, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model = load_pretrained_model()

    optimizer = optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )
    criterion = nn.NLLLoss()

    best_val_acc = 0.0

    print(f"\n{'=' * 50}")
    print("FINE-TUNING ON NEPALI DATASET")
    print(f"{'=' * 50}\n")

    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0

        for waveform, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
        ):
            waveform = waveform.to(DEVICE)
            labels = labels.to(DEVICE)

            hidden = model.init_hidden(waveform.size(0), DEVICE)

            optimizer.zero_grad()
            output, _ = model(waveform, hidden)
            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1) * 100

        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for waveform, labels in val_loader:
                waveform = waveform.to(DEVICE)
                labels = labels.to(DEVICE)
                hidden = model.init_hidden(waveform.size(0), DEVICE)
                output, _ = model(waveform, hidden)
                preds = output.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1) * 100
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {train_loss / max(len(train_loader), 1):.4f} | "
            f"Train: {train_acc:.1f}% | "
            f"Val: {val_acc:.1f}%",
            end="",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "emotions": EMOTIONS,
                    "dataset_path": NEPALI_DATA_PATH,
                    "preprocessing": {
                        "sample_rate": SAMPLE_RATE,
                        "target_duration_seconds": TARGET_DURATION,
                        "max_raw_duration_seconds": MAX_RAW_DURATION,
                        "min_clean_duration_seconds": MIN_CLEAN_DURATION,
                        "trim_top_db": TRIM_TOP_DB,
                    },
                },
                os.path.join(SAVE_PATH, "best_model.pth"),
            )
            print(" ✅ SAVED")
        else:
            print()

    print(f"\nDistress emotions: {DISTRESS_EMOTIONS}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(SAVE_PATH, 'best_model.pth')}")


if __name__ == "__main__":
    fine_tune()
