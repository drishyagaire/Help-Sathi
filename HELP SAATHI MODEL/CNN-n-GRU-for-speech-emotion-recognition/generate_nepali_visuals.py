#!/usr/bin/env python
# coding: utf-8

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

import config
from datasets import get_dataloader, get_emotion_classes
from models import CNN5GRU, CNN18GRU
from utils.plotting import (
    load_history_json,
    plot_confusion_matrix_visual,
    plot_training_visual,
)


MODEL_SPECS = {
    "cnn18gru": {
        "display_name": "CNN18GRU",
        "class": CNN18GRU,
        "checkpoint": "experiments/nepali/cnn18gru/best_model.pth",
        "history": "experiments/nepali/cnn18gru/history.json",
        "training_plot": "experiments/nepali/CNN18GRU_training_curves.png",
        "confusion_plot": "experiments/nepali/CNN18GRU_confusion_matrix.png",
    },
    "cnn5gru": {
        "display_name": "CNN5GRU",
        "class": CNN5GRU,
        "checkpoint": "experiments/nepali/cnn5gru/best_model.pth",
        "history": "experiments/nepali/cnn5gru/history.json",
        "training_plot": "experiments/nepali/CNN5GRU_training_curves.png",
        "confusion_plot": "experiments/nepali/CNN5GRU_confusion_matrix.png",
    },
}


def get_cpu_device():
    return torch.device("cpu")


def build_model(model_class, num_classes, device):
    model = model_class(
        n_input=1,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.GRU_LAYERS,
        n_output=num_classes,
        dropout=0.0,
    )
    return model.to(device)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    return checkpoint


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            hidden = model.init_hidden(data.size(0), device)
            output, _ = model(data, hidden)
            preds = output.argmax(dim=-1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    accuracy = (100.0 * correct / total) if total else 0.0
    return accuracy, all_targets, all_preds


def main():
    if config.DATASET.lower() != "nepali":
        raise ValueError("This script is only intended for the Nepali dataset. Set DATASET='nepali' in config.py.")

    device = get_cpu_device()
    _, _, test_loader, emotion_classes = get_dataloader(
        batch_size=config.BATCH_SIZE,
        shuffle_train=False,
        drop_last=False,
        num_workers=0,
    )

    print(f"Dataset: {config.DATASET}")
    print(f"Emotion classes: {emotion_classes}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Device: {device}")

    for model_key, spec in MODEL_SPECS.items():
        checkpoint_path = spec["checkpoint"]
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {spec['display_name']}: missing checkpoint at {checkpoint_path}")
            continue

        checkpoint = load_checkpoint(checkpoint_path, device)
        model = build_model(spec["class"], len(emotion_classes), device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
        checkpoint_metrics = {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_acc": checkpoint.get("val_acc"),
            "test_acc": test_acc,
        }

        history = load_history_json(spec["history"])
        plot_training_visual(
            history=history,
            checkpoint_metrics=checkpoint_metrics,
            output_path=spec["training_plot"],
            model_name=spec["display_name"],
            dataset_name=config.DATASET,
        )
        plot_confusion_matrix_visual(
            y_true=y_true,
            y_pred=y_pred,
            classes=get_emotion_classes(),
            output_path=spec["confusion_plot"],
            model_name=spec["display_name"],
            dataset_name=config.DATASET,
            accuracy_value=test_acc,
        )

        print(
            f"{spec['display_name']}: "
            f"epoch={checkpoint_metrics['epoch']}, "
            f"val_loss={checkpoint_metrics['val_loss']:.4f}, "
            f"val_acc={checkpoint_metrics['val_acc']:.2f}%, "
            f"test_acc={test_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
