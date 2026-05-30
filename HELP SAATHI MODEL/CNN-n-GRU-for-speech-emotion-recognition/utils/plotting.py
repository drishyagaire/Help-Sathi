#!/usr/bin/env python
# coding: utf-8

import json
import os

from utils.common_imports import np, plt, sn, confusion_matrix


def _moving_average(values, window=3):
    if not values:
        return []
    if len(values) < window:
        return list(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def save_history_json(history, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def load_history_json(history_path):
    if not history_path or not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_visual(history, checkpoint_metrics, output_path, model_name, dataset_name):
    history = history or {}
    has_curves = all(history.get(key) for key in ("train_loss", "val_loss", "train_acc", "val_acc"))

    fig = plt.figure(figsize=(13, 6.5), facecolor="#08101d")

    if has_curves:
        epochs = np.arange(1, len(history["train_loss"]) + 1)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor("#101a2e")
            ax.grid(alpha=0.18, color="#c2d5ff")
            ax.tick_params(colors="#e8eefb")
            for spine in ax.spines.values():
                spine.set_color("#39517d")

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        train_acc = history["train_acc"]
        val_acc = history["val_acc"]

        ax1.plot(epochs, train_loss, color="#66c2ff", linewidth=2.2, label="Train loss")
        ax1.plot(epochs, val_loss, color="#ff6b8b", linewidth=2.2, label="Val loss")
        ax1.plot(epochs, _moving_average(train_loss), color="#66c2ff", linestyle="--", alpha=0.35)
        ax1.plot(epochs, _moving_average(val_loss), color="#ff6b8b", linestyle="--", alpha=0.35)
        ax1.set_title("Loss by Epoch", color="#f6f8ff", fontsize=15, fontweight="bold")
        ax1.set_xlabel("Epoch", color="#d9e2f5")
        ax1.set_ylabel("NLL loss", color="#d9e2f5")
        ax1.legend(frameon=False, labelcolor="#eef3ff")

        best_loss_epoch = int(np.argmin(val_loss)) + 1
        ax1.scatter([best_loss_epoch], [val_loss[best_loss_epoch - 1]], color="#ffd166", s=60, zorder=5)
        ax1.annotate(
            f"Best val loss\n{val_loss[best_loss_epoch - 1]:.3f}",
            (best_loss_epoch, val_loss[best_loss_epoch - 1]),
            textcoords="offset points",
            xytext=(10, -18),
            color="#fff3cf",
            fontsize=10,
        )

        ax2.plot(epochs, train_acc, color="#41d6a4", linewidth=2.2, label="Train acc")
        ax2.plot(epochs, val_acc, color="#ffbf47", linewidth=2.2, label="Val acc")
        ax2.plot(epochs, _moving_average(train_acc), color="#41d6a4", linestyle="--", alpha=0.35)
        ax2.plot(epochs, _moving_average(val_acc), color="#ffbf47", linestyle="--", alpha=0.35)
        ax2.set_title("Accuracy by Epoch", color="#f6f8ff", fontsize=15, fontweight="bold")
        ax2.set_xlabel("Epoch", color="#d9e2f5")
        ax2.set_ylabel("Accuracy (%)", color="#d9e2f5")
        ax2.legend(frameon=False, labelcolor="#eef3ff")

        best_acc_epoch = int(np.argmax(val_acc)) + 1
        ax2.scatter([best_acc_epoch], [val_acc[best_acc_epoch - 1]], color="#ffd166", s=60, zorder=5)
        ax2.annotate(
            f"Best val acc\n{val_acc[best_acc_epoch - 1]:.2f}%",
            (best_acc_epoch, val_acc[best_acc_epoch - 1]),
            textcoords="offset points",
            xytext=(10, -18),
            color="#fff3cf",
            fontsize=10,
        )
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#101a2e")
        ax.axis("off")

        lines = [
            f"{model_name} · {dataset_name.upper()}",
            "",
            "Per-epoch training history was not saved for this run.",
            "This panel shows the real checkpoint metrics only.",
            "",
            f"Saved epoch: {checkpoint_metrics.get('epoch', 'N/A')}",
            f"Best validation accuracy: {checkpoint_metrics.get('val_acc', float('nan')):.2f}%",
            f"Best validation loss: {checkpoint_metrics.get('val_loss', float('nan')):.4f}",
        ]
        ax.text(
            0.05,
            0.82,
            "\n".join(lines),
            color="#eef3ff",
            fontsize=16,
            va="top",
            ha="left",
            linespacing=1.5,
            family="monospace",
        )

    fig.suptitle(
        f"{model_name} Training Overview on {dataset_name.title()}",
        color="#ffffff",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_confusion_matrix_visual(y_true, y_pred, classes, output_path, model_name, dataset_name, accuracy_value):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annotations = np.empty_like(cm, dtype=object)

    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            count = cm[row_idx, col_idx]
            percentage = normalized[row_idx, col_idx] * 100
            annotations[row_idx, col_idx] = f"{count}\n{percentage:.1f}%"

    fig = plt.figure(figsize=(14, 7), facecolor="#08101d")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.15], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("#101a2e")
    side.set_facecolor("#101a2e")

    heatmap = sn.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap=sn.color_palette(["#0d1628", "#174b78", "#2d8bc6", "#ffd166"], as_cmap=True),
        cbar=True,
        square=True,
        linewidths=0.8,
        linecolor="#223455",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"fontsize": 9},
    )
    heatmap.collections[0].colorbar.ax.tick_params(colors="#e8eefb")
    ax.set_title("Row-normalized confusion matrix", color="#f6f8ff", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted label", color="#d9e2f5", fontsize=11)
    ax.set_ylabel("True label", color="#d9e2f5", fontsize=11)
    ax.tick_params(axis="x", rotation=25, colors="#e8eefb")
    ax.tick_params(axis="y", rotation=0, colors="#e8eefb")

    side.axis("off")
    diagonal = np.diag(normalized) * 100
    supports = cm.sum(axis=1)
    lines = [
        f"Test accuracy: {accuracy_value:.2f}%",
        "",
        "Per-class recall",
    ]
    lines.extend(
        f"{label:<9} {recall:>6.2f}%  n={support}"
        for label, recall, support in zip(classes, diagonal, supports)
    )
    side.text(
        0.08,
        0.92,
        "\n".join(lines),
        color="#eef3ff",
        fontsize=12,
        va="top",
        ha="left",
        linespacing=1.5,
        family="monospace",
    )

    fig.suptitle(
        f"{model_name} Confusion Matrix on {dataset_name.title()}",
        color="#ffffff",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08, wspace=0.16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
