import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

OUT_DIR = "../out"
SAVE_DIR = "../visualization/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def group_files():
    groups = defaultdict(dict)
    pattern = re.compile(r"^(.+)__(.+)\.csv$")
    for fname in os.listdir(OUT_DIR):
        m = pattern.match(fname)
        if m:
            ds, opt = m.group(1), m.group(2)
            groups[ds][opt] = os.path.join(OUT_DIR, fname)
    return groups


def plot_dataset(ds_name, methods, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Обучение нейронной сети — {ds_name}", fontsize=14, fontweight="bold")

    ax_loss, ax_f1 = axes

    for color, (opt_name, path) in zip(COLORS, methods.items()):
        df = pd.read_csv(path)
        label = opt_name.replace("_", " ")
        ax_loss.plot(df["epoch"], df["train_loss"], color=color, label=label, linewidth=1.8)
        ax_f1.plot(df["epoch"], df["test_f1"], color=color, label=label, linewidth=1.8)

    ax_loss.set_title("Train Loss")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, which="both", ls="--", alpha=0.4)

    ax_f1.set_title("Test F1-score")
    ax_f1.set_xlabel("Эпоха")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.legend()
    ax_f1.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{ds_name}.png")
    plt.savefig(path, dpi=150)
    print(f"Сохранён: {path}")
    plt.close()


if __name__ == "__main__":
    groups = group_files()
    if not groups:
        print(f"CSV не найдены в {OUT_DIR}. Сначала запустите go run ./cmd/metopt-lab3/")
    else:
        for ds_name, methods in sorted(groups.items()):
            plot_dataset(ds_name, methods, SAVE_DIR)
        print("Готово.")
