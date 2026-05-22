import os
import re
import polars as pl
import matplotlib.pyplot as plt

OUT = "../out"
DATA = ".."
SAVE = "plots"
os.makedirs(SAVE, exist_ok=True)

COLORS = {"SGD": "#1f77b4", "Momentum": "#ff7f0e", "Adam": "#2ca02c"}


def opt_color(name):
    for k, c in COLORS.items():
        if name.startswith(k):
            return c
    return "#555555"


def history():
    g = {}
    for f in os.listdir(OUT):
        m = re.match(r"(.+)__(.+)\.csv", f)
        if m:
            g.setdefault(m.group(1), {})[m.group(2)] = pl.read_csv(os.path.join(OUT, f))
    return g


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(SAVE, name + ".png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(p)


def plot_curves(ds, runs):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for opt, df in sorted(runs.items()):
        c = opt_color(opt)
        ep = df["epoch"].to_numpy()
        ax[0].plot(ep, df["train_loss"].to_numpy(), color=c, label=opt)
        ax[0].plot(ep, df["test_loss"].to_numpy(), color=c, ls="--", alpha=0.6)
        ax[1].plot(ep, df["test_f1"].to_numpy(), color=c, label=opt)
        ax[2].plot(ep, df["test_acc"].to_numpy(), color=c, label=opt)

    ax[0].set(title="loss (— train, -- test)", xlabel="epoch", ylabel="BCE")
    ax[0].set_yscale("log")
    ax[1].set(title="test F1", xlabel="epoch", ylim=(0, 1.05))
    ax[2].set(title="test accuracy", xlabel="epoch", ylim=(0, 1.05))
    for a in ax:
        a.grid(True, ls="--", alpha=0.3)
        a.legend()
    fig.suptitle(ds)
    save(fig, ds + "_curves")


def plot_corr(ds, df):
    cols = df.columns
    corr = df.corr().to_numpy()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2, 0.9 * n + 1.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), cols, rotation=45, ha="right")
    ax.set_yticks(range(n), cols)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(ds + ": correlation")
    save(fig, ds + "_corr")


def plot_scatter(ds, df):
    if df.width != 3:  # 2 признака + target
        return
    x, y, t = df.columns
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, color in [(0, "#1f77b4"), (1, "#d62728")]:
        s = df.filter(pl.col(t).cast(pl.Int64) == cls)
        ax.scatter(s[x].to_numpy(), s[y].to_numpy(), s=12, color=color, alpha=0.7, label=f"class {cls}")
    ax.set(title=ds + ": classes", xlabel=x, ylabel=y)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    save(fig, ds + "_scatter")


if __name__ == "__main__":
    for ds, runs in sorted(history().items()):
        plot_curves(ds, runs)

    for f in sorted(os.listdir(DATA)):
        if re.match(r"dataset.*\.csv", f):
            df = pl.read_csv(os.path.join(DATA, f))
            plot_corr(f[:-4], df)
            plot_scatter(f[:-4], df)
