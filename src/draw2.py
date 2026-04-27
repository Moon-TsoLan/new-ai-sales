import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# ---------- 风格设置 ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.4,
    "text.usetex": False,
})
sns.set_style("ticks")

# ---------- 读取数据 ----------
project_root = Path(__file__).resolve().parent.parent
report_dir = project_root / "reports" / "linux_bert_ner_agent_lr1e5_bs32"
df = pd.read_csv(report_dir / "epoch_metrics.csv")
epochs = df["epoch"].values


def parse_label_metrics(report_path: Path) -> dict:
    """Parse BRAND/COLOR/FABRIC precision/recall/f1 from one report txt."""
    metrics = {}
    pattern = re.compile(
        r"^\s*(BRAND|COLOR|FABRIC)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s*$"
    )
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            label = match.group(1)
            metrics[label] = {
                "precision": float(match.group(2)),
                "recall": float(match.group(3)),
                "f1": float(match.group(4)),
            }
    return metrics

# ========== 图1：训练与验证损失曲线 ==========
color_train = "#2c7bb6"  # 深蓝
color_val = "#d7191c"    # 深红

fig1, ax1 = plt.subplots(figsize=(8, 4.2))
ax1.plot(
    epochs,
    df["train_loss"],
    color=color_train,
    marker="o",
    markersize=4,
    linewidth=1.2,
    label="Training Loss",
    markeredgewidth=0.5,
    markeredgecolor="white",
)
ax1.plot(
    epochs,
    df["val_loss"],
    color=color_val,
    marker="s",
    markersize=4,
    linewidth=1.2,
    label="Validation Loss",
    markeredgewidth=0.5,
    markeredgecolor="white",
)

best_loss_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
best_val_loss = float(df["val_loss"].min())
ax1.axvline(x=best_loss_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.annotate(
    f"Best Val Loss: {best_val_loss:.4f} (Ep.{best_loss_epoch})",
    xy=(best_loss_epoch, best_val_loss),
    xytext=(best_loss_epoch + 0.8, best_val_loss * 1.45),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    fontsize=9,
    color="gray",
)

ax1.set_ylabel("Loss")
ax1.set_xlabel("Epoch")
ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray")
ax1.grid(True, linestyle="--", linewidth=0.5)
ax1.set_title(
    "NER Training and Validation Loss",
    loc="center",
    fontweight="semibold",
    fontname="DejaVu Serif",
)
ax1.set_xticks(range(1, int(df["epoch"].max()) + 1, 1))
ax1.set_xlim(1, int(df["epoch"].max()))
ax1.set_yscale("log")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig1.tight_layout()
fig1.savefig(report_dir / "3.7figure_ner_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# ========== 图2：验证 F1 曲线 ==========
color_f1 = "#2ca25f"  # 绿色

fig2, ax2 = plt.subplots(figsize=(8, 4.2))
ax2.plot(
    epochs,
    df["val_f1"] * 100,
    color=color_f1,
    marker="D",
    markersize=4,
    linewidth=1.2,
    label="Validation F1",
    markeredgewidth=0.5,
    markeredgecolor="white",
)

best_f1_epoch = int(df.loc[df["val_f1"].idxmax(), "epoch"])
best_f1 = float(df["val_f1"].max() * 100)
ax2.axvline(x=best_f1_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.annotate(
    f"Best Val F1: {best_f1:.2f}% (Ep.{best_f1_epoch})",
    xy=(best_f1_epoch, best_f1),
    xytext=(best_f1_epoch + 0.8, best_f1 - 3.0),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    fontsize=9,
    color="gray",
)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1 Score (%)")
ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="gray")
ax2.grid(True, linestyle="--", linewidth=0.5)
ax2.set_title(
    "NER Validation F1 Curve",
    loc="center",
    fontweight="semibold",
    fontname="DejaVu Serif",
)
ax2.set_xticks(range(1, int(df["epoch"].max()) + 1, 1))
ax2.set_xlim(1, int(df["epoch"].max()))
ax2.set_ylim(max(0, df["val_f1"].min() * 100 - 5), min(101, df["val_f1"].max() * 100 + 2))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig2.tight_layout()
fig2.savefig(report_dir / "3.7figure_ner_f1_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

# ========== 图3：各标签 F1 曲线 ==========
label_records = []
for epoch in sorted(df["epoch"].astype(int).tolist()):
    report_path = report_dir / f"classification_report_epoch_{epoch}.txt"
    if not report_path.exists():
        continue
    parsed = parse_label_metrics(report_path)
    label_records.append(
        {
            "epoch": epoch,
            "brand_f1": parsed.get("BRAND", {}).get("f1", 0.0) * 100,
            "color_f1": parsed.get("COLOR", {}).get("f1", 0.0) * 100,
            "fabric_f1": parsed.get("FABRIC", {}).get("f1", 0.0) * 100,
        }
    )

if label_records:
    label_df = pd.DataFrame(label_records)
    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    ax3.plot(
        label_df["epoch"],
        label_df["brand_f1"],
        color="#1f78b4",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="BRAND F1",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.plot(
        label_df["epoch"],
        label_df["color_f1"],
        color="#33a02c",
        marker="s",
        markersize=4,
        linewidth=1.2,
        label="COLOR F1",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.plot(
        label_df["epoch"],
        label_df["fabric_f1"],
        color="#e6550d",
        marker="^",
        markersize=4,
        linewidth=1.2,
        label="FABRIC F1",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )

    best_label_epoch = int(
        label_df.loc[label_df[["brand_f1", "color_f1", "fabric_f1"]].mean(axis=1).idxmax(), "epoch"]
    )
    mean_best = float(
        label_df.loc[label_df["epoch"] == best_label_epoch, ["brand_f1", "color_f1", "fabric_f1"]]
        .mean(axis=1)
        .values[0]
    )
    ax3.axvline(x=best_label_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.annotate(
        f"Best Mean F1 Epoch: {best_label_epoch}\nMean F1: {mean_best:.2f}%",
        xy=(best_label_epoch, mean_best),
        xytext=(best_label_epoch + 0.8, mean_best - 6),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        fontsize=9,
        color="gray",
    )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1 Score (%)")
    ax3.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="gray")
    ax3.grid(True, linestyle="--", linewidth=0.5)
    ax3.set_title(
        "NER Label-wise F1 Curves (BRAND/COLOR/FABRIC)",
        loc="center",
        fontweight="semibold",
        fontname="DejaVu Serif",
    )
    ax3.set_xticks(range(1, int(df["epoch"].max()) + 1, 1))
    ax3.set_xlim(1, int(df["epoch"].max()))
    y_min = min(label_df["brand_f1"].min(), label_df["color_f1"].min(), label_df["fabric_f1"].min())
    y_max = max(label_df["brand_f1"].max(), label_df["color_f1"].max(), label_df["fabric_f1"].max())
    ax3.set_ylim(max(0, y_min - 5), min(101, y_max + 3))
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig3.tight_layout()
    fig3.savefig(report_dir / "3.7figure_ner_labelwise_f1_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

plt.show()
