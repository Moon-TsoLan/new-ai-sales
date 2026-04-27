import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path

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
sns.set_style("ticks")  # 带边框的网格样式

# ---------- 读取数据 ----------
project_root = Path(__file__).resolve().parent.parent
report_dir = project_root / "reports" / "linux_bert_classifier_lr5e6_bs32"
df = pd.read_csv(report_dir / "epoch_metrics.csv")
epochs = df["epoch"].values

# ========== 图1：训练与验证损失曲线 ==========
color_train = "#2c7bb6"  # 深蓝
color_val   = "#d7191c"  # 深红

fig1, ax1 = plt.subplots(figsize=(8, 4.2))
ax1.plot(epochs, df["train_loss"], color=color_train, marker="o", markersize=4,
         linewidth=1.2, label="Training Loss", markeredgewidth=0.5, markeredgecolor="white")
ax1.plot(epochs, df["val_loss"], color=color_val, marker="s", markersize=4,
         linewidth=1.2, label="Validation Loss", markeredgewidth=0.5, markeredgecolor="white")

# 标注最佳轮次（统一按验证损失最小）
best_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
best_val = float(df["val_loss"].min())
ax1.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.annotate(
    f"Best Val Loss: {best_val:.4f} (Ep.{best_epoch})",
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 1, best_val * 1.35),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    fontsize=9,
    color="gray",
)

ax1.set_ylabel("Loss")
ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray")
ax1.grid(True, linestyle="--", linewidth=0.5)
ax1.set_title(
    "Training and Validation Loss",
    loc="center",
    fontweight="semibold",
    fontname="DejaVu Serif",
)
ax1.set_xlabel("Epoch")
ax1.set_xticks(range(1, int(df["epoch"].max()) + 1, 2))
ax1.set_xlim(1, int(df["epoch"].max()))
ax1.set_yscale("log")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
fig1.tight_layout()
fig1.savefig(report_dir / "3.6figure_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# ========== 图2：大类与子类准确率曲线 ==========
color_cat = "#2ca25f"   # 绿色
color_sub = "#e6550d"   # 橙色

fig2, ax2 = plt.subplots(figsize=(8, 4.2))
ax2.plot(epochs, df["category_acc"] * 100, color=color_cat, marker="D", markersize=4,
         linewidth=1.2, label="Category Accuracy", markeredgewidth=0.5, markeredgecolor="white")
ax2.plot(epochs, df["sub_acc"] * 100, color=color_sub, marker="^", markersize=4,
         linewidth=1.2, label="Sub-category Accuracy", markeredgewidth=0.5, markeredgecolor="white")

# 标注最佳轮次（与图1统一：使用验证损失最小轮次）
best_cat_acc = float(df.loc[df["epoch"] == best_epoch, "category_acc"].values[0] * 100)
best_sub_acc = float(df.loc[df["epoch"] == best_epoch, "sub_acc"].values[0] * 100)
ax2.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.annotate(
    f"Best Epoch {best_epoch}\nCat: {best_cat_acc:.2f}% | Sub: {best_sub_acc:.2f}%",
    xy=(best_epoch, best_sub_acc),
    xytext=(best_epoch + 1.0, best_sub_acc - 2.4),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    fontsize=9,
    color="gray",
)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="gray")
ax2.grid(True, linestyle="--", linewidth=0.5)
ax2.set_title(
    "Category and Sub-category Accuracy",
    loc="center",
    fontweight="semibold",
    fontname="DejaVu Serif",
)

# 横轴刻度
ax2.set_xticks(range(1, int(df["epoch"].max()) + 1, 2))
ax2.set_xlim(1, int(df["epoch"].max()))

# 纵轴微调
ax2.set_ylim(min(df["sub_acc"].min() * 100, df["category_acc"].min() * 100) - 2, 101)

# 移除顶部和右侧边框
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ---------- 保存 ----------
fig2.tight_layout()
fig2.savefig(report_dir / "3.6figure_accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

plt.show()