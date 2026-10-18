"""Generate basic exploratory data analysis outputs for pcos_cleaned.csv."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "pcos_cleaned.csv"
OUT = ROOT / "eda_outputs"
TARGET = "PCOS (Y/N)"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    features = frame.drop(columns=[TARGET], errors="ignore")
    features = features.drop(columns=[c for c in features if c.lower().startswith("unnamed") or "file no" in c.lower()], errors="ignore")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    OUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = frame[TARGET].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#6c63ff")
    ax.set(title="PCOS target distribution", xlabel="PCOS (0 = No, 1 = Yes)", ylabel="Records")
    fig.tight_layout()
    fig.savefig(OUT / "target_distribution.png", dpi=160)
    plt.close(fig)

    missing = numeric.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, len(missing) * 0.28)))
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#f4a261")
        ax.set(title="Missing values by input", xlabel="Missing values (%)", ylabel="")
        fig.tight_layout()
        fig.savefig(OUT / "missingness.png", dpi=160)
        plt.close(fig)

    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title("Clinical input correlations")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    summary = {
        "source": DATA_PATH.name,
        "rows": int(frame.shape[0]),
        "input_features": int(numeric.shape[1]),
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "missing_percent_by_feature": {str(k): float(v) for k, v in missing.items()},
        "numeric_summary": numeric.describe().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"EDA outputs written to {OUT}")


if __name__ == "__main__":
    main()
