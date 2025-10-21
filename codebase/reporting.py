import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from codebase.mailing import send_email


def get_cm(trues, preds) -> str:
    names = sorted(set(trues) | set(preds))
    cm = confusion_matrix(trues, preds)
    cm_str = "\n"
    cm_str += "-" * 50 + "\n"
    max_name_width = max(len(name) for name in names)
    max_value_width = max(len(str(cm.max())), 4)
    cell_width = max(8, max_name_width + 1, max_value_width + 1)
    cm_str += " " * (max_name_width + 2)
    for name in names:
        cm_str += f"{name:>{cell_width}}"
    cm_str += "\n"
    cm_str += "-" * (max_name_width + 2 + cell_width * len(names)) + "\n"
    for i, name in enumerate(names):
        cm_str += f"{name:<{max_name_width}} | "
        for j in range(len(names)):
            cm_str += f"{cm[i, j]:>{cell_width}}"
        cm_str += "\n"
    return cm_str


def get_report(trues, preds) -> str:
    return str(
        classification_report(
            trues,
            preds,
            zero_division=0,
            digits=4,
        )
    )


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_xp_title(xp_title: str):
    parts = [p.strip() for p in xp_title.split("|")]
    if len(parts) != 3:
        raise ValueError(f"Unexpected xp_title format: {xp_title}")
    use_other = "with" in parts[0].lower()
    textualizer = parts[1]
    encoder = parts[2]
    return use_other, textualizer, encoder


def map_family(textualizer: str):
    name = textualizer.lower()
    sequential_keywords = ["tdost", "tsb", "tst", "temporal", "basic"]
    if any(k in name for k in sequential_keywords) and "plain" not in name:
        return "sequential"
    if "plain" in name:
        return "summarized"
    summarized_keywords = ["dts", "cts", "css", "summary"]
    if any(k in name for k in summarized_keywords):
        return "summarized"
    return "summarized"


def to_tidy_dataframe(all_results: dict) -> pd.DataFrame:
    rows = []
    for xp_title, xp_result in all_results.items():
        use_other, textualizer, encoder = parse_xp_title(xp_title)
        for src, tgt_dict in xp_result.items():
            for tgt, acc in tgt_dict.items():
                setting = "in_domain" if str(src) == str(tgt) else "cross_domain"
                rows.append(
                    {
                        "use_other": use_other,
                        "textualizer": textualizer,
                        "encoder": encoder,
                        "source": str(src),
                        "target": str(tgt),
                        "setting": setting,
                        "balanced_accuracy": float(acc),
                        "family": map_family(textualizer),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["use_other", "setting", "textualizer", "encoder", "source", "target"]
        ).reset_index(drop=True)
    return df


def analyze_and_send_results(results, outdir):
    print("hi", outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = to_tidy_dataframe(results)
    tidy_path = os.path.join(outdir, "tidy_results.csv")
    df.to_csv(tidy_path, index=False)
    print(f"[Data] Saved tidy CSV -> {tidy_path}")
    attachments = {
        "tidy_results.csv": open(tidy_path, "rb").read(),
    }
    send_email(
        subject="Experiment Results - Analysis Files",
        body=f"Attached are the analysis results from the experiment.\n\nTotal files: {len(attachments)}",
        attachments=attachments,
    )
    print(f"[Email] Sent {len(attachments)} files from {outdir}")
