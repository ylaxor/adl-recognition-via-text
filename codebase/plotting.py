from pathlib import Path

from matplotlib import rcParams
from matplotlib.pyplot import savefig, show, subplots
from numpy import arange, nan

rcParams["font.family"] = "serif"
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 13
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["legend.fontsize"] = 10
rcParams["figure.titlesize"] = 14


def plot_transfer_results(
    res: dict,
    save_path: str | None = None,
    figsize=(10, 6),
    title: str = "Cross-Dataset Transfer Learning Performance",
):
    if not res:
        raise ValueError("Results dictionary is empty")

    src_datasets = sorted(list(res.keys()), key=lambda x: str(x))
    tgt_datasets_set = set()
    for source in src_datasets:
        if isinstance(res[source], dict):
            tgt_datasets_set.update(res[source].keys())
    tgt_datasets = sorted(list(tgt_datasets_set), key=lambda x: str(x))
    if not tgt_datasets:
        raise ValueError("No target datasets found in results")

    def clean_name(ds):
        name = str(ds).split(".")[-1].replace("'", "").replace(">", "")
        return name

    tgt_labels = [clean_name(ds) for ds in tgt_datasets]
    x_positions = arange(len(tgt_datasets))
    colors = [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
        "#AA4499",
    ]
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]
    linestyles = ["-", "--", "-.", ":"]
    fig, ax = subplots(figsize=figsize, constrained_layout=True)
    for idx, source in enumerate(src_datasets):
        src_label = clean_name(source)
        scores = [res[source].get(target, nan) for target in tgt_datasets]
        ax.plot(
            x_positions,
            scores,
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.5,
            color=colors[idx % len(colors)],
            alpha=0.85,
            zorder=1,
            label=src_label,
        )

        for i, (target, score) in enumerate(zip(tgt_datasets, scores)):
            if target == source:
                ax.plot(
                    x_positions[i],
                    score,
                    marker=markers[idx % len(markers)],
                    markersize=11,
                    markerfacecolor=colors[idx % len(colors)],
                    markeredgecolor="black",
                    markeredgewidth=2.0,
                    zorder=3,
                )
            else:
                ax.plot(
                    x_positions[i],
                    score,
                    marker=markers[idx % len(markers)],
                    markersize=8,
                    markerfacecolor=colors[idx % len(colors)],
                    markeredgecolor=colors[idx % len(colors)],
                    markeredgewidth=0.5,
                    alpha=0.9,
                    zorder=2,
                )
    ax.set_xlabel("Target Dataset", fontweight="bold", fontsize=13)
    ax.set_ylabel("Balanced Accuracy", fontweight="bold", fontsize=13)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tgt_labels, fontsize=12, fontweight="normal")
    ax.set_xlim(-0.3, len(tgt_datasets) - 0.7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.8, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(
        title="Source Dataset",
        title_fontsize=11,
        fontsize=10,
        loc="best",
        framealpha=0.98,
        edgecolor="#333333",
        fancybox=False,
        shadow=False,
        borderpad=0.8,
        labelspacing=0.6,
    )
    legend.get_title().set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")
    ax.text(
        0.02,
        0.98,
        "Note: Markers with black borders indicate self-evaluation",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC", alpha=0.9
        ),
    )

    if save_path:
        path = Path(save_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: could not create directory for {save_path}: {e}")

        savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {path}")
    else:
        show()

    return fig, ax
