from collections import Counter
from os import environ
from pathlib import Path
from pickle import dumps, loads
from warnings import filterwarnings

from matplotlib import style
from matplotlib.pyplot import close, savefig, subplots, tight_layout
from numpy import array, vstack
from seaborn import color_palette, set_palette
from streamlit import (
    button,
    cache_data,
    cache_resource,
    columns,
    dataframe,
    expander,
    header,
    markdown,
    metric,
    multiselect,
    number_input,
    pyplot,
    selectbox,
    set_page_config,
    sidebar,
    spinner,
    subheader,
    success,
    title,
    warning,
    write,
)
from toml import load as load_toml
from umap import UMAP

from codebase.domain import TestBed
from codebase.loading import read_dataset, read_sensors_metadata
from codebase.textualization import TextualizerType, WindowTextualizerFactory
from codebase.vectorization import SentenceVectorizerFactory, SentenceVectorizerType
from codebase.windowing import get_ground_truth_windows

environ["TOKENIZERS_PARALLELISM"] = "false"
filterwarnings("ignore")
style.use("seaborn-v0_8-darkgrid")
set_palette("husl")

set_page_config(page_title="Activity Analysis", layout="wide")


@cache_resource
def load_configs():
    cfg = load_toml("config/experiment.toml")
    data_cfg = load_toml("config/data.toml")
    return cfg, data_cfg


@cache_data
def load_dataset_cached(
    dataset_name, nb_days, simplify_sensors, simplify_activities, rename_activities
):
    dataset = TestBed[dataset_name]
    sen_acts, act_occs = read_dataset(
        dataset,
        days_to_include=nb_days,
        simplify_sensors=simplify_sensors,
        simplify_activities=simplify_activities,
        rename_activities=rename_activities,
    )
    sen_meta = read_sensors_metadata(dataset)

    sen_acts_pickled = dumps(sen_acts)
    act_occs_pickled = dumps(act_occs)

    return sen_acts_pickled, act_occs_pickled, sen_meta


@cache_data
def get_windows_cached(sen_acts_pickled, act_occs_pickled, use_other):
    sen_acts = loads(sen_acts_pickled)
    act_occs = loads(act_occs_pickled)

    windows = get_ground_truth_windows(sen_acts, act_occs, use_idle=use_other)
    return windows


def get_vectorizer_args(vect_type, cfg):
    if vect_type == SentenceVectorizerType.TFIDF:
        args = cfg["tfidf_args"].copy()
        if "ngram_range" in args:
            args["ngram_range"] = tuple(args["ngram_range"])
        return args
    elif vect_type == SentenceVectorizerType.WORD2VEC:
        return cfg["word2vec_args"]
    elif vect_type == SentenceVectorizerType.SBERT:
        return cfg["sbert_args"]
    return {}


def generate_example_sentences(
    dataset,
    textualization,
    nb_days,
    simplify_sensors,
    simplify_activities,
    rename_activities,
    use_other,
    max_examples=3,
):
    output_dir = Path("output/exploration/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    sen_acts_pickled, act_occs_pickled, sen_meta = load_dataset_cached(
        dataset.name, nb_days, simplify_sensors, simplify_activities, rename_activities
    )

    windows = get_windows_cached(sen_acts_pickled, act_occs_pickled, use_other)

    if len(windows) == 0:
        warning("No windows found")
        return None, None

    descriptor = WindowTextualizerFactory.create(textualization.value)
    descriptor.setup(sen_meta)
    descriptions = descriptor.describe(windows)

    class_examples = {}
    for window, desc_list in zip(windows, descriptions):
        label = window.label
        if label not in class_examples:
            class_examples[label] = []
        class_examples[label].append(desc_list)

    output_lines = []
    output_lines.append(f"Dataset: {dataset.value}")
    output_lines.append(f"Textualization: {textualization.value}")
    output_lines.append(f"Total Windows: {len(windows)}")
    output_lines.append(f"Number of Classes: {len(class_examples)}")
    output_lines.append("=" * 80)
    output_lines.append("")

    for label in sorted(class_examples.keys()):
        examples = class_examples[label][:max_examples]
        output_lines.append(f"\nActivity: {label}")
        output_lines.append(f"Total instances: {len(class_examples[label])}")
        output_lines.append("-" * 80)

        for i, desc_list in enumerate(examples, 1):
            output_lines.append(f"\nExample {i}:")
            for sentence in desc_list:
                output_lines.append(f"  {sentence}")
        output_lines.append("")

    filename = f"{dataset.value}_{textualization.value}_examples.txt"
    save_path = output_dir / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return class_examples, save_path


def generate_activity_statistics(
    dataset, nb_days, simplify_sensors, simplify_activities, use_other, data_cfg
):
    output_dir = Path("output/exploration/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    sen_acts_orig_pickled, act_occs_orig_pickled, sen_meta_orig = load_dataset_cached(
        dataset.name, nb_days, simplify_sensors, simplify_activities, False
    )
    sen_acts_pickled, act_occs_pickled, sen_meta = load_dataset_cached(
        dataset.name, nb_days, simplify_sensors, simplify_activities, True
    )

    windows_orig = get_windows_cached(
        sen_acts_orig_pickled, act_occs_orig_pickled, use_other
    )
    windows = get_windows_cached(sen_acts_pickled, act_occs_pickled, use_other)

    orig_counts = Counter([win.label for win in windows_orig])
    renamed_counts = Counter([win.label for win in windows])

    orig_to_renamed = {}
    for orig_win, renamed_win in zip(windows_orig, windows):
        if orig_win.label != renamed_win.label:
            orig_to_renamed[orig_win.label] = renamed_win.label

    output_lines = []
    output_lines.append(f"Dataset: {dataset.value}")
    output_lines.append(f"Number of Days: {nb_days}")
    output_lines.append(f"Total Windows (original): {len(windows_orig)}")
    output_lines.append(f"Total Windows (after processing): {len(windows)}")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("ORIGINAL ACTIVITIES (before renaming):")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Activity Name':<30} {'# Windows':<15} {'Mapped To':<30}")
    output_lines.append("-" * 80)
    for activity in sorted(orig_counts.keys()):
        count = orig_counts[activity]
        mapped = orig_to_renamed.get(activity, activity)
        arrow = " → " if activity != mapped else "   "
        output_lines.append(
            f"{activity:<30} {count:<15} {arrow}{mapped if activity != mapped else '(unchanged)'}"
        )

    output_lines.append("")
    output_lines.append("")
    output_lines.append("PROCESSED ACTIVITIES (after renaming):")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Activity Name':<30} {'# Windows':<15}")
    output_lines.append("-" * 80)
    for activity in sorted(renamed_counts.keys()):
        count = renamed_counts[activity]
        output_lines.append(f"{activity:<30} {count:<15}")

    output_lines.append("")
    output_lines.append(f"Total unique activities (original): {len(orig_counts)}")
    output_lines.append(f"Total unique activities (processed): {len(renamed_counts)}")

    filename = f"{dataset.value}_activity_stats.txt"
    save_path = output_dir / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return orig_counts, renamed_counts, save_path


def generate_sensor_metadata(dataset, nb_days, simplify_sensors):
    output_dir = Path("output/exploration/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    sen_acts_pickled, _, sen_meta = load_dataset_cached(
        dataset.name, nb_days, simplify_sensors, False, False
    )

    sen_acts = loads(sen_acts_pickled)

    sensor_usage = Counter([sa.identifier for sa in sen_acts])
    used_sensors = set(sensor_usage.keys())

    output_lines = []
    output_lines.append(f"Dataset: {dataset.value}")
    output_lines.append(f"Number of Days: {nb_days}")
    output_lines.append(f"Total Sensors in Metadata: {len(sen_meta)}")
    output_lines.append(f"Sensors Used (after simplification): {len(used_sensors)}")
    output_lines.append(f"Total Sensor Activations: {sum(sensor_usage.values())}")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("SENSOR METADATA:")
    output_lines.append("-" * 120)
    output_lines.append(
        f"{'Sensor ID':<15} {'Type':<15} {'Room':<20} {'Object':<20} {'Used?':<10} {'# Activations':<15}"
    )
    output_lines.append("-" * 120)
    for sensor_id in sorted(sen_meta.keys()):
        meta = sen_meta[sensor_id]
        sensor_type = meta.sensor_type or "N/A"
        sensor_room = meta.sensor_room or "N/A"
        sensor_object = meta.sensor_object or "N/A"
        is_used = "Yes" if sensor_id in used_sensors else "No"
        num_activations = sensor_usage.get(sensor_id, 0)
        output_lines.append(
            f"{sensor_id:<15} {sensor_type:<15} {sensor_room:<20} {sensor_object:<20} "
            f"{is_used:<10} {num_activations:<15}"
        )

    output_lines.append("")
    output_lines.append("")
    output_lines.append("SUMMARY BY SENSOR TYPE:")
    output_lines.append("-" * 80)
    type_counts = Counter(
        [meta.sensor_type for meta in sen_meta.values() if meta.sensor_type]
    )
    for sensor_type in sorted(type_counts.keys()):
        count = type_counts[sensor_type]
        output_lines.append(f"{sensor_type:<20} {count} sensors")

    output_lines.append("")
    output_lines.append("")
    output_lines.append("SUMMARY BY ROOM:")
    output_lines.append("-" * 80)
    room_counts = Counter(
        [meta.sensor_room for meta in sen_meta.values() if meta.sensor_room]
    )
    for room in sorted(room_counts.keys()):
        count = room_counts[room]
        output_lines.append(f"{room:<20} {count} sensors")

    filename = f"{dataset.value}_sensor_metadata.txt"
    save_path = output_dir / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return sen_meta, sensor_usage, save_path


def generate_umap_plot(
    dataset,
    textualization,
    vectorizer,
    nb_days,
    simplify_sensors,
    simplify_activities,
    rename_activities,
    use_other,
    cfg,
):
    output_dir = Path("output/exploration/umap")
    output_dir.mkdir(parents=True, exist_ok=True)

    sen_acts_pickled, act_occs_pickled, sen_meta = load_dataset_cached(
        dataset.name, nb_days, simplify_sensors, simplify_activities, rename_activities
    )

    windows = get_windows_cached(sen_acts_pickled, act_occs_pickled, use_other)

    if len(windows) == 0:
        warning("No windows found")
        return None, None

    descriptor = WindowTextualizerFactory.create(textualization.value)
    descriptor.setup(sen_meta)
    descriptions = descriptor.describe(windows)

    descriptions_flat = []
    lengths = []
    for desc in descriptions:
        descriptions_flat.extend(desc)
        lengths.append(len(desc))

    ve_args = get_vectorizer_args(vectorizer, cfg)
    vec = SentenceVectorizerFactory.create(vectorizer.value, **ve_args)
    vectors_flat = vec.fit_transform(descriptions_flat)

    vectors = []
    idx = 0
    for length in lengths:
        window_vectors = vectors_flat[idx : idx + length]
        vectors.append(window_vectors.mean(axis=0))
        idx += length

    X = vstack(vectors)
    labels = [win.label for win in windows]

    if X.shape[0] < 15:
        n_neighbors = max(2, X.shape[0] - 1)
    else:
        n_neighbors = 15

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = subplots(figsize=(12, 8))
    unique_labels = sorted(set(labels))
    colors = color_palette("husl", len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    for label in unique_labels:
        mask = array([ll == label for ll in labels])
        ax.scatter(
            embedding[mask, 0],  # type: ignore
            embedding[mask, 1],  # type: ignore
            c=[label_to_color[label]],
            label=label,
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(
        f"{dataset.value} | {textualization.value} | {vectorizer.value}\n"
        f"2D UMAP Projection ({len(windows)} windows, {len(unique_labels)} classes)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    tight_layout()

    filename = f"{dataset.value}_{textualization.value}_{vectorizer.value}.png"
    save_path = output_dir / filename
    savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, save_path


def analyze_dataset_comparison(
    selected_datasets, nb_days, simplify_sensors, simplify_activities, data_cfg
):
    output_dir = Path("output/exploration/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_data = {}
    for dataset_name in selected_datasets:
        dataset = TestBed[dataset_name]
        sen_meta = read_sensors_metadata(dataset)
        _, act_occs_orig_pickled, _ = load_dataset_cached(
            dataset_name, nb_days, simplify_sensors, simplify_activities, False
        )
        _, act_occs_renamed_pickled, _ = load_dataset_cached(
            dataset_name, nb_days, simplify_sensors, simplify_activities, True
        )

        act_occs_orig = loads(act_occs_orig_pickled)
        act_occs_renamed = loads(act_occs_renamed_pickled)
        sensor_types = [
            meta.sensor_type for meta in sen_meta.values() if meta.sensor_type
        ]
        sensor_rooms = [
            meta.sensor_room for meta in sen_meta.values() if meta.sensor_room
        ]
        sensor_objects = [
            meta.sensor_object for meta in sen_meta.values() if meta.sensor_object
        ]
        activities_orig = [act.identifier for act in act_occs_orig]
        activities_renamed = [act.identifier for act in act_occs_renamed]

        datasets_data[dataset_name] = {
            "sensor_types": sensor_types,
            "sensor_rooms": sensor_rooms,
            "sensor_objects": sensor_objects,
            "activities_orig": activities_orig,
            "activities_renamed": activities_renamed,
            "sensor_meta": sen_meta,
        }

    return datasets_data


def generate_comparison_report(datasets_data, output_dir):
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("DATASET COMPARISON ANALYSIS")
    output_lines.append("=" * 100)
    output_lines.append("")
    dataset_names = list(datasets_data.keys())
    output_lines.append("(A) SENSOR TYPE DISTRIBUTIONS")
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Sensor Type':<25} " + " ".join([f"{ds:<15}" for ds in dataset_names])
    )
    output_lines.append("-" * 100)
    all_sensor_types = set()
    for data in datasets_data.values():
        all_sensor_types.update(data["sensor_types"])

    for sensor_type in sorted(all_sensor_types):
        counts = []
        for ds_name in dataset_names:
            count = datasets_data[ds_name]["sensor_types"].count(sensor_type)
            counts.append(f"{count:<15}")
        output_lines.append(f"{sensor_type:<25} " + " ".join(counts))
    output_lines.append("")
    output_lines.append("(B) SENSOR TYPE OVERLAP")
    output_lines.append("-" * 100)
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1 :]:
            types1 = set(datasets_data[ds1]["sensor_types"])
            types2 = set(datasets_data[ds2]["sensor_types"])
            overlap = types1 & types2
            union = types1 | types2
            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            output_lines.append(f"{ds1} ↔ {ds2}:")
            output_lines.append(f"  Common types: {sorted(overlap)}")
            output_lines.append(
                f"  Overlap: {len(overlap)}/{len(union)} ({overlap_pct:.1f}%)"
            )
            output_lines.append(f"  Only in {ds1}: {sorted(types1 - types2)}")
            output_lines.append(f"  Only in {ds2}: {sorted(types2 - types1)}")
            output_lines.append("")
    output_lines.append("")
    output_lines.append("(C) SENSOR OBJECT DISTRIBUTIONS")
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Object':<25} " + " ".join([f"{ds:<15}" for ds in dataset_names])
    )
    output_lines.append("-" * 100)

    all_objects = set()
    for data in datasets_data.values():
        all_objects.update(data["sensor_objects"])

    for obj in sorted(all_objects):
        counts = []
        for ds_name in dataset_names:
            count = datasets_data[ds_name]["sensor_objects"].count(obj)
            counts.append(f"{count:<15}")
        output_lines.append(f"{obj:<25} " + " ".join(counts))
    output_lines.append("")
    output_lines.append("(D) SENSOR OBJECT OVERLAP")
    output_lines.append("-" * 100)
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1 :]:
            objs1 = set(datasets_data[ds1]["sensor_objects"])
            objs2 = set(datasets_data[ds2]["sensor_objects"])
            overlap = objs1 & objs2
            union = objs1 | objs2
            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            output_lines.append(f"{ds1} ↔ {ds2}:")
            output_lines.append(f"  Common objects: {sorted(overlap)}")
            output_lines.append(
                f"  Overlap: {len(overlap)}/{len(union)} ({overlap_pct:.1f}%)"
            )
            output_lines.append(f"  Only in {ds1}: {sorted(objs1 - objs2)}")
            output_lines.append(f"  Only in {ds2}: {sorted(objs2 - objs1)}")
            output_lines.append("")
    output_lines.append("")
    output_lines.append("(E) SENSOR ROOM DISTRIBUTIONS")
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Room':<25} " + " ".join([f"{ds:<15}" for ds in dataset_names])
    )
    output_lines.append("-" * 100)

    all_rooms = set()
    for data in datasets_data.values():
        all_rooms.update(data["sensor_rooms"])

    for room in sorted(all_rooms):
        counts = []
        for ds_name in dataset_names:
            count = datasets_data[ds_name]["sensor_rooms"].count(room)
            counts.append(f"{count:<15}")
        output_lines.append(f"{room:<25} " + " ".join(counts))
    output_lines.append("")
    output_lines.append("(F) SENSOR ROOM OVERLAP")
    output_lines.append("-" * 100)
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1 :]:
            rooms1 = set(datasets_data[ds1]["sensor_rooms"])
            rooms2 = set(datasets_data[ds2]["sensor_rooms"])
            overlap = rooms1 & rooms2
            union = rooms1 | rooms2
            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            output_lines.append(f"{ds1} ↔ {ds2}:")
            output_lines.append(f"  Common rooms: {sorted(overlap)}")
            output_lines.append(
                f"  Overlap: {len(overlap)}/{len(union)} ({overlap_pct:.1f}%)"
            )
            output_lines.append(f"  Only in {ds1}: {sorted(rooms1 - rooms2)}")
            output_lines.append(f"  Only in {ds2}: {sorted(rooms2 - rooms1)}")
            output_lines.append("")
    output_lines.append("")
    output_lines.append("(G) ACTIVITY DISTRIBUTIONS (ORIGINAL)")
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Activity':<30} " + " ".join([f"{ds:<15}" for ds in dataset_names])
    )
    output_lines.append("-" * 100)

    all_activities_orig = set()
    for data in datasets_data.values():
        all_activities_orig.update(set(data["activities_orig"]))

    for activity in sorted(all_activities_orig):
        counts = []
        for ds_name in dataset_names:
            count = datasets_data[ds_name]["activities_orig"].count(activity)
            counts.append(f"{count:<15}")
        output_lines.append(f"{activity:<30} " + " ".join(counts))
    output_lines.append("")
    output_lines.append("(H) ACTIVITY VOCABULARY OVERLAP (ORIGINAL)")
    output_lines.append("-" * 100)
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1 :]:
            acts1 = set(datasets_data[ds1]["activities_orig"])
            acts2 = set(datasets_data[ds2]["activities_orig"])
            overlap = acts1 & acts2
            union = acts1 | acts2
            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            output_lines.append(f"{ds1} ↔ {ds2}:")
            output_lines.append(f"  Common activities: {sorted(overlap)}")
            output_lines.append(
                f"  Overlap: {len(overlap)}/{len(union)} ({overlap_pct:.1f}%)"
            )
            output_lines.append(f"  Only in {ds1}: {sorted(acts1 - acts2)}")
            output_lines.append(f"  Only in {ds2}: {sorted(acts2 - acts1)}")
            output_lines.append("")
    output_lines.append("")
    output_lines.append("(I) ACTIVITY DISTRIBUTIONS (AFTER RENAMING)")
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Activity':<30} " + " ".join([f"{ds:<15}" for ds in dataset_names])
    )
    output_lines.append("-" * 100)

    all_activities_renamed = set()
    for data in datasets_data.values():
        all_activities_renamed.update(set(data["activities_renamed"]))

    for activity in sorted(all_activities_renamed):
        counts = []
        for ds_name in dataset_names:
            count = datasets_data[ds_name]["activities_renamed"].count(activity)
            counts.append(f"{count:<15}")
        output_lines.append(f"{activity:<30} " + " ".join(counts))
    output_lines.append("")
    output_lines.append("(J) ACTIVITY VOCABULARY OVERLAP (AFTER RENAMING)")
    output_lines.append("-" * 100)
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1 :]:
            acts1 = set(datasets_data[ds1]["activities_renamed"])
            acts2 = set(datasets_data[ds2]["activities_renamed"])
            overlap = acts1 & acts2
            union = acts1 | acts2
            overlap_pct = (len(overlap) / len(union) * 100) if union else 0
            output_lines.append(f"{ds1} ↔ {ds2}:")
            output_lines.append(f"  Common activities: {sorted(overlap)}")
            output_lines.append(
                f"  Overlap: {len(overlap)}/{len(union)} ({overlap_pct:.1f}%)"
            )
            output_lines.append(f"  Only in {ds1}: {sorted(acts1 - acts2)}")
            output_lines.append(f"  Only in {ds2}: {sorted(acts2 - acts1)}")
            output_lines.append("")
    filename = "dataset_comparison_report.txt"
    save_path = output_dir / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    return save_path, output_lines


def generate_comparison_visualizations(datasets_data, output_dir):
    dataset_names = list(datasets_data.keys())
    figures = []
    fig1, ax1 = subplots(figsize=(12, 6))
    all_sensor_types = set()
    for data in datasets_data.values():
        all_sensor_types.update(data["sensor_types"])

    type_counts = {
        ds: Counter(datasets_data[ds]["sensor_types"]) for ds in dataset_names
    }
    x_pos = range(len(all_sensor_types))
    width = 0.8 / len(dataset_names)

    for i, ds_name in enumerate(dataset_names):
        counts = [type_counts[ds_name].get(st, 0) for st in sorted(all_sensor_types)]
        ax1.bar([x + i * width for x in x_pos], counts, width, label=ds_name, alpha=0.8)

    ax1.set_xlabel("Sensor Type", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(
        "Sensor Type Distribution Across Datasets", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks([x + width * (len(dataset_names) - 1) / 2 for x in x_pos])
    ax1.set_xticklabels(sorted(all_sensor_types), rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    tight_layout()
    fig1_path = output_dir / "sensor_type_distribution.png"
    savefig(fig1_path, dpi=300, bbox_inches="tight")
    figures.append((fig1, "Sensor Type Distribution"))
    fig2, ax2 = subplots(figsize=(14, 6))
    all_objects = set()
    for data in datasets_data.values():
        all_objects.update(data["sensor_objects"])

    object_counts = {
        ds: Counter(datasets_data[ds]["sensor_objects"]) for ds in dataset_names
    }
    x_pos = range(len(all_objects))

    for i, ds_name in enumerate(dataset_names):
        counts = [object_counts[ds_name].get(obj, 0) for obj in sorted(all_objects)]
        ax2.bar([x + i * width for x in x_pos], counts, width, label=ds_name, alpha=0.8)

    ax2.set_xlabel("Sensor Object", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(
        "Sensor Object Distribution Across Datasets", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks([x + width * (len(dataset_names) - 1) / 2 for x in x_pos])
    ax2.set_xticklabels(sorted(all_objects), rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    tight_layout()
    fig2_path = output_dir / "sensor_object_distribution.png"
    savefig(fig2_path, dpi=300, bbox_inches="tight")
    figures.append((fig2, "Sensor Object Distribution"))
    fig3, ax3 = subplots(figsize=(12, 6))
    all_rooms = set()
    for data in datasets_data.values():
        all_rooms.update(data["sensor_rooms"])

    room_counts = {
        ds: Counter(datasets_data[ds]["sensor_rooms"]) for ds in dataset_names
    }
    x_pos = range(len(all_rooms))

    for i, ds_name in enumerate(dataset_names):
        counts = [room_counts[ds_name].get(room, 0) for room in sorted(all_rooms)]
        ax3.bar([x + i * width for x in x_pos], counts, width, label=ds_name, alpha=0.8)

    ax3.set_xlabel("Room", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title(
        "Sensor Room Distribution Across Datasets", fontsize=14, fontweight="bold"
    )
    ax3.set_xticks([x + width * (len(dataset_names) - 1) / 2 for x in x_pos])
    ax3.set_xticklabels(sorted(all_rooms), rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    tight_layout()
    fig3_path = output_dir / "sensor_room_distribution.png"
    savefig(fig3_path, dpi=300, bbox_inches="tight")
    figures.append((fig3, "Sensor Room Distribution"))
    fig4, ax4 = subplots(figsize=(14, 7))
    all_activities_orig = set()
    for data in datasets_data.values():
        all_activities_orig.update(set(data["activities_orig"]))

    activity_orig_counts = {
        ds: Counter(datasets_data[ds]["activities_orig"]) for ds in dataset_names
    }
    x_pos = range(len(all_activities_orig))

    for i, ds_name in enumerate(dataset_names):
        counts = [
            activity_orig_counts[ds_name].get(act, 0)
            for act in sorted(all_activities_orig)
        ]
        ax4.bar([x + i * width for x in x_pos], counts, width, label=ds_name, alpha=0.8)

    ax4.set_xlabel("Activity (Original)", fontsize=12)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.set_title(
        "Activity Distribution Across Datasets (Original Names)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xticks([x + width * (len(dataset_names) - 1) / 2 for x in x_pos])
    ax4.set_xticklabels(sorted(all_activities_orig), rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    tight_layout()
    fig4_path = output_dir / "activity_distribution_original.png"
    savefig(fig4_path, dpi=300, bbox_inches="tight")
    figures.append((fig4, "Activity Distribution (Original)"))
    fig5, ax5 = subplots(figsize=(12, 6))
    all_activities_renamed = set()
    for data in datasets_data.values():
        all_activities_renamed.update(set(data["activities_renamed"]))

    activity_renamed_counts = {
        ds: Counter(datasets_data[ds]["activities_renamed"]) for ds in dataset_names
    }
    x_pos = range(len(all_activities_renamed))

    for i, ds_name in enumerate(dataset_names):
        counts = [
            activity_renamed_counts[ds_name].get(act, 0)
            for act in sorted(all_activities_renamed)
        ]
        ax5.bar([x + i * width for x in x_pos], counts, width, label=ds_name, alpha=0.8)

    ax5.set_xlabel("Activity (Renamed)", fontsize=12)
    ax5.set_ylabel("Count", fontsize=12)
    ax5.set_title(
        "Activity Distribution Across Datasets (After Renaming)",
        fontsize=14,
        fontweight="bold",
    )
    ax5.set_xticks([x + width * (len(dataset_names) - 1) / 2 for x in x_pos])
    ax5.set_xticklabels(sorted(all_activities_renamed), rotation=45, ha="right")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
    tight_layout()
    fig5_path = output_dir / "activity_distribution_renamed.png"
    savefig(fig5_path, dpi=300, bbox_inches="tight")
    figures.append((fig5, "Activity Distribution (After Renaming)"))

    return figures


def main():
    title("🔍 Activity Recognition Analysis")

    cfg, data_cfg = load_configs()

    sidebar.header("Configuration")

    task = sidebar.selectbox(
        "Select Task",
        [
            "Example Sentences",
            "Activity Statistics",
            "Sensor Metadata",
            "UMAP Visualization",
            "Dataset Comparison",
        ],
    )

    dataset_options = ["ORDA", "ORDB", "CASA", "CASM"]
    dataset_name = sidebar.selectbox("Dataset", dataset_options)
    dataset = TestBed[dataset_name]

    nb_days = sidebar.number_input(
        "Number of Days", min_value=1, max_value=100, value=cfg.get("nb_days", 14)
    )
    simplify_sensors = sidebar.checkbox(
        "Simplify Sensors", value=cfg.get("simplify_sensors", True)
    )
    simplify_activities = sidebar.checkbox(
        "Simplify Activities", value=cfg.get("simplify_activities", True)
    )
    rename_activities = sidebar.checkbox(
        "Rename Activities", value=cfg.get("rename_activities", True)
    )
    use_other = sidebar.checkbox("Use OTHER class", value=False)

    if task == "Example Sentences":
        header("📝 Example Sentences")

        textualization_options = [t.value for t in TextualizerType]
        textualization_name = selectbox("Textualization Method", textualization_options)
        textualization = TextualizerType(textualization_name)

        max_examples = number_input(
            "Max Examples per Class", min_value=1, max_value=10, value=3
        )

        if button("Generate Examples"):
            with spinner("Generating example sentences..."):
                class_examples, save_path = generate_example_sentences(
                    dataset,
                    textualization,
                    nb_days,
                    simplify_sensors,
                    simplify_activities,
                    rename_activities,
                    use_other,
                    max_examples,
                )

                if class_examples:
                    success(f"✓ Saved to {save_path}")

                    for label in sorted(class_examples.keys()):
                        with expander(
                            f"Activity: {label} ({len(class_examples[label])} instances)"
                        ):
                            examples = class_examples[label][:max_examples]
                            for i, desc_list in enumerate(examples, 1):
                                markdown(f"**Example {i}:**")
                                for sentence in desc_list:
                                    write(f"  • {sentence}")

    elif task == "Activity Statistics":
        header("📊 Activity Statistics")

        if button("Generate Statistics"):
            with spinner("Generating activity statistics..."):
                orig_counts, renamed_counts, save_path = generate_activity_statistics(
                    dataset,
                    nb_days,
                    simplify_sensors,
                    simplify_activities,
                    use_other,
                    data_cfg,
                )

                success(f"✓ Saved to {save_path}")

                col1, col2 = columns(2)

                with col1:
                    subheader("Original Activities")
                    metric("Unique Activities", len(orig_counts))
                    dataframe(
                        {
                            "Activity": list(orig_counts.keys()),
                            "Count": list(orig_counts.values()),
                        },
                        height=400,
                    )

                with col2:
                    subheader("Processed Activities")
                    metric("Unique Activities", len(renamed_counts))
                    dataframe(
                        {
                            "Activity": list(renamed_counts.keys()),
                            "Count": list(renamed_counts.values()),
                        },
                        height=400,
                    )

    elif task == "Sensor Metadata":
        header("🔧 Sensor Metadata")

        if button("Generate Metadata"):
            with spinner("Generating sensor metadata..."):
                sen_meta, sensor_usage, save_path = generate_sensor_metadata(
                    dataset, nb_days, simplify_sensors
                )

                success(f"✓ Saved to {save_path}")

                subheader("Summary")
                col1, col2, col3 = columns(3)
                col1.metric("Total Sensors", len(sen_meta))
                col2.metric("Used Sensors", len(sensor_usage))
                col3.metric("Total Activations", sum(sensor_usage.values()))

                sensor_data = []
                for sensor_id in sorted(sen_meta.keys()):
                    meta = sen_meta[sensor_id]
                    sensor_data.append(
                        {
                            "Sensor ID": sensor_id,
                            "Type": meta.sensor_type or "N/A",
                            "Room": meta.sensor_room or "N/A",
                            "Object": meta.sensor_object or "N/A",
                            "Used": "Yes" if sensor_id in sensor_usage else "No",
                            "Activations": sensor_usage.get(sensor_id, 0),
                        }
                    )

                dataframe(sensor_data, height=500)

    elif task == "UMAP Visualization":
        header("🗺️ UMAP Visualization")

        textualization_options = [t.value for t in TextualizerType]
        textualization_name = selectbox("Textualization Method", textualization_options)
        textualization = TextualizerType(textualization_name)

        vectorizer_options = [v.value for v in SentenceVectorizerType]
        vectorizer_name = selectbox("Vectorizer", vectorizer_options)
        vectorizer = SentenceVectorizerType(vectorizer_name)

        if button("Generate UMAP Plot"):
            with spinner("Generating UMAP visualization..."):
                fig, save_path = generate_umap_plot(
                    dataset,
                    textualization,
                    vectorizer,
                    nb_days,
                    simplify_sensors,
                    simplify_activities,
                    rename_activities,
                    use_other,
                    cfg,
                )

                if fig:
                    success(f"✓ Saved to {save_path}")
                    pyplot(fig)
                    close()

    elif task == "Dataset Comparison":
        header("📊 Dataset Comparison")

        markdown(
            """
            Compare sensor and activity distributions across multiple datasets.
            Analyze overlaps and differences in sensor types, objects, rooms, and activity vocabularies.
            """
        )

        dataset_options = ["ORDA", "ORDB", "CASA", "CASM"]
        selected_datasets = multiselect(
            "Select Datasets to Compare",
            dataset_options,
            default=dataset_options[:2],
        )

        if len(selected_datasets) < 2:
            warning("⚠️ Please select at least 2 datasets to compare")
        else:
            comparison_nb_days = sidebar.number_input(
                "Days for Comparison",
                min_value=1,
                max_value=100,
                value=nb_days,
                key="comparison_days",
            )

            if button("Generate Comparison Analysis"):
                with spinner("Analyzing datasets..."):
                    datasets_data = analyze_dataset_comparison(
                        selected_datasets,
                        comparison_nb_days,
                        simplify_sensors,
                        simplify_activities,
                        data_cfg,
                    )

                    output_dir = Path("output/exploration/comparison")
                    report_path, report_lines = generate_comparison_report(
                        datasets_data, output_dir
                    )
                    figures = generate_comparison_visualizations(
                        datasets_data, output_dir
                    )

                    success(f"✓ Report saved to {report_path}")
                    subheader("📈 Summary Metrics")
                    cols = columns(len(selected_datasets))
                    for i, ds_name in enumerate(selected_datasets):
                        with cols[i]:
                            metric(f"{ds_name}", "Dataset")
                            metric(
                                "Sensor Types",
                                len(set(datasets_data[ds_name]["sensor_types"])),
                            )
                            metric(
                                "Sensor Objects",
                                len(set(datasets_data[ds_name]["sensor_objects"])),
                            )
                            metric(
                                "Rooms",
                                len(set(datasets_data[ds_name]["sensor_rooms"])),
                            )
                            metric(
                                "Activities (Orig)",
                                len(set(datasets_data[ds_name]["activities_orig"])),
                            )
                            metric(
                                "Activities (Renamed)",
                                len(set(datasets_data[ds_name]["activities_renamed"])),
                            )
                    subheader("📊 Visualizations")
                    for fig, fig_title in figures:
                        with expander(fig_title):
                            pyplot(fig)
                            close(fig)
                    subheader("🔍 Overlap Analysis")
                    dataset_names = selected_datasets
                    with expander("Sensor Type Overlap"):
                        for i, ds1 in enumerate(dataset_names):
                            for ds2 in dataset_names[i + 1 :]:
                                types1 = set(datasets_data[ds1]["sensor_types"])
                                types2 = set(datasets_data[ds2]["sensor_types"])
                                overlap = types1 & types2
                                union = types1 | types2
                                overlap_pct = (
                                    (len(overlap) / len(union) * 100) if union else 0
                                )

                                markdown(f"**{ds1} ↔ {ds2}**")
                                metric(
                                    "Overlap Percentage",
                                    f"{overlap_pct:.1f}%",
                                    f"{len(overlap)}/{len(union)}",
                                )
                                write(f"Common: {sorted(overlap)}")
                                write(f"Only in {ds1}: {sorted(types1 - types2)}")
                                write(f"Only in {ds2}: {sorted(types2 - types1)}")
                                markdown("---")
                    with expander("Sensor Object Overlap"):
                        for i, ds1 in enumerate(dataset_names):
                            for ds2 in dataset_names[i + 1 :]:
                                objs1 = set(datasets_data[ds1]["sensor_objects"])
                                objs2 = set(datasets_data[ds2]["sensor_objects"])
                                overlap = objs1 & objs2
                                union = objs1 | objs2
                                overlap_pct = (
                                    (len(overlap) / len(union) * 100) if union else 0
                                )

                                markdown(f"**{ds1} ↔ {ds2}**")
                                metric(
                                    "Overlap Percentage",
                                    f"{overlap_pct:.1f}%",
                                    f"{len(overlap)}/{len(union)}",
                                )
                                write(f"Common: {sorted(overlap)}")
                                write(f"Only in {ds1}: {sorted(objs1 - objs2)}")
                                write(f"Only in {ds2}: {sorted(objs2 - objs1)}")
                                markdown("---")
                    with expander("Room Overlap"):
                        for i, ds1 in enumerate(dataset_names):
                            for ds2 in dataset_names[i + 1 :]:
                                rooms1 = set(datasets_data[ds1]["sensor_rooms"])
                                rooms2 = set(datasets_data[ds2]["sensor_rooms"])
                                overlap = rooms1 & rooms2
                                union = rooms1 | rooms2
                                overlap_pct = (
                                    (len(overlap) / len(union) * 100) if union else 0
                                )

                                markdown(f"**{ds1} ↔ {ds2}**")
                                metric(
                                    "Overlap Percentage",
                                    f"{overlap_pct:.1f}%",
                                    f"{len(overlap)}/{len(union)}",
                                )
                                write(f"Common: {sorted(overlap)}")
                                write(f"Only in {ds1}: {sorted(rooms1 - rooms2)}")
                                write(f"Only in {ds2}: {sorted(rooms2 - rooms1)}")
                                markdown("---")
                    with expander("Activity Vocabulary Overlap (Original)"):
                        for i, ds1 in enumerate(dataset_names):
                            for ds2 in dataset_names[i + 1 :]:
                                acts1 = set(datasets_data[ds1]["activities_orig"])
                                acts2 = set(datasets_data[ds2]["activities_orig"])
                                overlap = acts1 & acts2
                                union = acts1 | acts2
                                overlap_pct = (
                                    (len(overlap) / len(union) * 100) if union else 0
                                )

                                markdown(f"**{ds1} ↔ {ds2}**")
                                metric(
                                    "Overlap Percentage",
                                    f"{overlap_pct:.1f}%",
                                    f"{len(overlap)}/{len(union)}",
                                )
                                write(f"Common: {sorted(overlap)}")
                                write(f"Only in {ds1}: {sorted(acts1 - acts2)}")
                                write(f"Only in {ds2}: {sorted(acts2 - acts1)}")
                                markdown("---")
                    with expander("Activity Vocabulary Overlap (After Renaming)"):
                        for i, ds1 in enumerate(dataset_names):
                            for ds2 in dataset_names[i + 1 :]:
                                acts1 = set(datasets_data[ds1]["activities_renamed"])
                                acts2 = set(datasets_data[ds2]["activities_renamed"])
                                overlap = acts1 & acts2
                                union = acts1 | acts2
                                overlap_pct = (
                                    (len(overlap) / len(union) * 100) if union else 0
                                )

                                markdown(f"**{ds1} ↔ {ds2}**")
                                metric(
                                    "Overlap Percentage",
                                    f"{overlap_pct:.1f}%",
                                    f"{len(overlap)}/{len(union)}",
                                )
                                write(f"Common: {sorted(overlap)}")
                                write(f"Only in {ds1}: {sorted(acts1 - acts2)}")
                                write(f"Only in {ds2}: {sorted(acts2 - acts1)}")
                                markdown("---")


if __name__ == "__main__":
    main()
