from dataclasses import dataclass
from functools import partial
from os import environ
from time import time
from warnings import filterwarnings

from numpy import arange, array
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from toml import load as load_toml

from codebase.loading import TestBed, read_dataset, read_sensors_metadata
from codebase.lstm import LSTMClassifier
from codebase.mailing import send_email
from codebase.plotting import plot_transfer_results
from codebase.preprocessing import get_case_by_shape, pad_truncate, stack
from codebase.reporting import analyze_and_send_results, get_cm, get_report
from codebase.textualization import TextualizerType, WindowTextualizerFactory
from codebase.vectorization import SentenceVectorizerFactory, SentenceVectorizerType
from codebase.windowing import get_ground_truth_windows

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
filterwarnings("ignore")


@dataclass
class Config:
    nb_days: int
    simplify_sensors: bool
    simplify_activities: bool
    rename_activities: bool
    tfidf_args: dict
    word2vec_args: dict
    sbert_args: dict
    mlp_args: dict
    lstm_args: dict
    pad_trunc_length: int
    n_splits: int
    dts_args: dict
    css_args: dict

    def __post_init__(self):
        if "ngram_range" in self.tfidf_args:
            self.tfidf_args["ngram_range"] = tuple(self.tfidf_args["ngram_range"])


def run_experiment(
    config_path: str = "./config/experiment.toml",
    output_dir: str = "./output",
    sources: list[TestBed] | None = None,
    targets: list[TestBed] | None = None,
    textualizers: list[TextualizerType] | None = None,
    vectorizers: list[SentenceVectorizerType] | None = None,
    use_others: list[bool] | None = None,
) -> dict:
    cfg = Config(**load_toml(config_path))
    reader = partial(
        read_dataset,
        simplify_sensors=cfg.simplify_sensors,
        simplify_activities=cfg.simplify_activities,
        rename_activities=cfg.rename_activities,
    )
    windower = get_ground_truth_windows
    mlp_clf = MLPClassifier(**cfg.mlp_args)
    lstm_mlp_clf = LSTMClassifier(**cfg.lstm_args)
    cross_validator = StratifiedKFold(cfg.n_splits)
    all_results = {}
    for use_other in use_others or [True, False]:
        for desc in textualizers or list(TextualizerType):
            if desc == TextualizerType.DTS:
                desc_args = cfg.dts_args
            elif desc == TextualizerType.CSS:
                desc_args = cfg.css_args
            else:
                desc_args = {}
            descriptor = WindowTextualizerFactory.create(desc.value, **desc_args)
            for vect in vectorizers or list(SentenceVectorizerType):
                if vect == SentenceVectorizerType.TFIDF:
                    ve_args = cfg.tfidf_args
                elif vect == SentenceVectorizerType.WORD2VEC:
                    ve_args = cfg.word2vec_args
                elif vect == SentenceVectorizerType.SBERT:
                    ve_args = cfg.sbert_args
                else:
                    ve_args = {}
                vectorizer = SentenceVectorizerFactory.create(vect.value, **ve_args)
                desc_name = descriptor.__class__.__name__
                vect_name = vectorizer.__class__.__name__
                xp_use_other = "With 'OTHER'" if use_other else "No 'OTHER'"
                xp_title = f"{xp_use_other} | {desc_name} | {vect_name}"
                print(f"Experiment: {xp_title}")
                xp_result = {}
                mail = (
                    "Welcome to the HAR Cross-Source Transfer Learning Experiment\n\n"
                )
                for src_name in sources or list(TestBed):
                    print("=" * 50)
                    print(f"Source: {src_name}")
                    xp_result[src_name] = {}
                    mail += f"Source: {src_name}\n"
                    mail += f"Nb Days: {cfg.nb_days}\n"
                    mail += f"Use of 'Other' class: {use_other}\n"
                    mail += f"Simplify Sensors: {cfg.simplify_sensors}\n"
                    mail += f"Simplify Activities: {cfg.simplify_activities}\n"
                    mail += f"Rename Activities: {cfg.rename_activities}\n"
                    mail += f"Descriptor: {descriptor.__class__.__name__}\n"
                    mail += f"Vectorizer: {vectorizer.__class__.__name__}\n"
                    mail += f"Vectorizer Args: {ve_args}\n"
                    src_sen_acts, src_act_occs = reader(src_name, cfg.nb_days)
                    src_sen_meta = read_sensors_metadata(src_name)
                    src_gt_windows = windower(src_sen_acts, src_act_occs, use_other)
                    descriptor = descriptor.setup(src_sen_meta)
                    src_descriptions = descriptor.describe(src_gt_windows)
                    src_descriptions_flat, src_lengths = [], []
                    for desc in src_descriptions:
                        src_descriptions_flat.extend(desc)
                        src_lengths.append(len(desc))
                    src_vect_time = time()
                    src_vectors_flat = vectorizer.fit_transform(src_descriptions_flat)
                    src_vect_time = time() - src_vect_time
                    print(f"Source Vectorization Time: {src_vect_time:.2f} sec")
                    mail += f"Source Vectorization Time: {src_vect_time:.2f} sec\n"
                    src_vectors = []
                    idx = 0
                    for length in src_lengths:
                        src_vectors.append(src_vectors_flat[idx : idx + length])
                        idx += length
                    case_by_shape = get_case_by_shape(src_vectors)
                    if case_by_shape == "sequence":
                        mail += "Case by Shape: sequence\n"
                        mail += f"Pad/Trunc Length: {cfg.pad_trunc_length}\n"
                        src_X, src_masks = pad_truncate(
                            src_vectors, cfg.pad_trunc_length
                        )
                        src_clf = lstm_mlp_clf
                    elif case_by_shape == "single":
                        mail += "Case by Shape: single\n"
                        mail += "Pad/Trunc Length: N/A\n"
                        src_X = stack(src_vectors)
                        src_masks = None
                        src_clf = mlp_clf
                    else:
                        raise RuntimeError("Unrecognized case_by_shape")
                    src_labels = sorted(
                        list(set([win.label for win in src_gt_windows]))
                    )
                    mail += f"Source Labels: {src_labels}\n"
                    src_Y = array([win.label for win in src_gt_windows])
                    label_encoder = None
                    if case_by_shape == "single":
                        label_encoder = LabelEncoder()
                        src_Y_for_split = label_encoder.fit_transform(src_Y)
                        src_Y_for_split = array(src_Y_for_split)
                    else:
                        src_Y_for_split = src_Y
                    src_splits = list(
                        cross_validator.split(
                            arange(len(src_Y_for_split)), src_Y_for_split
                        )
                    )
                    src_trues, src_preds = [], []
                    mail += f"Cross-Validation Splits: {cfg.n_splits}\n"
                    src_cv_time = time()
                    for i, (train_idx, test_idx) in enumerate(src_splits):
                        print(f"S {i}: train: {len(train_idx)}, test: {len(test_idx)}")
                        src_X_train, src_X_test = src_X[train_idx], src_X[test_idx]
                        src_Y_train, src_Y_test = src_Y[train_idx], src_Y[test_idx]
                        src_split_clf = clone(src_clf)
                        if case_by_shape == "sequence":
                            assert src_masks is not None
                            src_split_clf.fit(
                                src_X_train,
                                src_Y_train,
                                src_masks[train_idx],
                            )
                            src_Y_test_pred = src_split_clf.predict(
                                src_X_test,
                                src_masks[test_idx],
                            )
                        elif case_by_shape == "single":
                            src_Y_train_encoded = src_Y_for_split[train_idx]
                            src_split_clf.fit(src_X_train, src_Y_train_encoded)
                            src_Y_test_pred_encoded = src_split_clf.predict(src_X_test)
                            assert label_encoder is not None, (
                                "Label encoder should be initialized for single case"
                            )
                            src_Y_test_pred = label_encoder.inverse_transform(
                                src_Y_test_pred_encoded
                            )
                        else:
                            raise RuntimeError("Unrecognized case_by_shape")
                        acc_balanced = balanced_accuracy_score(
                            src_Y_test, src_Y_test_pred
                        )
                        print(f"Split {i}: Acc Balanced: {acc_balanced:.4f}")
                        mail += f"Split {i}: Acc Balanced: {acc_balanced:.4f}\n"
                        src_trues.extend(src_Y_test)
                        src_preds.extend(src_Y_test_pred)
                    src_cv_time = time() - src_cv_time
                    print(f"Source Cross-Validation Time: {src_cv_time:.2f} sec")
                    mail += f"Source Cross-Validation Time: {src_cv_time:.2f} sec\n"
                    acc_balanced = balanced_accuracy_score(src_trues, src_preds)
                    print("Source Acc Balanced (On Agg. CV):")
                    mail += f"Source Acc Balanced (On Agg. CV): {acc_balanced:.4f}\n"
                    print(f"{acc_balanced:.4f}")
                    xp_result[src_name][src_name] = acc_balanced
                    src_cm_str = get_cm(src_trues, src_preds)
                    print("Source Confusion Matrix (On Agg. CV):")
                    print(src_cm_str)
                    mail += "Source Confusion Matrix (On Agg. CV):\n"
                    mail += src_cm_str + "\n"
                    src_report_str = get_report(src_trues, src_preds)
                    print("Source Classification Report (On Agg. CV):")
                    print(src_report_str)
                    mail += "Source Classification Report (On Agg. CV):\n"
                    mail += src_report_str + "\n"
                    final_clf = clone(src_clf)
                    if case_by_shape == "sequence":
                        assert src_masks is not None
                        final_clf.fit(src_X, src_Y, src_masks)
                    elif case_by_shape == "single":
                        final_clf.fit(src_X, src_Y_for_split)
                    else:
                        raise RuntimeError("Unrecognized case_by_shape")
                    tgt_names = [x for x in targets or list(TestBed) if x != src_name]
                    for tgt_name in tgt_names:
                        print("=" * 50)
                        print(f"Target: {tgt_name}")
                        mail += f"Target: {tgt_name}\n"
                        tgt_sen_acts, tgt_act_occs = reader(tgt_name, cfg.nb_days)
                        tgt_sen_meta = read_sensors_metadata(tgt_name)
                        tgt_gt_windows = windower(tgt_sen_acts, tgt_act_occs, use_other)
                        tgt_windows = [
                            win for win in tgt_gt_windows if win.label in src_labels
                        ]
                        tgt_descriptions = descriptor.setup(tgt_sen_meta).describe(
                            tgt_windows
                        )
                        tgt_descriptions_flat, tgt_lengths = [], []
                        for desc in tgt_descriptions:
                            tgt_descriptions_flat.extend(desc)
                            tgt_lengths.append(len(desc))
                        tgt_vect_time = time()
                        tgt_vectors_flat = vectorizer.encode(tgt_descriptions_flat)
                        tgt_vect_time = time() - tgt_vect_time
                        print(f"Target Vectorization Time: {tgt_vect_time:.2f} sec")
                        mail += f"Target Vectorization Time: {tgt_vect_time:.2f} sec\n"
                        tgt_vectors = []
                        idx = 0
                        for length in tgt_lengths:
                            tgt_vectors.append(tgt_vectors_flat[idx : idx + length])
                            idx += length
                        if case_by_shape == "sequence":
                            tgt_X, tgt_masks = pad_truncate(
                                tgt_vectors, cfg.pad_trunc_length
                            )
                            tgt_pred_time = time()
                            tgt_Y_pred = final_clf.predict(tgt_X, tgt_masks)
                            tgt_pred_time = time() - tgt_pred_time
                        elif case_by_shape == "single":
                            tgt_X = stack(tgt_vectors)
                            tgt_pred_time = time()
                            tgt_Y_pred_encoded = final_clf.predict(tgt_X)
                            assert label_encoder is not None, (
                                "Label encoder should be initialized for single case"
                            )
                            tgt_Y_pred = label_encoder.inverse_transform(
                                tgt_Y_pred_encoded
                            )
                            tgt_pred_time = time() - tgt_pred_time
                        else:
                            raise RuntimeError("Unrecognized case_by_shape")
                        print(f"Target Prediction Time: {tgt_pred_time:.2f} sec")
                        mail += f"Target Prediction Time: {tgt_pred_time:.2f} sec\n"
                        tgt_Y = array([win.label for win in tgt_windows])
                        acc_balanced = balanced_accuracy_score(tgt_Y, tgt_Y_pred)
                        print(f"Target Overall Acc Balanced: {acc_balanced:.4f}")
                        mail += f"Target Overall Acc Balanced: {acc_balanced:.4f}\n"
                        tgt_cm_str = get_cm(tgt_Y, tgt_Y_pred)
                        print("Target Confusion Matrix:")
                        print(tgt_cm_str)
                        mail += "Target Confusion Matrix:\n"
                        mail += tgt_cm_str + "\n"
                        tgt_report_str = get_report(tgt_Y, tgt_Y_pred)
                        print("Target Classification Report:")
                        print(tgt_report_str)
                        mail += "Target Classification Report:\n"
                        mail += tgt_report_str + "\n"
                        xp_result[src_name][tgt_name] = acc_balanced
                for src in xp_result:
                    for tgt in xp_result[src]:
                        print(
                            f"Src: {src}, Tgt: {tgt}, Score: {xp_result[src][tgt]:.4f}"
                        )
                print("=" * 50)
                print("Generating transfer results plot...")
                plot_transfer_results(
                    xp_result,
                    save_path=f"{output_dir}/figures/{xp_title}.pdf",
                    title=xp_title,
                )
                all_results[xp_title] = xp_result
                with open(f"{output_dir}/figures/{xp_title}.pdf", "rb") as f:
                    xp_figure = f.read()
                send_email(
                    subject=f"Transfer Learning Results: {xp_title}",
                    body="Please find attached the transfer learning results.",
                    attachments={
                        f"{xp_title}.pdf": xp_figure,
                        f"{xp_title}.txt": mail.encode(),
                    },
                )
    analyze_and_send_results(all_results, output_dir)
    return all_results
