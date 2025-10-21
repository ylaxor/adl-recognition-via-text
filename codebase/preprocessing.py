from collections import Counter

from numpy import float32, full, ndarray, vstack, zeros
from torch import Tensor


def pad_truncate(
    embeddings: list[Tensor | ndarray],
    max_length: int = 100,
    padding_strategy: str = "post",
    truncation_strategy: str = "post",
    padding_value: float = 0.0,
) -> tuple[ndarray, ndarray]:
    if not embeddings:
        raise ValueError("embeddings list is empty")
    n_samples = len(embeddings)
    embedding_dim = embeddings[0].shape[1]
    padded_sequences = full(
        (n_samples, max_length, embedding_dim), padding_value, dtype=float32
    )
    masks = zeros((n_samples, max_length), dtype=float32)
    for i, emb in enumerate(embeddings):
        seq_len = emb.shape[0]
        if isinstance(emb, Tensor):
            emb_array = emb.cpu().numpy()
        else:
            emb_array = emb
        if seq_len > max_length:
            if truncation_strategy == "post":
                emb_array = emb_array[:max_length]
            elif truncation_strategy == "pre":
                emb_array = emb_array[-max_length:]
            else:
                raise ValueError(f"Unknown truncation_strategy: {truncation_strategy}")
            actual_len = max_length
        else:
            actual_len = seq_len
        if padding_strategy == "post":
            padded_sequences[i, :actual_len] = emb_array[:actual_len]
            masks[i, :actual_len] = 1.0
        elif padding_strategy == "pre":
            padded_sequences[i, -actual_len:] = emb_array[:actual_len]
            masks[i, -actual_len:] = 1.0
        else:
            raise ValueError(f"Unknown padding_strategy: {padding_strategy}")
    return padded_sequences, masks


def get_case_by_shape(vectors) -> str:
    unique_shapes = Counter([v.shape for v in vectors])
    if len(unique_shapes) > 1:
        return "sequence"
    else:
        unique_shape = vectors[0].shape
        assert unique_shape[0] == 1
        return "single"


def stack(vectors) -> ndarray:
    return vstack([v[0] for v in vectors])
