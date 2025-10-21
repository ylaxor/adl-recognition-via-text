from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Optional

from gensim import downloader as api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from numpy import array, asarray, float32, ndarray, vstack, zeros
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as _TV
from sklearn.preprocessing import normalize as _l2_normalize
from torch import cuda, no_grad


def default_tokenizer(s: str) -> list[str]:
    return simple_preprocess(s, deacc=False, min_len=1)


class SentenceVectorizerType(Enum):
    TFIDF = "tfidf"
    SBERT = "sbert"
    WORD2VEC = "word2vec"


class SentenceVectorizer(ABC):
    def fit_transform(self, corpus: Iterable[str]) -> ndarray:
        self.fit(corpus)
        return self.encode(corpus)

    @abstractmethod
    def fit(self, corpus: Iterable[str] | None = None) -> "SentenceVectorizer": ...

    def encode(self, texts: Iterable[str]) -> ndarray:
        return array([self.encode_sentence(text) for text in texts])

    @abstractmethod
    def encode_sentence(self, text: str) -> ndarray: ...


class TfidfVectorizer(SentenceVectorizer):
    def __init__(self, max_features: int | None = 20000, ngram_range=(1, 1)):
        self._tv = _TV(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="word",
            tokenizer=default_tokenizer,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,  # type: ignore
        )
        self._fitted = False

    def fit(self, corpus: Iterable[str] | None = None) -> "TfidfVectorizer":
        if corpus is None:
            raise ValueError("TF-IDF requires fit(corpus=...) at least once.")
        self._tv.fit(corpus)
        self._fitted = True
        return self

    def encode_sentence(self, text: str) -> ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfVectorizer is not fitted. Call fit(corpus) first.")
        return self._tv.transform([text]).toarray()[0].astype(float32)  # type: ignore


class Word2VecVectorizer(SentenceVectorizer):
    def __init__(
        self,
        model_name: str = "word2vec-google-news-300",
        aggregation: str = "mean",
        use_tfidf: bool = True,
        idf_power: float = 1.0,
        tfidf_params: Optional[Mapping] = None,
        l2_normalize: bool = False,
    ):
        self.tokenizer = default_tokenizer
        self.aggregation = aggregation
        self.l2_normalize = l2_normalize
        self.use_tfidf = use_tfidf
        self.idf_power = float(idf_power)
        self.tfidf_params = tfidf_params or {}
        self.model_name = model_name
        self._fitted = False
        self._kv: Optional[KeyedVectors] = None
        self._dim: Optional[int] = None
        self._idf: Optional[Mapping[str, float]] = None

    def _load_pretrained(self) -> KeyedVectors:
        return api.load(self.model_name)  # type: ignore

    def _ensure_fitted(self):
        if not (self._fitted and self._kv is not None and self._dim is not None):
            raise RuntimeError(
                "Word2VecVectorizer is not fitted. Call fit(corpus=...) or set pretrained_path and call fit()."
            )

    def _compute_idf(self, corpus: Iterable[str]) -> Mapping[str, float]:
        tv = _TV(
            analyzer="word",
            tokenizer=self.tokenizer,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,  # type: ignore
            **self.tfidf_params,
        )
        tv.fit(corpus)
        feats = tv.get_feature_names_out()
        idf = tv.idf_
        powered = idf**self.idf_power if self.idf_power != 1.0 else idf
        return {tok: float(w) for tok, w in zip(feats, powered)}

    def fit(self, corpus: Iterable[str] | None = None) -> "Word2VecVectorizer":
        self._kv = self._load_pretrained()
        self._dim = int(self._kv.vector_size)
        self._idf = self._compute_idf(corpus) if self.use_tfidf and corpus else None
        self._fitted = True
        return self

    def encode_sentence(self, text: str) -> ndarray:
        self._ensure_fitted()
        assert self._kv is not None and self._dim is not None
        toks = list(self.tokenizer(text))
        if not toks:
            vec = zeros(self._dim, dtype=float32)
            return (
                _l2_normalize(vec.reshape(1, -1)).ravel() if self.l2_normalize else vec
            )
        vecs = []
        wts = []
        for t in toks:
            if t in self._kv:
                vecs.append(self._kv[t])
                if self._idf is not None:
                    wts.append(self._idf.get(t, 1.0))
                else:
                    wts.append(1.0)
        if not vecs:
            vec = zeros(self._dim, dtype=float32)
        else:
            V = vstack(vecs).astype(float32)
            w = asarray(wts, dtype=float32).reshape(-1, 1)
            if self.aggregation == "sum":
                vec = (V * w).sum(axis=0)
            else:
                denom = max(w.sum(), 1e-12)
                vec = (V * w).sum(axis=0) / denom
        if self.l2_normalize:
            vec = _l2_normalize(vec.reshape(1, -1)).ravel().astype(float32)
        return vec.astype(float32)


class SentenceTransformersVectorizer(SentenceVectorizer):
    def __init__(
        self,
        model_name: str,
        l2_normalize: bool = False,
        max_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.l2_normalize = l2_normalize
        self.max_length = max_length
        self._device = "cuda" if cuda.is_available() else "cpu"
        self._model: Optional[SentenceTransformer] = None
        self._dim: Optional[int] = None
        self._fitted = False

    def fit(
        self, corpus: Iterable[str] | None = None
    ) -> "SentenceTransformersVectorizer":
        self._model = SentenceTransformer(self.model_name, device=self._device)
        if self.max_length is not None:
            self._model.max_seq_length = int(self.max_length)
        probe = self._model.encode(
            [""], convert_to_numpy=True, normalize_embeddings=False
        )
        self._dim = int(probe.shape[1])
        self._fitted = True
        return self

    def encode_sentence(self, text: str) -> ndarray:
        if not self._fitted or self._model is None or self._dim is None:
            raise RuntimeError(
                "SentenceTransformersVectorizer is not fitted. Call fit() first."
            )
        self._model.eval()
        with no_grad():
            vec = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.l2_normalize,
                batch_size=1,
                show_progress_bar=False,
            ).astype(float32)
            return vec


class SentenceVectorizerFactory:
    _registry = {
        "tfidf": TfidfVectorizer,
        "word2vec": Word2VecVectorizer,
        "sbert": SentenceTransformersVectorizer,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> SentenceVectorizer:
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown vectorizer '{name}'. Choose from: {', '.join(cls._registry)}"
            )
        return cls._registry[key](**kwargs)
