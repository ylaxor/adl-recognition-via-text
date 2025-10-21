from logging import WARNING, getLogger
from multiprocessing import cpu_count
from warnings import filterwarnings

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from numpy import arange, array, ndarray, ones, random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_X_y
from torch import (
    FloatTensor,
    LongTensor,
    Tensor,
    argmax,
    clamp,
    manual_seed,
    no_grad,
    optim,
    set_float32_matmul_precision,
    softmax,
)
from torch.nn import LSTM, CrossEntropyLoss, Dropout, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

set_float32_matmul_precision("high")


def device_info_filter(record):
    return "PU available: " not in record.getMessage()


getLogger("lightning.pytorch.utilities.rank_zero").addFilter(device_info_filter)
getLogger("pytorch_lightning").setLevel(WARNING)
getLogger("lightning.pyutilities.rank_zero").setLevel(WARNING)
getLogger("lightning.pyaccelerators.cuda").setLevel(WARNING)
getLogger("lightning").setLevel(0)
getLogger("lightning.pytorch.accelerators.cuda").setLevel(WARNING)


class LSTMModule(LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        learning_rate_init: float = 0.001,
        weight_decay: float = 0.0,
        solver: str = "adam",
        class_weights: Tensor | None = None,
        use_masking: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = Dropout(dropout)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = Linear(lstm_output_size, num_classes)
        self.criterion = CrossEntropyLoss(weight=class_weights)
        self.train_acc = []
        self.val_acc = []

    def forward(self, x, mask=None):
        if mask is not None and self.hparams.get("use_masking", True):
            lengths = mask.sum(dim=1).long().cpu()
            lengths = clamp(lengths, min=1)
            packed_input = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
            batch_size = x.size(0)
            last_output = lstm_out[range(batch_size), lengths - 1]
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
            last_output = lstm_out[:, -1, :]

        last_output = self.dropout(last_output)
        output = self.classifier(last_output)
        return output

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None

        y_hat = self(x, mask)
        loss = self.criterion(y_hat, y)
        preds = argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None

        y_hat = self(x, mask)
        loss = self.criterion(y_hat, y)
        preds = argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc, "preds": preds, "targets": y}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.hparams["solver"].lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams["learning_rate_init"],
                weight_decay=self.hparams["weight_decay"],
            )
        elif self.hparams["solver"].lower() == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams["learning_rate_init"],
                weight_decay=self.hparams["weight_decay"],
                momentum=0.9,
            )
        elif self.hparams["solver"].lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams["learning_rate_init"],
                weight_decay=self.hparams["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams['optimizer']}")
        return optimizer


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        learning_rate_init: float = 0.001,
        weight_decay: float = 0.0,
        solver: str = "adam",
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_fraction: float = 0.2,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        class_weight: str | dict | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        use_masking: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate_init = learning_rate_init
        self.weight_decay = weight_decay
        self.solver = solver
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.use_masking = use_masking

    def _validate_params(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= self.validation_fraction <= 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_iter_no_change <= 0:
            raise ValueError("patience must be positive")

    def _setup_random_state(self):
        if self.random_state is not None:
            seed_everything(self.random_state, workers=True)
            manual_seed(self.random_state)
            random.seed(self.random_state)

    def _compute_class_weights(self, y: ndarray) -> Tensor | None:
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str) and self.class_weight == "balanced":
            class_weights = compute_class_weight(
                "balanced", classes=arange(self.n_classes_), y=y
            )
            return FloatTensor(class_weights)
        elif isinstance(self.class_weight, dict):
            weights = ones(self.n_classes_)
            for class_idx, weight in self.class_weight.items():
                weights[class_idx] = weight
            return FloatTensor(weights)
        else:
            raise ValueError("class_weight must be 'balanced', dict, or None")

    def _create_data_loaders(
        self, X: ndarray, y: ndarray, masks: ndarray | None = None
    ) -> tuple[DataLoader, DataLoader | None]:
        X_tensor = FloatTensor(X)
        y_tensor = LongTensor(y)

        masks_tensor = None
        if masks is not None:
            masks_tensor = FloatTensor(masks)

        if self.validation_fraction > 0:
            n_samples = len(X)
            n_val = int(n_samples * self.validation_fraction)
            indices = arange(n_samples)
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(indices)
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            if masks_tensor is not None:
                train_dataset = TensorDataset(
                    X_tensor[train_indices],
                    y_tensor[train_indices],
                    masks_tensor[train_indices],
                )
                val_dataset = TensorDataset(
                    X_tensor[val_indices],
                    y_tensor[val_indices],
                    masks_tensor[val_indices],
                )
            else:
                train_dataset = TensorDataset(
                    X_tensor[train_indices],
                    y_tensor[train_indices],
                )
                val_dataset = TensorDataset(
                    X_tensor[val_indices],
                    y_tensor[val_indices],
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cpu_count() or 0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=cpu_count() or 0,
            )
            return train_loader, val_loader
        else:
            if masks_tensor is not None:
                train_dataset = TensorDataset(X_tensor, y_tensor, masks_tensor)
            else:
                train_dataset = TensorDataset(X_tensor, y_tensor)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cpu_count() or 0,
            )
            return train_loader, None

    def fit(self, X, y, masks=None):
        self._validate_params()
        self._setup_random_state()
        X, y = check_X_y(X, y, allow_nd=True)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif X.ndim != 3:
            raise ValueError("X must be 2D or 3D array")
        if masks is not None:
            if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[1]:
                raise ValueError(
                    f"masks shape {masks.shape} incompatible with X shape {X.shape}"
                )

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.input_size_ = X.shape[2]
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        y_encoded = array(y_encoded)
        class_weights = self._compute_class_weights(y_encoded)
        self.model_ = LSTMModule(
            input_size=self.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.n_classes_,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            learning_rate_init=self.learning_rate_init,
            weight_decay=self.weight_decay,
            solver=self.solver,
            class_weights=class_weights,
            use_masking=self.use_masking,
        )
        train_loader, val_loader = self._create_data_loaders(X, y_encoded, masks)
        callbacks = []
        if self.early_stopping and val_loader is not None:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=self.n_iter_no_change,
                mode="min",
                verbose=self.verbose,
            )
            callbacks.append(early_stop_callback)
        if not self.verbose:
            filterwarnings("ignore", ".*does not have many workers.*")
            self.trainer_ = Trainer(
                default_root_dir="./log",
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                deterministic=self.random_state is not None,
                enable_progress_bar=self.verbose,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
                log_every_n_steps=1,
            )
        else:
            self.trainer_ = Trainer(
                default_root_dir="./log",
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                deterministic=self.random_state is not None,
                enable_progress_bar=self.verbose,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
        self.trainer_.fit(
            self.model_,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        return self

    def predict(self, X, masks=None):
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before making predictions")
        X = check_array(X, allow_nd=True)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif X.ndim != 3:
            raise ValueError("X must be 2D or 3D array")
        if X.shape[2] != self.input_size_:
            raise ValueError(
                f"X has {X.shape[2]} features, but model expects {self.input_size_}"
            )
        if masks is not None:
            if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[1]:
                raise ValueError(
                    f"masks shape {masks.shape} incompatible with X shape {X.shape}"
                )

        X_tensor = FloatTensor(X)
        if masks is not None:
            masks_tensor = FloatTensor(masks)
            dataset = TensorDataset(X_tensor, masks_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.model_.eval()
        predictions = []
        with no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    X_batch, mask_batch = batch
                else:
                    X_batch = batch[0]
                    mask_batch = None

                y_pred = self.model_(X_batch, mask_batch)
                pred_labels = argmax(y_pred, dim=1)
                predictions.extend(pred_labels.cpu().numpy())
        predictions = array(predictions)
        return self.label_encoder_.inverse_transform(predictions)

    def predict_proba(self, X, masks=None):
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before making predictions")
        X = check_array(X, allow_nd=True)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif X.ndim != 3:
            raise ValueError("X must be 2D or 3D array")
        if X.shape[2] != self.input_size_:
            raise ValueError(
                f"X has {X.shape[2]} features, but model expects {self.input_size_}"
            )
        if masks is not None:
            if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[1]:
                raise ValueError(
                    f"masks shape {masks.shape} incompatible with X shape {X.shape}"
                )
        X_tensor = FloatTensor(X)
        if masks is not None:
            masks_tensor = FloatTensor(masks)
            dataset = TensorDataset(X_tensor, masks_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.model_.eval()
        probabilities = []
        with no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    X_batch, mask_batch = batch
                else:
                    X_batch = batch[0]
                    mask_batch = None
                y_logits = self.model_(X_batch, mask_batch)
                y_proba = softmax(y_logits, dim=1)
                probabilities.extend(y_proba.cpu().numpy())
        return array(probabilities)

    def score(self, X, y, sample_weight=None, masks=None):
        y_pred = self.predict(X, masks)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "learning_rate_init": self.learning_rate_init,
            "weight_decay": self.weight_decay,
            "solver": self.solver,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "use_masking": self.use_masking,
        }

    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self
        return self
        return self
        return self
