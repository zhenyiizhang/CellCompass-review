"""Shared protein-PCA phenotype readout for CITE-PCFM biology figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ProteinPhenotypeReadout:
    """MLP probability readout from observed protein PCA space to phenotype labels."""

    model: object
    label_encoder: LabelEncoder
    pheno_order: list[str]

    def predict(self, protein: np.ndarray) -> np.ndarray:
        encoded = self.model.predict(np.asarray(protein, dtype=np.float32))
        return self.label_encoder.inverse_transform(np.asarray(encoded, dtype=int))

    def predict_proba_ordered(self, protein: np.ndarray) -> np.ndarray:
        protein = np.asarray(protein, dtype=np.float32)
        raw = self.model.predict_proba(protein)
        out = np.zeros((protein.shape[0], len(self.pheno_order)), dtype=np.float64)
        model_classes = self.label_encoder.inverse_transform(np.asarray(self.model.classes_, dtype=int))
        class_to_col = {str(label): i for i, label in enumerate(model_classes)}
        for j, label in enumerate(self.pheno_order):
            if label in class_to_col:
                out[:, j] = raw[:, class_to_col[label]]
        row_sum = out.sum(axis=1, keepdims=True)
        np.divide(out, row_sum, out=out, where=row_sum > 0)
        return out


def _make_mlp(random_state: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    )


def train_mlp_protein_phenotype_readout(
    protein: np.ndarray,
    labels: np.ndarray,
    pheno_order: list[str],
    random_state: int,
    holdout_fraction: float = 0.25,
) -> tuple[ProteinPhenotypeReadout, dict[str, object], pd.DataFrame]:
    """Train final MLP on all observed protein PCA and audit on a held-out split."""

    protein = np.asarray(protein, dtype=np.float32)
    labels = np.asarray(labels, dtype=str)
    label_encoder = LabelEncoder().fit(labels)
    y = label_encoder.transform(labels)

    idx = np.arange(protein.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=y,
    )

    audit_model = _make_mlp(random_state)
    audit_model.fit(protein[train_idx], y[train_idx])
    pred = audit_model.predict(protein[test_idx])
    pred_labels = label_encoder.inverse_transform(np.asarray(pred, dtype=int))
    true_labels = labels[test_idx]
    confusion = confusion_matrix(true_labels, pred_labels, labels=pheno_order)
    confusion_df = pd.DataFrame(confusion, index=pheno_order, columns=pheno_order)

    per_class_recall = {}
    for i, label in enumerate(pheno_order):
        denom = float(confusion[i, :].sum())
        per_class_recall[label] = float(confusion[i, i] / denom) if denom > 0 else float("nan")

    metrics = {
        "classifier": "StandardScaler + MLPClassifier",
        "feature_space": "observed protein PCA",
        "hidden_layer_sizes": [64, 32],
        "holdout_fraction": float(holdout_fraction),
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted")),
        "per_class_recall": per_class_recall,
    }

    final_model = _make_mlp(random_state)
    final_model.fit(protein, y)
    readout = ProteinPhenotypeReadout(final_model, label_encoder, list(pheno_order))
    return readout, metrics, confusion_df

