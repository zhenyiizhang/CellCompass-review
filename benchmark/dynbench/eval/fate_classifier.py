"""
Unified MLP fate classifier for DynBench v2.

Trained on training data with known fate labels.
Used consistently across all metrics that need fate assignment:
  - GT generation (BoolODE simulation endpoints)
  - M3: classify predicted holdout cells
  - M4: classify SDE forward simulation endpoints  
  - M5: classify perturbation simulation endpoints
"""

import numpy as np
import pickle
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _build_mlp():
    """Build a simple MLP for fate classification using sklearn."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=False,
        random_state=42,
        verbose=False,
    )


def _concat_time(expr: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Concatenate expression matrix with time column."""
    time_col = np.asarray(time, dtype=np.float32).reshape(-1, 1)
    return np.hstack([expr, time_col])


class _PortableMLPClassifier:
    """Small inference-only proxy for sklearn MLPClassifier exports."""

    def __init__(
        self,
        *,
        classes: np.ndarray,
        coefs: list[np.ndarray],
        intercepts: list[np.ndarray],
        activation: str,
        out_activation: str,
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.coefs_ = [np.asarray(x, dtype=float) for x in coefs]
        self.intercepts_ = [np.asarray(x, dtype=float) for x in intercepts]
        self.activation = activation
        self.out_activation_ = out_activation

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -709, 709)))
        if self.activation == "identity":
            return x
        raise ValueError(f"Unsupported MLP activation: {self.activation}")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(np.clip(shifted, -709, 709))
        return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-300)

    def _decision(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float)
        for weights, bias in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            out = self._activate(out @ weights + bias)
        return out @ self.coefs_[-1] + self.intercepts_[-1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._decision(x)
        if self.out_activation_ == "softmax":
            return self._softmax(logits)
        if self.out_activation_ == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -709, 709)))
            if p.ndim == 2 and p.shape[1] == 1 and len(self.classes_) == 2:
                p1 = p[:, 0]
                return np.column_stack([1.0 - p1, p1])
            return p
        if self.out_activation_ == "identity":
            return logits
        raise ValueError(f"Unsupported MLP output activation: {self.out_activation_}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


def train_fate_classifier(
    expr: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    save_path: Optional[str] = None,
) -> object:
    """
    Train MLP fate classifier on ALL training data (including Progenitor).
    
    Features = gene expression + time (so the classifier can distinguish
    early progenitors from late transitioning cells at similar expression).
    
    Args:
        expr: (n_cells, n_genes) gene expression
        labels: (n_cells,) fate labels (e.g. 'Fate_A', 'Fate_B', 'Progenitor')
        time: (n_cells,) time values for each cell
        save_path: optional path to save the trained model
        
    Returns:
        Trained MLPClassifier
    """
    # Filter out only unknown/invalid labels, keep ALL biological categories
    valid_mask = np.array([l not in ['unknown', ''] for l in labels])
    
    if valid_mask.sum() < 10:
        raise ValueError(f"Too few valid cells ({valid_mask.sum()}) to train classifier")
    
    train_features = _concat_time(expr[valid_mask], time[valid_mask])
    train_labels = labels[valid_mask]
    
    clf = _build_mlp()
    clf.fit(train_features, train_labels)
    
    # Report accuracy
    train_acc = clf.score(train_features, train_labels)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"  Fate classifier trained: {valid_mask.sum()} cells, "
          f"features=expr({expr.shape[1]})+time, "
          f"accuracy={train_acc:.3f}, classes={list(clf.classes_)}")
    print(f"  Label distribution: {label_dist}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"  Saved to: {save_path}")
    
    return clf


def load_fate_classifier(path: str) -> object:
    """Load a trained fate classifier."""
    # Week-1 classifier pickles may encode NumPy's MT19937 bit generator as
    # the class object rather than the module/name string expected by newer
    # NumPy. Keep the compatibility shim scoped to this unpickle call.
    try:
        import numpy.random._pickle as np_random_pickle
    except Exception:
        np_random_pickle = None

    if np_random_pickle is None:
        with open(path, 'rb') as f:
            return pickle.load(f)

    original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        return _load_portable_fate_classifier(path, exc)
    finally:
        np_random_pickle.__bit_generator_ctor = original_ctor


def _load_portable_fate_classifier(path: str, original_error: Exception) -> object:
    """Load an sklearn MLP classifier through a portable weight export.

    This fallback is only used when direct unpickling fails because the task
    package was created with an incompatible NumPy/sklearn stack.
    """
    candidates = []
    env_python = os.environ.get("DYNBENCH_CLASSIFIER_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend(
        Path(p)
        for p in [
            "/opt/anaconda3/envs/iam_env/bin/python",
            "/opt/anaconda3/envs/gene/bin/python",
            "/opt/anaconda3/envs/gwot/bin/python",
        ]
    )

    script = """
import json
import pickle
import sys
from pathlib import Path

import numpy as np

with open(sys.argv[1], "rb") as handle:
    clf = pickle.load(handle)

meta = {
    "activation": str(getattr(clf, "activation", "")),
    "out_activation": str(getattr(clf, "out_activation_", "")),
    "n_layers": int(getattr(clf, "n_layers_", len(getattr(clf, "coefs_", [])) + 1)),
    "n_outputs": int(getattr(clf, "n_outputs_", 1)),
}
arrays = {"classes": np.asarray(clf.classes_)}
for idx, coef in enumerate(clf.coefs_):
    arrays[f"coef_{idx}"] = np.asarray(coef)
for idx, intercept in enumerate(clf.intercepts_):
    arrays[f"intercept_{idx}"] = np.asarray(intercept)

np.savez(sys.argv[2], **arrays)
Path(sys.argv[3]).write_text(json.dumps(meta), encoding="utf-8")
"""

    errors = []
    with tempfile.TemporaryDirectory(prefix="dynbench-clf-") as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / "classifier_arrays.npz"
        meta_path = tmp_path / "classifier_meta.json"
        for python in candidates:
            if not python.exists():
                continue
            try:
                subprocess.run(
                    [str(python), "-c", script, str(path), str(npz_path), str(meta_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                )
                with np.load(npz_path, allow_pickle=False) as payload:
                    classes = payload["classes"]
                    coefs = [
                        payload[f"coef_{idx}"]
                        for idx in range(len([k for k in payload.files if k.startswith("coef_")]))
                    ]
                    intercepts = [
                        payload[f"intercept_{idx}"]
                        for idx in range(len([k for k in payload.files if k.startswith("intercept_")]))
                    ]
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return _PortableMLPClassifier(
                    classes=classes,
                    coefs=coefs,
                    intercepts=intercepts,
                    activation=meta["activation"],
                    out_activation=meta["out_activation"],
                )
            except Exception as exc:
                errors.append(f"{python}: {exc}")

    raise RuntimeError(
        "Could not load fate_classifier.pkl directly or via portable export. "
        f"Original error: {original_error}. Fallback errors: {errors}"
    ) from original_error


def classify_cells(
    clf,
    expr: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """
    Classify cells into fates using trained MLP.
    
    Args:
        clf: trained classifier
        expr: (n_cells, n_genes) gene expression
        time: (n_cells,) time values
        
    Returns:
        (n_cells,) predicted fate labels
    """
    features = _concat_time(expr, time)
    return clf.predict(features)
