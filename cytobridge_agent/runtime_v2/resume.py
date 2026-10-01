from __future__ import annotations

import base64
from collections import defaultdict
import os
from pathlib import Path
import pickle
import re
import sqlite3
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import InMemorySaver

try:  # Optional dependency in older installs; requirements-agent.txt installs it.
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover - exercised only when optional dep is absent.
    SqliteSaver = None  # type: ignore[assignment]

from .state import ensure_runtime_v2_state

SQLITE_CHECKPOINT_FORMAT = "langgraph_sqlite_v1"
INMEMORY_CHECKPOINT_FORMAT = "langgraph_inmemory_v1"


def restore_runtime_state(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    return ensure_runtime_v2_state(agent_state)


def _safe_checkpoint_component(value: str) -> str:
    text = str(value or "").strip() or "runtime-v2-default"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:160] or "runtime-v2-default"


def _checkpoint_base_dir() -> Path:
    override = os.getenv("CYTOBRIDGE_LANGGRAPH_CHECKPOINT_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cellcompass" / "conversations" / "checkpoints"


def default_checkpoint_path(state: Optional[Dict[str, Any]] = None) -> Path:
    state = state or {}
    session_id = _safe_checkpoint_component(str(state.get("session_id") or "runtime-v2-default"))
    return _checkpoint_base_dir() / f"{session_id}.sqlite"


def _annotate_checkpointer(checkpointer: Any, *, backend: str, path: Optional[Path] = None) -> Any:
    try:
        setattr(checkpointer, "_cytobridge_checkpoint_backend", backend)
        if path is not None:
            setattr(checkpointer, "_cytobridge_checkpoint_path", str(path))
    except Exception:
        pass
    return checkpointer


def create_runtime_checkpointer(
    state: Optional[Dict[str, Any]] = None,
    *,
    checkpoint_path: Optional[str] = None,
) -> Any:
    """Create the runtime checkpointer.

    Prefer LangGraph's SQLite saver so graph checkpoints are persisted by the
    official checkpointer backend. ``InMemorySaver`` remains as a fallback for
    minimal installs and tests that explicitly force it.
    """
    force_memory = os.getenv("CYTOBRIDGE_LANGGRAPH_CHECKPOINTER", "").strip().lower() in {
        "memory",
        "inmemory",
        "in_memory",
    }
    if force_memory or SqliteSaver is None:
        return _annotate_checkpointer(InMemorySaver(), backend="memory")

    path = Path(checkpoint_path).expanduser() if checkpoint_path else default_checkpoint_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")
    saver = SqliteSaver(conn)
    saver.setup()
    return _annotate_checkpointer(saver, backend="sqlite", path=path)


def _storage_to_plain(storage: Any) -> Dict[str, Any]:
    return {
        thread_id: {checkpoint_ns: dict(checkpoints) for checkpoint_ns, checkpoints in namespaces.items()}
        for thread_id, namespaces in dict(storage).items()
    }


def _writes_to_plain(writes: Any) -> Dict[str, Any]:
    return {key: dict(value) for key, value in dict(writes).items()}


def _blobs_to_plain(blobs: Any) -> Dict[str, Any]:
    return dict(blobs)


def _plain_to_storage(data: Dict[str, Any]) -> Any:
    storage = defaultdict(lambda: defaultdict(dict))
    for thread_id, namespaces in (data or {}).items():
        for checkpoint_ns, checkpoints in dict(namespaces).items():
            storage[thread_id][checkpoint_ns] = dict(checkpoints)
    return storage


def _plain_to_writes(data: Dict[str, Any]) -> Any:
    writes = defaultdict(dict)
    for key, value in (data or {}).items():
        writes[key] = dict(value)
    return writes


def _plain_to_blobs(data: Dict[str, Any]) -> Any:
    blobs = defaultdict()
    blobs.update(data or {})
    return blobs


def export_checkpointer_snapshot(checkpointer: Any) -> Dict[str, Any]:
    backend = str(getattr(checkpointer, "_cytobridge_checkpoint_backend", "") or "").strip()
    if backend == "sqlite":
        path = str(getattr(checkpointer, "_cytobridge_checkpoint_path", "") or "").strip()
        return {
            "format": SQLITE_CHECKPOINT_FORMAT,
            "backend": "sqlite",
            "path": path,
        }

    if not isinstance(checkpointer, InMemorySaver):
        return {
            "format": "langgraph_external_v1",
            "backend": backend or type(checkpointer).__name__,
        }

    payload = {
        "storage": _storage_to_plain(checkpointer.storage),
        "writes": _writes_to_plain(checkpointer.writes),
        "blobs": _blobs_to_plain(checkpointer.blobs),
    }
    encoded = base64.b64encode(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
    return {
        "format": INMEMORY_CHECKPOINT_FORMAT,
        "payload_b64": encoded,
    }


def checkpoint_snapshot_path(snapshot: Optional[Dict[str, Any]]) -> str:
    if not snapshot:
        return ""
    if snapshot.get("format") != SQLITE_CHECKPOINT_FORMAT:
        return ""
    return str(snapshot.get("path") or "").strip()


def restore_checkpointer_snapshot(checkpointer: Any, snapshot: Optional[Dict[str, Any]]) -> bool:
    if not snapshot:
        return False
    if snapshot.get("format") == SQLITE_CHECKPOINT_FORMAT:
        path = checkpoint_snapshot_path(snapshot)
        return bool(path and Path(path).expanduser().exists())

    if snapshot.get("format") != INMEMORY_CHECKPOINT_FORMAT:
        return False
    if not isinstance(checkpointer, InMemorySaver):
        return False
    encoded = snapshot.get("payload_b64")
    if not encoded:
        return False
    payload = pickle.loads(base64.b64decode(encoded.encode("ascii")))
    checkpointer.storage = _plain_to_storage(payload.get("storage") or {})
    checkpointer.writes = _plain_to_writes(payload.get("writes") or {})
    checkpointer.blobs = _plain_to_blobs(payload.get("blobs") or {})
    return True
