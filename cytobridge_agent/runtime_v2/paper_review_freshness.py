from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List


_PAPER_ROOT_FILES = {
    "main.tex",
    "supplement.tex",
    "paper_plan.md",
    "source_obligation_matrix.md",
    "theory_obligation_ledger.md",
    "algorithm_detail_ledger.md",
    "evidence_ledger.md",
    "citation_ledger.md",
    "narrative_report.md",
}


def _is_tracked_file(path: Path) -> bool:
    if path.name in _PAPER_ROOT_FILES:
        return True
    parent = path.parent.name
    if parent == "sections" and path.suffix == ".tex":
        return True
    if parent == "checks" and path.suffix == ".md":
        return True
    return False


def _paper_files_from_dir(path: Path) -> List[Path]:
    if path.name == "sections":
        return sorted(child for child in path.glob("*.tex") if child.is_file())
    if path.name == "checks":
        return sorted(child for child in path.glob("*.md") if child.is_file())

    files: List[Path] = []
    for name in sorted(_PAPER_ROOT_FILES):
        candidate = path / name
        if candidate.is_file():
            files.append(candidate)
    sections_dir = path / "sections"
    if sections_dir.is_dir():
        files.extend(sorted(child for child in sections_dir.glob("*.tex") if child.is_file()))
    checks_dir = path / "checks"
    if checks_dir.is_dir():
        files.extend(sorted(child for child in checks_dir.glob("*.md") if child.is_file()))
    return files


def collect_paper_review_files(paths: Iterable[Any]) -> List[Path]:
    found: Dict[str, Path] = {}
    for raw_path in paths or []:
        text = str(raw_path or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.is_dir():
            for child in _paper_files_from_dir(path):
                try:
                    resolved = child.resolve()
                except Exception:
                    resolved = child
                found[str(resolved)] = resolved
        elif path.is_file() and _is_tracked_file(path):
            found[str(path)] = path
    return [found[key] for key in sorted(found)]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paper_review_manifest(paths: Iterable[Any]) -> Dict[str, Any]:
    source_paths = [str(item) for item in (paths or []) if str(item or "").strip()]
    files = []
    for path in collect_paper_review_files(source_paths):
        try:
            stat = path.stat()
            file_hash = _hash_file(path)
        except Exception:
            continue
        files.append(
            {
                "path": str(path),
                "sha256": file_hash,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", 0)),
            }
        )

    overall = hashlib.sha256()
    for item in files:
        overall.update(str(item["path"]).encode("utf-8", errors="surrogateescape"))
        overall.update(b"\0")
        overall.update(str(item["sha256"]).encode("ascii"))
        overall.update(b"\0")

    return {
        "version": 1,
        "algorithm": "sha256",
        "source_paths": source_paths,
        "digest": overall.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def paper_review_manifest_status(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        return {"fresh": False, "reason": "review has no file-hash manifest"}
    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list) or not source_paths:
        source_paths = [
            str(item.get("path") or "")
            for item in (manifest.get("files") or [])
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        ]
    if not source_paths:
        return {"fresh": False, "reason": "review hash manifest has no source paths"}

    current = build_paper_review_manifest(source_paths)
    expected_digest = str(manifest.get("digest") or "").strip()
    current_digest = str(current.get("digest") or "").strip()
    if not expected_digest:
        return {"fresh": False, "reason": "review hash manifest has no digest", "current": current}
    if int(current.get("file_count") or 0) <= 0:
        return {"fresh": False, "reason": "no tracked paper files found for review freshness check", "current": current}
    if current_digest != expected_digest:
        return {
            "fresh": False,
            "reason": "tracked paper file hashes changed after reviewer approval",
            "expected_digest": expected_digest,
            "current_digest": current_digest,
            "current": current,
        }
    return {"fresh": True, "reason": "tracked paper file hashes match reviewer approval", "current": current}
