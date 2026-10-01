from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, StreamObject


def replace_page_streams(reader: PdfReader, replacements: list[tuple[bytes, bytes]]) -> dict[bytes, int]:
    """Replace literal text in page and nested Form XObject streams."""
    counts = {old: 0 for old, _ in replacements}
    seen: set[int] = set()

    def visit(obj: object) -> None:
        try:
            resolved = obj.get_object()  # type: ignore[attr-defined]
        except AttributeError:
            resolved = obj

        object_id = id(resolved)
        if object_id in seen:
            return
        seen.add(object_id)

        if isinstance(resolved, StreamObject):
            data = resolved.get_data()
            updated = data
            for old, new in replacements:
                occurrence_count = updated.count(old)
                if occurrence_count:
                    updated = updated.replace(old, new)
                    counts[old] += occurrence_count
            if updated != data:
                resolved.set_data(updated)

        if isinstance(resolved, DictionaryObject):
            for value in resolved.values():
                visit(value)
        elif isinstance(resolved, ArrayObject):
            for value in resolved:
                visit(value)

    for page in reader.pages:
        visit(page)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct literal labels in a PDF-compatible Illustrator file.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--replace",
        action="append",
        nargs=2,
        metavar=("OLD", "NEW"),
        required=True,
        help="Literal UTF-8 text replacement; may be supplied more than once.",
    )
    args = parser.parse_args()

    replacements = [(old.encode("utf-8"), new.encode("utf-8")) for old, new in args.replace]
    reader = PdfReader(args.source)
    counts = replace_page_streams(reader, replacements)

    missing = [old.decode("utf-8") for old, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"Replacement target not found in visible page streams: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    with args.output.open("wb") as handle:
        writer.write(handle)

    for old, new in replacements:
        print(f"{old.decode('utf-8')} -> {new.decode('utf-8')}: {counts[old]} replacement(s)")


if __name__ == "__main__":
    main()
