from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REFERENCE_WIDTH = 1240.0
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
INK = (27, 26, 29, 255)
WHITE = (255, 255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace legacy MetabolicBridge labels in Figure 4.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pdf-output", type=Path)
    args = parser.parse_args()

    image = Image.open(args.source).convert("RGBA")
    scale = image.width / REFERENCE_WIDTH
    draw = ImageDraw.Draw(image)

    def rect(box: tuple[float, float, float, float]) -> None:
        draw.rectangle(tuple(round(value * scale) for value in box), fill=WHITE)

    def font(size: float) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(FONT_PATH), round(size * scale))

    def label(x: float, y: float, text: str, size: float, anchor: str = "la") -> None:
        draw.text(
            (round(x * scale), round(y * scale)),
            text,
            font=font(size),
            fill=INK,
            anchor=anchor,
        )

    # Panel a model heading and selected-model label.
    rect((728, 44, 1240, 82))
    label(739, 50, "PulseBridge drug-response dynamics", 27)
    rect((462, 503, 627, 534))
    label(544, 519, "PulseBridge", 18, anchor="mm")

    # Panel b title and right-hand comparison heading.
    rect((0, 560, 1240, 600))
    label(0, 568, "b CellCompass self-evolves PulseBridge through metabolic-labeling-guided optimization", 25)
    rect((757, 598, 1240, 666))
    label(761, 605, "Selected PulseBridge lowers pulse-", 24)
    label(888, 637, "identifiability error", 24)

    # Panel c title.
    rect((0, 1097, 1240, 1135))
    label(9, 1103, "c PulseBridge reveals timed drug-response programs and adaptive response branches", 25)

    # Flatten the transparent Illustrator canvas so that opaque correction
    # rectangles cannot acquire faint edges in Word/LibreOffice rendering.
    flattened = Image.alpha_composite(Image.new("RGBA", image.size, WHITE), image).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flattened.save(args.output, dpi=image.info.get("dpi", (300, 300)))
    if args.pdf_output is not None:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        flattened.save(args.pdf_output, "PDF", resolution=600.0)


if __name__ == "__main__":
    main()
