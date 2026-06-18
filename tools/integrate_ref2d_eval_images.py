#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


DEFAULT_INPUT_DIR = Path("/gs/bs/tga-RLA/qdeng/RAP/outputs/ref2d_eval_e10")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "integrated"
DEFAULT_SUBDIRS = (
    "baseline",
    "shift_only",
    "recovery_only",
    "offset_recovery",
)
ALLOWED_SUBDIRS = set(DEFAULT_SUBDIRS)
DEFAULT_CROP = (175, 0, 145, 0)  # top, right, bottom, left
GRID_GAP = 28
OUTER_PADDING = 28
PANEL_PADDING = 8
LABEL_PAD_X = 22
LABEL_PAD_Y = 12
LABEL_MARGIN = 22
BG_COLOR = (255, 255, 255)
PANEL_BG = (248, 248, 248)
PANEL_BORDER = (220, 220, 220)
LABEL_BG = (20, 25, 31, 220)
LABEL_FG = (255, 255, 255)
MISSING_FG = (120, 120, 120)
INVALID_FG = (190, 70, 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge same-token ref2d evaluation PNGs from multiple subfolders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root directory containing evaluation subfolders. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for integrated PNGs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=list(DEFAULT_SUBDIRS),
        help="Subfolders to merge, in display order.",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default=list(DEFAULT_CROP),
        help=(
            "Pixels cropped from each source PNG before layout. "
            f"Default: {' '.join(map(str, DEFAULT_CROP))}"
        ),
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also output tokens that are missing from one or more subfolders.",
    )
    return parser.parse_args()


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def resolve_image_dir(subdir):
    vis_dir = subdir / "visualizations"
    if vis_dir.is_dir():
        return vis_dir
    return subdir


def collect_images(input_dir, subdirs):
    image_map = {}
    for name in subdirs:
        if name not in ALLOWED_SUBDIRS:
            allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
            raise ValueError(f"Unsupported subdir label '{name}'. Allowed labels: {allowed}")

        subdir = input_dir / name
        image_dir = resolve_image_dir(subdir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        token_to_path = {}
        for png_path in sorted(image_dir.glob("*.png")):
            token_to_path[png_path.stem] = png_path
        image_map[name] = token_to_path
    return image_map


def tokens_to_merge(image_map, include_missing):
    token_sets = [set(paths) for paths in image_map.values()]
    if include_missing:
        return sorted(set.union(*token_sets)) if token_sets else []
    return sorted(set.intersection(*token_sets)) if token_sets else []


def text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def crop_image(image, crop):
    top, right, bottom, left = crop
    if min(crop) < 0:
        raise ValueError(f"Crop values must be non-negative: {crop}")

    width, height = image.size
    crop_box = (left, top, width - right, height - bottom)
    if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
        raise ValueError(f"Crop {crop} is too large for image size {image.size}")
    return image.crop(crop_box)


def draw_label(image, label, font):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_w, text_h = text_size(draw, label, font)
    label_w = text_w + 2 * LABEL_PAD_X
    label_h = text_h + 2 * LABEL_PAD_Y
    x0 = LABEL_MARGIN
    y0 = LABEL_MARGIN
    x1 = x0 + label_w
    y1 = y0 + label_h

    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=LABEL_BG)
    else:
        draw.rectangle((x0, y0, x1, y1), fill=LABEL_BG)
    draw.text((x0 + LABEL_PAD_X, y0 + LABEL_PAD_Y), label, fill=LABEL_FG, font=font)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_panel(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill=PANEL_BG, outline=PANEL_BORDER, width=1)


def fit_into_panel(image, panel_size):
    panel_w, panel_h = panel_size
    image_w, image_h = image.size
    if image_w <= panel_w and image_h <= panel_h:
        return image

    scale = min(panel_w / image_w, panel_h / image_h)
    resized = (
        max(1, int(round(image_w * scale))),
        max(1, int(round(image_h * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def draw_centered_message(draw, box, message, font, fill):
    x0, y0, x1, y1 = box
    text_w, text_h = text_size(draw, message, font)
    draw.text(
        (x0 + max(0, (x1 - x0 - text_w) // 2), y0 + max(0, (y1 - y0 - text_h) // 2)),
        message,
        fill=fill,
        font=font,
    )


def make_canvas(token, subdirs, image_map, crop, font):
    opened = {}
    invalid = set()
    for name in subdirs:
        png_path = image_map[name].get(token)
        if png_path is not None:
            try:
                with Image.open(png_path) as image:
                    cleaned = crop_image(image.convert("RGB"), crop)
                    opened[name] = draw_label(cleaned, name, font)
            except (UnidentifiedImageError, OSError) as exc:
                invalid.add(name)
                print(f"warning: cannot read {png_path}: {exc}", file=sys.stderr)

    if not opened:
        return None

    max_w = max(image.width for image in opened.values())
    max_h = max(image.height for image in opened.values())
    cols = 2 if len(subdirs) > 1 else 1
    rows = int(math.ceil(len(subdirs) / cols))
    panel_w = max_w + 2 * PANEL_PADDING
    panel_h = max_h + 2 * PANEL_PADDING
    image_panel_size = (max_w, max_h)
    canvas_w = 2 * OUTER_PADDING + cols * panel_w + (cols - 1) * GRID_GAP
    canvas_h = 2 * OUTER_PADDING + rows * panel_h + (rows - 1) * GRID_GAP

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for index, name in enumerate(subdirs):
        col = index % cols
        row = index // cols
        panel_x = OUTER_PADDING + col * (panel_w + GRID_GAP)
        panel_y = OUTER_PADDING + row * (panel_h + GRID_GAP)
        image_x = panel_x + PANEL_PADDING
        image_y = panel_y + PANEL_PADDING

        draw_panel(draw, (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h))
        image = opened.get(name)
        image_area = (image_x, image_y, image_x + max_w, image_y + max_h)
        if image is None:
            if name in invalid:
                draw_centered_message(draw, image_area, f"invalid image: {name}", font, INVALID_FG)
            else:
                draw_centered_message(draw, image_area, f"missing: {name}", font, MISSING_FG)
            continue

        image = fit_into_panel(image, image_panel_size)
        paste_x = image_x + (max_w - image.width) // 2
        paste_y = image_y + (max_h - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))

    return canvas


def main():
    args = parse_args()
    image_map = collect_images(args.input_dir, args.subdirs)
    tokens = tokens_to_merge(image_map, args.include_missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(54)
    for token in tokens:
        canvas = make_canvas(token, args.subdirs, image_map, tuple(args.crop), font)
        if canvas is not None:
            canvas.save(args.output_dir / f"{token}.png")

    print(f"Wrote {len(tokens)} integrated images to {args.output_dir}")


if __name__ == "__main__":
    main()
