#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, UnidentifiedImageError


DEFAULT_INPUT_DIR = Path("/gs/bs/tga-RLA/qdeng/RAP/outputs/ref2d_eval_e10")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "integrated"
DEFAULT_SUBDIRS = (
    "baseline",
    "shift_only_log",
    "recovery_only_log",
    "recovery_aux_only_log_l03",
    "offset_recovery_log",
    "offset_recovery_aux_log_l03",
)
DEFAULT_EXCLUDE_SUBDIRS = ("legacy", "integrated")
DEFAULT_CROP = (0, 0, 0, 0)  # top, right, bottom, left
DEFAULT_TRIM_PADDING = 18
DEFAULT_TRIM_THRESHOLD = 8
DEFAULT_MAX_PANEL_WIDTH = 1800
DEFAULT_MAX_PANEL_HEIGHT = 1200
DEFAULT_LABEL_FONT_SIZE = 44
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
        default=None,
        help=(
            "Subfolders to merge, in display order. "
            "Default: auto-discover non-legacy visualization folders, ordered by known ref2d variants first."
        ),
    )
    parser.add_argument(
        "--exclude-subdirs",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_SUBDIRS),
        help=(
            "Top-level input subfolders excluded during auto-discovery. "
            f"Default: {' '.join(DEFAULT_EXCLUDE_SUBDIRS)}"
        ),
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default=list(DEFAULT_CROP),
        help=(
            "Fixed pixels cropped from each source PNG before auto-trim/layout. "
            f"Default: {' '.join(map(str, DEFAULT_CROP))}"
        ),
    )
    parser.add_argument(
        "--no-auto-trim",
        dest="auto_trim",
        action="store_false",
        help="Disable automatic white-margin trimming around each source PNG.",
    )
    parser.set_defaults(auto_trim=True)
    parser.add_argument(
        "--trim-padding",
        type=int,
        default=DEFAULT_TRIM_PADDING,
        help=f"Pixels kept around the auto-trim bounding box. Default: {DEFAULT_TRIM_PADDING}",
    )
    parser.add_argument(
        "--trim-threshold",
        type=int,
        default=DEFAULT_TRIM_THRESHOLD,
        help=f"RGB difference threshold used by auto-trim. Default: {DEFAULT_TRIM_THRESHOLD}",
    )
    parser.add_argument(
        "--max-panel-width",
        type=int,
        default=DEFAULT_MAX_PANEL_WIDTH,
        help=(
            "Resize each source PNG to this maximum width before integration. "
            "Use <=0 to disable. "
            f"Default: {DEFAULT_MAX_PANEL_WIDTH}"
        ),
    )
    parser.add_argument(
        "--max-panel-height",
        type=int,
        default=DEFAULT_MAX_PANEL_HEIGHT,
        help=(
            "Resize each source PNG to this maximum height before integration. "
            "Use <=0 to disable. "
            f"Default: {DEFAULT_MAX_PANEL_HEIGHT}"
        ),
    )
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=DEFAULT_LABEL_FONT_SIZE,
        help=f"Font size for the cell label overlay. Default: {DEFAULT_LABEL_FONT_SIZE}",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also output tokens that are missing from one or more subfolders.",
    )
    parser.add_argument(
        "--require-subdirs",
        action="store_true",
        help="Fail if any requested evaluation subfolder is missing. By default missing subfolders are skipped.",
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


def discover_subdirs(input_dir, output_dir, excluded):
    excluded = set(excluded)
    discovered = []
    output_dir = output_dir.resolve()

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in excluded:
            continue
        if subdir.resolve() == output_dir:
            continue

        image_dir = resolve_image_dir(subdir)
        if image_dir.is_dir() and any(image_dir.glob("*.png")):
            discovered.append(subdir.name)

    known_order = [name for name in DEFAULT_SUBDIRS if name in discovered]
    remaining = [name for name in discovered if name not in DEFAULT_SUBDIRS]
    return known_order + remaining


def collect_images(input_dir, subdirs, require_subdirs):
    image_map = {}
    for name in subdirs:
        subdir = input_dir / name
        image_dir = resolve_image_dir(subdir)
        if not image_dir.is_dir():
            if require_subdirs:
                raise FileNotFoundError(f"Missing image directory: {image_dir}")
            print(f"warning: missing image directory, skipped: {image_dir}", file=sys.stderr)
            continue

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


def auto_trim_image(image, padding, threshold):
    if padding < 0:
        raise ValueError(f"Trim padding must be non-negative: {padding}")
    if threshold < 0:
        raise ValueError(f"Trim threshold must be non-negative: {threshold}")

    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, BG_COLOR)
    diff = ImageChops.difference(rgb, bg).convert("L")
    if threshold > 0:
        diff = diff.point(lambda value: 255 if value > threshold else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return image

    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def resize_to_max(image, max_width, max_height):
    limits = []
    if max_width and max_width > 0:
        limits.append(max_width / image.width)
    if max_height and max_height > 0:
        limits.append(max_height / image.height)
    if not limits:
        return image

    scale = min(limits)
    if scale >= 1.0:
        return image

    resized = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


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


def prepare_image(image, label, crop, auto_trim, trim_padding, trim_threshold, max_panel_width, max_panel_height, font):
    image = crop_image(image.convert("RGB"), crop)
    if auto_trim:
        image = auto_trim_image(image, trim_padding, trim_threshold)
    image = resize_to_max(image, max_panel_width, max_panel_height)
    return draw_label(image, label, font)


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


def make_canvas(
    token,
    subdirs,
    image_map,
    crop,
    auto_trim,
    trim_padding,
    trim_threshold,
    max_panel_width,
    max_panel_height,
    font,
):
    opened = {}
    invalid = set()
    for name in subdirs:
        png_path = image_map[name].get(token)
        if png_path is not None:
            try:
                with Image.open(png_path) as image:
                    opened[name] = prepare_image(
                        image,
                        name,
                        crop,
                        auto_trim,
                        trim_padding,
                        trim_threshold,
                        max_panel_width,
                        max_panel_height,
                        font,
                    )
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
    subdirs_to_merge = args.subdirs
    if subdirs_to_merge is None:
        subdirs_to_merge = discover_subdirs(args.input_dir, args.output_dir, args.exclude_subdirs)
        if not subdirs_to_merge:
            raise RuntimeError(f"No non-excluded PNG visualization subfolders found under {args.input_dir}")

    image_map = collect_images(args.input_dir, subdirs_to_merge, args.require_subdirs)
    subdirs = [name for name in subdirs_to_merge if name in image_map]
    if not subdirs:
        raise RuntimeError(f"No image subfolders found under {args.input_dir}")
    tokens = tokens_to_merge(image_map, args.include_missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(args.label_font_size)
    for token in tokens:
        canvas = make_canvas(
            token,
            subdirs,
            image_map,
            tuple(args.crop),
            args.auto_trim,
            args.trim_padding,
            args.trim_threshold,
            args.max_panel_width,
            args.max_panel_height,
            font,
        )
        if canvas is not None:
            canvas.save(args.output_dir / f"{token}.png")

    print(f"Wrote {len(tokens)} integrated images to {args.output_dir}")


if __name__ == "__main__":
    main()
