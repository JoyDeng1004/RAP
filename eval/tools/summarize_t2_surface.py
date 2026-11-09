"""T2 surface-correspondence diagnostic: summarize the run_t2_surface_v1.py run of one tag into
eval/t2_results/t2_surface_<tag>.md.

Means over tokens with 95% log-clustered bootstrap CIs (summarize_t1.boot_ci), c1 = raster, identity rig.

Pre-registered reading (height presets h_p1.0 and h_m0.7, both must agree):
  GEOM  the encoder sees one surface the same way and the T2 anchor drop comes from *which* surface the
        fixed 3D reference points land on: surf_cos CI low >= 0.97 and (surf_cos - anchor_cos_drawn)
        CI low > 0.03  ->  same-surface feature alignment (entry point 1's loss) has nothing to fix
  ENC   the same surface itself looks different to the encoder: surf_cos CI high <= 0.93
  otherwise inconclusive. Longitudinal / lateral / focal presets are the controls.
Raster only: no texture, and the renderer shades by depth (surf_rgb_diff), so this bounds what a
textured image would show; it does not replace a real-image test.

    python tools/summarize_t2_surface.py --tag full
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, token_logs  # noqa: E402

ORDER = ("ident", "h_p1.0", "h_m0.7", "d_p1.0", "d_m1.0", "l_p0.5", "l_m0.5", "f_0.8", "f_1.25")
HEIGHT = ("h_p1.0", "h_m0.7")
GEOM_SURF_MIN, GEOM_GAP_MIN, ENC_SURF_MAX = 0.97, 0.03, 0.93


def ci(s: pd.Series, logs: pd.Series, rng, n_boot: int):
    s = s.dropna()
    return None if s.empty else boot_ci(s, logs, n_boot, rng)


def cell(t, fmt="{:.3f}") -> str:
    return "–" if t is None else f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"


def mean(s, fmt="{:.3f}") -> str:
    s = s.dropna() if s is not None else None
    return "–" if s is None or s.empty else fmt.format(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "t2_results")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    logs = token_logs(args.subset_csv)

    run_dir = args.exp_root / f"t2surf_{args.tag}"
    csv = latest_csv(run_dir) if run_dir.is_dir() else None
    if csv is None:
        raise SystemExit(f"no run under {run_dir}")
    df = pd.read_csv(csv, index_col=0)
    presets = [p for p in ORDER if p in set(df["preset"])] + sorted(set(df["preset"]) - set(ORDER))

    rows1, rows2, gap, surf = [], [], {}, {}
    for p in presets:
        g = df[df["preset"] == p]
        ok = g[g["valid"].astype(bool)].set_index("token")
        surf[p] = ci(ok["surf_cos"], logs, rng, args.n_boot)
        gap[p] = ci(ok["surf_cos"] - ok["anchor_cos_drawn"], logs, rng, args.n_boot)
        rows1.append(f"| {p} | {len(ok)} | {len(g) - len(ok)} | {cell(surf[p])} | "
                     f"{cell(ci(ok['surf_samepix_cos'], logs, rng, args.n_boot))} | "
                     f"{cell(ci(ok['anchor_cos'], logs, rng, args.n_boot))} | "
                     f"{cell(ci(ok['anchor_cos_drawn'], logs, rng, args.n_boot))} | {cell(gap[p], '{:+.3f}')} | "
                     f"{mean(ok.get('anchor_cos_t2'))} |")
        rows2.append(f"| {p} | {mean(ok['surf_match_frac'])} | {mean(ok['n_out'] / ok['n_surface'])} | "
                     f"{mean(ok['n_mismatch'] / ok['n_surface'])} | {mean(ok['surf_shift_cells'], '{:.2f}')} | "
                     f"{mean(ok['surf_depth'], '{:.1f}')} | {mean(ok['surf_rgb_diff'], '{:.1f}')} | "
                     f"{mean(ok['surf_cos_box'])} / {mean(ok['surf_cos_map'])} | "
                     f"{mean(ok['surf_cos_z0'])} / {mean(ok['surf_cos_z1'])} / {mean(ok['surf_cos_z2'])} | "
                     f"{mean(ok['anchor_cos_bg'])} | {mean(ok['anchor_n_drawn'] / ok['anchor_n'], '{:.2f}')} |")

    have = [p for p in HEIGHT if surf.get(p) is not None and gap.get(p) is not None]
    geom = len(have) == len(HEIGHT) and all(surf[p][1] >= GEOM_SURF_MIN and gap[p][1] > GEOM_GAP_MIN for p in have)
    enc = len(have) == len(HEIGHT) and all(surf[p][2] <= ENC_SURF_MAX for p in have)
    verdict = "GEOM" if geom else "ENC" if enc else "inconclusive"
    detail = ", ".join(f"{p}: surf {cell(surf[p])}, surf − anchor_drawn {cell(gap[p], '{:+.3f}')}" for p in have)

    lines = [
        f"# T2 surface-correspondence diagnostic — tag {args.tag}",
        "",
        f"Official RAP v1 weights, F0, raster, c1 = identity rig; source `{csv}`. Means over tokens, 95% "
        "log-clustered bootstrap CI.",
        "",
        "surf: feature cosine at pixels of c1 and of the changed rig that show the same surface point; "
        "samepix: same pairs, changed rig read at the c1 pixel; anchor: T2's cosine at the projections of the "
        "same round-1 3D reference points (all / on a drawn pixel); gap = surf − anchor_drawn per token; "
        "anchor (T2 run) = the value in the T2 CSV (cross-check).",
        "",
        "## 1 cosines",
        "",
        "| preset | n | invalid | surf | samepix | anchor | anchor drawn | gap | anchor (T2 run) |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows1,
        "",
        "## 2 correspondences",
        "",
        "match / out / occl: fraction of c1 surface samples matched / projecting outside the new view / "
        "showing another surface there; shift: mean correspondence displacement in feature cells (16 px input); "
        "rgb diff: mean |ΔRGB| (0–255) of the raster at matched pixels (renderer depth shading); "
        "z bins: depth < 10 / 10–25 / ≥ 25 m; anchor bg: anchors on black in both inputs; drawn share of anchors.",
        "",
        "| preset | match | out | occl | shift | depth (m) | rgb diff | surf box / map | surf z0 / z1 / z2 | anchor bg | drawn share |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows2,
        "",
        "## 3 pre-registered reading",
        "",
        f"- GEOM: for both height presets surf CI low ≥ {GEOM_SURF_MIN} and gap CI low > {GEOM_GAP_MIN}",
        f"- ENC: for both height presets surf CI high ≤ {ENC_SURF_MAX}",
        f"- **verdict: {verdict}** ({detail or 'height presets missing'})",
        "",
        "Raster only (no texture, depth-shaded colours): a bound for textured images, not a substitute.",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"t2_surface_{args.tag}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}; verdict {verdict}")


if __name__ == "__main__":
    main()
