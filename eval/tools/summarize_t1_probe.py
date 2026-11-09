"""Exp-0 / T1 step 8: summarize run_t1_probe_v1.py runs of one tag into eval/t1_results/t1_probe_<tag>.md.

Per preset, paired on the same tokens (Δ = variant - raw, PDMS points ×100 in the text, raw in the
tables; CIs by log-clustered bootstrap as in summarize_t1.py):
  8a  Δ img (= T1) vs Δ fix_all / fix_last (raw candidates re-ranked under the new rig: selection
      only), Δ oracle@64 / mean64 (candidate pool), capture and Spearman of the scorer
  8c  Δ crop, Δ feat (equivariant-backbone features), backbone share Δ img - Δ feat,
      downstream share Δ feat - Δ crop
  8b  feature similarity of the real shifted features to the moved crop features
It also checks the probe's img / raw picks against the T1 CSVs of the same preset (t1_sub_<t1tag>_*).

    python tools/summarize_t1_probe.py --tag official
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, load, token_logs  # noqa: E402


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Spearman (average ranks for ties); NaN where either row is constant."""
    def ranks(x):
        return pd.DataFrame(x).rank(axis=1).to_numpy(dtype=np.float64, copy=True)
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


def ci_cell(d: pd.Series, logs, rng, n_boot) -> str:
    d = d.dropna()
    if d.empty:
        return "–"
    m, lo, hi = boot_ci(d, logs, n_boot, rng)
    return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="official")
    ap.add_argument("--t1-tag", default="official", help="T1 sweep tag to cross-check against")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "t1_results")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    logs = token_logs(args.subset_csv)

    runs = {}
    pat = re.compile(rf"^t1probe_{re.escape(args.tag)}_(.+)$")
    for d in sorted(args.exp_root.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir() and (csv := latest_csv(d)) is not None:
            df = load(csv)
            npz = csv.with_name(csv.stem + "_arrays.npz")
            runs[m.group(1)] = (csv, df, np.load(npz) if npz.exists() else None)
    if not runs:
        raise SystemExit(f"no t1probe_{args.tag}_* runs under {args.exp_root}")

    a_rows, c_rows, b_rows, chk_rows, l_rows, s_rows, o_rows = [], [], [], [], [], [], []
    for name, (csv, df, npz) in sorted(runs.items()):
        ok = df[df["valid"].astype(bool)]
        n_bad = len(df) - len(ok)

        def delta(k):
            return ok[f"pdms_{k}"] - ok["pdms_raw"] if f"pdms_{k}" in ok else pd.Series(dtype=float)

        sel = {}
        if npz is not None:
            tok = pd.Index(npz["token"])
            for k in ("raw", "img", "fix_all", "fix_last", "crop", "feat", "raw_oob", "img_oob"):
                pool = "raw" if k.startswith("fix") else k
                if f"score_{k}" in npz and f"s64_{pool}" in npz:
                    sel[k] = pd.Series(spearman_rows(npz[f"score_{k}"], npz[f"s64_{pool}"]), index=tok)
        cap = {k: (ok[f"pdms_{k}"] / ok[f"oracle_{k}"].where(ok[f"oracle_{k}"] > 0)) for k in ("raw", "img")}
        a_rows.append(
            f"| {name} | {len(ok)} | {n_bad} | {ci_cell(delta('img'), logs, rng, args.n_boot)} | "
            f"{ci_cell(delta('fix_all'), logs, rng, args.n_boot)} | {ci_cell(delta('fix_last'), logs, rng, args.n_boot)} | "
            f"{ci_cell(ok['oracle_img'] - ok['oracle_raw'], logs, rng, args.n_boot)} | "
            f"{ci_cell(ok['mean64_img'] - ok['mean64_raw'], logs, rng, args.n_boot)} | "
            f"{cap['raw'].mean():.3f} / {cap['img'].mean():.3f} | "
            + " / ".join(f"{sel[k].mean():.3f}" if k in sel else "–" for k in ("raw", "img", "fix_all"))
            + " |")

        if "pdms_feat" in ok:
            d_img, d_feat, d_crop = delta("img"), delta("feat"), delta("crop")
            c_rows.append(
                f"| {name} | {ok['n_valid_pos'].mean():.0f} | {ci_cell(d_img, logs, rng, args.n_boot)} | "
                f"{ci_cell(d_crop, logs, rng, args.n_boot)} | {ci_cell(d_feat, logs, rng, args.n_boot)} | "
                f"{ci_cell(d_img - d_feat, logs, rng, args.n_boot)} | {ci_cell(d_feat - d_crop, logs, rng, args.n_boot)} | "
                f"{ci_cell(ok['oracle_feat'] - ok['oracle_raw'], logs, rng, args.n_boot)} |")
            b_rows.append(
                f"| {name} | " + " | ".join(
                    f"{ok[f'{k}_f0'].mean():.4f} / {ok[f'{k}_mean'].mean():.4f}"
                    for k in ("fpn_aligned", "fpn_unaligned", "vit_aligned", "vit_unaligned", "fpn_crop_vs_raw", "fpn_noise",
                              "rl2_fpn_aligned", "rl2_fpn_crop_vs_raw", "rl2_fpn_noise")) + " |")

        # boundary test: SCA samples on black fill / padding treated as outside the image
        if "pdms_img_oob" in ok:
            d_img, d_oob, d_roob = delta("img"), delta("img_oob"), delta("raw_oob")
            o_rows.append(
                f"| {name} | {ci_cell(d_img, logs, rng, args.n_boot)} | {ci_cell(d_roob, logs, rng, args.n_boot)} | "
                f"{ci_cell(d_oob, logs, rng, args.n_boot)} | {ci_cell(d_oob - d_img, logs, rng, args.n_boot)} | "
                f"{ci_cell(ok['oracle_img_oob'] - ok['oracle_raw'], logs, rng, args.n_boot)} | "
                f"{ci_cell(ok['mean64_img_oob'] - ok['mean64_raw'], logs, rng, args.n_boot)} |")

        # layer-wise divergence ladder: image features -> queries per round -> proposals -> scores -> PDMS
        for k in ("img", "feat", "crop", "fix_all", "fix_last", "raw_oob", "img_oob"):
            if f"pdms_{k}" not in ok:
                continue
            img_cos = {"img": ("vit_aligned_f0", "fpn_aligned_f0"), "crop": (None, "fpn_crop_vs_raw_f0")}.get(k, (None, None))
            cells = [f"{ok[c].mean():.4f}" if c and c in ok else "–" for c in img_cos]
            cells += [f"{ok[f'qcos_r{r}_{k}'].mean():.4f}" if f"qcos_r{r}_{k}" in ok else "–" for r in range(1, 5)]
            cells += [f"{ok[f'pade_r4_{k}'].mean():.2f}" if f"pade_r4_{k}" in ok else "–",
                      f"{ok[f'cover_{k}'].mean():.2f}" if f"cover_{k}" in ok else "–",
                      f"{sel[k].mean():.3f}" if k in sel else "–",
                      ci_cell(delta(k), logs, rng, args.n_boot)]
            l_rows.append(f"| {name} | {k} | " + " | ".join(cells) + " |")

        # F0 SCA attention mass, first and last refiner round, averaged over the 8 scored waypoints.
        # Round 1 reads the images at the same initial proposals for every variant: the clean comparison.
        if npz is not None:
            for rnd in (0, -1):
                for k in ("raw", "img", "fix_all", "crop", "feat", "raw_oob", "img_oob"):
                    if f"sca_{k}" not in npz:
                        continue
                    a = npz[f"sca_{k}"][:, rnd, :8, :]  # (N, waypoints, [participation, out, band, pad, content])
                    with np.errstate(invalid="ignore"):
                        m = np.nanmean(a.reshape(-1, a.shape[-1]), axis=0)
                        near = np.nanmean(a[:, :2].reshape(-1, a.shape[-1]), axis=0)  # first 1 s
                    s_rows.append(f"| {name} | {npz[f'sca_{k}'].shape[1] if rnd == -1 else 1} | {k} | "
                                  + " | ".join(f"{x:.3f}" for x in m) + " | "
                                  + " / ".join(f"{x:.3f}" for x in near[[0, 2]]) + " |")

        # cross-check against the T1 sweep: same preset through RAPV2RigAgent / run_pdm_score_v1
        for k, t1name in (("img", name), ("raw", "raw")):
            d = args.exp_root / f"t1_sub_{args.t1_tag}_{t1name}"
            t1csv = latest_csv(d) if d.is_dir() else None
            if t1csv is None:
                continue
            t1 = load(t1csv)
            common = ok.index.intersection(t1.index)
            diff = (ok.loc[common, f"pdms_{k}"] - t1.loc[common, "score"]).abs()
            chk_rows.append(f"| {name} | {k} vs T1 {t1name} | {len(common)} | {(diff < 1e-6).mean():.4f} | "
                            f"{100 * (ok.loc[common, f'pdms_{k}'] - t1.loc[common, 'score']).mean():+.3f} |")

    lines = [f"# T1 step 8 probes — tag {args.tag}", "",
             "ΔPDMS in points (×100) vs raw on the same tokens, 95% log-clustered bootstrap CI.", "",
             "## 8a generation vs selection", "",
             "fix_all / fix_last: the raw-rig candidates, re-ranked by the scorer under the new rig (selection only; "
             "fix_all reads the new images in all 4 rounds with raw proposals teacher-forced, fix_last only in the last). "
             "oracle / mean64: the regenerated candidate pool. capture = picked / oracle; Spearman(scorer, true PDMS) over 64.", "",
             "| preset | n | invalid | Δ img (T1) | Δ fix_all | Δ fix_last | Δ oracle@64 | Δ mean64 | capture raw / img | Spearman raw / img / fix_all |",
             "|---|---|---|---|---|---|---|---|---|---|", *a_rows, ""]
    if c_rows:
        lines += ["## 8c feature translation (pure translations)", "",
                  "crop: same visible content at the original position. feat: decoder fed the crop features moved by the shift "
                  "(what an exactly equivariant backbone would give; bilinear for fractional shifts) with the shifted calibration; "
                  "band, padding row and partial patches come from the real shifted features. "
                  "backbone share = Δ img − Δ feat; downstream share = Δ feat − Δ crop.", "",
                  "| preset | moved positions / cam | Δ img | Δ crop | Δ feat | backbone share | downstream share | Δ oracle@64 feat |",
                  "|---|---|---|---|---|---|---|---|", *c_rows, "",
                  "## 8b backbone translation equivariance (F0 / mean of 4 cams)", "",
                  "cos = mean cosine over the moved positions, embeddings removed. aligned: real shifted vs moved crop features; "
                  "unaligned: the same positions without moving (scale of a real mismatch); crop_vs_raw: kept content, crop vs raw "
                  "(context loss, PDMS-neutral in T1); noise: raw vs resample_ctrl (0.2 px). rl2 = relative L2.", "",
                  "| preset | cos FPN aligned | cos FPN unaligned | cos ViT aligned | cos ViT unaligned | cos FPN crop_vs_raw | cos FPN noise | rl2 FPN aligned | rl2 crop_vs_raw | rl2 noise |",
                  "|---|---|---|---|---|---|---|---|---|---|", *b_rows, ""]
    if o_rows:
        lines += ["## boundary test: SCA samples on black fill / padding count as outside", "",
                  "img_oob: img with every SCA sampling location on black fill or RAP's padding row moved outside the "
                  "image (adds 0, like the raw-image samples of the same 3D points); raw_oob: the same rule on raw input "
                  "(control). recovered = Δ img_oob − Δ img.", "",
                  "| preset | Δ img | Δ raw_oob | Δ img_oob | recovered | Δ oracle@64 img_oob | Δ mean64 img_oob |",
                  "|---|---|---|---|---|---|---|", *o_rows, ""]
    lines += ["## layer-wise divergence from raw", "",
              "Image level: F0 cosine (img: real shifted vs moved crop features; crop: crop vs raw on kept content). "
              "Query rounds: cosine of the trajectory queries to raw after each refiner round (slot-matched; "
              "fix_last runs only round 4). prop ADE r4: slot-matched ADE (m) of the final proposals; cover: mean ADE "
              "from each raw candidate to the closest new one (slot-free). Spearman: scorer vs true PDMS over 64.", "",
              "| preset | variant | ViT cos | FPN cos | query r1 | r2 | r3 | r4 | prop ADE r4 | cover | Spearman | ΔPDMS |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|", *l_rows, ""]
    if s_rows:
        lines += ["## F0 spatial cross-attention sampling (refiner rounds 1 and 4, waypoints 0.5-4 s)", "",
                  "participation: fraction of candidates whose waypoint query attends F0; then the attention mass "
                  "of those queries landing outside the image / on black fill / on the padding row / on content. "
                  "Last column: participation / black-fill mass for the first 1 s.", "",
                  "Round 1 samples at the same initial proposals for every variant (clean comparison); round 4 already "
                  "includes the query drift. *_oob rows record the moved locations (black / pad should be ~0).", "",
                  "| preset | round | variant | participation | out | black | pad | content | first 1 s part. / black |",
                  "|---|---|---|---|---|---|---|---|---|", *s_rows, ""]
    if chk_rows:
        lines += ["## consistency with the T1 sweep", "",
                  "| preset | pick | tokens | identical | mean diff (pts) |", "|---|---|---|---|---|", *chk_rows, ""]
    text = "\n".join(lines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"t1_probe_{args.tag}.md"
    out.write_text(text + "\n")
    print(text)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
