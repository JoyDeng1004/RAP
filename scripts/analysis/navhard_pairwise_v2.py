"""navhard_two_stage: reproduce the 3 aggregate rows, then paired bootstrap per stage-1 scene group.

Devkit semantics (v2 run_pdm_score.py + SceneAggregator), entry e = (orig, prev, [(a_j, b_j)]):
  S1(t)   = score of stage-1 token t                        (stage-1 weight is always 1.0)
  S2w(t)  = sum_j w_j * score(child_j) / sum_j w_j,  w_j = exp(-|endpoint(t) - start(child_j)|^2 / 0.2),
            normalised; uniform if the sum isclose 0.  children of orig = a_j, of prev = b_j
  c_e     = (S1(orig) * S2w(orig) + S1(prev) * S2w(prev)) / 2
  combined = nanmean_e c_e ;  s1 = nanmean of the 2N S1 ;  s2 = nanmean of the 2N S2w
So every aggregate is a mean over entries -> resampling entries and re-averaging IS the devkit
re-aggregation. Stage-2 weights need each model's stage-1 endpoints (not in the CSV): pass them
with --endpoints; without them only s1 is reproduced/tested.

Read-only; prints to stdout.

  env -u PYTHONPATH $PY $RAP/scripts/analysis/navhard_pairwise_v2.py \
      --mapping $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml \
      --points <dir>/start_points.csv \
      --run E2a=$NAVSIM_EXP_ROOT/eval_v2_navhard_two_stage_E2a_scorer/8819498 ... \
      [--endpoints E2a=<csv with token,endpoint_x,endpoint_y> ...] \
      --pair E2b:E2a --pair E3:E2a --pair E3:E2b --pair official:E2a
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rap_analysis_common import V2_AGG_TOKENS, parse_named, read_eval_csv  # noqa: E402

SIGMA_SQ = 0.1  # SceneAggregator.sigma_squared
AGG = {"s1": "extended_pdm_score_stage_one", "s2": "extended_pdm_score_stage_two",
       "combined": "extended_pdm_score_combined"}
TOL = 1e-9


def load_mapping(path, csv_tokens):
    d = yaml.safe_load(open(path))
    raw = d.get("reactive_all_mapping")
    if not raw:
        raise SystemExit(f"[FAIL] {path}: reactive_all_mapping is empty")
    entries = []
    for orig, prev, pairs in raw:
        if orig in csv_tokens or prev in csv_tokens:  # same filter as run_pdm_score.main (scene_loader.tokens)
            entries.append((str(orig), str(prev), [tuple(map(str, p)) for p in pairs]))
    s1 = [t for e in entries for t in e[:2]]
    a = [p[0] for e in entries for p in e[2] if len(p) > 0]
    b = [p[1] for e in entries for p in e[2] if len(p) > 1]
    print(f"[mapping] {path}")
    print(f"  entries {len(raw)} -> kept {len(entries)}; stage-1 refs {len(s1)} (unique {len(set(s1))}); "
          f"children now {len(a)} (unique {len(set(a))}), prev {len(b)} (unique {len(set(b))}), "
          f"now&prev overlap {len(set(a) & set(b))}")
    dup = len(s1) != len(set(s1)) or len(a) + len(b) != len(set(a) | set(b))
    if dup:
        print("  [warn] 有 token 出现在多个条目里：devkit 的 DataFrame.update 只保留最后一次的 weight，本脚本按条目各算各的，可能不一致")
    return entries


def weighted(scores, w):
    tot = w.sum()
    if len(scores) == 0 or tot == 0 or np.isnan(tot):
        return np.nan
    return float((scores * w).sum() / tot)


def pcl_weights(endpoint, starts):
    """SceneAggregator.calculate_pseudo_closed_loop_weights."""
    d2 = (endpoint[0] - starts[:, 0]) ** 2 + (endpoint[1] - starts[:, 1]) ** 2
    w = np.exp(-d2 / (2 * SIGMA_SQ))
    s = w.sum()
    if np.isclose(s, 0.0) or np.isnan(s):
        return np.full(len(starts), 1.0 / len(starts)), True
    return w / s, False


class Model:
    def __init__(self, name, run_dir, entries, points, endpoints):
        self.name = name
        rows, agg, self.csv = read_eval_csv(run_dir, V2_AGG_TOKENS)
        self.agg = agg.set_index("token")
        self.rows = rows.set_index("token")
        n_inv = int((~rows["valid"]).sum())
        st1 = rows["ego_progress_stage_one"].notna().sum()
        st2 = rows["ego_progress_stage_two"].notna().sum()
        print(f"[load] {name}: {self.csv}  tokens {len(rows)} (stage1 {st1}, stage2 {st2}, invalid {n_inv})")
        self.metrics = [c[: -len("_stage_one")] for c in rows.columns if c.endswith("_stage_one")]
        sc = self.rows["score"]
        self.has_w = endpoints is not None
        recs, fallback, neff = [], 0, []
        for i, (orig, prev, pairs) in enumerate(entries):
            r = {"entry": i, "orig": orig, "prev": prev}
            for side, tok, kids in (("o", orig, [p[0] for p in pairs if len(p) > 0]),
                                    ("p", prev, [p[1] for p in pairs if len(p) > 1])):
                r[f"s1{side}"] = float(sc.get(tok, np.nan))
                for m in self.metrics:
                    r[f"{m}#s1{side}"] = float(self.rows[f"{m}_stage_one"].get(tok, np.nan))
                if not self.has_w:
                    continue
                kids = [k for k in kids if k in sc.index]
                if not kids:
                    r[f"s2{side}"] = np.nan
                    continue
                if tok not in endpoints.index:
                    raise SystemExit(f"[FAIL] {name}: no endpoint for stage-1 token {tok}")
                w, fb = pcl_weights(endpoints.loc[tok, ["endpoint_x", "endpoint_y"]].to_numpy(float),
                                    points.loc[kids, ["start_point_x", "start_point_y"]].to_numpy(float))
                fallback += int(fb)
                neff.append(1.0 / (w ** 2).sum())
                r[f"s2{side}"] = weighted(sc.loc[kids].to_numpy(float), w)
                for m in self.metrics:
                    r[f"{m}#s2{side}"] = weighted(self.rows.loc[kids, f"{m}_stage_two"].to_numpy(float), w)
            if self.has_w:
                r["c"] = (r["s1o"] * r["s2o"] + r["s1p"] * r["s2p"]) / 2
            recs.append(r)
        self.e = pd.DataFrame(recs).set_index("entry")
        if self.has_w:
            neff = np.array(neff)
            print(f"  stage-2 权重：{2 * len(entries)} 组中 {fallback} 组退化为均匀权重（最近的 stage-2 起点 > ~1.9 m）；"
                  f"有效场景数 1/Σw² 中位 {np.median(neff):.2f}，=1 的比例 {(neff < 1.001).mean():.1%}")

    def aggregates(self):
        e = self.e
        out = {"s1": np.nanmean(np.r_[e["s1o"], e["s1p"]])}
        if self.has_w:
            out["s2"] = np.nanmean(np.r_[e["s2o"], e["s2p"]])
            out["combined"] = np.nanmean(e["c"])
        return out

    def check(self):
        print(f"\n### 复现 {self.name}")
        ok = True
        got = self.aggregates()
        print("| 行 | CSV | 复算 | 差 |")
        print("|---|---|---|---|")
        for k, tok in AGG.items():
            ref = float(self.agg.loc[tok, "score"])
            if k not in got:
                print(f"| {tok} | {ref:.10f} | （缺 endpoint，跳过） | |")
                continue
            d = got[k] - ref
            ok &= abs(d) < TOL
            print(f"| {tok} | {ref:.10f} | {got[k]:.10f} | {d:.2e} |")
        # sub-metric columns of the stage rows
        worst = 0.0
        for m in self.metrics:
            e = self.e
            v1 = np.nanmean(np.r_[e[f"{m}#s1o"], e[f"{m}#s1p"]])
            worst = max(worst, abs(v1 - float(self.agg.loc[AGG["s1"], f"{m}_stage_one"])))
            if self.has_w:
                v2 = np.nanmean(np.r_[e[f"{m}#s2o"], e[f"{m}#s2p"]])
                worst = max(worst, abs(v2 - float(self.agg.loc[AGG["s2"], f"{m}_stage_two"])))
        ok &= worst < TOL
        print(f"- 分项列（{len(self.metrics)} 个指标）最大差 {worst:.2e}")
        print(f"- 结论：{'PASS' if ok else 'FAIL'}" + ("" if self.has_w else "（只核对了 s1）"))
        return ok


def entry_arrays(m, key):
    """Per-entry (sum, count) so that any resample's aggregate = sum(counts*sum)/sum(counts*count)."""
    e = m.e
    if key == "combined":
        v = e["c"].to_numpy(float)
        return np.nan_to_num(v), (~np.isnan(v)).astype(float)
    a, b = e[f"{key}o"].to_numpy(float), e[f"{key}p"].to_numpy(float)
    return np.nan_to_num(a) + np.nan_to_num(b), (~np.isnan(a)).astype(float) + (~np.isnan(b)).astype(float)


def bootstrap(models, pairs, block, n_boot, seed, logs):
    n = len(next(iter(models.values())).e)
    rng = np.random.default_rng(seed)
    if block == "log":
        uniq, inv = np.unique(logs, return_inverse=True)
        cl = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot)
        W = cl[:, inv].astype(float)
        unit = f"log（{len(uniq)} 个）"
    else:
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
        unit = f"条目（{n} 个 stage-1 场景对）"
    print(f"\n## 配对 bootstrap：按{unit}重采样，B={n_boot}，seed={seed}（×100，A−B）")
    print("| A − B | 指标 | 均值差 | 95% CI | 双侧 p | 胜/平/负（按条目） |")
    print("|---|---|---|---|---|---|")
    for A, B in pairs:
        ma, mb = models[A], models[B]
        keys = ["combined", "s1", "s2"] if (ma.has_w and mb.has_w) else ["s1"]
        for k in keys:
            sa, ca = entry_arrays(ma, k)
            sb, cb = entry_arrays(mb, k)
            if not np.array_equal(ca, cb):
                print(f"[warn] {A}/{B} {k}: NaN 模式不同")
            point = sa.sum() / ca.sum() - sb.sum() / cb.sum()
            dist = (W @ sa) / (W @ ca) - (W @ sb) / (W @ cb)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            p = min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))
            # per-entry comparison (entry value = entry's own mean over its non-NaN parts)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = sa / ca - sb / cb
            d = d[~np.isnan(d)]
            w, t, l_ = (d > TOL).sum(), (np.abs(d) <= TOL).sum(), (d < -TOL).sum()
            nn = len(d)
            print(f"| {A} − {B} | {k} | {point * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] | {p:.3g} | "
                  f"{w / nn:.0%} / {t / nn:.0%} / {l_ / nn:.0%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True, help="train_test_split/navhard_two_stage.yaml")
    ap.add_argument("--run", action="append", required=True, help="NAME=navhard result dir or csv")
    ap.add_argument("--points", default=None, help="start_points.csv from navhard_dump_points_v2.py")
    ap.add_argument("--endpoints", action="append", default=[], help="NAME=csv with token,endpoint_x,endpoint_y")
    ap.add_argument("--pair", action="append", default=[], help="A:B  (reports A − B)")
    ap.add_argument("--block", choices=["entry", "log"], default="entry")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs, eps = parse_named(a.run), parse_named(a.endpoints)
    first = read_eval_csv(next(iter(runs.values())), V2_AGG_TOKENS)[0]
    entries = load_mapping(a.mapping, set(first["token"]))
    points = None
    if a.points:
        points = pd.read_csv(a.points, dtype={"token": str}).set_index("token")
        print(f"[load] start points: {len(points)} tokens, {points['log_name'].nunique()} logs")
    if eps and points is None:
        raise SystemExit("[FAIL] --endpoints 需要 --points")

    models = {}
    for n, p in runs.items():
        ep = None
        if n in eps:
            ep = pd.read_csv(eps[n], dtype={"token": str}).set_index("token")
        models[n] = Model(n, p, entries, points, ep)

    ok = all([m.check() for m in models.values()])
    if not ok:
        print("\n[FAIL] 有模型没能逐位复现聚合行，不做 bootstrap")
        sys.exit(2)

    logs = None
    if a.block == "log":
        if points is None:
            raise SystemExit("[FAIL] --block log 需要 --points（取 log_name）")
        logs = points.loc[[e[0] for e in entries], "log_name"].to_numpy()
    pairs = [tuple(p.split(":")) for p in a.pair]
    for A, B in pairs:
        if A not in models or B not in models:
            raise SystemExit(f"[FAIL] 未知模型 {A}:{B}")
    bootstrap(models, pairs, a.block, a.n_boot, a.seed, logs)
    print("\n注：p 为 bootstrap 百分位法的双侧近似；4 组以上比较时请自行做多重比较校正。"
          "s2 的权重取决于各模型自己 stage-1 终点，s2/combined 的差包含“把车开到哪里”的差异。")


if __name__ == "__main__":
    main()
