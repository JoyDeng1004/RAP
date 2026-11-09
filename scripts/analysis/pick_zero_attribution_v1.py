"""Attribute the navtest v1 tokens where a model's scorer picked a 0-score trajectory.

Read-only; prints Markdown tables to stdout, writes nothing.

Steps (numbered like the task):
  0/1  consistency: scorer CSV score==0 <-> oracle CSV score_scorer==0 <-> npz score64[idx];
       zero cause from the picked trajectory's multiplicative terms (v1 PDMScorer:
       MultiMetricIndex = NO_COLLISION {0, 0.5, 1} x DRIVABLE_AREA {0, 1}) -> NC-only / DAC-only / both / other
  2    scorer-wrong (some candidate > 0) vs all-zero (no candidate > 0); rank / logit / ADE gaps
       between the picked and the best candidate; measured counterfactual gains
  3    distribution over log / scenario_type (v1 cache path) / token features read from the
       navsim log pkls (city, driving command, GT manoeuvre at +4 s, speed, nearby agents);
       enrichment vs the navtest share, Fisher exact p, BH q; groups with >= --min-bad bad tokens
  4    the same tokens under the reference models (score, pick-zero, NC, DAC, candidate availability)

Usage (TSUBAME, v1 env):
  env -u PYTHONPATH $PY $RAP/scripts/analysis/pick_zero_attribution_v1.py \
      --target E3 \
      --scorer E3=$NAVSIM_EXP_ROOT/eval_v1_navtest_E3_scorer/8825478 \
      --scorer E2b=$NAVSIM_EXP_ROOT/eval_v1_navtest_E2b_scorer/8825468 \
      --scorer official=$NAVSIM_EXP_ROOT/eval_v1_navtest_official_v1ckpt_scorer/8812832 \
      --oracle E3=$NAVSIM_EXP_ROOT/eval_v1_oracle64_navtest_E3/8825483 \
      --oracle E2b=$NAVSIM_EXP_ROOT/eval_v1_oracle64_navtest_E2b/8825474 \
      --oracle official=$NAVSIM_EXP_ROOT/eval_v1_oracle64_navtest_official/8819685 \
      --v1-cache $RAP/metric_cache_v1_navtest \
      --navsim-logs $OPENSCENE_DATA_ROOT/navsim_logs/test
"""
import argparse
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rap_analysis_common import (  # noqa: E402
    V1_AGG_TOKENS,
    bh_qvalues,
    find_one,
    fisher_two_sided,
    parse_named,
    quat_to_yaw,
    read_eval_csv,
    v1_cache_index,
)

NC, DAC = "no_at_fault_collisions", "drivable_area_compliance"
SUB = ["ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
CMD_NAMES = ("left", "forward", "right", "unknown")
AGENT_NAMES = {"vehicle", "pedestrian", "bicycle"}
EPS = 1e-9


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{nd}f}"


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "nan"


def q3(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return "n=0"
    return f"{np.median(v):.3g} [{np.percentile(v, 25):.3g}, {np.percentile(v, 75):.3g}] (mean {v.mean():.3g})"


def zero_cause(nc, dac, score):
    if score > EPS:
        return "nonzero"
    if nc < EPS and dac < EPS:
        return "both"
    if nc < EPS:
        return "NC_only"
    if dac < EPS:
        return "DAC_only"
    return "other"


# ----------------------------------------------------------------------------- loading
def load_oracle(path):
    rows, _, csv = read_eval_csv(path, V1_AGG_TOKENS)
    npz_path = find_one(str(Path(csv).parent) if Path(path).is_file() else path, "*_arrays.npz")
    z = np.load(npz_path, allow_pickle=True)
    arr = {k: z[k] for k in z.files}
    for k in ("token", "score64", "logit64", "ade64"):
        if k not in arr:
            raise SystemExit(f"[FAIL] {npz_path}: missing {k}; keys = {list(arr)}")
    arr["token"] = np.array([str(t) for t in arr["token"]])
    arr["pos"] = {t: i for i, t in enumerate(arr["token"])}
    return rows, arr, csv, npz_path


def log_features(job):
    """Per-token features from one navsim log pkl (only tokens in `need`)."""
    path, need, horizon_s = job
    with open(path, "rb") as fh:
        frames = sorted(pickle.load(fh), key=lambda fr: fr["timestamp"])
    if not frames:
        return []
    xy = np.array([fr["ego2global_translation"][:2] for fr in frames], dtype=np.float64)
    yaw = np.array([quat_to_yaw(fr["ego2global_rotation"]) for fr in frames])
    ts = np.array([fr["timestamp"] for fr in frames], dtype=np.float64) * 1e-6
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.5
    k = max(int(round(horizon_s / dt)), 1)
    out = []
    for i, fr in enumerate(frames):
        tok = str(fr["token"])
        if tok not in need:
            continue
        rec = {"token": tok, "city": str(fr.get("map_location", "na"))}
        dc = fr.get("driving_command")
        rec["command"] = CMD_NAMES[int(np.argmax(dc))] if dc is not None and len(dc) == 4 else "na"
        eds = fr.get("ego_dynamic_state")
        rec["speed"] = float(np.hypot(eds[0], eds[1])) if eds is not None else float("nan")
        if i + k < len(frames):
            d = xy[i + k] - xy[i]
            dist = float(np.hypot(*d))
            lat = float(-np.sin(yaw[i]) * d[0] + np.cos(yaw[i]) * d[1])
            dyaw = math.degrees((yaw[i + k] - yaw[i] + math.pi) % (2 * math.pi) - math.pi)
            if dist < 2.0:
                man = "stationary"
            elif dyaw > 20:
                man = "turn_left"
            elif dyaw < -20:
                man = "turn_right"
            elif abs(lat) > 1.5:
                man = "lateral_shift"
            else:
                man = "straight"
            rec.update(gt_dist4s=dist, maneuver=man)
        else:
            rec.update(gt_dist4s=float("nan"), maneuver="na")
        anns = fr.get("anns") or {}
        boxes, names = anns.get("gt_boxes"), anns.get("gt_names")
        if boxes is not None and names is not None and len(boxes):
            boxes = np.asarray(boxes, dtype=np.float64)
            names = np.asarray(names).astype(str)
            r = np.hypot(boxes[:, 0], boxes[:, 1])
            rec["n_agents30"] = int(((r < 30) & np.isin(names, list(AGENT_NAMES))).sum())
            rec["n_ped30"] = int(((r < 30) & (names == "pedestrian")).sum())
        else:
            rec["n_agents30"], rec["n_ped30"] = 0, 0
        out.append(rec)
    return out


def token_features(log_dir, token_log, workers, horizon_s):
    need_by_log = token_log.groupby("log_name").groups
    jobs, missing = [], []
    for log, toks in need_by_log.items():
        p = Path(log_dir) / f"{log}.pkl"
        if p.is_file():
            jobs.append((str(p), set(map(str, toks)), horizon_s))
        else:
            missing.append(log)
    if missing:
        print(f"[warn] {len(missing)} log pkls missing under {log_dir}, e.g. {missing[0]}")
    recs = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(log_features, jobs, chunksize=2):
            recs += r
    feat = pd.DataFrame(recs).set_index("token")
    feat["speed_bin"] = pd.cut(feat["speed"], [-1, 1, 5, 10, 100], labels=["<1", "1-5", "5-10", ">=10"]).astype(str)
    feat["agents30_bin"] = pd.cut(feat["n_agents30"], [-1, 5, 15, 30, 10000], labels=["0-5", "6-15", "16-30", ">30"]).astype(str)
    feat["ped30_bin"] = pd.cut(feat["n_ped30"], [-1, 0, 3, 10, 10000], labels=["0", "1-3", "4-10", ">10"]).astype(str)
    return feat


# ----------------------------------------------------------------------------- steps
def step1(name, sc, orc_rows, arr):
    print(f"\n## 0/1 一致性与 0 分归因（{name}）")
    n_all = len(sc)
    v = sc[sc["valid"]]
    print(f"- scorer CSV：{n_all} 行 token，valid {len(v)}，invalid {n_all - len(v)}")
    bad = set(v.loc[v["score"] <= EPS, "token"])
    ov = orc_rows[orc_rows["valid"]]
    if "score_scorer" not in ov.columns:
        raise SystemExit(f"[FAIL] oracle CSV has no score_scorer; columns = {list(ov.columns)}")
    bad_o = set(ov.loc[ov["score_scorer"] <= EPS, "token"])
    m = v.merge(ov[["token", "score_scorer"]], on="token", how="inner")
    dmax = float((m["score"] - m["score_scorer"]).abs().max())
    print(f"- score==0：scorer CSV {len(bad)} 个（{pct(len(bad), len(v))}），oracle CSV score_scorer==0 {len(bad_o)} 个；"
          f"交集 {len(bad & bad_o)}，仅 scorer {len(bad - bad_o)}，仅 oracle {len(bad_o - bad)}")
    print(f"- 两个 CSV 共同 valid token {len(m)}，|score − score_scorer| 最大 {dmax:.3g}")
    ok = bad == bad_o and dmax < 1e-6
    # npz self-consistency
    toks = [t for t in v["token"] if t in arr["pos"]]
    idx = np.array([arr["pos"][t] for t in toks])
    s64, l64 = arr["score64"][idx], arr["logit64"][idx]
    am = l64.argmax(1)
    if "idx_scorer" in ov.columns:
        idx_csv = ov.set_index("token").loc[toks, "idx_scorer"].to_numpy().astype(int)
        agree = float((idx_csv == am).mean())
        s_pick = s64[np.arange(len(idx)), idx_csv]
        dpick = float(np.abs(s_pick - v.set_index("token").loc[toks, "score"].to_numpy()).max())
        print(f"- npz：{len(toks)}/{len(v)} 个 token 有 N×64 数组；argmax(logit64)==idx_scorer {agree:.4f}；"
              f"|score64[idx_scorer] − score| 最大 {dpick:.3g}")
        ok = ok and agree == 1.0 and dpick < 1e-6
    else:
        print(f"- [warn] oracle CSV 没有 idx_scorer 列，用 argmax(logit64) 作为选中下标")
    print(f"- 一致性：{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[FAIL] 一致性不过，后面的数字不可信；先把上面的差异贴给我")

    for col in (NC, DAC, "score"):
        if col not in v.columns:
            raise SystemExit(f"[FAIL] scorer CSV missing {col}; columns = {list(v.columns)}")
    v = v.copy()
    v["cause"] = [zero_cause(a, b, c) for a, b, c in zip(v[NC], v[DAC], v["score"])]
    b = v[v["cause"] != "nonzero"]
    print(f"\n### 选中轨迹的 0 分原因（{name}，{len(b)} 个 token）")
    print("| 原因 | 个数 | 占 bad | 占全部 valid |")
    print("|---|---|---|---|")
    for c in ("NC_only", "DAC_only", "both", "other"):
        k = int((b["cause"] == c).sum())
        print(f"| {c} | {k} | {pct(k, len(b))} | {pct(k, len(v))} |")
    print(f"\nNC 取值分布（bad / 全部）："
          + "；".join(f"NC={x}: {int((b[NC] == x).sum())} / {int((v[NC] == x).sum())}" for x in (0.0, 0.5, 1.0)))
    print("（NC=0.5 只来自撞静态物体，不会单独让总分为 0；bad 里若有 NC=0.5，必然是 DAC=0 造成的）")
    good = v[v["cause"] == "nonzero"]
    print("\n| 分项均值 | bad | 非 bad |")
    print("|---|---|---|")
    for col in [NC, DAC] + [c for c in SUB if c in v.columns]:
        print(f"| {col} | {fmt(b[col].mean(), 3)} | {fmt(good[col].mean(), 3)} |")
    return v, ok


def step2(name, v, arr, n_total):
    print(f"\n## 2 候选池里有没有 >0 的候选（{name}）")
    v = v[v["token"].isin(arr["pos"].keys())].copy()
    idx = np.array([arr["pos"][t] for t in v["token"]])
    s64, l64, a64 = arr["score64"][idx], arr["logit64"][idx], arr["ade64"][idx]
    pick = l64.argmax(1)
    r = np.arange(len(idx))
    v["n_pos"] = (s64 > EPS).sum(1)
    v["oracle"] = s64.max(1)
    v["mean64"] = s64.mean(1)
    order = np.argsort(-l64, axis=1, kind="stable")          # logit rank 0 = picked
    s_sorted = np.take_along_axis(s64, order, 1)
    first_safe = np.where((s_sorted > EPS).any(1), (s_sorted > EPS).argmax(1) + 1, 0)  # 1-based, 0 = none
    best = s64.argmax(1)
    rank_of = np.empty_like(order)
    np.put_along_axis(rank_of, order, np.arange(64)[None, :].repeat(len(idx), 0), 1)
    v["first_safe_rank"] = first_safe
    v["best_logit_rank"] = rank_of[r, best] + 1
    v["logit_gap"] = l64[r, pick] - l64[r, best]
    v["logit_gap_z"] = v["logit_gap"] / np.maximum(l64.std(1), 1e-12)
    v["ade_pick"] = a64[r, pick]
    v["ade_best"] = a64[r, best]
    v["ade_rank_pick"] = (a64 < a64[r, pick][:, None]).sum(1) + 1
    v["score_first_safe"] = np.where(first_safe > 0, s_sorted[r, np.maximum(first_safe - 1, 0)], 0.0)
    v["safe_frac"] = v["n_pos"] / 64.0

    bad = v[v["cause"] != "nonzero"]
    good = v[v["cause"] == "nonzero"]
    wrong = bad[bad["n_pos"] > 0]
    allz = bad[bad["n_pos"] == 0]
    print(f"- bad {len(bad)} 个：scorer 选错（有 >0 候选）{len(wrong)}（{pct(len(wrong), len(bad))}），"
          f"候选全 0 {len(allz)}（{pct(len(allz), len(bad))}）")
    print("\n| 原因 \\ 类型 | scorer 选错 | 候选全 0 |")
    print("|---|---|---|")
    for c in ("NC_only", "DAC_only", "both", "other"):
        print(f"| {c} | {int((wrong['cause'] == c).sum())} | {int((allz['cause'] == c).sum())} |")
    print("\n| 候选池指标 | bad·选错 | bad·全 0 | 非 bad |")
    print("|---|---|---|---|")
    for col, lab in (("safe_frac", "安全候选比例"), ("oracle", "oracle64"), ("mean64", "mean64"),
                     ("ade_pick", "选中 ADE (m)")):
        print(f"| {lab} | {q3(wrong[col])} | {q3(allz[col])} | {q3(good[col])} |")
    print("\n### scorer 选错的 token：选中 vs 最优")
    print(f"- 最优候选在 logit 排序里的名次（1=被选中）：{q3(wrong['best_logit_rank'])}")
    print(f"- 第一个 >0 候选在 logit 排序里的名次：{q3(wrong['first_safe_rank'])}")
    bins = [(2, 2), (3, 4), (5, 8), (9, 16), (17, 64)]
    print("  - 分布：" + "，".join(
        f"{lo}–{hi}: {int(((wrong['first_safe_rank'] >= lo) & (wrong['first_safe_rank'] <= hi)).sum())}"
        if lo != hi else f"{lo}: {int((wrong['first_safe_rank'] == lo).sum())}" for lo, hi in bins))
    print(f"- logit 差（选中 − 最优）：{q3(wrong['logit_gap'])}；按 token 内 logit 标准差归一：{q3(wrong['logit_gap_z'])}")
    print(f"- 最优候选得分：{q3(wrong['oracle'])}；第一个 >0 候选得分：{q3(wrong['score_first_safe'])}")
    print(f"- 选中候选的 ADE 名次（1=最贴近 GT）：{q3(wrong['ade_rank_pick'])}；ADE 选中 {q3(wrong['ade_pick'])} vs 最优 {q3(wrong['ade_best'])}")
    closer = int((wrong["ade_pick"] < wrong["ade_best"]).sum())
    print(f"- 选中的 0 分候选比最优候选更贴近 GT：{closer}/{len(wrong)}（{pct(closer, len(wrong))}）")
    print("\n### 实测反事实（只改这些 bad token，其余不变；单位：PDMS 分 ×100）")
    g1 = wrong["oracle"].sum() / n_total * 100
    g2 = wrong["score_first_safe"].sum() / n_total * 100
    print(f"- 选中换成最优候选：+{g1:.2f}；换成 logit 顺位第一个 >0 候选：+{g2:.2f}"
          f"（{name} 全部 bad token 都换掉时的上限，等价于一个完美的安全过滤器）")
    return v


def enrich_table(df, col, min_bad, title, top=None, sort="p"):
    n, k = len(df), int(df["bad"].sum())
    rows = []
    for g, sub in df.groupby(col):
        a, ng = int(sub["bad"].sum()), len(sub)
        p = fisher_two_sided(a, ng - a, k - a, n - ng - (k - a))
        rows.append(dict(group=g, n=ng, bad=a, rate=a / ng, enr=(a / ng) / (k / n), p=p,
                         wrong=int(sub["wrong"].sum()), allzero=int(sub["allzero"].sum()),
                         nc=int(sub["nc0"].sum()), dac=int(sub["dac0"].sum())))
    t = pd.DataFrame(rows)
    t["q"] = bh_qvalues(t["p"].to_numpy())  # over all groups of this feature, before the display filter
    shown = t[t["bad"] >= min_bad].sort_values(["p", "enr"] if sort == "p" else ["enr"], ascending=[True, False] if sort == "p" else [False])
    print(f"\n### {title}（{t.shape[0]} 组，列出 bad ≥ {min_bad} 的 {len(shown)} 组；整体 bad 率 {pct(k, n)}；q 为本特征内 BH 校正）")
    print("| 组 | token 数 | bad | bad 率 | 富集比 | Fisher p | BH q | 选错/全0 | NC=0/DAC=0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in (shown.head(top) if top else shown).iterrows():
        print(f"| {r['group']} | {r['n']} | {r['bad']} | {r['rate'] * 100:.1f}% | {r['enr']:.2f} | {r['p']:.2g} | {r['q']:.2g} "
              f"| {r['wrong']}/{r['allzero']} | {r['nc']}/{r['dac']} |")
    return t


def step3(name, v, idx_cache, feat, min_bad, top_logs):
    print(f"\n## 3 bad token 的分布（{name}）")
    df = v.set_index("token").join(idx_cache, how="left")
    miss = int(df["log_name"].isna().sum())
    if miss:
        print(f"[warn] {miss} 个 token 在 v1 cache 索引里找不到 log")
    df["bad"] = df["cause"] != "nonzero"
    df["wrong"] = df["bad"] & (df["n_pos"] > 0)
    df["allzero"] = df["bad"] & (df["n_pos"] == 0)
    df["nc0"] = df["bad"] & (df[NC] <= EPS)
    df["dac0"] = df["bad"] & (df[DAC] <= EPS)
    st = df["scenario_type"].value_counts()
    print(f"- v1 cache 路径里的 scenario_type：{dict(st)}")
    if len(st) <= 1:
        print("  （v1 的 NavSimScenario.scenario_type 恒为 DUMMY_SCENARIO_TYPE='unknown'，这一维没有信息，改用下面从 log pkl 读出的特征）")
    else:
        enrich_table(df, "scenario_type", min_bad, "scenario_type")

    # concentration over logs
    per_log = df.groupby("log_name").agg(n=("bad", "size"), bad=("bad", "sum")).sort_values("bad", ascending=False)
    k = int(per_log["bad"].sum())
    cum = per_log["bad"].cumsum() / k
    for frac in (0.25, 0.5, 0.8):
        m = int((cum < frac).sum()) + 1
        print(f"- 前 {m} 个 log（共 {len(per_log)}）占了 {frac:.0%} 的 bad，这些 log 只占 token 的 "
              f"{pct(int(per_log['n'].head(m).sum()), len(df))}")
    print(f"- 有 bad 的 log：{int((per_log['bad'] > 0).sum())}/{len(per_log)}")
    enrich_table(df, "log_name", min_bad, "按 log", top=top_logs)

    if feat is not None:
        df = df.join(feat, how="left")
        nf = int(df["city"].isna().sum())
        if nf:
            print(f"[warn] {nf} 个 token 在 log pkl 里没找到特征")
        for col, title in (("city", "城市 map_location"), ("command", "driving_command"),
                           ("maneuver", "GT 未来 4 s 动作（Δyaw>20° 算转弯，位移<2 m 算静止）"),
                           ("speed_bin", "当前车速 m/s"), ("agents30_bin", "30 m 内车辆/行人/自行车数"),
                           ("ped30_bin", "30 m 内行人数")):
            enrich_table(df, col, min_bad, title, sort="enr")
    return df


def step4(target, df, others, oracles, n_total):
    print(f"\n## 4 同一批 token 在其他模型上的表现")
    bad_t = df.index[df["bad"]]
    nb = len(bad_t)
    zero_flags = {}
    for name, sc in others.items():
        s = sc[sc["valid"]].set_index("token")
        common = df.index.intersection(s.index)
        if len(common) != len(df):
            print(f"[warn] {name}: 与 {target} 共同 valid token {len(common)}/{len(df)}")
        z_all = s.loc[common, "score"] <= EPS
        zb = z_all.reindex(bad_t)
        zero_flags[name] = z_all
        a = int(zb.sum())
        rate_all = float(z_all.mean())
        tbl = (a, nb - a, int(z_all.sum()) - a, len(common) - nb - (int(z_all.sum()) - a))
        p = fisher_two_sided(*tbl)
        sb = s.reindex(bad_t)
        cause_b = pd.Series([zero_cause(x, y, w) for x, y, w in zip(sb[NC], sb[DAC], sb["score"])], index=bad_t)
        print(f"\n### {name}")
        print(f"- 在 {target} 的 {nb} 个 bad token 上：{name} 平均分 {sb['score'].mean() * 100:.2f}"
              f"（{name} 全 navtest {s.loc[common, 'score'].mean() * 100:.2f}）")
        print(f"- {name} 也选中 0 分：{a}/{nb}（{pct(a, nb)}），{name} 全体 pick-zero 率 {rate_all * 100:.2f}% → "
              f"提升 {(a / nb) / max(rate_all, 1e-12):.1f} 倍，Fisher p={p:.2g}")
        print(f"- {name} 在这些 token 上：NC 分布 " + "，".join(f"{x}: {int((sb[NC] == x).sum())}" for x in (0.0, 0.5, 1.0))
              + f"；DAC=0 {int((sb[DAC] <= EPS).sum())}")
        both = cause_b[(cause_b != "nonzero")]
        if len(both):
            ct = pd.crosstab(df.loc[both.index, "cause"], both, rownames=[target], colnames=[name])
            print(f"- 两者都 0 分时的原因交叉表（行 {target}，列 {name}）：")
            print("  " + ct.to_string().replace("\n", "\n  "))
        contrib = (sb["score"].sum() - df.loc[bad_t, "score"].sum()) / n_total * 100
        total = (s.loc[common, "score"].mean() - df.loc[common, "score"].mean()) * 100
        print(f"- {name} − {target} 的总差 {total:+.2f} 分，其中这批 bad token 贡献 {contrib:+.2f} 分")
        if name in oracles:
            arr = oracles[name]
            tk = [t for t in bad_t if t in arr["pos"]]
            s64 = arr["score64"][[arr["pos"][t] for t in tk]]
            print(f"- {name} 候选池（这批 token）：有 >0 候选 {int((s64.max(1) > EPS).sum())}/{len(tk)}，"
                  f"oracle64 {s64.max(1).mean() * 100:.2f}，mean64 {s64.mean() * 100:.2f}，安全候选比例 {(s64 > EPS).mean() * 100:.1f}%")

    if len(zero_flags) >= 1:
        names = list(zero_flags)
        print(f"\n### 分类（{target} 的 bad token，按其他模型是否也选 0 分；'0'=也选 0，'+'=>0）")
        key = pd.Series("", index=bad_t)
        for n in names:
            key = key + n + zero_flags[n].reindex(bad_t).map({True: "=0 ", False: "=+ "})
        sub = df.loc[bad_t]
        tab = pd.DataFrame({"key": key.str.strip(), "wrong": sub["wrong"], "allzero": sub["allzero"],
                            "nc0": sub["nc0"], "dac0": sub["dac0"]})
        g = tab.groupby("key").agg(n=("wrong", "size"), wrong=("wrong", "sum"), allzero=("allzero", "sum"),
                                   nc0=("nc0", "sum"), dac0=("dac0", "sum")).sort_values("n", ascending=False)
        print(f"| 组合 | 个数 | 占 bad | {target} 选错 | {target} 全 0 | {target} NC=0 | {target} DAC=0 |")
        print("|---|---|---|---|---|---|---|")
        for k_, r in g.iterrows():
            print(f"| {k_} | {r['n']} | {pct(r['n'], nb)} | {r['wrong']} | {r['allzero']} | {r['nc0']} | {r['dac0']} |")
        # reverse direction
        for n in names:
            rev = int((zero_flags[n] & ~df["bad"].reindex(zero_flags[n].index).fillna(False)).sum())
            print(f"- 反向：{n} 选 0 分而 {target} >0 的 token {rev} 个")
    return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="E3")
    ap.add_argument("--scorer", action="append", required=True, help="NAME=v1 scorer result dir or csv")
    ap.add_argument("--oracle", action="append", required=True, help="NAME=v1 oracle64 result dir (csv + *_arrays.npz)")
    ap.add_argument("--v1-cache", required=True)
    ap.add_argument("--navsim-logs", default=None, help="navsim_logs/test dir; omit to skip token features")
    ap.add_argument("--min-bad", type=int, default=5)
    ap.add_argument("--top-logs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--horizon-s", type=float, default=4.0)
    ap.add_argument("--list-examples", type=int, default=15)
    a = ap.parse_args()

    sc_paths, or_paths = parse_named(a.scorer), parse_named(a.oracle)
    if a.target not in sc_paths or a.target not in or_paths:
        raise SystemExit("[FAIL] --target must appear in both --scorer and --oracle")
    scorers, oracles, orows = {}, {}, {}
    for n, p in sc_paths.items():
        rows, agg, f = read_eval_csv(p, V1_AGG_TOKENS)
        scorers[n] = rows
        v = rows[rows["valid"]]
        agg_s = float(agg["score"].iloc[0]) if len(agg) else float("nan")
        print(f"[load] scorer {n}: {f}  valid {len(v)}  mean {v['score'].mean() * 100:.4f}  average 行 {agg_s * 100:.4f}")
    for n, p in or_paths.items():
        rows, arr, f, npz = load_oracle(p)
        oracles[n], orows[n] = arr, rows
        print(f"[load] oracle {n}: {f}  +  {npz.name}  N={len(arr['token'])}  columns={list(rows.columns)}")

    v, ok = step1(a.target, scorers[a.target], orows[a.target], oracles[a.target])
    if not ok:
        sys.exit(2)
    n_total = len(v)
    v = step2(a.target, v, oracles[a.target], n_total)

    idx_cache = v1_cache_index(a.v1_cache)
    print(f"\n[load] v1 cache 索引：{len(idx_cache)} token，{idx_cache['log_name'].nunique()} log")
    feat = None
    if a.navsim_logs:
        tl = idx_cache.loc[idx_cache.index.intersection(v["token"])].reset_index()
        feat = token_features(a.navsim_logs, tl.set_index("token"), a.workers, a.horizon_s)
        print(f"[load] log pkl 特征：{len(feat)} token")
    df = step3(a.target, v, idx_cache, feat, a.min_bad, a.top_logs)

    others = {n: s for n, s in scorers.items() if n != a.target}
    step4(a.target, df, others, {n: o for n, o in oracles.items() if n != a.target}, n_total)

    if a.list_examples and others:
        print(f"\n### 例子：{a.target} 选错（有 >0 候选）且其他模型都 >0 的 token（按最优候选得分降序）")
        mask = df["wrong"].copy()
        for n, s in others.items():
            ss = s[s["valid"]].set_index("token")["score"]
            mask &= ss.reindex(df.index).fillna(0) > EPS
        ex = df[mask].sort_values("oracle", ascending=False).head(a.list_examples)
        cols = ["log_name", "cause", "n_pos", "first_safe_rank", "oracle", "ade_pick", "ade_best"]
        cols += [c for c in ("city", "maneuver", "command") if c in ex.columns]
        print(ex[cols].to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
