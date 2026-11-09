#!/usr/bin/env python3
"""Check that E4's Hydra config differs from E3's only where E4 is defined to differ.

E4 = E3 without the nuScenes native-rig pair (nusc_pair). Any difference outside
experiment_name and extra_train_sources fails, as does an E4 source list other
than E3's nusc_navsimrig_raster entry alone.

  python scripts/jobs/check_e4_config.py \
      $NAVSIM_EXP_ROOT/E3_mix_p010_s0/<ts>/code/hydra/config.yaml \
      $NAVSIM_EXP_ROOT/E4_navsimrig_raster_p010_s0/<ts>/code/hydra/config.yaml

--smoke also tolerates the two keys a fast_dev_run smoke changes
(trainer.params.fast_dev_run, resume_ckpt_path) and reports them separately.
Exit status: 0 pass, 1 check failed, 2 bad input.
"""

import argparse
import sys

import yaml

ALLOWED = {"experiment_name", "extra_train_sources"}
SMOKE_ONLY = {"trainer.params.fast_dev_run", "resume_ckpt_path"}
KEPT_SOURCE = "nusc_navsimrig_raster"
DROPPED_SOURCE = "nusc_pair"
MISSING = "<missing>"


def diff(a, b, path, out):
    """Append (path, e3_value, e4_value) for every leaf where a and b differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append((p, MISSING, b[k]))
            elif k not in b:
                out.append((p, a[k], MISSING))
            else:
                diff(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.<len>", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif type(a) is not type(b) or a != b:
        # type() keeps true/1 and 1.0/1 apart.
        out.append((path, a, b))


def root_key(path):
    return path.split(".", 1)[0].split("[", 1)[0]


def load(path):
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"无法读取 {path}: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(cfg, dict):
        print(f"{path} 不是 YAML 映射", file=sys.stderr)
        sys.exit(2)
    return cfg


def check_e4(e3, e4):
    """Return failed assertions on E4's own values and its relation to E3's sources."""
    errors = []

    def expect(cond, msg):
        if not cond:
            errors.append(msg)

    src = e4.get("extra_train_sources")
    expect(isinstance(src, list) and len(src) == 1, f"E4 extra_train_sources 应恰好 1 项，实际为 {src!r}")
    if isinstance(src, list) and len(src) == 1 and isinstance(src[0], dict):
        s = src[0]
        expect(s.get("use_real") is False, f"use_real 应为 false，实际为 {s.get('use_real')!r}")
        rig = s.get("rig_id")
        expect(type(rig) is int and rig == 0, f"rig_id 应为 0，实际为 {rig!r}")
        expect(s.get("score_mask") is False, f"score_mask 应为 false，实际为 {s.get('score_mask')!r}")
        expect(s.get("name") == KEPT_SOURCE, f"name 应为 {KEPT_SOURCE}，实际为 {s.get('name')!r}")
        cache = str(s.get("cache_path", ""))
        expect(cache.rstrip("/").endswith("/rap_nusc_navsimrig"), f"cache_path 应指向 rap_nusc_navsimrig，实际为 {cache!r}")
        expect("nativerig" not in cache, f"cache_path 不能是 nativerig: {cache!r}")

        e3_src = e3.get("extra_train_sources") or []
        e3_names = sorted(x.get("name") for x in e3_src if isinstance(x, dict))
        expect(e3_names == sorted([DROPPED_SOURCE, KEPT_SOURCE]),
               f"E3 extra_train_sources 应为 {DROPPED_SOURCE} + {KEPT_SOURCE}，实际为 {e3_names}")
        e3_kept = [x for x in e3_src if isinstance(x, dict) and x.get("name") == KEPT_SOURCE]
        if len(e3_kept) == 1:
            d = []
            diff(e3_kept[0], s, KEPT_SOURCE, d)
            for p, a, b in d:
                errors.append(f"{p} 与 E3 不同: E3={a!r} E4={b!r}")

    for key in ("restrict_train_logs", "train_score_mask"):
        expect(e4.get(key) is True, f"{key} 应为 true，实际为 {e4.get(key, MISSING)!r}")
    ckpt = (e4.get("agent") or {}).get("checkpoint_path", "")
    expect(ckpt in ("", None), f"E4 从头训练，agent.checkpoint_path 应为空，实际为 {ckpt!r}")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e3_config", help="E3 code/hydra/config.yaml")
    ap.add_argument("e4_config", help="E4 code/hydra/config.yaml")
    ap.add_argument("--smoke", action="store_true", help="also tolerate fast_dev_run and resume_ckpt_path")
    args = ap.parse_args()

    e3, e4 = load(args.e3_config), load(args.e4_config)
    diffs = []
    diff(e3, e4, "", diffs)

    bad = []
    print(f"E3: {args.e3_config}\nE4: {args.e4_config}\n")
    print(f"{len(diffs)} differences:")
    for p, a, b in diffs:
        if root_key(p) in ALLOWED:
            tag = "allowed"
        elif args.smoke and p in SMOKE_ONLY:
            tag = "smoke"
        else:
            tag = "FAIL"
            bad.append(p)
        print(f"  [{tag}] {p}: E3={a!r} -> E4={b!r}")

    errors = check_e4(e3, e4)
    print("\nE4 assertions:")
    for msg in errors:
        print(f"  [FAIL] {msg}")
    if not errors:
        print("  all passed")

    if bad or errors:
        print(f"\nRESULT: FAIL ({len(bad)} non-whitelisted differences, {len(errors)} failed assertions)")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
