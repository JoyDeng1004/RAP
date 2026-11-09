#!/usr/bin/env python3
"""Build the raster-only complement of a K-shot split as a directory of symlinks.

RAP §4.2 varies the real-data ratio by keeping real images for X% of navtrain and
replacing the rest with raster, at a fixed total. The kept part is the K-shot split
(e.g. ``navsim_kshot_p001_seed0``, the main source with real + raster); this script
links every other log of the pool split (``navsim_kshot_p100_seed0``, 978 logs) into
``--out`` so it can be added as an extra source with ``use_real:false,score_mask:true``.

CacheOnlyDataset lists log directories with ``iterdir`` and follows symlinks. The
manifest is written next to the subset, not inside it, because every entry of a
cache root is read as a log directory.

    python process_data/make_kshot_complement.py \
        --cache /gs/bs/tga-RLA/qdeng/RAP/cache/rap_ego \
        --pool  navsim/planning/script/config/training/train_val_test_log_split/navsim_kshot_p100_seed0.yaml \
        --keep  navsim/planning/script/config/training/train_val_test_log_split/navsim_kshot_p001_seed0.yaml \
        --out   /gs/bs/tga-RLA/qdeng/RAP/cache/rap_ego_rest_p001_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


def read_list(split_yaml: Path, key: str) -> List[str]:
    """Read one top-level list (train_logs / val_logs) from a split yaml without a yaml dependency."""
    out: List[str] = []
    inside = False
    for raw in split_yaml.read_text().replace("\r", "").splitlines():
        if raw.startswith(f"{key}:"):
            inside = True
            continue
        if inside and raw and not raw.startswith((" ", "-", "\t")):
            break
        item = raw.strip()
        if inside and item.startswith("- "):
            out.append(item[2:].strip().strip("'\""))
    return out


def complete_tokens(log_dir: Path) -> int:
    return sum(1 for t in log_dir.iterdir()
               if t.is_dir() and (t / "rap_feature.gz").is_file() and (t / "rap_target.gz").is_file())


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="full cache root (<log>/<token>/rap_*.gz)")
    p.add_argument("--pool", type=Path, required=True, help="split yaml whose train_logs is the full pool")
    p.add_argument("--keep", type=Path, required=True, help="split yaml whose train_logs keep real images")
    p.add_argument("--out", type=Path, required=True, help="complement root to create (must not exist)")
    args = p.parse_args(argv)

    cache = args.cache.resolve()
    if args.out.exists():
        print(f"FAIL {args.out} 已存在；换个名字，或确认后自己删掉", file=sys.stderr)
        return 1
    pool = read_list(args.pool, "train_logs")
    keep = read_list(args.keep, "train_logs")
    val = set(read_list(args.pool, "val_logs")) | set(read_list(args.keep, "val_logs"))
    if not pool or not keep:
        print(f"FAIL 没读到 train_logs：pool {len(pool)}，keep {len(keep)}", file=sys.stderr)
        return 1
    missing_keep = sorted(set(keep) - set(pool))
    if missing_keep:
        print(f"FAIL keep 里有 {len(missing_keep)} 个 log 不在 pool 里，例如 {missing_keep[:3]}", file=sys.stderr)
        return 1
    rest = [n for n in pool if n not in set(keep)]
    if set(rest) & val:
        print(f"FAIL 补集里混进了 val log：{sorted(set(rest) & val)[:3]}", file=sys.stderr)
        return 1
    absent = [n for n in rest if not (cache / n).is_dir()]
    if absent:
        print(f"FAIL {len(absent)} 个补集 log 不在 cache 里，例如 {absent[:3]}", file=sys.stderr)
        return 1

    t0 = time.time()
    tokens = {n: complete_tokens(cache / n) for n in rest}
    keep_tokens = sum(complete_tokens(cache / n) for n in keep if (cache / n).is_dir())
    args.out.mkdir(parents=True)
    for n in rest:
        (args.out / n).symlink_to(cache / n, target_is_directory=True)

    manifest = {
        "cache": str(cache), "pool": str(args.pool), "keep": str(args.keep),
        "pool_logs": len(pool), "keep_logs": len(keep), "rest_logs": len(rest),
        "keep_tokens": keep_tokens, "rest_tokens": sum(tokens.values()),
        "total_tokens": keep_tokens + sum(tokens.values()),
        "logs": tokens,
    }
    manifest_path = args.out.parent / f"{args.out.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"OK {args.out}: pool {len(pool)} − keep {len(keep)} = {len(rest)} logs; "
          f"tokens keep {keep_tokens} + rest {manifest['rest_tokens']} = {manifest['total_tokens']} "
          f"({time.time() - t0:.0f}s); manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
