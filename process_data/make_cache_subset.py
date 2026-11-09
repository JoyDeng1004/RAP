#!/usr/bin/env python3
"""Build a log-level subset of a feature cache as a directory of symlinks.

Logs (WOMD: one per scenario) are ordered by a salted SHA-1 of their name and
taken whole until at least ``--samples`` complete tokens are covered, so all
anchors of a scenario stay together and the same salt always yields the same
subset; a larger ``--samples`` with the same salt is a superset of a smaller one.

The subset is used as an ordinary cache path (``extra_train_sources``):
CacheOnlyDataset lists log directories with ``iterdir`` and follows symlinks.
The manifest is written next to the subset, not inside it, because the main
training dataset treats every entry of the cache root as a log directory.

    python process_data/make_cache_subset.py \
        --cache /gs/bs/tga-RLA/qdeng/RAP/cache/rap_womd_navsimrig \
        --out   /gs/bs/tga-RLA/qdeng/RAP/cache/rap_womd_navsimrig_23k --samples 23099
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List


def complete_tokens(log_dir: Path) -> int:
    return sum(1 for t in log_dir.iterdir()
               if t.is_dir() and (t / "rap_feature.gz").is_file() and (t / "rap_target.gz").is_file())


def salted_order(names: List[str], salt: str) -> List[str]:
    return sorted(names, key=lambda n: hashlib.sha1(f"{salt}:{n}".encode()).hexdigest())


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="full cache root (<log>/<token>/rap_*.gz)")
    p.add_argument("--out", type=Path, required=True, help="subset root to create (must not exist)")
    p.add_argument("--samples", type=int, required=True, help="minimum number of tokens to cover")
    p.add_argument("--salt", default="womd-subset-v1")
    args = p.parse_args(argv)

    cache = args.cache.resolve()
    if args.out.exists():
        print(f"FAIL {args.out} 已存在；换个名字，或确认后自己删掉", file=sys.stderr)
        return 1
    logs = [d.name for d in cache.iterdir() if d.is_dir()]
    if not logs:
        print(f"FAIL {cache} 下没有 log 目录", file=sys.stderr)
        return 1

    t0 = time.time()
    chosen: Dict[str, int] = {}
    total = 0
    for name in salted_order(logs, args.salt):
        n = complete_tokens(cache / name)
        if n == 0:
            continue
        chosen[name] = n
        total += n
        if total >= args.samples:
            break
    if total < args.samples:
        print(f"FAIL 全部 {len(logs)} 个 log 只有 {total} 个完整 token，不够 {args.samples}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True)
    for name in chosen:
        (args.out / name).symlink_to(cache / name, target_is_directory=True)

    manifest = args.out.parent / f"{args.out.name}.manifest.json"
    manifest.write_text(json.dumps({
        "source": str(cache), "subset": str(args.out.resolve()), "salt": args.salt,
        "requested_samples": args.samples, "samples": total, "logs": len(chosen),
        "source_logs": len(logs), "log_names": sorted(chosen),
    }, indent=1))
    print(f"[subset] {len(chosen)}/{len(logs)} 个 log，{total} 个 token（要求 ≥ {args.samples}，"
          f"每 log {total / len(chosen):.2f}），{time.time() - t0:.0f}s")
    print(f"[subset] -> {args.out}")
    print(f"[subset] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
