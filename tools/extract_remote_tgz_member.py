#!/usr/bin/env python3
"""Stream one member from a remote .tgz and stop once it is extracted."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("member_suffix")
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")

    with requests.get(args.url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for index, member in enumerate(archive, start=1):
                if index % 5000 == 0:
                    print(f"scanned {index} archive members", flush=True)
                if not member.isfile() or not member.name.endswith(args.member_suffix):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name}")
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(args.output)
                print(f"extracted {member.name} -> {args.output.resolve()}")
                return

    temporary.unlink(missing_ok=True)
    raise FileNotFoundError(args.member_suffix)


if __name__ == "__main__":
    main()
