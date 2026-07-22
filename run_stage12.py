"""1차 파이프라인: 데이터 수집 → master_dataset 생성."""
from __future__ import annotations

import argparse
import sys

from config import DEFAULT_END, DEFAULT_START
from data_collector import collect_all
from feature_builder import build_master_dataset

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()

    collected = collect_all(args.start, args.end)
    build_master_dataset(
        start=args.start,
        end=args.end,
        collected=collected,
        collect_if_missing=False,
    )


if __name__ == "__main__":
    main()
