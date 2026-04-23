#!/usr/bin/env python3
"""Parametric scale-fixture generator.

Per br-frankenpandas-k05t: the existing 1,249 packet fixtures are all
sub-4 KB. Pandas parity is untested on any frame size greater than
~30 rows. This generator produces fixtures at three tiers:

  Tier-S  (≤ 30 rows)         — status quo; skipped by default
  Tier-M  (1k-10k rows)       — runs under --features scale-m
  Tier-L  (100k+ rows)        — runs under nightly scheduled CI only

Usage:
    python3 crates/fp-conformance/oracle/generate_scale_fixtures.py \\
        --tier m \\
        --out crates/fp-conformance/fixtures/packets_scale/

Each emitted fixture is a standard packet-fixture JSON so it plugs
into the existing run_packet_suite harness. The generator is
deterministic (fixed seed) so fixture bytes are stable across runs.

Op coverage (first wave): series_add, series_sum, series_sort_values.
More op/class pairs can be added to `OP_GENERATORS` without a harness
change.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TIERS = {
    "s": [10, 30],
    "m": [1_000, 5_000, 10_000],
    "l": [100_000, 500_000],
}


def _int_series(rng: random.Random, length: int, prefix: str = "") -> dict[str, Any]:
    return {
        "index": [{"kind": "int64", "value": i} for i in range(length)],
        "values": [{"kind": "int64", "value": rng.randint(-1000, 1000)} for _ in range(length)],
    }


def _float_series(rng: random.Random, length: int) -> dict[str, Any]:
    return {
        "index": [{"kind": "int64", "value": i} for i in range(length)],
        "values": [
            {"kind": "float64", "value": round(rng.uniform(-1000.0, 1000.0), 4)}
            for _ in range(length)
        ],
    }


def _fixture_series_add(rng: random.Random, length: int, tier: str) -> dict[str, Any]:
    return {
        "packet_id": f"FP-P2D-SCALE-{tier.upper()}-SERIES-ADD-{length}",
        "case_id": f"series_add_{tier}_{length}",
        "operation": "series_add",
        "left": _int_series(rng, length),
        "right": _int_series(rng, length),
        "mode": "strict",
    }


def _fixture_series_sum(rng: random.Random, length: int, tier: str) -> dict[str, Any]:
    return {
        "packet_id": f"FP-P2D-SCALE-{tier.upper()}-SERIES-SUM-{length}",
        "case_id": f"series_sum_{tier}_{length}",
        "operation": "nan_sum",
        "series": _float_series(rng, length),
        "mode": "strict",
    }


def _fixture_series_sort(rng: random.Random, length: int, tier: str) -> dict[str, Any]:
    return {
        "packet_id": f"FP-P2D-SCALE-{tier.upper()}-SERIES-SORT-{length}",
        "case_id": f"series_sort_{tier}_{length}",
        "operation": "series_sort_values",
        "series": _int_series(rng, length),
        "ascending": True,
        "mode": "strict",
    }


OP_GENERATORS = [
    _fixture_series_add,
    _fixture_series_sum,
    _fixture_series_sort,
]


def generate(tier: str, out_dir: Path, seed: int) -> int:
    sizes = TIERS[tier]
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for length in sizes:
        for gen in OP_GENERATORS:
            fixture = gen(rng, length, tier)
            path = out_dir / f"{fixture['packet_id'].lower()}.json"
            path.write_text(json.dumps(fixture, indent=2))
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate scale-tier conformance fixtures (br-frankenpandas-k05t)"
    )
    parser.add_argument(
        "--tier",
        choices=sorted(TIERS.keys()),
        required=True,
        help="Fixture size tier (s=small, m=medium, l=large)",
    )
    parser.add_argument(
        "--out",
        default="crates/fp-conformance/fixtures/packets_scale",
        help="Output directory (default: crates/fp-conformance/fixtures/packets_scale)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed (default: 42)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) / args.tier
    count = generate(args.tier, out_dir, args.seed)
    print(f"generated {count} tier-{args.tier} scale fixtures → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
