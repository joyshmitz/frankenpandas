#!/usr/bin/env python3
"""Reverse-conformance oracle: pandas reads frankenpandas output.

Per br-frankenpandas-kdwn: the existing pandas_oracle.py proves
"frankenpandas produces the same answer as pandas" for in-memory
operations. Reverse conformance validates the claim that
"pandas can read files we write". Without this channel, we could
ship a parquet writer that produces bytes only our reader accepts.

Protocol:
  stdin  — JSON { "format": "<csv|parquet|feather|arrow_ipc|...>",
                  "bytes_base64": "<base64-encoded file bytes>",
                  "options": { ... per-format options ... } }
  stdout — JSON { "ok": true, "parsed_frame": { "index": [...], "columns": [...],
                  "rows": [[...]], "dtypes": {...} } }
           or    { "ok": false, "error": "<message>", "error_type": "<cls>" }

This script is invoked as a subprocess by
`crates/fp-conformance/tests/reverse_conformance.rs` (see br-kdwn).

Fresh pandas import per run (subprocess boundary) — no shared state
with the forward oracle.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from typing import Any


def _scalar_to_json(value: Any) -> Any:
    # Minimal normalizer shared with the forward oracle's convention.
    import math

    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if value is None:
        return {"kind": "null", "value": "null"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int64", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "null", "value": "na_n"}
        return {"kind": "float64", "value": value}
    return {"kind": "utf8", "value": str(value)}


def _read(fmt: str, payload: bytes, options: dict[str, Any]):
    import pandas as pd

    buf = io.BytesIO(payload)
    if fmt == "csv":
        return pd.read_csv(buf, **options)
    if fmt == "json":
        return pd.read_json(buf, **options)
    if fmt == "jsonl":
        return pd.read_json(buf, lines=True, **options)
    if fmt == "parquet":
        return pd.read_parquet(buf, **options)
    if fmt == "feather":
        return pd.read_feather(buf, **options)
    if fmt == "arrow_ipc":
        # IPC streaming format; pandas routes to pyarrow.
        import pyarrow.ipc as ipc

        reader = ipc.open_stream(buf)
        return reader.read_pandas()
    if fmt == "excel":
        return pd.read_excel(buf, **options)
    raise ValueError(f"unsupported reverse-oracle format: {fmt!r}")


def _frame_to_json(frame) -> dict[str, Any]:
    columns = [str(c) for c in frame.columns.tolist()]
    dtypes = {str(c): str(frame[c].dtype) for c in columns}
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([_scalar_to_json(v) for v in row])
    index = [_scalar_to_json(v) for v in frame.index.tolist()]
    return {"index": index, "columns": columns, "rows": rows, "dtypes": dtypes}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reverse-conformance oracle (pandas reads our output)"
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Path to JSON request (default: stdin)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Path to JSON response (default: stdout)",
    )
    args = parser.parse_args()

    if args.input == "-":
        request_raw = sys.stdin.read()
    else:
        with open(args.input, encoding="utf-8") as fh:
            request_raw = fh.read()

    try:
        request = json.loads(request_raw)
        fmt = request["format"]
        payload = base64.b64decode(request["bytes_base64"])
        options = request.get("options", {})
        frame = _read(fmt, payload, options)
        response = {"ok": True, "parsed_frame": _frame_to_json(frame)}
    except Exception as exc:  # noqa: BLE001
        response = {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }

    serialized = json.dumps(response, separators=(",", ":"))
    if args.output == "-":
        sys.stdout.write(serialized)
        sys.stdout.flush()
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(serialized)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
