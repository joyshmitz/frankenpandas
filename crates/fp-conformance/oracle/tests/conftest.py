"""Shared fixtures for pandas_oracle.py pytest suite.

Per br-frankenpandas-urhy: the oracle is the conformance spec's
"source of truth" but shipped with zero tests. These fixtures
make the oracle importable from within a pytest process and expose
the `pd` module so tests can validate op handlers without shelling
out through the stdin/stdout CLI protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


ORACLE_ROOT = Path(__file__).resolve().parents[1]
if str(ORACLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORACLE_ROOT))


@pytest.fixture(scope="session")
def oracle():
    """Import the oracle module once per session."""
    import pandas_oracle

    return pandas_oracle


@pytest.fixture(scope="session")
def pd():
    """Expose pandas for tests that construct reference values."""
    import pandas

    return pandas
