"""Run the whole suite: ``python tests/run.py`` (or ``python -m tests.run``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import harness  # noqa: E402

MODULES = [
    "tests.test_exceptions",
    "tests.test_config",
    "tests.test_schemas",
    "tests.test_jev_client",
    "tests.test_service",
    "tests.test_testing",
    "tests.test_api",
    "tests.test_cli",
]


def load():
    import importlib

    return [importlib.import_module(name) for name in MODULES]


if __name__ == "__main__":
    harness.main(load())
