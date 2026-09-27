"""A minimal test harness, because pytest is not installed.

Collects ``test_*`` functions from a module, runs them, and reports failures.
Async test functions are awaited automatically, so a test never has to build its
own event loop.

If pytest is ever added, these tests work under it unchanged: they are plain
functions using bare ``assert``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import time
import traceback
from typing import Any, Callable, Iterable

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def test_functions(module: Any) -> list[Callable[[], Any]]:
    return [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]


def run_all(modules: Iterable[Any], *, verbose: bool = False) -> int:
    """Run every test in the given modules. Returns a process exit code."""
    # Several tests deliberately provoke retries, auth failures, and bad
    # responses. Their warnings are expected output, not findings, so they are
    # silenced to keep real failures visible.
    logging.getLogger("app").setLevel(logging.CRITICAL)

    passed = 0
    failures: list[tuple[str, str]] = []

    for module in modules:
        name = module.__name__.split(".")[-1]
        module_tests = test_functions(module)
        if not module_tests:
            continue
        if verbose:
            print(f"\n{DIM}{name}{RESET}")

        for function in module_tests:
            label = f"{name}.{function.__name__}"
            started = time.monotonic()
            try:
                result = function()
                if inspect.iscoroutine(result):
                    asyncio.run(result)
            except Exception:
                failures.append((label, traceback.format_exc()))
                print(f"  {RED}FAIL{RESET} {function.__name__}")
                continue
            passed += 1
            if verbose:
                elapsed = (time.monotonic() - started) * 1000
                print(f"  {GREEN}ok{RESET}   {function.__name__} {DIM}{elapsed:.0f}ms{RESET}")

    total = passed + len(failures)
    print(f"\n{passed}/{total} passed")
    for label, tb in failures:
        print(f"\n{RED}{'=' * 70}{RESET}\n{RED}{label}{RESET}\n{tb.rstrip()}")
    return 1 if failures else 0


def main(modules: Iterable[Any]) -> None:
    sys.exit(run_all(modules, verbose=True))
