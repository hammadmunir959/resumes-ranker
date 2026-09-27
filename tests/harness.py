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
import os
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def test_functions(module: Any) -> list[Callable[[], Any]]:
    return [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]


def settings_env_names() -> set[str]:
    """Every environment variable ``Settings`` can read."""
    from app.config import Settings

    names = set(Settings.api_key_names)
    for field in Settings.model_fields.values():
        alias = field.validation_alias
        if alias is None:
            continue
        if hasattr(alias, "choices"):  # pydantic AliasChoices
            names.update(str(choice) for choice in alias.choices)
        else:
            names.add(str(alias))
    return names


@contextmanager
def hermetic_environment() -> Iterator[None]:
    """Run tests against a clean slate, whatever the machine exports.

    Several tests assert on default settings. A developer machine or CI runner
    that happens to export ``JEV_MAX_RPS`` would change those defaults and fail
    tests that have nothing to do with configuration, so a green run would
    depend on where it happened. Both the environment and the ``.env`` file are
    neutralized here, once, rather than in each test.

    Tests that want a specific value set it themselves, explicitly.
    """
    from app import config

    saved_env = {name: os.environ.pop(name, None) for name in settings_env_names()}
    saved_dotenv = config._dotenv_values
    config._dotenv_values = lambda: {}
    config.get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in saved_env.items():
            if value is not None:
                os.environ[name] = value
        config._dotenv_values = saved_dotenv
        config.get_settings.cache_clear()


def run_all(modules: Iterable[Any], *, verbose: bool = False) -> int:
    """Run every test in the given modules. Returns a process exit code."""
    # Several tests deliberately provoke retries, auth failures, and bad
    # responses. Their warnings are expected output, not findings, so they are
    # silenced to keep real failures visible.
    logging.getLogger("app").setLevel(logging.CRITICAL)

    passed = 0
    failures: list[tuple[str, str]] = []

    with hermetic_environment():
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
