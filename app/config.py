"""Runtime configuration: validated once, at the edge.

The environment is used for exactly one thing, the API key. Everything else is a
default in this file, or an argument a caller passes in code. That is a
deliberate trade: a deployment that wants different numbers changes the code or
passes ``Settings.build(...)``, and in exchange there is no ambient configuration
to be perturbed by a stray shell variable.

Key resolution order, which is the only place the environment is consulted:

    explicit argument  >  real environment  >  .env file

``Settings.build(...)`` therefore means the same thing on a developer machine
that happens to export variables as it does in CI, so tests cannot be perturbed
by their surroundings.

The key is looked up under both ``OPENROUTER_API_KEY`` and ``OPEN_ROUTER_KEY``
so an existing ``.env`` keeps working, and "which name won" is decided by code
below rather than by a library's source ordering.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.exceptions import ConfigError

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_FALLBACK_MODEL = "respan/span-01-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha/decisions"

#: Loopback hosts allowed to use plain http, so a local mock server works.
#: Anywhere else over http would put the API key on the wire in the clear.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@lru_cache(maxsize=1)
def _dotenv_values() -> dict[str, str]:
    """Parsed ``.env`` values, used to find the key under its alternate name."""
    if not ENV_PATH.exists():
        return {}
    try:
        from dotenv import dotenv_values

        return {k: v for k, v in dotenv_values(ENV_PATH).items() if v}
    except Exception:
        return {}


def is_safe_base_url(url: str) -> bool:
    """True if it is https, or http to loopback."""
    if url.startswith("https://"):
        return True
    if not url.startswith("http://"):
        return False
    return (urlparse(url).hostname or "") in _LOOPBACK_HOSTS


class Settings(BaseModel):
    """Validated configuration for one process.

    A plain model rather than an env-driven one, so a value is always either a
    default written here or something a caller passed explicitly. There is no
    third source that can change behaviour without appearing in the code.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    #: Checked in order; the canonical name wins when both are set.
    api_key_names: ClassVar[tuple[str, ...]] = ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY")

    openrouter_api_key: str = Field(default="", repr=False)
    jev_model: str = DEFAULT_MODEL
    #: Used when the primary model is the reason a call failed. Empty disables
    #: the fallback, which is how you turn it off without changing anything else.
    jev_fallback_model: str = DEFAULT_FALLBACK_MODEL
    jev_base_url: str = DEFAULT_BASE_URL

    # -- transport ---------------------------------------------------------- #

    jev_max_concurrency: int = Field(default=10, ge=1, le=64)
    jev_max_rps: float = Field(default=20.0, gt=0)
    jev_max_retries: int = Field(default=4, ge=1, le=10)
    jev_backoff_base: float = Field(default=1.5, gt=0)
    jev_backoff_max: float = Field(default=30.0, gt=0)
    jev_timeout: float = Field(default=60.0, gt=0)
    jev_max_batch_size: int = Field(default=100, ge=1)

    # -- server ------------------------------------------------------------- #

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    # -- construction ------------------------------------------------------- #

    def __init__(self, **values: Any):
        """Resolve the API key from the environment, then validate as usual."""
        # The environment spelling of the key is accepted as a keyword too, so
        # ``Settings.build(OPENROUTER_API_KEY=...)`` works the way a caller who
        # has the name in front of them would expect. It is consumed here
        # rather than declared as a field, because the key is the one value
        # resolved by hand.
        for name in self.api_key_names:
            alias = values.pop(name, None)
            if alias is not None and not values.get("openrouter_api_key"):
                values["openrouter_api_key"] = alias
        super().__init__(**{**values, "openrouter_api_key": self._find_key(values)})

    @classmethod
    def _find_key(cls, values: dict[str, Any]) -> str:
        """The first non-empty key among the explicit value, env, and .env."""
        for name in ("openrouter_api_key", *cls.api_key_names):
            explicit = values.get(name)
            if isinstance(explicit, str) and explicit.strip():
                return explicit.strip()
        for name in cls.api_key_names:
            for source in (os.environ.get(name), _dotenv_values().get(name)):
                if isinstance(source, str) and source.strip():
                    return source.strip()
        raise ValueError(
            "missing OpenRouter API key; set "
            + " or ".join(cls.api_key_names)
            + f" in the environment or in {ENV_PATH}"
        )

    # -- validation --------------------------------------------------------- #

    @field_validator("openrouter_api_key")
    @classmethod
    def _strip_key(cls, value: str) -> str:
        return value.strip()

    @field_validator("jev_base_url")
    @classmethod
    def _require_safe_url(cls, value: str) -> str:
        if not is_safe_base_url(value):
            raise ValueError(
                "jev_base_url must be https, except for loopback addresses "
                "(http://127.0.0.1:... to use a local mock server)"
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    # -- derived ------------------------------------------------------------ #

    @property
    def uses_mock_key(self) -> bool:
        """True when running against a local mock with a placeholder key."""
        return self.openrouter_api_key.startswith("mock-")

    @classmethod
    def build(cls, **overrides: Any) -> "Settings":
        """Settings for offline work, with a placeholder key.

        Used by ``--mock`` and the test suite so neither needs a real credential.
        Any keyword may be overridden, e.g. ``Settings.build(jev_model="...")``.

        A key passed in overrides is kept as-is; the placeholder is only added
        when the caller supplied neither name, so a real key is never shadowed.
        """
        values: dict[str, Any] = dict(overrides)
        if not any(name in values for name in ("openrouter_api_key", *cls.api_key_names)):
            values["openrouter_api_key"] = "mock-key-not-used"
        return cls(**values)

    @classmethod
    def for_reporting(cls) -> "Settings":
        """Best-effort settings for diagnostics. Never raises.

        Used by ``/health`` when validation failed, so the health payload can
        still describe what was configured. Unlike :meth:`build` this reports
        real key presence: a missing key must not be shown as configured, and a
        mock placeholder must never be invented for a process that may be about
        to make live calls.
        """
        try:
            return cls()
        except Exception:
            pass
        try:
            key = cls._find_key({})
        except Exception:
            key = ""
        return cls.model_construct(openrouter_api_key=key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Call ``get_settings.cache_clear()`` to reload."""
    try:
        return Settings()
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc


__all__ = [
    "Settings",
    "get_settings",
    "is_safe_base_url",
    "ConfigError",
    "DEFAULT_MODEL",
    "DEFAULT_FALLBACK_MODEL",
    "DEFAULT_BASE_URL",
    "ENV_PATH",
]
