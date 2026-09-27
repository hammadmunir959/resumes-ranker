"""Settings, loaded and validated once from the environment or a ``.env`` file.

Every field accepts either its environment variable or its Python name. The
Python name is listed first in each ``AliasChoices`` because pydantic resolves
the choices left to right, which gives the precedence a caller expects:

    explicit argument  >  real environment  >  .env file  >  default

That ordering matters: it means ``Settings.build(jev_max_rps=5)`` means the same
thing on a developer machine that happens to export ``JEV_MAX_RPS`` as it does in
CI, so tests cannot be perturbed by ambient configuration.

The only required value is the OpenRouter key, and it is looked up under both
``OPENROUTER_API_KEY`` and ``OPEN_ROUTER_KEY`` so an existing ``.env`` keeps
working.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.exceptions import ConfigError

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_MODEL = "typesafe/jev-1.13"
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


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    #: Checked in order; the canonical name wins when both are set.
    api_key_names: ClassVar[tuple[str, ...]] = ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY")

    # Deliberately has no env alias. The key is resolved explicitly in
    # __init__ so that "which of the two names won" is decided by code we can
    # read and test, rather than by pydantic-settings' source ordering.
    openrouter_api_key: str = Field(default="", repr=False)
    jev_model: str = Field(
        default=DEFAULT_MODEL,
        validation_alias=AliasChoices("jev_model", "JEV_MODEL"),
    )
    jev_base_url: str = Field(
        default=DEFAULT_BASE_URL,
        validation_alias=AliasChoices("jev_base_url", "JEV_BASE_URL"),
    )
    jev_max_concurrency: int = Field(
        default=10, ge=1, le=64,
        validation_alias=AliasChoices("jev_max_concurrency", "JEV_MAX_CONCURRENCY"),
    )
    jev_max_rps: float = Field(
        default=20.0, gt=0,
        validation_alias=AliasChoices("jev_max_rps", "JEV_MAX_RPS"),
    )
    jev_max_retries: int = Field(
        default=4, ge=1, le=10,
        validation_alias=AliasChoices("jev_max_retries", "JEV_MAX_RETRIES"),
    )
    jev_backoff_base: float = Field(
        default=1.5, gt=0,
        validation_alias=AliasChoices("jev_backoff_base", "JEV_BACKOFF_BASE"),
    )
    jev_backoff_max: float = Field(
        default=30.0, gt=0,
        validation_alias=AliasChoices("jev_backoff_max", "JEV_BACKOFF_MAX"),
    )
    jev_timeout: float = Field(
        default=60.0, gt=0,
        validation_alias=AliasChoices("jev_timeout", "JEV_TIMEOUT"),
    )
    jev_max_batch_size: int = Field(
        default=100, ge=1,
        validation_alias=AliasChoices("jev_max_batch_size", "JEV_MAX_BATCH_SIZE"),
    )
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("host", "HOST"))
    port: int = Field(default=8000, ge=1, le=65535,
                      validation_alias=AliasChoices("port", "PORT"))
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("log_level", "LOG_LEVEL"),
    )

    # -- construction ------------------------------------------------------ #

    def __init__(self, **values: Any):
        """Resolve the API key, then validate as usual.

        The key is looked for under both accepted names, in this order:
        an explicit argument, the real environment, then the ``.env`` file.
        Doing it here rather than in a validator keeps the rule in one readable
        place instead of depending on how pydantic-settings orders its sources.
        """
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

    # -- validation -------------------------------------------------------- #

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

    # -- derived ----------------------------------------------------------- #

    @property
    def uses_mock_key(self) -> bool:
        """True when running against a local mock with a placeholder key."""
        return self.openrouter_api_key.startswith("mock-")

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to expose over HTTP. Never includes the key."""
        return {
            "model": self.jev_model,
            "base_url": self.jev_base_url,
            "max_concurrency": self.jev_max_concurrency,
            "max_requests_per_second": self.jev_max_rps,
            "max_retries": self.jev_max_retries,
            "timeout_seconds": self.jev_timeout,
            "max_batch_size": self.jev_max_batch_size,
        }

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
    "DEFAULT_BASE_URL",
    "ENV_PATH",
]
