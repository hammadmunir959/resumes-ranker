"""Settings loading, validation, and key resolution."""

import os
import unittest.mock as mock

from app.config import DEFAULT_MODEL, Settings, get_settings, is_safe_base_url
from app.exceptions import ConfigError


def _without_key_env():
    """Patch out every way the key can be found, for the negative cases."""
    return mock.patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "", "OPEN_ROUTER_KEY": ""},
        clear=False,
    )


def test_defaults_apply():
    settings = Settings.build()
    assert settings.jev_model == DEFAULT_MODEL
    assert settings.jev_base_url == "https://openrouter.ai/api/alpha/decisions"
    assert settings.jev_max_rps == 20.0
    assert settings.jev_max_retries == 4
    assert settings.log_level == "INFO"


def test_build_uses_a_placeholder_key():
    assert Settings.build().uses_mock_key is True
    assert Settings.build(openrouter_api_key="sk-or-real").uses_mock_key is False


def test_env_var_name_is_accepted():
    assert Settings.build(OPENROUTER_API_KEY="sk-or-a").openrouter_api_key == "sk-or-a"


def test_python_field_name_is_accepted():
    assert Settings.build(openrouter_api_key="sk-or-b").openrouter_api_key == "sk-or-b"


def test_overrides_apply():
    settings = Settings.build(jev_model="~typesafe/jev-latest", jev_max_rps=2.5)
    assert settings.jev_model == "~typesafe/jev-latest"
    assert settings.jev_max_rps == 2.5


def test_non_positive_rps_is_rejected():
    for value in (0, -1, -0.5):
        try:
            Settings.build(jev_max_rps=value)
        except Exception:
            continue
        raise AssertionError(f"jev_max_rps={value} should be rejected")


def test_out_of_range_concurrency_is_rejected():
    for value in (0, 65):
        try:
            Settings.build(jev_max_concurrency=value)
        except Exception:
            continue
        raise AssertionError(f"jev_max_concurrency={value} should be rejected")


def test_unknown_log_level_is_rejected():
    try:
        Settings.build(log_level="chatty")
    except Exception:
        return
    raise AssertionError("log_level=chatty should be rejected")


def test_log_level_is_normalized():
    assert Settings.build(log_level="debug").log_level == "DEBUG"


def test_https_is_required_except_for_loopback():
    assert is_safe_base_url("https://openrouter.ai/api/alpha/decisions") is True
    assert is_safe_base_url("http://127.0.0.1:8078/x") is True
    assert is_safe_base_url("http://localhost:8078/x") is True
    # Plain http elsewhere would put the API key on the wire in the clear.
    assert is_safe_base_url("http://openrouter.ai/x") is False
    assert is_safe_base_url("http://example.com/x") is False
    assert is_safe_base_url("ftp://example.com/x") is False


def test_insecure_base_url_is_rejected():
    try:
        Settings.build(jev_base_url="http://evil.example.com/steal")
    except Exception:
        return
    raise AssertionError("a plain-http remote base URL should be rejected")


def test_loopback_base_url_is_allowed_for_the_mock():
    settings = Settings.build(jev_base_url="http://127.0.0.1:8078/api/alpha/decisions")
    assert settings.jev_base_url.endswith("/api/alpha/decisions")


def test_public_dict_never_exposes_the_key():
    settings = Settings.build(openrouter_api_key="sk-or-supersecret")
    public = settings.public_dict()
    assert "supersecret" not in repr(public)
    assert "api_key" not in public
    assert public["model"] == DEFAULT_MODEL


def test_repr_hides_the_key():
    assert "sk-or-supersecret" not in repr(
        Settings.build(openrouter_api_key="sk-or-supersecret")
    )


def test_model_copy_does_not_mutate_the_original():
    original = Settings.build(jev_max_rps=10.0)
    changed = original.model_copy(update={"jev_max_rps": 99.0})
    assert changed.jev_max_rps == 99.0
    assert original.jev_max_rps == 10.0, "model_copy must not share state"


def test_alternate_key_name_in_the_environment():
    with mock.patch.dict(os.environ, {"OPEN_ROUTER_KEY": "sk-or-alt"}, clear=False):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            assert Settings().openrouter_api_key == "sk-or-alt"


def test_real_environment_beats_the_file():
    with mock.patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "sk-or-from-env"}, clear=False
    ):
        assert Settings().openrouter_api_key == "sk-or-from-env"


def test_explicit_override_beats_the_environment():
    with mock.patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "sk-or-from-env"}, clear=False
    ):
        assert Settings(openrouter_api_key="sk-or-explicit").openrouter_api_key == (
            "sk-or-explicit"
        )


def test_missing_key_is_reported_clearly():
    import app.config as config

    with _without_key_env():
        with mock.patch.object(config, "_dotenv_values", lambda: {}):
            try:
                Settings()
            except Exception as exc:
                message = str(exc)
                assert "OPENROUTER_API_KEY" in message
                assert "OPEN_ROUTER_KEY" in message
                return
    raise AssertionError("a missing key should be rejected at construction")


def test_get_settings_wraps_the_error_as_a_config_error():
    import app.config as config

    get_settings.cache_clear()
    try:
        with _without_key_env():
            with mock.patch.object(config, "_dotenv_values", lambda: {}):
                get_settings()
    except ConfigError as exc:
        assert exc.http_status == 503
        return
    finally:
        get_settings.cache_clear()
    raise AssertionError("get_settings should raise ConfigError without a key")


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        # A key is supplied rather than read from the environment: the point is
        # the cache, and a machine with no .env would otherwise fail here.
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
