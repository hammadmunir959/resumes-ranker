"""The error hierarchy and the metadata the API serves from it."""

from app.exceptions import (
    ConfigError,
    InputValidationError,
    JevAuthError,
    JevCreditsError,
    JevError,
    JevProtocolError,
    JevTransientError,
    RankerError,
)


def test_everything_derives_from_one_base():
    for cls in (
        InputValidationError, ConfigError, JevError,
        JevAuthError, JevCreditsError, JevProtocolError, JevTransientError,
    ):
        assert issubclass(cls, RankerError), f"{cls.__name__} should be a RankerError"


def test_jev_errors_derive_from_jev_error():
    for cls in (JevAuthError, JevCreditsError, JevProtocolError, JevTransientError):
        assert issubclass(cls, JevError), f"{cls.__name__} should be a JevError"


def test_only_transient_errors_are_retriable():
    assert JevTransientError("x").retriable is True
    for cls in (JevAuthError, JevCreditsError, JevProtocolError):
        assert cls("x").retriable is False, f"{cls.__name__} must not be retried"


def test_http_statuses_match_intent():
    # A bad request is the caller's to fix, a missing key is ours.
    assert InputValidationError("x").http_status == 400
    assert ConfigError("x").http_status == 503
    # Out of credit is unambiguous, so it is not dressed up as a server fault.
    assert JevCreditsError("x").http_status == 402
    # Everything else upstream is either our problem to retry or a bad gateway.
    assert JevAuthError("x").http_status == 502
    assert JevProtocolError("x").http_status == 502
    assert JevTransientError("x").http_status == 503


def test_error_codes_are_unique():
    codes = [
        cls.error_code
        for cls in (
            RankerError, InputValidationError, ConfigError, JevError,
            JevAuthError, JevCreditsError, JevProtocolError, JevTransientError,
        )
    ]
    assert len(codes) == len(set(codes)), f"duplicate error codes: {codes}"


def test_payload_shape_is_stable():
    payload = JevCreditsError(
        "Insufficient credits.", status_code=402, detail={"account": "acme"}
    ).to_payload()
    assert payload == {
        "error": "jev_out_of_credit",
        "detail": "Insufficient credits.",
        "upstream_status": 402,
        "context": {"account": "acme"},
    }


def test_payload_omits_empty_context():
    payload = JevAuthError("bad key").to_payload()
    assert "context" not in payload
    assert "upstream_status" not in payload
    assert payload["error"] == "jev_auth_failed"


def test_message_is_readable():
    assert str(JevProtocolError("expected an object body")) == "expected an object body"


def test_retry_after_is_preserved():
    error = JevTransientError("slow down", retry_after=12.5)
    assert error.retry_after == 12.5
    assert error.detail == {}
