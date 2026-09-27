"""Test doubles for the Decisions API, shared by the tests and the CLI.

Having one implementation matters: the CLI used to carry its own copy of the
mock transport, and the two copies could answer differently, so a green test
run would not guarantee the demo worked.

Three tools here:

* :class:`MockTransport` - an httpx transport that answers in-process, so the
  CLI can demo with no server and no credit.
* :class:`ScriptedTransport` - replays a fixed sequence of responses, for
  exercising retries and error classification.
* :func:`serve` - the same logic as a real local HTTP server, for testing the
  API over a socket.

The mock is not a model. It scores by keyword overlap between the resume and
each question, which makes results stable and roughly meaningful, so a demo
produces a plausible ranking rather than noise.
"""

from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Sequence

import httpx

MOCK_HOST = "127.0.0.1"
MOCK_PORT = 8078
MOCK_PATH = "/api/alpha/decisions"
MOCK_BASE_URL = f"http://{MOCK_HOST}:{MOCK_PORT}{MOCK_PATH}"

#: Too common to be evidence of anything.
STOPWORDS = {
    "and", "the", "with", "for", "has", "have", "this", "that", "from",
    "years", "year", "not", "are", "was", "were", "you", "your", "our", "who",
}


def words(text: str) -> set[str]:
    """Meaningful lowercase words, for keyword overlap."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS}


def stable_jitter(seed: str) -> float:
    """A repeatable 0.0-1.0 value for a seed.

    Deterministic, so the same resume always scores the same and a demo or a test
    run is reproducible.
    """
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def coverage(resume_words: set[str], signal: str) -> float:
    """How much of a requirement the resume covers, from 0.0 to 1.0.

    Measured against the requirement itself, not the whole question. The
    instructions around a question are the same for every candidate and never
    appear in a resume, so including them would divide every score by a constant
    and flatten the whole pool.

    Three distinct words counts as full coverage: a requirement rarely shares
    more than that with a resume in the words both happen to use, while
    demanding all of them leaves every candidate near zero.
    """
    signal_words = words(signal)
    if not signal_words:
        return 0.0
    return min(1.0, len(signal_words & resume_words) / min(len(signal_words), 3))


def mock_answers(
    state: Any,
    questions: dict[str, Any],
) -> dict[str, Any]:
    """Answer every question by keyword overlap against the resume.

    Gates get a probability of yes; scores get a position on their rubric.

    The signal is the requirement named in the state's ``requirements`` entry for
    that question, falling back to the question's own instructions when the state
    carries no requirements. That second part matters: a score question's
    ``criteria`` is the rubric ("Not met", "Partially met", "Fully met"), which
    appears nowhere in a resume, so measuring against it alone would leave every
    scored criterion on pure noise.
    """
    requirements: dict[str, str] = {}
    if isinstance(state, dict):
        resume = str(state.get("resume", ""))
        candidate_id = str(state.get("candidate_id", ""))
        for entry in state.get("requirements") or []:
            if isinstance(entry, dict) and entry.get("id"):
                requirements[str(entry["id"])] = " ".join(
                    str(entry.get(key, "")) for key in ("requirement", "detail")
                )
    else:
        resume, candidate_id = str(state), ""

    resume_words = words(resume)

    answers: dict[str, Any] = {}
    for qid, question in questions.items():
        criteria = question.get("criteria")
        kind = question.get("type")
        instructions = str(question.get("instructions", ""))

        requirement = requirements.get(qid, "").strip()
        if not requirement:
            requirement = instructions

        density = coverage(resume_words, requirement)
        noise = stable_jitter(f"{candidate_id}|{qid}|{len(resume_words)}")

        if kind == "noul":
            # Gates need real evidence. The noise term is kept small on purpose:
            # with wide noise a resume sharing no words with the requirement
            # could still land above the threshold, and a mock that passes
            # everything teaches the wrong thing in a demo.
            answers[qid] = {
                "type": "noul",
                "noul": round(min(0.99, max(0.01, 0.08 + 0.72 * density + 0.20 * noise)), 4),
            }
        elif kind == "score":
            levels = [str(x) for x in criteria] if isinstance(criteria, list) else [
                "Not met", "Partially met", "Fully met",
            ]
            top = max(1, len(levels) - 1)
            # Truncation, not rounding, and noise scaled below one level. Together
            # these mean noise can only break a tie the evidence has already
            # earned: it can never promote a resume to a level its keyword
            # overlap does not support. Rounding here used to hand a resume with
            # no evidence at all a free "Partially met", which made every
            # candidate look mid-strength.
            index = min(top, int(density * top + 0.4 * noise))
            answers[qid] = {
                "type": "score",
                "score": float(index),
                "confidence": round(min(0.99, 0.35 + 0.6 * abs(noise - 0.5) + 0.2 * density), 4),
                "probabilities": {str(i): float(i == index) for i in range(len(levels))},
                "legend": {str(i): label for i, label in enumerate(levels)},
            }
        elif kind == "choice":
            options = sorted(criteria) if isinstance(criteria, dict) else ["yes", "no"]
            chosen = options[min(len(options) - 1, int(noise * len(options)))]
            answers[qid] = {
                "type": "choice",
                "choice": chosen,
                "confidence": round(0.5 + 0.45 * abs(noise - 0.5), 4),
                "probabilities": {opt: float(opt == chosen) for opt in options},
            }
    return answers


def mock_body(payload: dict[str, Any]) -> dict[str, Any]:
    """A complete, well-formed Decisions response for one request payload."""
    return {
        "id": "mock-dec",
        "model": payload.get("model", "typesafe/jev-1.13"),
        "provider": "mock",
        "answers": mock_answers(
            payload.get("state"), payload.get("questions") or {}
        ),
        "usage": {
            "input_tokens": len(json.dumps(payload, default=str)) // 4,
            "output_tokens": 30,
            "cost": 0.00002,
        },
    }


# --------------------------------------------------------------------------- #
# In-process transports
# --------------------------------------------------------------------------- #

class MockTransport(httpx.AsyncBaseTransport):
    """Answers Decisions requests in-process, with no server and no credit."""

    def __init__(self, responder: Optional[Callable[[dict[str, Any]], Any]] = None):
        #: Swap in to control the response, e.g. to inject a failure.
        self.responder = responder or mock_body
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        self.requests.append(payload)
        result = self.responder(payload)
        if isinstance(result, httpx.Response):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            status, body = result
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=result)


class ScriptedTransport(httpx.AsyncBaseTransport):
    """Replays a fixed sequence, for retry and classification tests.

    The last item repeats once the script runs out, so a test can assert on the
    number of attempts without counting the script length.
    """

    def __init__(self, responses: Sequence[Any]):
        if not responses:
            raise ValueError("need at least one scripted response")
        self.responses = list(responses)
        self.calls = 0
        #: Payloads in call order, so a test can assert what each attempt sent.
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        try:
            self.requests.append(json.loads(request.content or b"{}"))
        except ValueError:
            self.requests.append({})
        result = self.responses[index]
        if isinstance(result, httpx.Response):
            return result
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, tuple) and len(result) == 2:
            status, body = result
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=result)


# --------------------------------------------------------------------------- #
# Local HTTP server
# --------------------------------------------------------------------------- #

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(mock_body(payload)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Quiet: the mock server would otherwise log every request."""


class _Server(ThreadingHTTPServer):
    """Threaded, so concurrent candidates do not queue behind each other.

    Ranking a pool sends one request per candidate at the same time. A plain
    ``HTTPServer`` handles a single connection until it closes, and with
    HTTP/1.1 keep-alive that means the first connection is held open while the
    others wait: the second candidate blocks until the client times out, so the
    pool appears to hang rather than fail.
    """

    daemon_threads = True


def serve(host: str = MOCK_HOST, port: int = MOCK_PORT) -> None:
    """Run the mock Decisions API until interrupted.

    Point the app at it to exercise the full HTTP path without credit::

        JEV_BASE_URL=http://127.0.0.1:8078/api/alpha/decisions resume-ranker-api
    """
    server = _Server((host, port), _Handler)
    bound_host, bound_port = server.server_address[:2]
    print(f"mock Decisions API on http://{bound_host}:{bound_port}{MOCK_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()


__all__ = [
    "MOCK_BASE_URL",
    "MOCK_HOST",
    "MOCK_PATH",
    "MOCK_PORT",
    "MockTransport",
    "ScriptedTransport",
    "mock_answers",
    "mock_body",
    "serve",
    "stable_jitter",
    "words",
]
