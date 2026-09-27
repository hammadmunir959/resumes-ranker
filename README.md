# Resumes Ranker

Rank a pool of resumes against a job's criteria using
[TypeSafe Jev](https://typesafe.ai) via OpenRouter.

Each criterion becomes one Jev question. Required criteria act as hard gates,
optional ones as weighted scores. Every candidate is a single request, so cost
scales with the number of candidates, not candidates x criteria.

Works offline: an in-process mock ships with the project, so you can try the
whole thing without an API key or credit.

## Install

```bash
pip install -e .
```

Optional, for prettier CLI tables:

```bash
pip install -e ".[dev]"
```

## Try it without an API key

```bash
resume-ranker --mock
```

This loads a sample job and three candidates, then ranks them in-process. Type
`help` for the command list, `run` to rank, `show <id>` for a per-criterion
breakdown.

Scripted runs work too:

```bash
resume-ranker --mock -c run -c show cand_3 -c cost
```

## Run it against a real server

The API serves the same ranking over HTTP:

```bash
resume-ranker-api                  # http://0.0.0.0:8000
```

To exercise the real HTTP path with no credit, run the mock Decisions API in one
terminal and point the app at it in another. Only the key comes from the
environment, so the base URL is passed in code:

```bash
resume-ranker-mock
```

```python
# other terminal
import uvicorn
from app.config import Settings
from app.main import create_app
from app.testing import MOCK_BASE_URL

uvicorn.run(create_app(Settings.build(jev_base_url=MOCK_BASE_URL)))
```

Against live OpenRouter, set a key:

```bash
export OPENROUTER_API_KEY=sk-or-...
resume-ranker-api
```

## Endpoints

| Method | Path            | Purpose                                  |
| ------ | --------------- | ---------------------------------------- |
| GET    | `/health`       | Liveness plus the configuration loaded   |
| POST   | `/rank/single`  | Rank one candidate                       |
| POST   | `/rank/batch`   | Rank a pool concurrently                 |

Both rank endpoints return the same `RankingResult`, so a client only ever
parses one response shape. `single` is a batch of one.

```bash
curl -s localhost:8000/rank/batch -H 'content-type: application/json' -d '{
  "job_description": "Backend engineer, Python",
  "criteria": [
    {"id": "auth", "name": "Authorized to work", "required": true},
    {"id": "py", "name": "Python depth", "weight": 3.0}
  ],
  "candidates": [
    {"candidate_id": "c1", "resume_text": "Six years of Python with Django."}
  ]
}'
```

`GET /health` never contacts OpenRouter. A health check that fails when the
provider is down turns their outage into yours, and it reports less about the
local process than the loaded configuration does.

Errors are uniform, with the same shape for every failure:

```json
{"error": "jev_out_of_credit", "detail": "Insufficient credits.", "upstream_status": 402}
```

| Status | `error`                     | Meaning                                |
| ------ | --------------------------- | -------------------------------------- |
| 400    | `invalid_request`           | Rejected before any call was made       |
| 402    | `jev_out_of_credit`         | Out of credit                           |
| 502    | `jev_auth_failed`           | The key was rejected                    |
| 502    | `jev_bad_response`          | The provider replied with nonsense      |
| 503    | `jev_temporarily_unavailable` | Rate limited or upstream down, retried |
| 503    | `not_configured`            | No usable key or settings               |

The app still starts when it is misconfigured: it serves `/health` and returns
503 for ranking, rather than crash-looping on import.

## How ranking works

1. A job is a list of `Criterion`, each with an id, a name, a `weight`, and
   whether it is `required`.
2. Each criterion becomes one Jev question, built once per batch:
   - `required: true` becomes a `noul` question. The answer is a probability of
     "yes" and is compared against `GATE_THRESHOLD` (0.5). A candidate within
     `GATE_MARGIN` (0.15) of the threshold is flagged for review.
   - `required: false` becomes a `score` question on an ordered `rubric`. The
     answer is normalized to 0-100 and combined into a weighted average.
3. One request per candidate carries all of its questions.
4. Results are sorted: passing candidates first by score, then gated ones, then
   errors.

A transient failure marks only that one candidate as errored and the rest of the
pool still ranks. A systemic failure - bad key, no credit, a rejected request
shape - aborts the batch, because retrying it per candidate would only produce
the same answer N times.

### Interpreting the results

- `overall_score` is `None` when nothing was weighed, as for a job defined purely
  by hard requirements. `None` is not the same as `0.0`: `0.0` means "scored and
  earned nothing".
- `needs_review` means a gate was borderline or a confidence was below
  `REVIEW_CONFIDENCE_FLOOR` (0.60). It is a request for a human, not a rejection.
- `total_cost_usd` and `total_attempts` describe the whole batch, so a pool that
  partially failed still tells you what it cost.

## Configuration

The environment is used for exactly one thing: the API key, read under either
`OPENROUTER_API_KEY` or `OPEN_ROUTER_KEY`, with an explicit argument winning over
the environment, which wins over `.env`.

Everything else is a default in `app/config.py`, or an argument you pass:

```python
ResumeRanker.from_settings(Settings.build(jev_model="other/model"))
```

That is a deliberate trade. In exchange for a little editing when a deployment
needs different numbers, there is no ambient configuration that can change
behaviour without appearing in the code - which is also what lets the test suite
be immune to whatever the shell happens to export.

| `Settings` field        | Default                                       | Notes                              |
| ----------------------- | --------------------------------------------- | ---------------------------------- |
| `openrouter_api_key`    | required                                      | Never logged or returned by the API |
| `jev_model`             | `typesafe/jev-1.13`                           |                                    |
| `jev_fallback_model`    | `respan/span-01-lite`                         | Empty disables the fallback        |
| `jev_base_url`          | `https://openrouter.ai/api/alpha/decisions`   | https, except on loopback          |
| `jev_max_concurrency`   | `10`                                          | 1-64                               |
| `jev_max_rps`           | `20.0`                                        | Client-side rate limit             |
| `jev_max_retries`       | `4`                                           | 1-10                               |
| `jev_backoff_base`      | `1.5`                                         |                                    |
| `jev_backoff_max`       | `30.0`                                        |                                    |
| `jev_timeout`           | `60.0`                                        | Seconds                            |
| `jev_max_batch_size`    | `100`                                         |                                    |
| `host`, `port`          | `0.0.0.0`, `8000`                             | API bind address                   |
| `log_level`             | `INFO`                                        |                                    |

`jev_base_url` is rejected unless it is https or a loopback address: a plain
http endpoint anywhere else would put the key on the wire in the clear.

Rate limits and backoff are applied client-side, so a large pool does not
exhaust the account's in-flight budget.

### When the primary model is the problem

If a call fails for a reason that belongs to the model rather than the account -
unknown model, rejected request, or a provider error that survived the retries -
the client makes one more pass with `jev_fallback_model`. A bad key or an empty
balance does not fall back, because the fallback would be turned away for exactly
the same reason and the extra request would only delay the real message.

Results report the model that actually answered, per candidate and for the batch,
so a fallback is visible in the output rather than silent. Passing `model=` to
`JevClient.decide` means you want that one model, so it is never substituted.

## Limits

Enforced by the models, so they apply to the API, the CLI, and library use
alike: 50 criteria, 100 candidates per batch, 40k characters of resume text.
A request that cannot fit Jev's context window is rejected before any call is
made, rather than after paying for a partial run.

## Tests

```bash
python tests/run.py
```

273 tests, no pytest required. They are plain `test_*` functions with bare
asserts, so they also run unchanged under pytest if it is ever installed.

The mock is keyword-overlap, not a model. It exists to make demos and tests
reproducible and roughly plausible; a resume with none of a requirement's words
never passes a gate or earns a rubric level, whatever the noise term does.

## Layout

```
app/
  main.py         the process: ASGI app, lifespan, error handler, health
  config.py       defaults, key resolution from the environment, validation
  exceptions.py   one hierarchy; each class carries its own HTTP status
  cli.py          interactive tester
  testing.py      the mock, shared by the tests and the CLI
  schemas/        shapes, split by whose API they describe
    jev_schemas.py       the Decisions wire format
    ranking_schemas.py   criteria, candidates, scores, request envelopes
    health_schemas.py    the /health payload
  services/       the stateful objects
    jev_service.py       the Decisions client: retries, rate limit, fallback
    ranking_service.py   ResumeRanker
  utils/          pure functions, no state and no I/O
    jev_utils.py         fallback choice, Retry-After, non-JSON bodies
    ranking_utils.py     criteria -> questions -> scores, and the thresholds
  api/
    ranking_router.py    the two ranking routes
tests/            one module per app module, plus a small runner
```

The rule the layout follows: each package holds one kind of thing, and a question
has one place to be answered. `schemas` is shapes, `services` is stateful
objects, `utils` is pure functions, `api` is routes, and `main` is the process
that wires them together. The `__init__.py` of each subpackage re-exports its
public surface, so callers import `from app.services import ResumeRanker` rather
than reaching into a module.

`app/testing.py` is deliberately part of the package: the CLI demos and the test
suite share one mock, so a green test run also means the demo works.

## Not included

Observability and LangChain-based chains were considered and left out for now.
The seams are obvious if they are wanted later - `JevClient._call` for tracing
requests and `ResumeRanker.rank_batch` for tracing a batch.
