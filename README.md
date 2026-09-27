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
terminal and point the app at it in another:

```bash
resume-ranker-mock
JEV_BASE_URL=http://127.0.0.1:8078/api/alpha/decisions resume-ranker-api
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

Every field is read from the environment or a `.env` file, and any of them can
be passed explicitly. The key is read under either `OPENROUTER_API_KEY` or
`OPEN_ROUTER_KEY`; an explicit argument wins over the environment, which wins
over `.env`.

| Variable              | Default                                       | Notes                              |
| --------------------- | --------------------------------------------- | ---------------------------------- |
| `OPENROUTER_API_KEY`  | required                                      | Never logged or returned by the API |
| `JEV_MODEL`           | `typesafe/jev-1.13`                           |                                    |
| `JEV_BASE_URL`        | `https://openrouter.ai/api/alpha/decisions`  | https, except on loopback          |
| `JEV_MAX_CONCURRENCY` | `10`                                          | 1-64                               |
| `JEV_MAX_RPS`         | `20.0`                                        | Client-side rate limit             |
| `JEV_MAX_RETRIES`     | `4`                                           | 1-10                               |
| `JEV_BACKOFF_BASE`    | `1.5`                                         |                                    |
| `JEV_BACKOFF_MAX`     | `30.0`                                        |                                    |
| `JEV_TIMEOUT`         | `60.0`                                        | Seconds                            |
| `JEV_MAX_BATCH_SIZE`  | `100`                                         |                                    |
| `HOST`, `PORT`        | `0.0.0.0`, `8000`                             | API bind address                   |
| `LOG_LEVEL`           | `INFO`                                        |                                    |

`JEV_BASE_URL` is rejected unless it is https or a loopback address: a plain
http endpoint anywhere else would put the key on the wire in the clear.

Rate limits and backoff are applied client-side, so a large pool does not
exhaust the account's in-flight budget.

## Limits

Enforced by the models, so they apply to the API, the CLI, and library use
alike: 50 criteria, 100 candidates per batch, 40k characters of resume text.
A request that cannot fit Jev's context window is rejected before any call is
made, rather than after paying for a partial run.

## Tests

```bash
python tests/run.py
```

262 tests, no pytest required. They are plain `test_*` functions with bare
asserts, so they also run unchanged under pytest if it is ever installed.

The mock is keyword-overlap, not a model. It exists to make demos and tests
reproducible and roughly plausible; a resume with none of a requirement's words
never passes a gate or earns a rubric level, whatever the noise term does.

## Layout

```
app/
  exceptions.py   one hierarchy; each class carries its own HTTP status
  config.py       settings, key resolution, validation
  schemas.py      every Pydantic model, used by all layers
  jev_client.py   the Decisions API client, with retries and rate limiting
  service.py      criteria -> questions -> scores, and the ranker
  api.py          FastAPI app factory, routes, error mapping
  cli.py          interactive tester
  testing.py      the mock, shared by the tests and the CLI
tests/            one module per app module, plus a small runner
```

`app/testing.py` is deliberately part of the package: the CLI demos and the test
suite share one mock, so a green test run also means the demo works.

## Not included

Observability and LangChain-based chains were considered and left out for now.
The seams are obvious if they are wanted later - `JevClient._call` for tracing
requests and `ResumeRanker.rank_batch` for tracing a batch.
