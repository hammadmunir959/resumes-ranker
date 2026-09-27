"""CLI behavior: the command loop, the session, and the reports it prints.

The tests drive ``dispatch`` directly with a mock session, so no input is read
and no network is touched. Output is captured and the rich renderer is turned
off, so assertions match on plain text instead of terminal box drawing.
"""

import asyncio
import builtins
import io
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from app import cli
from app.schemas import Candidate, CandidateScore, Criterion

RESUME = "Authorized to work. Six years of python with django and aws."


@contextmanager
def stubbed_input(lines):
    """Feed the prompt functions a fixed script, one line per call."""
    scripted = iter(lines)
    original = builtins.input

    def fake_input(_prompt=""):
        try:
            return next(scripted)
        except StopIteration:
            raise EOFError from None

    builtins.input = fake_input
    try:
        yield
    finally:
        builtins.input = original


def run(session, line, answers=(), use_input=False):
    """Dispatch one command, returning whether to continue and what it printed.

    ``answers`` covers the single-line prompts. Pass ``use_input`` when the test
    exercises the multi-line ``ask_block`` loop, which reads ``input`` itself.
    """
    output = io.StringIO()
    prompts = iter(answers)
    saved = cli.ask, cli.ask_block, cli._RICH

    def fake_ask(prompt, default=""):
        try:
            return next(prompts)
        except StopIteration:
            return default

    def fake_block(prompt):
        try:
            return next(prompts)
        except StopIteration:
            return ""

    cli.ask = fake_ask
    if not use_input:
        cli.ask_block = fake_block
    cli._RICH = False
    try:
        with redirect_stdout(output):
            keep_going = cli.dispatch(session, line)
    finally:
        cli.ask, cli.ask_block, cli._RICH = saved
    return keep_going, output.getvalue()


def session(mock=True, **kwargs):
    return cli.Session(mock=mock, **kwargs)


# --- Dispatch --------------------------------------------------------------- #

def test_quit_ends_the_session():
    for line in ("quit", "exit", "q", "QUIT"):
        assert run(session(), line)[0] is False, line


def test_a_blank_line_is_ignored():
    assert run(session(), "")[0] is True
    assert run(session(), "    ")[0] is True


def test_an_unknown_command_does_not_stop_the_session():
    keep_going, output = run(session(), "flibbertigibbet")
    assert keep_going is True
    assert "unknown command" in output


def test_help_lists_every_command():
    _, output = run(session(), "help")
    for name, _ in cli.HELP:
        assert name.split()[0] in output


def test_status_reports_the_mode_and_pool():
    s = session()
    run(s, "demo")
    _, output = run(s, "status")
    assert "MOCK" in output
    assert "criteria    3" in output
    assert "candidates  3" in output


# --- Criteria --------------------------------------------------------------- #

def test_demo_loads_a_runnable_pool():
    s = session()
    run(s, "demo")
    assert [c.id for c in s.criteria] == ["auth", "py", "cloud"]
    assert [c.candidate_id for c in s.candidates] == ["cand_1", "cand_2", "cand_3"]


def test_adding_a_criterion_prompts_for_the_rest():
    s = session()
    run(s, "add py required", answers=["Python depth", "Senior python"])
    assert len(s.criteria) == 1
    added = s.criteria[0]
    assert added.id == "py"
    assert added.required is True
    assert added.name == "Python depth"
    assert added.description == "Senior python"


def test_a_scored_criterion_collects_a_weight_and_rubric():
    s = session()
    run(
        s, "add py scored",
        answers=["Python depth", "senior python", "3", "low, mid, high"],
    )
    added = s.criteria[0]
    assert added.required is False
    assert added.weight == 3.0
    assert added.rubric == ("low", "mid", "high")


def test_a_duplicate_criterion_id_is_refused():
    s = session()
    run(s, "add py", answers=["Python"])
    run(s, "add py", answers=["Python again"])
    assert len(s.criteria) == 1


def test_add_without_an_id_prints_usage():
    _, output = run(session(), "add")
    assert "usage" in output
    assert run(session(), "add")[1].count("usage") == 1


def test_add_rejects_an_unknown_kind():
    s = session()
    _, output = run(s, "add py maybe")
    assert "required or scored" in output
    assert s.criteria == []


def test_a_criterion_can_be_removed():
    s = session()
    run(s, "demo")
    run(s, "rmcrit cloud")
    assert "cloud" not in [c.id for c in s.criteria]
    _, output = run(s, "rmcrit nope")
    assert "no such criterion" in output


def test_criteria_lists_the_pool():
    s = session()
    run(s, "demo")
    _, output = run(s, "criteria")
    assert "auth" in output and "gate" in output
    assert "cloud" in output


# --- Candidates ------------------------------------------------------------- #

def test_adding_a_candidate_collects_resume_and_form_text():
    s = session()
    run(s, "addcand a", answers=[RESUME, "Available immediately"])
    added = s.candidates[0]
    assert added.candidate_id == "a"
    assert added.resume_text == RESUME
    assert added.application_form_text == "Available immediately"


def test_a_multi_line_resume_is_read_until_a_blank_line():
    # Driven through ``input`` rather than a stub, since ``ask_block`` reads one
    # line at a time and that loop is the thing under test.
    s = session()
    lines = ["line one", "line two", "", "form text", ""]
    with stubbed_input(lines):
        keep_going, _ = run(s, "addcand a", use_input=True)
    assert keep_going is True
    assert s.candidates[0].resume_text == "line one\nline two"
    assert s.candidates[0].application_form_text == "form text"


def test_an_empty_resume_cancels_the_add():
    s = session()
    _, output = run(s, "addcand a", answers=[""])
    assert "cancelled" in output
    assert s.candidates == []


def test_addcand_without_an_id_prints_usage():
    _, output = run(session(), "addcand")
    assert "usage" in output


def test_a_duplicate_candidate_id_is_refused():
    s = session()
    run(s, "addcand a", answers=[RESUME, ""])
    run(s, "addcand a", answers=[RESUME, ""])
    assert len(s.candidates) == 1


def test_a_candidate_can_be_removed():
    s = session()
    run(s, "demo")
    run(s, "rmcand cand_2")
    assert "cand_2" not in [c.candidate_id for c in s.candidates]
    assert "no such candidate" in run(s, "rmcand nope")[1]


def test_candidates_lists_the_pool():
    s = session()
    run(s, "demo")
    _, output = run(s, "candidates")
    assert "cand_1" in output


# --- Job and model ---------------------------------------------------------- #

def test_the_job_can_be_set_and_shown():
    s = session()
    run(s, "job Data engineer with dbt")
    assert s.job == "Data engineer with dbt"
    assert "Data engineer with dbt" in run(s, "job")[1]


def test_the_model_override_is_applied_to_settings():
    s = session(model="custom/model")
    assert s.settings().jev_model == "custom/model"
    run(s, "model another/model")
    assert s.model == "another/model"
    assert "another/model" in run(s, "model")[1]


def test_the_concurrency_override_is_applied_to_settings():
    assert session(concurrency=3).settings().jev_max_concurrency == 3


def test_mock_mode_toggles():
    s = session(mock=False)
    run(s, "mock on")
    assert s.mock is True
    run(s, "mock off")
    assert s.mock is False
    run(s, "mock")
    assert s.mock is True, "a bare 'mock' should toggle"


# --- Save and load ---------------------------------------------------------- #

def test_a_pool_survives_a_save_and_load():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pool.json"
        original = session()
        run(original, "demo")
        run(original, f"save {path}")
        reloaded = session()
        run(reloaded, f"load {path}")
        assert [c.id for c in reloaded.criteria] == [c.id for c in original.criteria]
        assert [c.candidate_id for c in reloaded.candidates] == [
            c.candidate_id for c in original.candidates
        ]
        assert reloaded.job == original.job


def test_the_saved_file_is_plain_json():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "pool.json"
        s = session()
        run(s, "demo")
        run(s, f"save {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"job", "model", "criteria", "candidates"}
        assert isinstance(data["criteria"], list)
        assert isinstance(data["criteria"][0]["rubric"], list), (
            "tuples must serialize as lists so the file reloads into the same model"
        )


def test_a_missing_file_is_reported_not_raised():
    s = session()
    keep_going, output = run(s, "load /nope/does-not-exist.json")
    assert keep_going is True, "a bad path must not kill the session"
    assert "no such file" in output


def test_invalid_json_is_reported_not_raised():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        keep_going, output = run(session(), f"load {path}")
        assert keep_going is True
        assert "could not read" in output


def test_save_and_load_without_a_path_print_usage():
    assert "usage" in run(session(), "save")[1]
    assert "usage" in run(session(), "load")[1]


# --- Running ---------------------------------------------------------------- #

def test_running_the_demo_ranks_every_candidate():
    s = session()
    run(s, "demo")
    _, output = run(s, "run")
    assert s.report is not None
    assert len(s.report.results) == 3
    assert "cand_1" in output and "cand_3" in output


def test_a_run_makes_one_call_per_candidate():
    s = session()
    run(s, "demo")
    run(s, "run")
    assert s.report.total_attempts == 3, "cost must scale with candidates"


def test_running_without_a_criterion_is_explained():
    s = session()
    run(s, "addcand a", answers=[RESUME, ""])
    keep_going, output = run(s, "run")
    assert keep_going is True
    assert "no criteria" in output


def test_running_without_a_candidate_is_explained():
    s = session()
    run(s, "demo")
    s.candidates = []
    _, output = run(s, "run")
    assert "no candidates" in output


def test_the_report_is_ordered_best_first():
    s = session()
    run(s, "demo")
    run(s, "run")
    scores = [r.overall_score or 0.0 for r in s.report.results]
    assert scores == sorted(scores, reverse=True)
    assert s.report.best().candidate_id == s.report.results[0].candidate_id


def test_json_before_and_after_a_run():
    s = session()
    assert "no run yet" in run(s, "json")[1]
    run(s, "demo")
    run(s, "run")
    _, output = run(s, "json")
    assert json.loads(output)["model"] == s.report.model


def test_cost_before_and_after_a_run():
    s = session()
    assert "no run yet" in run(s, "cost")[1]
    run(s, "demo")
    run(s, "run")
    _, output = run(s, "cost")
    assert "total cost" in output
    assert "errors          0" in output


def test_a_breakdown_is_shown_for_one_candidate():
    s = session()
    run(s, "demo")
    run(s, "run")
    _, output = run(s, "show cand_3")
    assert "auth" in output and "gate" in output
    assert "cloud" in output


def test_a_breakdown_for_an_unknown_id_is_reported():
    s = session()
    run(s, "demo")
    run(s, "run")
    assert "no result for" in run(s, "show nobody")[1]


def test_show_without_an_id_prints_usage():
    assert "usage" in run(session(), "show")[1]


# --- Formatting ------------------------------------------------------------- #

def test_a_gate_only_candidate_is_not_shown_as_zero():
    s = session()
    run(s, "demo")
    s.criteria = [c for c in s.criteria if c.required]
    run(s, "run")
    best = s.report.best()
    assert best.overall_score is None
    assert best.scored is True, "passing a gate-only job is still a real result"
    assert "gates only" in cli.score_text(best)


def test_a_failed_candidate_renders_as_an_error():
    failed = CandidateScore.failed("a", "JevTransientError: busy")
    assert "ERROR" in cli.badge(failed)
    assert cli.score_text(failed) == "-"


def test_a_gated_candidate_is_badged_as_gated():
    gated = CandidateScore(candidate_id="a", gate_passed=False, failed_gates=["auth"])
    assert "GATED" in cli.badge(gated)


def test_a_passing_candidate_is_badged_as_pass():
    assert "PASS" in cli.badge(CandidateScore(candidate_id="a"))


def test_an_empty_table_says_so():
    out = io.StringIO()
    with redirect_stdout(out):
        cli.table("Empty", ["a"], [])
    assert "none" in out.getvalue()


# --- Entry point ------------------------------------------------------------ #

def test_main_runs_scripted_commands_in_mock_mode():
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(["--mock", "-c", "run", "-c", "cost"])
    assert code == 0
    assert "total cost" in out.getvalue()


def test_main_starts_mock_mode_with_the_demo_loaded():
    out = io.StringIO()
    with redirect_stdout(out):
        cli.main(["--mock", "-c", "status"])
    assert "candidates  3" in out.getvalue()


def test_a_live_session_without_a_key_explains_itself():
    from app.exceptions import ConfigError

    s = session(mock=False)
    try:
        s.settings()
    except ConfigError as exc:
        assert "OpenRouter" in str(exc)
    else:
        # A machine that does have a key is a legitimate configuration.
        pass


# --- Prompted input --------------------------------------------------------- #

def test_ask_falls_back_to_the_default_at_eof():
    with stubbed_input([]):
        assert cli.ask("prompt", "fallback") == "fallback"
        assert cli.ask_block("prompt") == ""


def test_ask_yes_no_parses_the_common_answers():
    with stubbed_input(["y", "N", "", "yes"]):
        assert cli.ask_yes_no("go", False) is True
        assert cli.ask_yes_no("go", True) is False
        assert cli.ask_yes_no("go", True) is True, "an empty answer takes the default"
        assert cli.ask_yes_no("go", False) is True


def test_a_pool_built_only_from_prompts_is_valid():
    s = session()
    run(s, "add auth required", answers=["Authorized to work", "work authorization"])
    run(s, "addcand a", answers=[RESUME, ""])
    assert s.criteria[0] == Criterion(
        id="auth", name="Authorized to work", description="work authorization",
        required=True,
    )
    assert s.candidates[0] == Candidate(candidate_id="a", resume_text=RESUME)
    run(s, "run")
    assert s.report is not None and s.report.results[0].candidate_id == "a"


def test_criteria_and_candidates_are_immutable():
    # A criterion that could change mid-run would make a report unreproducible.
    s = session()
    run(s, "demo")
    try:
        s.criteria[0].id = "changed"
    except Exception:
        pass
    else:
        raise AssertionError("criteria must be frozen")


def test_asyncio_run_is_not_reused_across_commands():
    # Each 'run' opens and closes its own loop; a second one must still work.
    s = session()
    run(s, "demo")
    run(s, "run")
    run(s, "run")
    assert s.report is not None
    assert asyncio.iscoroutinefunction(cli.Session.run)
