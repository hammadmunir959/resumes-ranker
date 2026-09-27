"""Interactive tester for the ranker.

Lets you build a pool of criteria and candidates, then rank it against the real
Decisions API or a local mock, and inspect the result, all from a prompt::

    resume-ranker              # live
    resume-ranker --mock       # offline, no credit needed

The mock runs in-process through :mod:`app.testing`, so ``--mock`` needs neither
a server nor an API key and costs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from app.config import Settings
from app.exceptions import RankerError
from app.services import JevClient
from app.schemas import (
    Candidate,
    CandidateScore,
    Criterion,
    RankingResult,
)
from app.services import ResumeRanker
from app.testing import MockTransport

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"

#: The job context a new session starts with.
DEFAULT_JOB = "Backend engineer, Python"

DEMO_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        id="auth",
        name="Authorized to work",
        description="Authorized to work in the hiring country",
        required=True,
    ),
    Criterion(
        id="py",
        name="Python depth",
        description="Senior-level Python, web frameworks, testing",
        weight=3.0,
    ),
    Criterion(
        id="cloud",
        name="Cloud experience",
        description="Deployed and operated services in AWS or GCP",
        weight=1.0,
    ),
)

DEMO_CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        candidate_id="cand_1",
        resume_text=(
            "Frontend developer, 4 years JavaScript and React. Authorized to "
            "work. No backend or cloud experience."
        ),
    ),
    Candidate(
        candidate_id="cand_2",
        resume_text=(
            "1 year of Python, mostly scripting and notebooks. Authorized to "
            "work. No cloud experience."
        ),
    ),
    Candidate(
        candidate_id="cand_3",
        resume_text=(
            "Senior engineer, 6 years Python, migrated a monolith to AWS, "
            "mentored 3 junior engineers. Authorized to work in the US."
        ),
    ),
)

HELP: tuple[tuple[str, str], ...] = (
    ("help", "show this help"),
    ("status", "current pool, mode, and model"),
    ("demo", "load the sample criteria and 3 candidates"),
    ("mock on|off", "toggle offline simulation"),
    ("criteria", "list the criteria"),
    ("add <id> [required|scored]", "add a criterion, prompted for the rest"),
    ("rmcrit <id>", "remove a criterion"),
    ("candidates", "list candidates"),
    ("addcand <id>", "add a candidate, prompted for resume text"),
    ("rmcand <id>", "remove a candidate"),
    ("job [text]", "show or set the role context"),
    ("model [name]", "show or set the Jev model id"),
    ("load <file.json>", "replace the pool from a JSON file"),
    ("save <file.json>", "write the current pool to JSON"),
    ("run", "rank the pool"),
    ("show <id>", "per-criterion breakdown for one candidate"),
    ("json", "the last report as JSON"),
    ("cost", "tokens, cost, and timing for the last run"),
    ("clear", "clear the screen"),
    ("quit", "exit"),
)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def say(message: str = "") -> None:
    print(message)


def table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        say(f"{BOLD}{title}{RESET}\n  {DIM}(none){RESET}")
        return
    if _RICH:
        grid = Table(title=title, header_style="bold", title_justify="left")
        for column in columns:
            grid.add_column(column, overflow="fold")
        for row in rows:
            grid.add_row(*row)
        Console().print(grid)
        return
    widths = [
        max(len(str(row[i])) for row in [list(columns), *rows])
        for i in range(len(columns))
    ]
    say(f"{BOLD}{title}{RESET}")
    say("  " + "  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    say("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        say("  " + "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def badge(score: CandidateScore) -> str:
    if score.error:
        return f"{RED}ERROR{RESET}"
    if not score.gate_passed:
        return f"{RED}GATED{RESET}"
    if score.needs_review:
        return f"{YELLOW}PASS?{RESET}"
    return f"{GREEN}PASS{RESET}"


def score_text(score: CandidateScore) -> str:
    if score.error:
        return "-"
    if score.overall_score is None:
        return f"{DIM}gates only{RESET}"
    return f"{score.overall_score:.1f}"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

def ask(prompt: str, default: str = "") -> str:
    try:
        answer = input(f"{CYAN}{prompt}{RESET} ").strip()
    except EOFError:
        return default
    return answer or default


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = ask(f"{prompt} [{suffix}]").lower()
    if not answer:
        return default
    return answer.startswith("y")


def ask_block(prompt: str) -> str:
    """Read several lines until a blank one, for resume text."""
    say(f"{CYAN}{prompt}{RESET} {DIM}(blank line to finish){RESET}")
    lines: list[str] = []
    while True:
        try:
            line = input("  ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

class Session:
    """Everything one CLI session holds, and the operations on it."""

    def __init__(
        self,
        mock: bool = False,
        model: Optional[str] = None,
        concurrency: Optional[int] = None,
    ):
        self.mock = mock
        self.model = model
        self.concurrency = concurrency
        self.job = DEFAULT_JOB
        self.criteria: list[Criterion] = []
        self.candidates: list[Candidate] = []
        self.report: Optional[RankingResult] = None

    # -- settings ---------------------------------------------------------- #

    def settings(self) -> Settings:
        """The real environment, with the session's overrides applied.

        ``model_copy`` replaces what used to be a hand-written copy of every
        field, which silently dropped any field added to ``Settings`` later.
        """
        base = Settings.build() if self.mock else _real_settings()
        overrides: dict[str, Any] = {}
        if self.model:
            overrides["jev_model"] = self.model
        if self.concurrency:
            overrides["jev_max_concurrency"] = self.concurrency
        return base.model_copy(update=overrides)

    # -- pool -------------------------------------------------------------- #

    def load_demo(self) -> None:
        self.criteria = list(DEMO_CRITERIA)
        self.candidates = list(DEMO_CANDIDATES)
        self.job = DEFAULT_JOB
        say(f"{GREEN}loaded{RESET} {len(self.criteria)} criteria, "
            f"{len(self.candidates)} candidates")

    def load_file(self, path: Path) -> None:
        """Replace the pool from a JSON file written by ``save``."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.criteria = [Criterion(**c) for c in data.get("criteria", [])]
        self.candidates = [Candidate(**c) for c in data.get("candidates", [])]
        self.job = str(data.get("job") or DEFAULT_JOB)
        if data.get("model"):
            self.model = str(data["model"])
        say(f"{GREEN}loaded{RESET} {path}: {len(self.criteria)} criteria, "
            f"{len(self.candidates)} candidates")

    def save_file(self, path: Path) -> None:
        """Write the pool as JSON.

        The models are dumped with ``mode="json"`` so tuples come back as lists
        and the file reloads into the same objects.
        """
        payload = {
            "job": self.job,
            "model": self.model or self.settings().jev_model,
            "criteria": [c.model_dump(mode="json") for c in self.criteria],
            "candidates": [c.model_dump(mode="json") for c in self.candidates],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        say(f"{GREEN}saved{RESET} {path}")

    def find(self, candidate_id: str) -> Optional[CandidateScore]:
        if not self.report:
            return None
        for result in self.report.results:
            if result.candidate_id == candidate_id:
                return result
        return None

    # -- ranking ----------------------------------------------------------- #

    async def run(self) -> None:
        if not self.criteria:
            raise RankerError("no criteria yet - try 'demo' or 'add <id>'")
        if not self.candidates:
            raise RankerError("no candidates yet - try 'demo' or 'addcand <id>'")

        settings = self.settings()
        mode = "MOCK" if self.mock else "LIVE"
        say(
            f"{DIM}{mode}: {len(self.candidates)} candidate(s) x "
            f"{len(self.criteria)} question(s) = {len(self.candidates)} call(s), "
            f"model={settings.jev_model}{RESET}"
        )

        # The client is given a transport rather than being built in mock mode,
        # so the same code path runs either way and only the transport differs.
        jev = (
            JevClient(settings, transport=MockTransport())
            if self.mock
            else JevClient(settings)
        )
        ranker = ResumeRanker(settings, jev)
        try:
            self.report = await ranker.rank_batch(
                self.job, self.criteria, self.candidates
            )
        finally:
            await ranker.aclose()
        self.show_report(self.report)

    # -- reporting --------------------------------------------------------- #

    def show_report(self, report: RankingResult) -> None:
        rows = []
        for rank, result in enumerate(report.results, start=1):
            detail = (
                result.error
                or (f"failed: {', '.join(result.failed_gates)}" if result.failed_gates else "")
                or ("needs review" if result.needs_review else "")
            )
            rows.append([
                str(rank),
                result.candidate_id,
                badge(result),
                score_text(result),
                detail,
            ])
        table("Ranking", ["#", "candidate", "status", "score", "notes"], rows)

        if report.skipped:
            say(f"{YELLOW}skipped{RESET} {', '.join(report.skipped)}")
        say(
            f"{DIM}{report.elapsed_seconds}s, "
            f"${report.total_cost_usd or 0.0:.5f}, "
            f"{report.total_attempts} upstream attempt(s), "
            f"{report.error_count} error(s){RESET}"
        )

    def show_candidate(self, candidate_id: str) -> None:
        result = self.find(candidate_id)
        if result is None:
            say(f"{YELLOW}no result for{RESET} {candidate_id}")
            return
        rows = []
        for outcome in result.per_criterion:
            kind = "gate" if outcome.required else "scored"
            verdict = (
                outcome.error
                or ("pass" if outcome.passed else "fail" if outcome.passed is False else "-")
            )
            rows.append([
                outcome.criterion_id,
                kind,
                outcome.label or "-",
                f"{outcome.score_0_100:.1f}",
                "-" if outcome.confidence is None else f"{outcome.confidence:.2f}",
                verdict,
            ])
        table(
            f"Breakdown: {candidate_id}",
            ["criterion", "kind", "level", "score", "conf", "verdict"],
            rows,
        )
        say(
            f"  overall {score_text(result)}  {badge(result)}"
            + (f"  cost ${result.cost_usd:.5f}" if result.cost_usd else "")
        )

    def show_cost(self) -> None:
        if not self.report:
            say(f"{DIM}no run yet{RESET}")
            return
        say(
            f"model           {self.report.model}\n"
            f"elapsed         {self.report.elapsed_seconds}s\n"
            f"total cost      ${self.report.total_cost_usd or 0.0:.5f}\n"
            f"upstream calls  {self.report.total_attempts}\n"
            f"errors          {self.report.error_count}"
        )


def _real_settings() -> Settings:
    """Live settings, or a clear message about what is missing."""
    from app.config import get_settings

    return get_settings()


# --------------------------------------------------------------------------- #
# Pool editing
# --------------------------------------------------------------------------- #

def add_criterion(session: Session, args: list[str]) -> None:
    if not args:
        say(f"{YELLOW}usage:{RESET} add <id> [required|scored]")
        return
    criterion_id = args[0]
    if any(c.id == criterion_id for c in session.criteria):
        say(f"{YELLOW}already exists:{RESET} {criterion_id}")
        return

    kind = args[1].lower() if len(args) > 1 else ""
    if kind not in {"required", "scored", ""}:
        say(f"{YELLOW}kind must be{RESET} required or scored")
        return
    required = kind == "required" if kind else ask_yes_no("required (a hard gate)?", False)

    name = ask("name", criterion_id)
    description = ask("what does it require?", name)
    weight = 1.0
    rubric: Optional[tuple[str, ...]] = None
    if not required:
        weight = _ask_float("weight", 1.0)
        raw = ask("rubric levels, comma separated", "Not met, Partially met, Fully met")
        rubric = tuple(level.strip() for level in raw.split(",") if level.strip())

    session.criteria.append(
        Criterion(
            id=criterion_id,
            name=name,
            description=description,
            required=required,
            weight=weight,
            **({"rubric": rubric} if rubric else {}),
        )
    )
    say(f"{GREEN}added{RESET} {criterion_id} ({'gate' if required else 'scored'})")


def add_candidate(session: Session, args: list[str]) -> None:
    if not args:
        say(f"{YELLOW}usage:{RESET} addcand <id>")
        return
    candidate_id = args[0]
    if any(c.candidate_id == candidate_id for c in session.candidates):
        say(f"{YELLOW}already exists:{RESET} {candidate_id}")
        return
    resume = ask_block(f"resume text for {candidate_id}")
    if not resume:
        say(f"{YELLOW}cancelled{RESET}")
        return
    form = ask_block("application form text (optional)")
    session.candidates.append(
        Candidate(
            candidate_id=candidate_id,
            resume_text=resume,
            application_form_text=form,
        )
    )
    say(f"{GREEN}added{RESET} {candidate_id}")


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            say(f"{YELLOW}enter a number{RESET}")


def _guard(path: Path, action: Any) -> None:
    """Report a bad path instead of dropping a traceback on the user.

    A mistyped filename is the most likely mistake here, and a REPL that dies on
    it loses the pool the user just built.
    """
    try:
        action(path)
    except FileNotFoundError:
        say(f"{YELLOW}no such file:{RESET} {path}")
    except IsADirectoryError:
        say(f"{YELLOW}that is a directory:{RESET} {path}")
    except PermissionError:
        say(f"{YELLOW}permission denied:{RESET} {path}")
    except (ValueError, OSError) as exc:
        # Invalid JSON lands here, and is the likeliest cause worth naming.
        say(f"{YELLOW}could not read {path}:{RESET} {exc}")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def dispatch(session: Session, line: str) -> bool:
    """Run one command. Returns False when the session should end."""
    parts = shlex.split(line)
    if not parts:
        return True
    command, args = parts[0].lower(), parts[1:]

    if command in {"quit", "exit", "q"}:
        return False

    if command in {"help", "h", "?"}:
        table("Commands", ["command", "what it does"],
              [[name, description] for name, description in HELP])

    elif command == "status":
        settings = session.settings()
        say(
            f"mode        {'MOCK' if session.mock else 'LIVE'}\n"
            f"model       {settings.jev_model}\n"
            f"base url    {settings.jev_base_url}\n"
            f"job         {session.job}\n"
            f"criteria    {len(session.criteria)}\n"
            f"candidates  {len(session.candidates)}"
        )

    elif command == "demo":
        session.load_demo()

    elif command == "mock":
        wanted = args[0].lower() if args else ""
        session.mock = (
            not session.mock if wanted not in {"on", "off"} else wanted == "on"
        )
        say(f"{GREEN}mock mode{RESET} {'on' if session.mock else 'off'}")

    elif command == "criteria":
        table(
            "Criteria",
            ["id", "kind", "weight", "levels", "name"],
            [
                [
                    c.id,
                    "gate" if c.required else "scored",
                    "-" if c.required else f"{c.weight:g}",
                    str(len(c.rubric)),
                    c.name,
                ]
                for c in session.criteria
            ],
        )

    elif command == "add":
        add_criterion(session, args)

    elif command in {"rmcrit", "rm-criterion"}:
        if args and args[0] in {c.id for c in session.criteria}:
            session.criteria = [c for c in session.criteria if c.id != args[0]]
            say(f"{GREEN}removed{RESET} {args[0]}")
        else:
            say(f"{YELLOW}no such criterion{RESET}")

    elif command == "candidates":
        table(
            "Candidates",
            ["id", "resume chars", "form chars", "preview"],
            [
                [
                    c.candidate_id,
                    str(len(c.resume_text)),
                    str(len(c.application_form_text)),
                    c.resume_text.replace("\n", " ")[:60],
                ]
                for c in session.candidates
            ],
        )

    elif command == "addcand":
        add_candidate(session, args)

    elif command in {"rmcand", "rm-candidate"}:
        if args and args[0] in {c.candidate_id for c in session.candidates}:
            session.candidates = [
                c for c in session.candidates if c.candidate_id != args[0]
            ]
            say(f"{GREEN}removed{RESET} {args[0]}")
        else:
            say(f"{YELLOW}no such candidate{RESET}")

    elif command == "job":
        if args:
            session.job = " ".join(args)
            say(f"{GREEN}job{RESET} {session.job}")
        else:
            say(f"job {session.job}")

    elif command == "model":
        if args:
            session.model = args[0]
            say(f"{GREEN}model{RESET} {session.model}")
        else:
            say(f"model {session.model or session.settings().jev_model}")

    elif command == "load":
        if not args:
            say(f"{YELLOW}usage:{RESET} load <file.json>")
        else:
            _guard(Path(args[0]), session.load_file)

    elif command == "save":
        if not args:
            say(f"{YELLOW}usage:{RESET} save <file.json>")
        else:
            _guard(Path(args[0]), session.save_file)

    elif command == "run":
        try:
            asyncio.run(session.run())
        except RankerError as exc:
            say(f"{RED}{type(exc).__name__}:{RESET} {exc}")

    elif command == "show":
        if not args:
            say(f"{YELLOW}usage:{RESET} show <id>")
        else:
            session.show_candidate(args[0])

    elif command == "json":
        if not session.report:
            say(f"{DIM}no run yet{RESET}")
        else:
            print(session.report.model_dump_json(indent=2))

    elif command == "cost":
        session.show_cost()

    elif command == "clear":
        print("\033[2J\033[H", end="")

    else:
        say(f"{YELLOW}unknown command{RESET} {command} - try 'help'")

    return True


def repl(session: Session) -> None:
    say(f"{BOLD}Resumes Ranker{RESET} {DIM}- 'help' for commands, 'quit' to exit{RESET}")
    say(f"{DIM}mode {'MOCK' if session.mock else 'LIVE'}{RESET}")
    while True:
        try:
            line = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not dispatch(session, line):
            break


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resume-ranker",
        description="Interactive tester for the resume ranker.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="run against an in-process mock; no API key or credit needed",
    )
    parser.add_argument("--model", help="override the Jev model id")
    parser.add_argument(
        "--concurrency", type=int, help="override max concurrent requests"
    )
    parser.add_argument(
        "--command", "-c", action="append", default=[],
        help="run a command and exit; repeatable",
    )
    args = parser.parse_args(argv)

    session = Session(mock=args.mock, model=args.model, concurrency=args.concurrency)
    if session.mock:
        # Mock mode always starts from the sample pool, so a demo or a scripted
        # -c run has something to rank without any setup.
        session.load_demo()

    if args.command:
        for line in args.command:
            if not dispatch(session, line):
                break
        return 0

    repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Session", "dispatch", "main", "repl"]
