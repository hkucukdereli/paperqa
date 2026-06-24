"""Ask one question against an exploration's corpus; print + persist the result."""

import os
from pathlib import Path

from paperqa import ask

from engine import report
from engine.config import load_config, pdf_count
from engine.settings import make_settings


def _require_papers(d: Path) -> None:
    if pdf_count(d) == 0:
        raise SystemExit(
            f"No PDFs in {d / 'papers'} — add some before running.\n"
            f"(Drop PDFs there, then: pqe index {d.name}  and  pqe run {d.name} \"...\")"
        )


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it or put it in .env "
            "(copy .env.example). Local embeddings need no key, but the answer/summary LLMs do."
        )


def run_question(name: str, question: str, *, save: bool = True):
    """Resolve the exploration, build settings, ask, print evidence, and save the run."""
    from engine.config import exploration_dir

    d = exploration_dir(name)
    _require_papers(d)
    _require_key()

    cfg = load_config(d)
    settings = make_settings(d, cfg)

    print(f"\nQ: {question}")
    print(f"   exploration={name}  embedding={cfg['embedding']}  evidence_k={cfg['evidence_k']}")
    print("-" * 70)

    answer_response = ask(question, settings=settings)   # sync context → returns AnswerResponse

    print(report.render_run(question, answer_response))

    if save:
        path = report.save_run(d, question, answer_response)
        print(f"\n[saved] {path}")
    return answer_response
