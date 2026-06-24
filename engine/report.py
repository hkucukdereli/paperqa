"""Read grounding evidence off the result, print it, and persist runs.

FIX #2 (see docs/ERRATA.md): on paper-qa 2026.3.18 the answer text and evidence are NOT
top-level on AnswerResponse — `answer.formatted_answer` / `answer.contexts` are None. They
live under `answer.session.*` (PQASession). Helpers below extract defensively because the
PQASession/Context schema was verified from remote source, not a live install.
"""

import re
from datetime import datetime
from pathlib import Path

SNIPPET_CHARS = 300


def get_session(answer_response):
    """Return the PQASession. Field is `.session` (aliased 'answer' for construction)."""
    return getattr(answer_response, "session", None) or getattr(answer_response, "answer", None)


def answer_text(answer_response) -> str:
    ses = get_session(answer_response)
    if ses is None:
        return str(answer_response)
    return getattr(ses, "formatted_answer", None) or getattr(ses, "answer", None) or str(ses)


def _context_source(ctx) -> str:
    """Best human-readable source id for a retrieved chunk."""
    text = getattr(ctx, "text", None)
    name = getattr(text, "name", None)
    return str(name) if name else "?"


def _context_blob(ctx) -> str:
    """All identifier strings for a chunk (name + doc citation/docname/filepath), for matching."""
    parts = []
    text = getattr(ctx, "text", None)
    parts.append(getattr(text, "name", "") or "")
    doc = getattr(text, "doc", None)
    for attr in ("docname", "citation", "formatted_citation", "filepath", "dockey"):
        parts.append(str(getattr(doc, attr, "") or ""))
    return " | ".join(p for p in parts if p)


def context_rows(answer_response) -> list[dict]:
    ses = get_session(answer_response)
    contexts = getattr(ses, "contexts", None) or []   # NOT answer_response.contexts
    rows = []
    for i, ctx in enumerate(contexts, 1):
        summary = (getattr(ctx, "context", "") or "").replace("\n", " ").strip()
        rows.append({
            "i": i,
            "source": _context_source(ctx),
            "score": getattr(ctx, "score", "?"),
            "snippet": summary[:SNIPPET_CHARS],
            "blob": _context_blob(ctx),
        })
    return rows


def format_evidence(answer_response) -> str:
    rows = context_rows(answer_response)
    if not rows:
        return "(no retrieved evidence chunks — judge whether the answer should have refused)"
    out = []
    for r in rows:
        out.append(f"[{r['i']}] score={r['score']}  source={r['source']}\n    {r['snippet']}…")
    return "\n".join(out)


def render_run(question: str, answer_response) -> str:
    """Full text block: answer + the evidence to judge grounding against."""
    return (
        f"Q: {question}\n"
        + "-" * 70 + "\n"
        + answer_text(answer_response) + "\n\n"
        + "=" * 70 + "\n"
        + "RETRIEVED EVIDENCE (judge grounding HERE, not the prose):\n"
        + "=" * 70 + "\n"
        + format_evidence(answer_response) + "\n"
    )


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "query"


def save_run(exploration_dir: Path, question: str, answer_response, prefix: str = "") -> Path:
    runs = Path(exploration_dir) / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}__{prefix}{_slug(question)}.md"
    path = runs / fname
    path.write_text(render_run(question, answer_response))
    return path
