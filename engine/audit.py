"""Phase-2 grounding audit — the real go/no-go (see docs/GROUNDING_AUDIT.md).

Runs eval_questions.yaml and surfaces, per question, the retrieved chunks + heuristic
signals. The heuristics are a convenience; the actual judgment is yours — read the chunks.
"""

import re
from datetime import datetime
from pathlib import Path

import yaml

from engine import report
from engine.config import load_config
from engine.run import _require_key, _require_papers
from engine.settings import make_settings

# Phrases a healthy system uses when the library can't answer (negative-control check).
REFUSAL_MARKERS = [
    "insufficient", "cannot answer", "cannot be answered", "not enough", "no evidence",
    "unable to", "i don't have", "do not have", "not provide", "cannot be determined",
    "no relevant", "not found", "not available", "lacks", "unanswerable", "i cannot",
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _source_found(expect_source: str, answer_response) -> bool:
    """Heuristic: did a normalized form of the expected source appear among retrieved chunks?"""
    target = _normalize(Path(expect_source).stem)
    if not target:
        return False
    blob = _normalize(" ".join(r["blob"] + " " + r["source"] for r in report.context_rows(answer_response)))
    return target in blob


def _refused(answer_response) -> bool:
    text = report.answer_text(answer_response).lower()
    return any(m in text for m in REFUSAL_MARKERS)


def run_audit(name: str, *, save: bool = True):
    from engine.config import exploration_dir

    d = exploration_dir(name)
    _require_papers(d)
    _require_key()

    eval_file = d / "eval_questions.yaml"
    if not eval_file.exists():
        raise SystemExit(f"No {eval_file} — fill in the template (≥4 answerable + 1 negative control).")
    questions = yaml.safe_load(eval_file.read_text()) or []
    if not questions:
        raise SystemExit(f"{eval_file} is empty — add eval questions you know the answer to.")

    cfg = load_config(d)
    settings = make_settings(d, cfg)

    blocks = [
        f"# Grounding audit — {name}",
        f"embedding={cfg['embedding']}  evidence_k={cfg['evidence_k']}  questions={len(questions)}",
        "Heuristics below are a convenience. JUDGE THE CHUNKS YOURSELF — fluency ≠ grounding.\n",
    ]
    print("\n".join(blocks))

    for n, item in enumerate(questions, 1):
        q = item.get("q", "")
        expect = item.get("expect", "answerable")
        expect_source = item.get("expect_source")

        answer_response = ask_q(q, settings)

        if expect == "insufficient_evidence":
            ok = _refused(answer_response)
            verdict = "PASS (refused)" if ok else "FLAG (answered a question the library can't — possible confabulation)"
            head = f"[{n}] NEGATIVE CONTROL — expect: insufficient_evidence → heuristic {verdict}"
        else:
            found = _source_found(expect_source, answer_response) if expect_source else None
            if found is None:
                verdict = "no expect_source given"
            else:
                verdict = "PASS (expected source retrieved)" if found else "FLAG (expected source NOT among retrieved chunks)"
            head = f"[{n}] expect: answerable  expect_source={expect_source} → heuristic {verdict}"

        block = head + "\n" + report.render_run(q, answer_response)
        print("\n" + "=" * 72 + "\n" + block)
        blocks.append("\n" + "=" * 72 + "\n" + block)

    if save:
        runs = d / "runs"
        runs.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = runs / f"{stamp}__audit.md"
        path.write_text("\n".join(blocks))
        print(f"\n[saved] {path}")


def ask_q(question: str, settings):
    from paperqa import ask
    return ask(question, settings=settings)
