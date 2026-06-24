# CLAUDE.md — paperqa

Claude-backed, OpenAI-free grounded literature retrieval over a PDF library, built on paper-qa
(PaperQA2). Purpose: retrieval + **attributable** synthesis, where every claim ties to an
inspectable chunk. Hypothesis framing stays a conversation with the user — the machine grounds,
the user ideates. Read `docs/BUILDSPEC.md` for full intent.

## Architecture
- **One shared engine** (`engine/`) serves **many self-contained explorations** (`explorations/<name>/`).
- Each exploration owns its `papers/` (PDFs), a co-located `index/`, `eval_questions.yaml`, and
  `runs/` (saved answers + evidence). Nothing leaks into `~/.pqa`; explorations never collide.

## engine/ map
- `settings.py` — model strings (`anthropic/claude-opus-4-8`, `anthropic/claude-haiku-4-5`,
  default `st-` local embedding) + `make_settings(exploration_dir, cfg)`.
- `config.py` — `DEFAULTS`, `load_config`, path helpers, `load_env` (reads `.env`).
- `report.py` — reads grounding off `answer.session.*` and persists runs.
- `run.py` / `audit.py` — single question / full eval set.
- `cli.py` — the `pqe` command (`list`, `index`, `run`, `audit`).

## Critical paper-qa facts (verified vs 2026.3.18 — see docs/ERRATA.md)
- `paper_directory` is **not** a top-level `Settings` kwarg (`extra="ignore"` silently drops it).
  Set `agent.index.paper_directory`. → in `make_settings`.
- Answer + evidence are under `answer.session.*` (`session.formatted_answer`, `session.contexts`),
  **not** top-level on `AnswerResponse`. → in `report.get_session`.
- Use the `anthropic/` prefix on all model strings (version-independent LiteLLM routing).
- Explicit index `name` + co-located `index_directory` per exploration. Indexing is incremental.
  An embedder swap under a fixed name does NOT auto-invalidate → use `pqe index <name> --rebuild`.

## Conventions
- Add a new exploration by copying `explorations/_template` (mirrors the `engine/` + per-unit
  pattern used elsewhere in this lab's code, e.g. VRFarm).
- All of `pqe index`/`run`/`audit` need `ANTHROPIC_API_KEY`. Embeddings are local, but paper-qa
  calls the LLM during indexing (citation/metadata via `Docs.aadd`) too — verified, see ERRATA.
- Cost: `summary_llm` (Haiku) is called ~`evidence_k`× per question and dominates spend; lower
  `evidence_k` in `config.yaml` before touching the answer model.
- When auditing grounding, judge the **evidence chunks**, not the prose (docs/GROUNDING_AUDIT.md).
