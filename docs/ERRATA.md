# ERRATA — corrections to BUILDSPEC.md, verified against paper-qa 2026.3.18

The buildspec (`docs/BUILDSPEC.md`) was a vetted starting point whose run path was never
executed end-to-end. Before building, the engine code was checked against the **current**
paper-qa source (v2026.3.18 — confirmed the latest PyPI release; the buildspec's version is
current, no drift). These corrections are baked into `engine/`.

## Bug 1 — `Settings(paper_directory="papers")` is a silent no-op
`Settings` uses `model_config = SettingsConfigDict(extra="ignore")`, so an unknown top-level
kwarg is **dropped without error**. `paper_directory` is not a top-level field — it lives at
`Settings.agent.index.paper_directory` (default = `Path.cwd()`). The buildspec line would have
quietly indexed the **current working directory**, not `./papers`.
→ Fixed in `engine/settings.py`: set it under `agent.index.paper_directory`.
Source: `src/paperqa/settings.py` (IndexSettings).

## Bug 2 — answer text/evidence are not top-level on the result
`ask()` returns an `AnswerResponse` whose only relevant field is `session` (a `PQASession`,
aliased `"answer"`). There is **no** top-level `formatted_answer`, `contexts`, or `answer`
attribute and no `__getattr__` fallback, so the buildspec's `getattr(answer, "formatted_answer")`
/ `getattr(answer, "contexts")` both return `None`. Grounding lives under `answer.session.*`:
`session.answer`, `session.formatted_answer`, `session.contexts`. Per-context access
(`ctx.context`, `ctx.score`, `ctx.text.name`) is correct once you iterate `session.contexts`.
→ Fixed in `engine/report.py` (`get_session`, `context_rows`).
Source: `src/paperqa/agents/models.py`, `src/paperqa/types.py`.

## Model strings — use the `anthropic/` prefix
paper-qa passes `llm`/`summary_llm`/`agent.agent_llm` straight to LiteLLM, which routes by the
model string. New Claude IDs (`claude-opus-4-8`, `claude-haiku-4-5`) are newer than what the
LiteLLM Anthropic docs enumerate, so a bare name risks a "provider not provided" error depending
on the pinned litellm. The `anthropic/` prefix makes routing version-independent.
→ `engine/settings.py` uses `anthropic/claude-opus-4-8` and `anthropic/claude-haiku-4-5`.
Note: `claude-opus-4-8` has **no** date suffix (the alias is the complete id).

## Index management — multi-exploration design
- Index config is entirely under `Settings.agent.index` (`IndexSettings`): `name`,
  `paper_directory`, `index_directory` (default `~/.pqa/indexes`), `sync_with_paper_directory`
  (default True — adds new / removes deleted PDFs on load), `recurse_subdirectories`.
- We pin an **explicit `name`** (the folder name) and a **co-located `index_directory`**
  (`<exploration>/index/`) so each exploration is self-contained and corpora never collide.
- Indexing is **incremental** (filecheck on filename + body-hash); adding PDFs later only
  embeds the new ones. Local embeddings make this free.
- ⚠ With an explicit `name`, changing the embedder does **NOT** auto-invalidate the index
  (the auto-hash that would differentiate it is bypassed). → `pqe index <name> --rebuild`
  clears `index/` first. Do this for the Phase-3 embedder swap.
Source: `src/paperqa/settings.py`, `src/paperqa/agents/search.py`, `src/paperqa/utils.py`,
GitHub Discussion #854.

## Embeddings — local, OpenAI-free
`embedding="st-<hf-model>"` selects `SentenceTransformerEmbeddingModel` (needs the `[local]`
extra, installed). `hybrid-st-…` adds sparse keyword vectors; `sparse` is keyword-only.
Phase-3 biomedical candidates: `st-pritamdeka/S-PubMedBert-MS-MARCO` (best asymmetric
query→passage fit; truncates ~350 tokens, keep chunks modest) and
`st-NeuML/pubmedbert-base-embeddings`. Avoid `BioSimCSE-BioLinkBERT` as a retriever (it is
STS/symmetric). Worth an eval later: `st-abhinand5/MedEmbed-*`.

## Phase-0 smoke-check results (this install, 2026-06-24)
Verified on paper-qa 2026.3.18 / sentence-transformers 5.6.0 / litellm 1.84.1, Python 3.13.3:
- **Bug 1 fixed & confirmed:** `make_settings` yields
  `agent.index.paper_directory = …/sc_brainstorming/papers` (not the cwd).
- Model strings resolve to `anthropic/claude-opus-4-8` / `anthropic/claude-haiku-4-5`.
- Native `pqa --help` works; `pqe list|run|audit` work and error cleanly (missing key, no PDFs,
  unknown exploration → friendly `SystemExit`, no traceback).
- Local embedding loads offline (`multi-qa-MiniLM-L6-cos-v1`, 384-d) — OpenAI-free path good.

### Correction to the build plan: **indexing REQUIRES the API key**
The plan assumed `pqe index` needs no key. **Wrong.** paper-qa calls the LLM during
`Docs.aadd` while indexing (citation/metadata, and — if `multimodal` is on — figure enrichment),
so a real index build needs `ANTHROPIC_API_KEY`. A keyless `pqe index` now fails fast with a
friendly message (`cli.cmd_index` calls `_require_key()`). Setting `parsing.use_doc_details=False`
does **not** make indexing offline — `aadd` still issues an LLM call. The primary indexing call
uses `settings.llm` (our `anthropic/…` model), so it is OpenAI-free.

### Chunk sizing — the real field (resolves the buildspec conflict)
`parsing.chunk_size` does **not** exist (`Settings(parsing={"chunk_size": …})` →
`extra_forbidden`). The buildspec was right that it's absent; the README CLI example is stale.
Chunking lives at **`parsing.reader_config["chunk_chars"]`** (default 5000) and
`["overlap"]` (default 250). `engine.make_settings` exposes it via the optional config key
`chunk_chars`. Leave at defaults unless recall is poor.

### OpenAI leak closed
`parsing.enrichment_llm` defaults to **`gpt-4o-2024-11-20`** (OpenAI) for multimodal figure/table
enrichment. `engine.make_settings` overrides it to `SUMMARY_LLM` (Claude Haiku, vision-capable,
cheap) so multimodal stays OpenAI-free. `multimodal` is on by default (=1) → enrichment calls the
LLM per figure (cost/latency). Set `multimodal: 0` in an exploration's `config.yaml` for faster,
text-only indexing.

### Bug 3 — `temperature` is deprecated for claude-opus-4-8 (found on first keyed index)
The first real `pqe index` (with a key) hit:
`BadRequestError: AnthropicException - "temperature is deprecated for this model". Model=anthropic/claude-opus-4-8`.
paper-qa's `make_default_litellm_model_list_settings` hard-codes `temperature=0.0` into every
model's `litellm_params`; the Anthropic API now rejects the param for opus-4-8. litellm 1.84.1
still lists `temperature` as **supported** for this model (`get_supported_openai_params` includes
it), so `litellm.drop_params=True` does **not** drop it — the param must simply not be sent.
→ Fixed in `engine/settings.py` (`_model_config`): `make_settings` sets explicit
`llm_config` / `summary_llm_config` / `agent.agent_llm_config` / `parsing.enrichment_llm_config`
that mirror paper-qa's default **minus** the `temperature` key (keeping prompt-cache injection).
Verified: `temperature` is absent from every router deployment's `litellm_params`.
Covers the default `agent_type="ToolSelector"` (its LLM goes through `get_agent_llm().get_router()`).
Notes: (a) we omit temperature for Haiku too, for uniformity/safety — summaries use the model
default instead of 0.0; (b) the hard-coded `llm_model={"temperature": self.temperature}` at
settings.py:1035/1040 is only used by `ldp`-based agent types — if you ever switch `agent_type`
away from ToolSelector, temperature handling there would need revisiting.

### Still confirm at first real run (needs the key)
- Live `session.contexts[0]` schema (`.context`/`.score`/`.text.name`) — `engine/report.py`
  guards defensively, but eyeball it once.
- That `litellm 1.84.1` routes `anthropic/claude-opus-4-8` without cost-map warnings.
