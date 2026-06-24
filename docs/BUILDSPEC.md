# Grounded literature retrieval for hypothesis framing — build spec (v2)

A handoff for Claude Code. Builds a Claude-backed, OpenAI-free **grounded-retrieval
layer** over a large, unfiltered PDF library, with hypothesis framing kept in
conversation with Claude on top of what it surfaces.

> v2 changes (audited since v1): regime is now committed to **large unfiltered
> corpus** (hundreds–thousands of papers), which (a) makes the retrieval layer
> worth building, (b) makes the general-purpose embedder the main risk → adds a
> biomedical-embedding upgrade phase, (c) raises confabulation risk → hardens the
> grounding audit with a required negative control, and (d) adds a hybrid Layer-2
> workflow (retrieve to find, full-text to think).

---

## Goal, in one paragraph

Not an "AI scientist" that emits hypotheses from a black box. A **grounding engine**:
point it at my whole reference library, ask a pointed mechanistic question, get an
answer where every claim ties to a specific chunk of a specific paper I can inspect.
The hypothesis generation — the leap, the framing, the "is this a grant aim" call —
stays a conversation between me and Claude, fed by retrieved, attributable evidence.
Machine does retrieval + synthesis (controllable, where hallucination is catchable);
I keep ideation.

## Why this shape

**Why a retrieval layer at all (not just PDFs into Claude Code).** Decided: the corpus
is large and *unfiltered* — papers I have not pre-selected per question. That doesn't fit
in context and can't be reasoned over directly, so I need persistent indexing + retrieval.
(If I were interrogating 10–20 papers I'd already chosen, I'd skip this and read them
full-text in Claude Code — that's strictly better at small scale. This build is for the
large-corpus case only.)

**Why not a knowledge graph (BioDisco / SciAgents).** Public biomedical KGs are thin on
circuit-level relationships (DRN→DMH serotonergic projections, Htr subtypes, tanycyte
biology); I'd be building my own KG from the corpus first — that graph construction is the
real cost. And KG idea-mining surfaces *associations*, not evidence-grounded testable
hypotheses (the lamm-mit GraphAgents 2026 follow-up concedes this gap). Grounding still has
to come from the literature, so do that well first; revisit a KG only if retrieval+Claude
proves insufficient.

## Architecture (two layers)

```
Layer 1 — GROUNDED RETRIEVAL  (this build)
  PaperQA2 + Claude (via LiteLLM) + LOCAL biomedical sentence-transformers embeddings
  in:  ./papers/*.pdf (large)  +  a pointed question
  out: an answer + retrieved evidence chunks (source file + score + text)

Layer 2 — HYPOTHESIS FRAMING  (mostly conversation, hybrid)
  Use Layer 1 to NARROW the large corpus to the ~5–10 papers that actually matter,
  then pull THOSE full PDFs into Claude Code for deep reasoning. Retrieve to find;
  full-text to think. The hypothesis work happens here, evidence stays attributable.
```

---

## Who does what

**Claude Code can do:** create the venv, install deps, write/​run the runner, build the
index on a small smoke-test folder, run the eval question set, print and help interpret
the evidence chunks, swap the embedding model and rebuild, tune `evidence_k`.

**Only I (Hakan) can do:** provide the PDFs, provide `ANTHROPIC_API_KEY` (never pasted into
a chat — set as an env var locally), and make the **Phase 2 go/no-go judgment** by reading
the chunks against questions I know the answer to.

---

## Build — phased

### Phase 0 — environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "paper-qa" "sentence-transformers"
export ANTHROPIC_API_KEY=sk-ant-...
mkdir papers
```

### Phase 1 — smoke test on a SMALL folder (prove the pipeline)
Put ~5–8 PDFs in `papers/`, start with the fast generic embedder, confirm it runs and
produces attributed chunks. Don't index the whole library yet — validate plumbing first.
```bash
python paperqa_claude_demo.py "a question you know the answer to"
```

### Phase 2 — AUDIT the grounding (the real go/no-go — see protocol below)
Run the eval set. Read the **evidence chunks**, not the prose. Decide whether retrieval is
honest. If it fails here, no downstream cleverness fixes it — be willing to fix retrieval
or walk away.

### Phase 3 — upgrade the embedder, then scale up
Only after Phase 2 passes on the small set: swap `EMBEDDING` to a **biomedical,
retrieval-tuned** sentence-transformers model and **rebuild the index** (changing the
embedder invalidates the old vectors — clear the cache / re-index, don't reuse).
Candidate: `st-pritamdeka/S-PubMedBert-MS-MARCO` (MS-MARCO-tuned = good for asymmetric
query→passage search). Alternatives to A/B: `st-NeuML/pubmedbert-base-embeddings`,
`st-kamalkraj/BioSimCSE-BioLinkBERT-BASE`. **Verify the `st-…` string resolves to a real
HF model on first use.** Then point `papers/` at the full library and build the index once
(local embeddings = free; this can take a while for hundreds of PDFs, but is reused after).

### Phase 4 — hybrid hypothesis framing (Layer 2)
Query the full index to surface the handful of relevant papers + grounded summary. Pull
those full PDFs into a Claude Code / Claude session and do the actual reasoning there.

---

## Phase 2 — grounding audit protocol (do not skip)

The trap: the prose answer is generated by Claude and is only loosely coupled to whether
retrieval actually found support. Fluency ≠ correctness. A confident paragraph can sit on
top of retrieval that missed. You only catch this by opening the cited chunks.

**Three failure modes, all invisible in the prose:**
1. **Retrieval missed → pretraining fallback.** No relevant chunk found; model answers from
   its own memory and staples on the nearest chunk. You think you're reasoning over your
   library; you're not.
2. **Citation doesn't support the claim.** Chunk is real and from your library, but says
   something weaker/adjacent (e.g. "Htr2c is *expressed* in DMH" cited for a claim that it
   *drives* thermogenesis). The leap is the model's.
3. **Strength inflation.** Source says "correlation, male mice, one condition"; answer says
   "X causes Y." Grounding exists but got overclaimed — the exact failure to refuse to
   inherit into a grant aim.

**Test design (build an `eval_questions.yaml`):**
- 4–6 questions **you already know the answer to** and know which paper holds it. You can't
  audit grounding without ground truth.
- For each: did it retrieve the paper you know is right? Do the chunks actually support the
  claim, at the right strength, without an unlicensed leap?
- **One negative control** — a question the library demonstrably *cannot* answer. A healthy
  system says "insufficient evidence"; a broken one confabulates. This matters MORE at
  scale: with hundreds of papers there's almost always a superficially-close chunk, so the
  temptation to confabulate is higher. This single test is the cleanest read on honesty.

```yaml
# eval_questions.yaml  (template)
- q: "<question whose answer you know>"
  expect_source: "Author_Year.pdf"      # the paper that should be retrieved
  expect: answerable
- q: "<question your library cannot answer>"
  expect: insufficient_evidence          # MUST refuse, not confabulate
```

---

## The runner

```python
#!/usr/bin/env python3
"""PaperQA2 grounding demo — Claude-first, OpenAI-free."""
import sys
from paperqa import Settings, ask

# Align with LiteLLM's current Anthropic naming; if a bare name isn't recognised,
# try the explicit "anthropic/…" prefix.
ANSWER_LLM  = "claude-opus-4-8"        # final synthesis: smartest model
SUMMARY_LLM = "claude-haiku-4-5"       # per-chunk summaries: called ~evidence_k times
                                       # per question → use a cheap/fast model here
# Phase 1: fast generic embedder to smoke-test.  Phase 3: swap to biomedical + reindex.
EMBEDDING   = "st-multi-qa-MiniLM-L6-cos-v1"
# EMBEDDING = "st-pritamdeka/S-PubMedBert-MS-MARCO"   # Phase 3 (verify string resolves)

def build_settings() -> Settings:
    s = Settings(
        llm=ANSWER_LLM,
        summary_llm=SUMMARY_LLM,
        embedding=EMBEDDING,
        paper_directory="papers",
    )
    s.agent.agent_llm = ANSWER_LLM
    s.answer.evidence_k = 12           # chunks retrieved per question (recall vs cost lever)
    s.answer.answer_max_sources = 8    # sources allowed into the final answer
    return s

def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "What evidence links DRN serotonergic projections to DMH thermoregulation?"
    )
    print(f"\nQ: {question}\n" + "-" * 70)
    answer = ask(question, settings=build_settings())

    print(getattr(answer, "formatted_answer", None) or str(answer))

    contexts = getattr(answer, "contexts", None)
    if contexts:
        print("\n" + "=" * 70)
        print("RETRIEVED EVIDENCE (judge grounding HERE, not the prose):")
        print("=" * 70)
        for i, ctx in enumerate(contexts, 1):
            src = getattr(getattr(ctx, "text", None), "name", "?")
            score = getattr(ctx, "score", "?")
            snippet = (getattr(ctx, "context", "") or "")[:280].replace("\n", " ")
            print(f"\n[{i}] source={src}  score={score}\n    {snippet}…")
    else:
        print("\n(No .contexts on result — check this version's schema; grounding "
              "data may be under .session or .answer.)")

if __name__ == "__main__":
    main()
```

For the large-corpus index, also check whether this PaperQA version ships the `pqa` CLI
(`pqa --help`): a CLI with a settings file + persistent index may be more robust for
indexing hundreds of PDFs and re-running many questions than the hand-rolled script. Use
whichever proves reliable; the script is the guaranteed-minimal path.

---

## Config knobs — verified vs not

Verified on **paper-qa 2026.3.18** that these exist/are settable: `llm`, `summary_llm`,
`embedding`, `paper_directory`, `agent.agent_llm`, `answer.evidence_k`,
`answer.answer_max_sources`. The package ships `SentenceTransformerEmbeddingModel`, so the
local-embedding path is real.

Do **not** use `parsing.chunk_size` — it does not exist in this version (raised
AttributeError). If recall is poor and you want finer chunks, find the current field name
first; leave chunking at defaults otherwise.

## First-run gotchas (expected, cheap)
- **Claude model string:** if the bare names aren't recognised, use the `anthropic/` prefix
  or list valid strings via LiteLLM.
- **Result attribute names:** `.formatted_answer` / `.contexts` drift between versions; the
  script guards for it.
- **Reindex on embedder change:** switching `EMBEDDING` requires rebuilding the index.

## Honest status
Install, import, and the config surface were tested in a sandbox. The run path was **not**
executed end-to-end (no key, HF blocked, no PDFs there), so expect the two small tweaks
above on first run. Vetted starting point, not observed output. The first real test is
Phase 2 on your own PDFs.

## Cost note
`summary_llm` is called ~`evidence_k` times per question and dominates token spend → Haiku
there, Opus only for final synthesis. Local embeddings make indexing free regardless of
corpus size. If a run feels expensive, lower `evidence_k` before touching the answer model.
