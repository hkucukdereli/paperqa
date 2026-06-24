# paperqa — grounded literature retrieval

A Claude-backed **grounding engine** over your PDF library, built on
[paper-qa](https://github.com/Future-House/paper-qa) (PaperQA2). Point it at a corpus, ask a
pointed mechanistic question, get an answer where **every claim ties to a specific chunk of a
specific paper you can inspect**. The machine does retrieval + attributable synthesis (where
hallucination is catchable); you keep the hypothesis framing.

One shared **engine** serves many self-contained **explorations** — drop a new folder under
`explorations/` for each project, each with its own PDFs, index, and eval questions.

## Layout
```
engine/         shared retrieval engine (settings, run, audit, cli) — reused by every exploration
explorations/   one folder per project; copy _template to start a new one
docs/           BUILDSPEC (intent), ERRATA (paper-qa API corrections), GROUNDING_AUDIT (Phase-2)
```

## Setup (one-time)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # paper-qa[local] + pyyaml
cp .env.example .env             # then put your ANTHROPIC_API_KEY in .env  (local embeddings need no key)
```

## Workflow
```bash
pqe list                                  # what explorations exist
# 1. start an exploration (or use the seeded sc_brainstorming)
cp -r explorations/_template explorations/my_topic     # edit config.yaml: name: my_topic
#    drop PDFs into explorations/my_topic/papers/
# 2. build the index (embeddings are local; paper-qa uses the LLM for citations → needs the key)
pqe index my_topic
# 3. ask — prints the answer AND the evidence chunks; saves to runs/
pqe run my_topic "what evidence links X to Y?"
# 4. Phase-2 grounding audit (fill eval_questions.yaml first)
pqe audit my_topic
```

## The phased build (see docs/BUILDSPEC.md)
1. **Smoke test** — ~5–8 PDFs, generic embedder, confirm attributed chunks.
2. **Grounding audit** — `pqe audit`; judge the *chunks*, not the prose. The real go/no-go
   (docs/GROUNDING_AUDIT.md). Includes a required negative control.
3. **Upgrade embedder + scale** — set a biomedical embedder in `config.yaml`, `pqe index
   <name> --rebuild`, point `papers/` at the full library.
4. **Hybrid framing** — use retrieval to narrow to the ~5–10 papers that matter, then pull
   those full PDFs into a Claude session for the actual reasoning.

## Notes
- Models: `anthropic/claude-opus-4-8` (answer/agent), `anthropic/claude-haiku-4-5` (per-chunk
  summaries — cheap, the cost driver). Embeddings are local (`st-…`). Indexing still calls the
  LLM (citations + optional figure enrichment) — set `multimodal: 0` in config.yaml to skip the
  figure step and make indexing cheaper/faster.
- `papers/` and `index/` are gitignored (data, large/private); `runs/` is kept as your audit
  trail. Two buildspec bugs were caught and fixed before building — see docs/ERRATA.md.
