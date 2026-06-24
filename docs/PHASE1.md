# Phase 1 — smoke test (prove the pipeline)

**Goal:** confirm the plumbing works end-to-end on a small corpus and produces *attributed*
evidence chunks — not to judge answer quality yet (that's Phase 2). You're checking that
indexing runs, retrieval returns chunks from your PDFs, and each claim ties to a source you can
open. Uses the seeded `sc_brainstorming` exploration (your 8 superior-colliculus PDFs are already
in `explorations/sc_brainstorming/papers/`).

> Start small (the 8 PDFs already there is perfect). Do **not** point at your full library yet —
> validate the pipeline first, scale in Phase 3.

---

## 0. One-time setup

```bash
cd /Users/hakan/Library/CloudStorage/Dropbox/Hakan/lab/code/paperqa
source .venv/bin/activate          # the venv already has paper-qa[local] + the `pqe` CLI

cp .env.example .env               # then edit .env and paste your real key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

Sanity check (no LLM calls, no cost):

```bash
pqe list
# sc_brainstorming should show  pdfs=8  index=—  embedding=st-multi-qa-MiniLM-L6-cos-v1
```

If the key isn't picked up, confirm it's exported in the shell or present in `.env`
(`pqe` auto-loads `.env`). Local embeddings need no key; the LLM calls do.

---

## 1. Build the index

```bash
pqe index sc_brainstorming
```

- First run downloads the embedding model (~80 MB, one time) and parses + embeds the 8 PDFs.
- Embeddings are local/free; paper-qa **does** call Claude during indexing for citation metadata,
  so this costs a little and needs the key.
- Expect a few minutes for 8 PDFs. The index lands in
  `explorations/sc_brainstorming/index/sc_brainstorming/` (co-located, gitignored).
- Re-running is incremental — only new/changed PDFs are re-embedded.

If a PDF logs `Error parsing <file>, skipping index for this file` — note which one. A scanned or
image-only PDF may not extract text; set it aside or OCR it later. One skipped file won't block
Phase 1.

---

## 2. Ask a question you know the answer to

Pick something the 8 papers genuinely cover, where you already know the right answer and which
paper holds it (that's what makes it a *test*).

```bash
pqe run sc_brainstorming "What pathway carries superior colliculus signals to the amygdala for threat processing?"
```

The output has two parts — **read the bottom part**:

```
Q: ...
----------------------------------------------------------------------
<answer prose with [citations]>

======================================================================
RETRIEVED EVIDENCE (judge grounding HERE, not the prose):
======================================================================
[1] score=8  source=Author2023 chunk 3
    <the actual chunk text the answer is standing on>…
[2] ...
```

Every run is saved to `explorations/sc_brainstorming/runs/<timestamp>__<slug>.md` so you can
compare later.

---

## 3. What "passing Phase 1" looks like

You're validating *plumbing*, not correctness yet. Phase 1 passes if:

- [ ] `pqe index` completed and built an index (`pqe list` shows `index=yes`).
- [ ] `pqe run` returned an answer **plus** a non-empty `RETRIEVED EVIDENCE` block.
- [ ] The evidence chunks are real text **from your PDFs** (source names match your papers), not
      generic-sounding prose with no chunk behind it.
- [ ] At least one chunk is recognizably from the paper you expected.

If retrieval returns nothing, or the "answer" has no chunks under it, the pipeline isn't grounding
— stop and fix that before Phase 2 (see Troubleshooting).

You do **not** need the answer to be *good* yet. Judging honesty/strength is Phase 2
(`pqe audit`, see [GROUNDING_AUDIT.md](GROUNDING_AUDIT.md)).

---

## 4. First-run gotchas (expected, cheap)

| Symptom | Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` | Put the key in `.env` or `export` it. |
| `No PDFs in …/papers` | Add PDFs to the exploration's `papers/` folder. |
| `Error parsing <file>, skipping` | That PDF didn't yield text (scanned/image-only). Skip or OCR it. |
| Indexing feels slow/expensive | Figure enrichment is on. Set `multimodal: 0` in `config.yaml`, then `pqe index sc_brainstorming --rebuild`. |
| LiteLLM provider / cost-map warning | Routing still works (we use the `anthropic/` prefix). If it errors, `pip install -U litellm`. |
| Want to see the raw evidence schema once | The run output already prints it; chunks live under `answer.session.contexts` (see docs/ERRATA.md). |

---

## 5. When Phase 1 passes → Phase 2

Fill in `explorations/sc_brainstorming/eval_questions.yaml`: 4–6 questions you know the answers to
(with `expect_source`) **plus one negative control** the library can't answer
(`expect: insufficient_evidence`). Then:

```bash
pqe audit sc_brainstorming
```

Phase 2 is the real go/no-go — judge the **chunks**, not the prose. Protocol:
[GROUNDING_AUDIT.md](GROUNDING_AUDIT.md). Only after it passes do you swap to the biomedical
embedder and scale up (Phase 3).
