"""paper-qa Settings factory — Claude-backed, OpenAI-free, per-exploration.

Bakes in the two buildspec corrections verified against paper-qa 2026.3.18 source
(see docs/ERRATA.md):
  FIX #1  paper_directory is NOT a top-level Settings kwarg — Settings(extra="ignore")
          silently drops it and the index defaults to the cwd. Set it under
          agent.index.paper_directory instead.
  Models  use the `anthropic/` prefix so LiteLLM routes new Claude IDs version-independently.
"""

from pathlib import Path

from paperqa import Settings
from paperqa.settings import AgentSettings, IndexSettings

# --- shared model strings (every exploration uses these unless config overrides embedding) ---
ANSWER_LLM = "anthropic/claude-opus-4-8"      # final synthesis + agent reasoning (1M ctx)
SUMMARY_LLM = "anthropic/claude-haiku-4-5"    # per-chunk summaries, called ~evidence_k× → keep cheap
# Phase 1 = fast generic embedder (smoke test). Phase 3 = swap to biomedical in config.yaml:
#   st-pritamdeka/S-PubMedBert-MS-MARCO   or   st-NeuML/pubmedbert-base-embeddings
DEFAULT_EMBEDDING = "st-multi-qa-MiniLM-L6-cos-v1"


def _model_config(model_name: str) -> dict:
    """A LiteLLM model_list config that OMITS `temperature`.

    claude-opus-4-8 rejects the `temperature` param (the Anthropic API returns
    "`temperature` is deprecated for this model"), but paper-qa's default
    `make_default_litellm_model_list_settings` hard-codes `temperature=0.0` into
    litellm_params. litellm 1.84.x still lists temperature as "supported" for this model,
    so `drop_params` won't help — the param must simply not be sent. This mirrors the
    paper-qa default minus the temperature key, keeping prompt-cache injection.
    See docs/ERRATA.md.
    """
    return {
        "name": model_name,
        "model_list": [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "cache_control_injection_points": [
                        {"location": "message", "role": "system"}
                    ],
                },
            }
        ],
    }


def make_settings(exploration_dir: Path, cfg: dict) -> Settings:
    """Build paper-qa Settings for one exploration folder.

    exploration_dir/papers/  -> corpus to index
    exploration_dir/index/   -> co-located persistent index (self-contained, portable)

    cfg keys: name, embedding, evidence_k, answer_max_sources (see engine.config.load_config).
    """
    exploration_dir = Path(exploration_dir).resolve()
    papers = exploration_dir / "papers"
    index = exploration_dir / "index"

    s = Settings(
        llm=ANSWER_LLM,
        summary_llm=SUMMARY_LLM,
        embedding=cfg["embedding"],
        agent=AgentSettings(
            agent_llm=ANSWER_LLM,
            index=IndexSettings(
                name=cfg["name"],                 # explicit name -> reproducible, never auto-hashed
                paper_directory=str(papers),      # FIX #1: the real field (not a top-level kwarg)
                index_directory=str(index),       # co-located per-exploration store
                sync_with_paper_directory=True,   # add new / drop deleted PDFs on each load
                recurse_subdirectories=True,
            ),
        ),
    )
    s.answer.evidence_k = cfg["evidence_k"]            # chunks retrieved per question (recall vs cost)
    s.answer.answer_max_sources = cfg["answer_max_sources"]

    # OpenAI-free guarantee: paper-qa's multimodal figure enrichment defaults to gpt-4o.
    # Route it to a cheap, vision-capable Claude model so nothing ever touches OpenAI.
    s.parsing.enrichment_llm = SUMMARY_LLM

    # Don't send `temperature` (deprecated for claude-opus-4-8 → Anthropic 400s on it).
    # Explicit per-model configs that omit it; covers get_llm / get_summary_llm /
    # get_agent_llm (used by the default ToolSelector agent) / get_enrichment_llm.
    s.llm_config = _model_config(ANSWER_LLM)
    s.summary_llm_config = _model_config(SUMMARY_LLM)
    s.agent.agent_llm_config = _model_config(ANSWER_LLM)
    s.parsing.enrichment_llm_config = _model_config(SUMMARY_LLM)

    # Optional advanced knobs (set in config.yaml only if you need them):
    #   multimodal: 0       -> skip LLM figure/table enrichment (faster + cheaper; text-only)
    #   chunk_chars: 5000   -> chunk size (lives at parsing.reader_config["chunk_chars"], NOT parsing.chunk_size)
    if cfg.get("multimodal") is not None:
        s.parsing.multimodal = cfg["multimodal"]
    if cfg.get("chunk_chars") is not None:
        s.parsing.reader_config = {**s.parsing.reader_config, "chunk_chars": cfg["chunk_chars"]}
    return s
