"""paperqa engine — a thin, reusable layer over paper-qa (PaperQA2).

One shared retrieval engine serves many self-contained exploration folders under
`explorations/`. Each exploration owns its own PDFs, its own co-located index, and
its own eval questions. See docs/BUILDSPEC.md (intent) and docs/ERRATA.md (the
paper-qa API corrections this code is built on).
"""

from engine.settings import ANSWER_LLM, SUMMARY_LLM, DEFAULT_EMBEDDING, make_settings

__all__ = ["ANSWER_LLM", "SUMMARY_LLM", "DEFAULT_EMBEDDING", "make_settings"]
