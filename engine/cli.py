"""pqe — drive grounded retrieval across exploration folders.

    pqe list                          # what explorations exist, papers/index status
    pqe index <exploration> [--rebuild]   # build/refresh that folder's index (uses the LLM for citations)
    pqe run   <exploration> "question"    # ask; print answer + evidence; save to runs/
    pqe audit <exploration>           # run eval_questions.yaml (Phase-2 grounding go/no-go)

Named `pqe` to avoid clashing with paper-qa's own `pqa` CLI.
"""

import argparse
import shutil
import sys

from engine.config import (
    list_explorations,
    load_config,
    load_env,
    pdf_count,
)


def cmd_list(_args) -> int:
    explorations = list_explorations()
    if not explorations:
        print("No explorations yet. Copy explorations/_template to start one.")
        return 0
    print(f"{'exploration':24}  {'pdfs':>5}  {'index':>6}  embedding")
    print("-" * 78)
    for d in explorations:
        cfg = load_config(d)
        has_index = "yes" if (d / "index").is_dir() and any((d / "index").iterdir()) else "—"
        print(f"{d.name:24}  {pdf_count(d):>5}  {has_index:>6}  {cfg['embedding']}")
    return 0


def cmd_index(args) -> int:
    import asyncio

    from engine.config import exploration_dir
    from engine.run import _require_key, _require_papers
    from engine.settings import make_settings

    d = exploration_dir(args.exploration)
    _require_papers(d)
    _require_key()   # paper-qa calls the LLM during indexing (citation/metadata) — key required
    cfg = load_config(d)

    index_dir = d / "index"
    if args.rebuild and index_dir.is_dir():
        # An explicit index name does NOT auto-invalidate on embedder change — clear first.
        print(f"[--rebuild] clearing {index_dir}")
        shutil.rmtree(index_dir)

    settings = make_settings(d, cfg)
    print(f"Indexing {pdf_count(d)} PDF(s) in {d / 'papers'} with embedding={cfg['embedding']} …")

    from paperqa.agents.search import get_directory_index

    asyncio.run(get_directory_index(settings=settings))
    print(f"[done] index at {index_dir / cfg['name']}")
    return 0


def cmd_run(args) -> int:
    from engine.run import run_question

    run_question(args.exploration, args.question)
    return 0


def cmd_audit(args) -> int:
    from engine.audit import run_audit

    run_audit(args.exploration)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pqe", description="Grounded retrieval over per-exploration PDF corpora.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list exploration folders").set_defaults(func=cmd_list)

    pi = sub.add_parser("index", help="build/refresh an exploration's index (needs ANTHROPIC_API_KEY)")
    pi.add_argument("exploration")
    pi.add_argument("--rebuild", action="store_true", help="clear the index first (use after an embedder swap)")
    pi.set_defaults(func=cmd_index)

    pr = sub.add_parser("run", help="ask a question against an exploration")
    pr.add_argument("exploration")
    pr.add_argument("question")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("audit", help="run eval_questions.yaml (Phase-2 grounding audit)")
    pa.add_argument("exploration")
    pa.set_defaults(func=cmd_audit)
    return p


def main(argv=None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
