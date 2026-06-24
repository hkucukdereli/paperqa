"""Paths, per-exploration config loading, and a tiny .env loader (no extra deps)."""

from pathlib import Path

import yaml

from engine.settings import DEFAULT_EMBEDDING

ROOT = Path(__file__).resolve().parent.parent          # repo root (.../paperqa)
EXPLORATIONS = ROOT / "explorations"

# Engine defaults; a per-exploration config.yaml overrides any of these.
DEFAULTS = {
    "embedding": DEFAULT_EMBEDDING,
    "evidence_k": 12,            # chunks retrieved per question
    "answer_max_sources": 8,     # sources allowed into the final answer
}


def load_env() -> None:
    """Load ROOT/.env into os.environ (KEY=VALUE lines). Avoids a python-dotenv dep.

    Existing environment variables win, so an already-exported ANTHROPIC_API_KEY is kept.
    """
    import os

    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def exploration_dir(name: str) -> Path:
    """Resolve an exploration name to its folder, erroring clearly if absent."""
    d = EXPLORATIONS / name
    if not d.is_dir():
        available = ", ".join(sorted(p.name for p in EXPLORATIONS.iterdir() if p.is_dir())) or "(none)"
        raise SystemExit(f"No exploration '{name}' under {EXPLORATIONS}.\nAvailable: {available}")
    return d


def list_explorations() -> list[Path]:
    if not EXPLORATIONS.is_dir():
        return []
    return sorted(p for p in EXPLORATIONS.iterdir() if p.is_dir())


def load_config(d: Path) -> dict:
    """Merge an exploration's config.yaml over the engine DEFAULTS.

    `name` defaults to the folder name (used as the paper-qa index name).
    """
    cfg = dict(DEFAULTS)
    cfg_file = d / "config.yaml"
    if cfg_file.exists():
        loaded = yaml.safe_load(cfg_file.read_text()) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    cfg.setdefault("name", d.name)
    return cfg


def pdf_count(d: Path) -> int:
    papers = d / "papers"
    if not papers.is_dir():
        return 0
    return sum(1 for p in papers.rglob("*") if p.suffix.lower() == ".pdf")
