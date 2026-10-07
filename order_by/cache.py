"""On-disk caches shared by every module that talks to an LLM.

`cache` stores LLM responses keyed by `hash_prompt(prompt, model)`: the model's
provider-agnostic short name plus the exact prompt. A response is therefore
reused across providers, algorithms and scripts, and re-running an experiment
whose calls are all cached costs nothing. See cache_tools.py for exporting and
importing it.

Caches live at the repo root by default. Set SORT_CACHE_DIR (LLM responses),
JEV_CACHE_DIR (Jev responses, see jev.py) or WIKI_CACHE_DIR (Wikipedia / web
search lookups, see tools/web_search.py) in the shell to use caches kept
elsewhere, e.g. one shared between checkouts.
"""
import os

from diskcache import Cache

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def open_cache(default_dir: str, env_var: str, size_limit: int) -> Cache:
    """Open the cache at $env_var, or at <repo root>/<default_dir> when unset."""
    path = os.environ.get(env_var) or os.path.join(PROJECT_ROOT, default_dir)
    return Cache(path, size_limit=size_limit, eviction_policy='least-recently-used')


def open_wiki_cache() -> Cache:
    """The Wikipedia / web search lookup cache (opened only by code that needs it)."""
    return open_cache('wiki_cache', 'WIKI_CACHE_DIR', 2 * 1024**3)


def open_jev_cache() -> Cache:
    """Responses of the Jev model (see jev.py); kept apart from the LLM cache so
    it can be exported and shared as a separate, much smaller file."""
    return open_cache('jev_cache', 'JEV_CACHE_DIR', 10 * 1024**3)


cache = open_cache('sort_cache', 'SORT_CACHE_DIR', 50 * 1024**3)
