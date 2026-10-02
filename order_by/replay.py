"""Cache-only replay: re-run algorithms purely from the LLM response cache.

Once an experiment's LLM responses are cached, it can be replayed for free.
The helpers here make that a guarantee rather than a hope:

* `NoNetworkClient` (also selectable as `--provider cache`) refuses every API
  call, so a cache miss raises `RealCall` instead of silently spending money.
* `install_memcache()` routes cache reads through an in-memory read-through
  layer: each key is read from disk at most once and nothing is written back,
  so a replay neither modifies the on-disk cache nor pays its LRU bookkeeping
  on every read.
"""
from . import optimizer, pair_comparison, pointwise
from .clients import NoNetworkClient, RealCall  # noqa: F401  (re-exported)
from .tools import web_search


class MemCache:
    """Read-through in-memory wrapper over the on-disk cache. Reads load each
    key from disk once; writes and deletes stay in RAM."""

    def __init__(self, real):
        self.real = real
        self.mem = {}

    def __contains__(self, key):
        if key in self.mem:
            return True
        try:
            self.mem[key] = self.real[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        if key not in self.mem:
            self.mem[key] = self.real[key]
        return self.mem[key]

    def __setitem__(self, key, value):
        self.mem[key] = value

    def __delitem__(self, key):
        self.mem.pop(key, None)


def install_memcache() -> MemCache:
    """Swap the response cache used by every LLM-calling module for a MemCache."""
    mem = MemCache(pointwise.cache)
    for module in (pointwise, pair_comparison, optimizer, web_search):
        module.cache = mem
    return mem
