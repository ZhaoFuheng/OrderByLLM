#!/usr/bin/env python3
"""Export and import the on-disk caches as compressed JSONL files, so cached
responses can be shared and experiments re-run without calling any API.

Two caches exist (see order_by/cache.py):
  sort  the LLM response cache, sort_cache/ (or SORT_CACHE_DIR)   [default]
  wiki  the Wikipedia / web search lookups used by the *_with_search
        algorithms, wiki_cache/ (or WIKI_CACHE_DIR)

Usage:
    # Export the LLM response cache to sort_cache_export.jsonl.gz
    python cache_tools.py export

    # Export with zstd (better compression, requires pyzstd)
    python cache_tools.py export --format zstd

    # Import from a compressed JSONL file
    python cache_tools.py import sort_cache_export.jsonl.gz

    # Import without overwriting existing entries
    python cache_tools.py import sort_cache_export.jsonl.gz --no-overwrite

    # The same for the wiki cache
    python cache_tools.py export --cache wiki
    python cache_tools.py import wiki_cache_export.jsonl.gz --cache wiki
"""

import argparse
import gzip
import itertools
import json
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

from order_by.cache import cache as sort_cache, open_wiki_cache


IMPORT_BATCH = 10_000  # entries written per transaction
# Import-time SQLite settings. Keys are hashes, so every insert lands at a random
# place in the key index; a page cache large enough to hold that index (~1 GB at
# the default 4 KB pages) keeps a multi-million-entry import from slowing to a
# crawl. Eviction checks are pointless while filling the cache, so they are off.
IMPORT_SETTINGS = {"cull_limit": 0, "sqlite_cache_size": 2**18}


def _cache(name: str):
    return sort_cache if name == "sort" else open_wiki_cache()


def _open_writer(output: str, fmt: str):
    if fmt == "zstd":
        import pyzstd
        return pyzstd.ZstdFile(output, "wb", level_or_option=10)
    return gzip.open(output, "wb", compresslevel=6)


def _open_reader(path: str):
    if path.endswith(".zst"):
        import pyzstd
        return pyzstd.ZstdFile(path, "rb")
    return gzip.open(path, "rb")


def export_cache(name: str, output: str | None, fmt: str):
    db_path = Path(_cache(name).directory) / "cache.db"

    if output is None:
        ext = "jsonl.zst" if fmt == "zstd" else "jsonl.gz"
        output = f"{name}_cache_export.{ext}"

    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM Cache").fetchone()[0]
    print(f"Exporting {total:,} entries from {db_path}")

    start = time.time()
    count = 0

    skipped = 0
    cursor = conn.execute("SELECT key, value FROM Cache")
    with _open_writer(output, fmt) as fh:
        for raw_key, raw_value in cursor:
            try:
                key = raw_key if isinstance(raw_key, str) else pickle.loads(raw_key)
                value = raw_value if isinstance(raw_value, (str, int, float)) else pickle.loads(raw_value)
                line = json.dumps({"key": key, "value": value}, ensure_ascii=False)
                fh.write(line.encode("utf-8"))
                fh.write(b"\n")
                count += 1
            except Exception:
                skipped += 1
            if (count + skipped) % 100_000 == 0:
                elapsed = time.time() - start
                pct = (count + skipped) / total * 100
                print(f"  {count:>10,} exported, {skipped:>6,} skipped / {total:,} ({pct:.1f}%)  [{elapsed:.0f}s]", flush=True)

    conn.close()
    elapsed = time.time() - start
    size_mb = os.path.getsize(output) / (1024 ** 2)
    print(f"Done: {count:,} exported, {skipped:,} skipped -> {output} ({size_mb:.1f} MB) in {elapsed:.0f}s")
    if skipped:
        print(f"  ({skipped:,} entries contained non-serializable objects and were skipped)")


def import_cache(name: str, input_path: str, overwrite: bool):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    cache = _cache(name)
    existing = len(cache)
    print(f"Importing into {cache.directory} (existing entries: {existing:,})")

    saved = {setting: getattr(cache, setting) for setting in IMPORT_SETTINGS}
    for setting, value in IMPORT_SETTINGS.items():
        cache.reset(setting, value)
    try:
        _import_entries(cache, input_path, overwrite)
    finally:
        for setting, value in saved.items():
            cache.reset(setting, value)
    print(f"Cache now has {len(cache):,} entries")


def _import_entries(cache, input_path: str, overwrite: bool):
    start = time.time()
    imported = 0
    skipped = 0

    with _open_reader(input_path) as fh:
        while True:
            batch = list(itertools.islice(fh, IMPORT_BATCH))
            if not batch:
                break
            with cache.transact():   # one transaction per batch: far faster than one per entry
                for line in batch:
                    entry = json.loads(line)
                    key = entry["key"]
                    val = entry["value"]
                    if not overwrite and key in cache:
                        skipped += 1
                    else:
                        cache[key] = val
                        imported += 1
                    if (imported + skipped) % 100_000 == 0:
                        elapsed = time.time() - start
                        print(f"  imported: {imported:>10,}  skipped: {skipped:>10,}  [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - start
    print(f"Done: imported {imported:,}, skipped {skipped:,} in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Export/import the LLM response cache or the wiki cache")
    sub = parser.add_subparsers(dest="command")

    exp = sub.add_parser("export", help="Export cache to compressed JSONL")
    exp.add_argument("--cache", choices=["sort", "wiki"], default="sort",
                     help="Which cache to export (default: sort, the LLM response cache)")
    exp.add_argument("-o", "--output", default=None, help="Output file path")
    exp.add_argument("--format", choices=["gzip", "zstd"], default="gzip",
                     help="Compression format (default: gzip)")

    imp = sub.add_parser("import", help="Import cache from compressed JSONL")
    imp.add_argument("file", help="Path to the compressed JSONL file")
    imp.add_argument("--cache", choices=["sort", "wiki"], default="sort",
                     help="Which cache to import into (default: sort, the LLM response cache)")
    imp.add_argument("--no-overwrite", action="store_true",
                     help="Skip entries that already exist in cache")

    args = parser.parse_args()
    if args.command == "export":
        export_cache(args.cache, args.output, args.format)
    elif args.command == "import":
        import_cache(args.cache, args.file, overwrite=not args.no_overwrite)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
