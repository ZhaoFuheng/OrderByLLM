import asyncio
import json
import logging
import os
import re
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

from ddgs import DDGS
from diskcache import Cache
from pydantic import BaseModel

from ..utils import count_tokens, hash_prompt, create_numbered_passages, create_numbered_SQLs, create_numbered_reviews
from prompts.all_prompts import web_search_system_prompt as _WEB_POINTWISE_SYSTEM_PROMPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
cache = Cache(os.path.join(PROJECT_ROOT, "sort_cache"), size_limit=50 * 1024**3, eviction_policy='least-recently-used')
_wiki_cache = Cache(os.path.join(PROJECT_ROOT, "wiki_cache"), size_limit=2 * 1024**3, eviction_policy='least-recently-used')

_CACHE_VERSION = "v5"
_WIKI_CACHE_VERSION = "v2"


class WebSearchPointwiseResult(BaseModel):
    explanation: str
    value: float


class WebSearchExternalResult(BaseModel):
    explanation: str
    values: list[float]


# ── Wikipedia helpers ─────────────────────────────────────────────────────────

def _fetch_page_info(entity: str) -> tuple[str, str] | None:
    """Return (title, extract) for the best-matching Wikipedia page.

    Tries a direct title lookup first (fast, avoids rate-limited search API),
    then falls back to the search API. Returns None on complete failure.
    """
    title_slug = urllib.parse.quote(entity.replace(" ", "_"))

    def _from_summary(slug: str):
        req = urllib.request.Request(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            headers={"User-Agent": "OrderByLLM/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            page = json.loads(resp.read())
        if page.get("type") == "disambiguation":
            return None
        return page.get("title", ""), page.get("extract", "")

    try:
        result = _from_summary(title_slug)
        if result:
            return result
    except Exception:
        pass

    try:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": entity, "format": "json", "srlimit": 1,
        })
        req = urllib.request.Request(
            f"https://en.wikipedia.org/w/api.php?{params}",
            headers={"User-Agent": "OrderByLLM/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return None
        slug = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
        return _from_summary(slug)
    except Exception:
        return None


def _fetch_infobox_field(title: str, field: str) -> str | None:
    """Extract a field value from a Wikipedia infobox using regex.

    Fetches section 0 HTML via the MediaWiki parse API and searches for the
    pattern ``{field}</th> <td>VALUE</td>``.

    If the cell contains a parenthesised metric value like "(2.01 m)", only
    that metric part is returned for a clean, unambiguous context string.
    Returns None if the field is not present in the infobox.
    """
    title_enc = urllib.parse.quote(title.replace(" ", "_"))
    params = urllib.parse.urlencode({
        "action": "parse", "page": title_enc,
        "prop": "text", "section": "0", "format": "json",
    })
    req = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "OrderByLLM/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    html = data.get("parse", {}).get("text", {}).get("*", "")

    pat = rf'{re.escape(field)}</th>\s*<td[^>]*>(.*?)</td>'
    m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    clean = re.sub(r"<[^>]+>", "", m.group(1))
    clean = clean.replace("&#160;", " ").replace("&nbsp;", " ").strip()
    if not clean:
        return None

    # Prefer the parenthesised metric value when present (e.g. "6 ft 7 in (2.01 m)" → "2.01 m")
    metric = re.search(r"\(([\d.]+)\s*m\)", clean)
    return f"{metric.group(1)} m" if metric else clean


def wiki_search(entity: str, wiki_field: str | None = None, max_chars: int = 500) -> str:
    cache_key = f"wiki_{_WIKI_CACHE_VERSION}:{entity}|field:{wiki_field or ''}"
    chars = 0 if wiki_field else max_chars
    if cache_key in _wiki_cache:
        cached = _wiki_cache[cache_key]
        if cached is None:
            return ""
        return f"{cached['header']} {cached['extract'][:chars]}".strip()

    try:
        page_info = _fetch_page_info(entity)
        if not page_info:
            _wiki_cache[cache_key] = None
            return ""
        title, extract = page_info

        field_value = None
        if wiki_field:
            try:
                field_value = _fetch_infobox_field(title, wiki_field)
            except Exception:
                pass
            if field_value is None:
                _wiki_cache[cache_key] = None
                return ""

        header = f"Wikipedia — {title}"
        if wiki_field and field_value:
            header += f" [{wiki_field}: {field_value}]"
        header += ":"

        _wiki_cache[cache_key] = {"header": header, "extract": extract}
        return f"{header} {extract[:chars]}".strip()

    except Exception:
        return ""


# ── Pointwise value scoring ───────────────────────────────────────────────────

async def web_search_pointwise_value(
    client,
    modelname: str,
    prompt: str,
    wiki_entity: str,
    wiki_field: str | None = None,
):
    """Return a float score for a single item, optionally grounded by Wikipedia.

    Parameters
    ----------
    wiki_entity:
        Bare entity name used for the Wikipedia lookup (e.g. "LeBron James").
    wiki_field:
        Infobox label to extract (e.g. "Listed height").
        If found: the prompt is augmented with that fact as context.
        If not found or omitted: the raw prompt is sent so the LLM uses its
        own knowledge.
    """
    field_tag = f"[field={wiki_field}]" if wiki_field else ""
    cache_key = f"[web_search_{_CACHE_VERSION}][wiki={wiki_entity}]{field_tag}{prompt}"
    key_hash = hash_prompt(cache_key, modelname)

    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = WebSearchPointwiseResult(**cached["parsed"])
            input_tokens = cached.get("input_tokens", count_tokens(prompt))
            output_tokens = cached["tokens"] - input_tokens
            return parsed.value, 0, input_tokens, output_tokens
        except Exception:
            del cache[key_hash]

    wiki_context = wiki_search(wiki_entity, wiki_field=wiki_field)

    if wiki_context:
        augmented_prompt = (
            f"Use the following context.\n\n"
            f"Context:\n{wiki_context}\n\n"
            f"{prompt}"
        )
    else:
        augmented_prompt = prompt

    for attempt in range(1, 11):
        try:
            response = await client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt},
                ],
                temperature=0.0,
                response_format=WebSearchPointwiseResult,
                max_completion_tokens=8192,
            )
            break
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = 30.0 * attempt
                log.warning("web_search_pointwise_value %s: rate limited (attempt %d/10), sleeping %.0fs",
                            modelname, attempt, wait)
                await asyncio.sleep(wait)
            else:
                log.error("web_search_pointwise_value %s [ERROR] attempt %d: %s", modelname, attempt, e)
                return 0.0, 1, 0, 0
    else:
        log.error("web_search_pointwise_value %s: all 10 retries exhausted", modelname)
        return 0.0, 1, 0, 0

    parsed = response.choices[0].message.parsed
    input_tokens = (
        getattr(response.usage, "input_tokens", None)
        or getattr(response.usage, "prompt_tokens", None)
        or count_tokens(augmented_prompt)
    )
    output_tokens = response.usage.total_tokens - input_tokens
    cache[key_hash] = {
        "parsed": parsed.model_dump(),
        "tokens": response.usage.total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return float(parsed.value), 1, input_tokens, output_tokens


# ── External pointwise scoring ───────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5, max_chars_per_result: int = 300) -> str:
    """Return DuckDuckGo search results as a text snippet."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception:
        return "No results found."
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['body'][:max_chars_per_result]}" for r in results)


def ddg_search_cached(query: str, max_results: int = 5, max_chars_per_result: int = 300) -> str:
    """Cached DuckDuckGo search — same query always returns the same result."""
    cache_key = f"ddg_v1:{query}:{max_results}:{max_chars_per_result}"
    if cache_key in _wiki_cache:
        return _wiki_cache[cache_key]
    result = _ddg_search(query, max_results=max_results, max_chars_per_result=max_chars_per_result)
    _wiki_cache[cache_key] = result
    return result


def _extract_vals(parsed, schema):
    for attr in ("values", "relevance_scores", "correctness_scores", "review_scores"):
        if hasattr(parsed, attr):
            return getattr(parsed, attr)
    return []


async def wiki_search_external_values(data, client, prompt_template, modelname, output_type, schema, wiki_field=None):
    """Score a batch of items using Wikipedia infobox-augmented prompts.

    For each item in the batch, fetches the ``wiki_field`` value from Wikipedia
    (e.g. "Listed height"). Found values are prepended as context; items with no
    data get a "Not found" note so the LLM falls back to its own knowledge.
    """
    base_prompt = prompt_template.format(keys=str(data))

    contexts = [wiki_search(str(item), wiki_field=wiki_field) for item in data]
    context_lines = [
        f"[Item {i+1}] {ctx}" if ctx else f"[Item {i+1}] No data found."
        for i, ctx in enumerate(contexts)
    ]
    combined_context = "\n".join(context_lines)

    augmented_prompt = (
        f"Use the following Wikipedia context for each item.\n\n"
        f"Context per item:\n{combined_context}\n\n"
        f"{base_prompt}"
    )

    field_tag = f"[field={wiki_field}]" if wiki_field else ""
    cache_key = f"[wiki_search_ext_{_CACHE_VERSION}]{field_tag}{augmented_prompt}"
    key_hash = hash_prompt(cache_key, modelname)
    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = schema(**cached["parsed"])
            vals = _extract_vals(parsed, schema)
            input_tokens = cached.get("input_tokens", count_tokens(augmented_prompt))
            output_tokens = cached["tokens"] - input_tokens
            if len(vals) == len(data):
                return [output_type(v) for v in vals], 0, input_tokens, output_tokens
            del cache[key_hash]
        except Exception:
            del cache[key_hash]

    for attempt in range(1, 11):
        # From attempt 2 onward, remind the model of the required output length.
        suffix = (
            f"\nRespond in JSON. The values list must contain exactly {len(data)} floats, one per item.\n"
            if attempt > 1 else ""
        )
        temperature = 0.0 if attempt <= 5 else 0.3
        try:
            response = await client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt + suffix},
                ],
                temperature=temperature,
                response_format=schema,
                max_completion_tokens=8192,
            )
            parsed = response.choices[0].message.parsed
            vals = _extract_vals(parsed, schema)
            input_tokens = (
                getattr(response.usage, "input_tokens", None)
                or getattr(response.usage, "prompt_tokens", None)
                or count_tokens(augmented_prompt)
            )
            output_tokens = response.usage.total_tokens - input_tokens

            if len(vals) == len(data):
                cache[key_hash] = {
                    "parsed": parsed.model_dump(),
                    "tokens": response.usage.total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                return [output_type(v) for v in vals], 1, input_tokens, output_tokens

            log.warning("wiki_search_external_values %s: length mismatch attempt %d/10, got %d expected %d",
                        modelname, attempt, len(vals), len(data))
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = 30.0 * attempt
                log.warning("wiki_search_external_values %s: rate limited (attempt %d/10), sleeping %.0fs",
                            modelname, attempt, wait)
                await asyncio.sleep(wait)
            else:
                log.error("wiki_search_external_values %s [ERROR] attempt %d: %s", modelname, attempt, e)

    log.error("wiki_search_external_values %s: all 10 retries exhausted, returning zeros", modelname)
    return [output_type(0) for _ in data], 1, 0, 0


async def web_search_external_values(data, client, prompt_template, modelname, output_type, schema):
    """Score a batch of items using DuckDuckGo-augmented prompts via chat completions."""
    from ..pointwise import (
        PassageExternalPointwiseReasoning,
        SQLExternalPointwiseReasoning,
        ReviewExternalPointwiseReasoning,
    )

    if schema == PassageExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_passages(data)))
    elif schema == SQLExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_SQLs(data)))
    elif schema == ReviewExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_reviews(data)))
    else:
        base_prompt = prompt_template.format(keys=str(data))

    contexts = [_ddg_search(str(item)[:150], max_results=3) for item in data]
    combined_context = "\n\n".join(f"[Item {i+1}] {ctx}" for i, ctx in enumerate(contexts))

    augmented_prompt = (
        f"Use the following web search results as context for each item.\n\n"
        f"Web context per item:\n{combined_context}\n\n"
        f"{base_prompt}"
    )

    cache_key = f"[web_search_ext]{augmented_prompt}"
    key_hash = hash_prompt(cache_key, modelname)
    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = schema(**cached["parsed"])
            vals = _extract_vals(parsed, schema)
            input_tokens = cached.get("input_tokens", count_tokens(augmented_prompt))
            output_tokens = cached["tokens"] - input_tokens
            if len(vals) == len(data):
                return [output_type(v) for v in vals], 0, input_tokens, output_tokens
            del cache[key_hash]
        except Exception:
            del cache[key_hash]

    response = await client.beta.chat.completions.parse(
        model=modelname,
        messages=[
            {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ],
        temperature=0.0,
        response_format=schema,
        max_completion_tokens=8192,
    )
    parsed = response.choices[0].message.parsed
    vals = _extract_vals(parsed, schema)
    input_tokens = getattr(response.usage, "prompt_tokens", None) or count_tokens(augmented_prompt)
    output_tokens = response.usage.total_tokens - input_tokens

    if len(vals) == len(data):
        cache[key_hash] = {
            "parsed": parsed.model_dump(),
            "tokens": response.usage.total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    else:
        print(f"web_search_external_values: length mismatch, got {len(vals)}, expected {len(data)}")
        vals = [0.0] * len(data)

    return [output_type(v) for v in vals], 1, input_tokens, output_tokens
