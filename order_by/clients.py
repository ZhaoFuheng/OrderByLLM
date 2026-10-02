"""LLM clients for every supported provider.

The sorting code only ever talks to the small AsyncOpenAI surface
(`chat.completions.create` and `beta.chat.completions.parse`), using a model's
provider-agnostic short name (e.g. "llama3.1-70b", "claude-haiku-4-5"). Each
builder below returns a client with that surface and rewrites the short name to
the provider's own model id on the wire. The response cache, pricing table and
result filenames stay keyed on the short name, so a model's cache is shared no
matter which provider served it.

Use `build_client(provider)`; the provider-specific builders document the
environment variables they read. The `cache` provider needs no credentials: it
answers only from the response cache and refuses to call any API.
"""
import asyncio
import json
import os

import httpx
from openai import AsyncOpenAI


def _strip_titles(obj):
    """Recursively remove 'title' keys from a JSON schema dict so it
    matches the clean format expected by Snowflake Cortex."""
    if isinstance(obj, dict):
        return {k: _strip_titles(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [_strip_titles(v) for v in obj]
    return obj


def pydantic_to_response_format(schema_cls) -> dict:
    """Convert a Pydantic model class to the Snowflake Cortex REST API
    response_format dict (json_schema type)."""
    schema = _strip_titles(schema_cls.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_cls.__name__,
            "schema": schema,
        },
    }


def cortex_extra_body(model: str) -> dict:
    """Return extra_body kwargs for the OpenAI chat completions call.
    Disables guardrails for Claude models on Snowflake Cortex."""
    if 'claude' in model.lower():
        return {"guardrails": False}
    return {}


class _SnowflakeAuth(httpx.Auth):
    """httpx auth handler that reads the live session token from the
    Snowflake connection on every request.  Proactively reconnects
    every hour, and also reconnects on 401 responses."""

    REFRESH_INTERVAL = 3600  # seconds (1 hour)

    def __init__(self, conn, connection_name: str):
        import time
        self.conn = conn
        self.connection_name = connection_name
        self._last_connect_time = time.monotonic()

    def _reconnect(self):
        import time
        import snowflake.connector
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = snowflake.connector.connect(
            connection_name=self.connection_name,
            client_session_keep_alive=True,
        )
        self._last_connect_time = time.monotonic()
        import logging
        logging.getLogger(__name__).info(
            "Snowflake session re-established for %s", self.connection_name)

    def auth_flow(self, request):
        import time
        if time.monotonic() - self._last_connect_time > self.REFRESH_INTERVAL:
            self._reconnect()

        token = self.conn.rest.token
        request.headers["Authorization"] = f'Snowflake Token="{token}"'
        response = yield request

        if response.status_code == 401:
            self._reconnect()
            token = self.conn.rest.token
            request.headers["Authorization"] = f'Snowflake Token="{token}"'
            yield request


# Map the provider-agnostic short model name (used as the sort_cache key,
# the tokens2price key, and the results_<model>.json filename) to the
# provider-specific model id each REST API expects.  Only the outbound
# request is rewritten — caching stays keyed on the short name so a model's
# cache is shared across providers (model-dependent, not provider-dependent).
FIREWORKS_MODEL_MAP = {
    # The dedicated Fireworks deployment of Llama-3.1-70B the experiments ran on.
    # Point FIREWORKS_LLAMA_70B_MODEL at your own deployment (or a serverless id
    # such as accounts/fireworks/models/llama-v3p1-70b-instruct) to override.
    "llama3.1-70b": "accounts/zhaofuheng1/deployments/llama31-70b",
}

HF_MODEL_MAP = {
    # HuggingFace Inference Router; ":featherless-ai" selects the serving provider.
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct:featherless-ai",
}

OPENAI_MODEL_MAP = {
    # Short name (Cortex-style, used for cache/pricing/filename) -> OpenAI API id.
    "openai-gpt-4.1": "gpt-4.1",
    "openai-gpt-5-nano": "gpt-5-nano",
    "openai-gpt-5-mini": "gpt-5-mini",
}

# OpenAI reasoning models (gpt-5*, o-series): take `reasoning_effort` and reject
# a non-default `temperature`, unlike gpt-4.1.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_openai_reasoning_model(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith(_OPENAI_REASONING_PREFIXES)


def _build_mapped_openai_client(
    base_url: str,
    api_key: str,
    model_map: dict,
    *,
    rename_max_tokens: bool = False,
    max_concurrency: int | None = None,
    strict_json_schema: bool = False,
    reasoning_effort: str | None = None,
) -> AsyncOpenAI:
    """Build an OpenAI-compatible AsyncOpenAI client whose chat.completions
    call rewrites the short model name to the provider's model id (so the
    sort_cache stays keyed on the short name and is shared across providers).

    rename_max_tokens: if True, translate max_completion_tokens ->
        max_tokens for providers that only accept the older param.
    max_concurrency: if set, a global semaphore caps the number of in-flight
        requests through this client — the sorting algorithms fan out many
        concurrent LLM calls, which exceeds strict per-user concurrency caps
        (e.g. HF/featherless allows only ~2 concurrent requests).
    strict_json_schema: if True, upgrade a json_schema response_format to
        OpenAI Structured Outputs (add strict:true and normalize the schema to
        additionalProperties:false + all-required) so the JSON is guaranteed to
        conform — the Cortex-shaped response_format lacks these.
    reasoning_effort: if set (e.g. "minimal"/"low"), applied to OpenAI reasoning
        models (gpt-5*, o-series) via the chat-completions reasoning_effort
        param; temperature is dropped for those models (they reject a non-default
        temperature). Ignored for non-reasoning models like gpt-4.1.
    """
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=60.0, read=900.0, write=60.0, pool=60.0),
    )
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    def _transform(kwargs):
        short = kwargs.get("model")
        if short in model_map:
            kwargs["model"] = model_map[short]
        # Cortex-only guardrails knob is not accepted by other providers.
        eb = kwargs.get("extra_body")
        if isinstance(eb, dict) and "guardrails" in eb:
            eb = {k: v for k, v in eb.items() if k != "guardrails"}
            kwargs["extra_body"] = eb or None
        if rename_max_tokens and "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        if reasoning_effort and _is_openai_reasoning_model(kwargs.get("model", "")):
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs.pop("temperature", None)  # reasoning models reject non-default temperature
        if strict_json_schema:
            # Only dict json_schema response_format (create); parse() passes a
            # pydantic class and handles structured output itself.
            rf = kwargs.get("response_format")
            if isinstance(rf, dict) and rf.get("type") == "json_schema":
                js = dict(rf["json_schema"])
                js["strict"] = True
                js["schema"] = _normalize_schema_for_strict(js["schema"])
                kwargs["response_format"] = {"type": "json_schema", "json_schema": js}
        return kwargs

    def _wrap(orig):
        async def _wrapped(*args, **kwargs):
            kwargs = _transform(kwargs)
            if sem is not None:
                async with sem:
                    return await orig(*args, **kwargs)
            return await orig(*args, **kwargs)
        return _wrapped

    # chat.completions.create — the sorting algorithms' path.
    client.chat.completions.create = _wrap(client.chat.completions.create)  # type: ignore[assignment]
    # beta.chat.completions.parse — the web_search tools' path (search variants).
    try:
        client.beta.chat.completions.parse = _wrap(client.beta.chat.completions.parse)  # type: ignore[assignment]
    except AttributeError:
        pass
    return client


def build_fireworks_client() -> AsyncOpenAI:
    """AsyncOpenAI client for the Fireworks REST API.  OpenAI-compatible
    (same chat.completions + json_schema response_format as Cortex); only the
    model id differs, which _build_mapped_openai_client rewrites on the wire.

    A concurrency gate defaults ON (file-descriptor safety; the sorting
    algorithms fan out heavily — pooled HellaSwag = 400 docs/query). The default
    is high (256) because a dedicated Fireworks deployment can serve many
    concurrent requests.

    Env vars : FIREWORKS_API_KEY (falls back to OPENAI_API_KEY)
               FIREWORKS_BASE_URL (default https://api.fireworks.ai/inference/v1)
               FIREWORKS_MAX_CONCURRENCY (default 256; set 0 to disable the gate)
               FIREWORKS_LLAMA_70B_MODEL (overrides the llama3.1-70b model id)
    """
    api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "FIREWORKS_API_KEY (or OPENAI_API_KEY) is required in environment or .env"
        )
    base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    mc = int(os.getenv("FIREWORKS_MAX_CONCURRENCY", "256"))
    model_map = dict(FIREWORKS_MODEL_MAP)
    if os.getenv("FIREWORKS_LLAMA_70B_MODEL"):
        model_map["llama3.1-70b"] = os.environ["FIREWORKS_LLAMA_70B_MODEL"]
    return _build_mapped_openai_client(
        base_url, api_key, model_map,
        max_concurrency=mc if mc > 0 else None,
    )


def build_hf_client() -> AsyncOpenAI:
    """AsyncOpenAI client for the HuggingFace Inference Router.
    OpenAI-compatible; the router uses max_tokens rather than
    max_completion_tokens, so that param is translated on the wire.

    HF/featherless enforces a strict per-user concurrency cap (~10 units,
    ~4 units per request => ~2 concurrent requests), so requests are gated
    by a global semaphore (default 2, override with HF_MAX_CONCURRENCY).

    Env vars : HF_TOKEN
               HF_BASE_URL (default https://router.huggingface.co/v1)
               HF_MAX_CONCURRENCY (default 2)
    """
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN is required in environment or .env")
    base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    max_concurrency = int(os.getenv("HF_MAX_CONCURRENCY", "2"))
    return _build_mapped_openai_client(
        base_url, api_key, HF_MODEL_MAP,
        rename_max_tokens=True, max_concurrency=max_concurrency,
    )


def build_openai_client() -> AsyncOpenAI:
    """AsyncOpenAI client for the OpenAI API (GPT-4.1). Natively OpenAI-shaped,
    so it reuses the shared helper; structured output is upgraded to OpenAI
    Structured Outputs (strict json_schema) so scores parse reliably. A
    concurrency gate defaults ON (rate-limit + file-descriptor safety; the
    sorting algorithms fan out heavily — pooled HellaSwag = 400 docs/query).

    Env vars : OPENAI_API_KEY
               OPENAI_BASE_URL (default https://api.openai.com/v1 — make sure it
                   is not pointed at another provider)
               OPENAI_MAX_CONCURRENCY (default 128; set 0 to disable the gate)
               OPENAI_REASONING_EFFORT (default "minimal"; applied only to
                   reasoning models like gpt-5-nano, ignored for gpt-4.1)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required in environment or .env")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    mc = int(os.getenv("OPENAI_MAX_CONCURRENCY", "128"))
    return _build_mapped_openai_client(
        base_url, api_key, OPENAI_MODEL_MAP,
        strict_json_schema=True,
        max_concurrency=mc if mc > 0 else None,
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "minimal"),
    )


ANTHROPIC_MODEL_MAP = {
    # Short name (cache/pricing/filename key) -> Anthropic Messages API model id.
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
}


def _normalize_schema_for_strict(schema):
    """Recursively enforce the constraints Anthropic strict tool use requires:
    every object sets additionalProperties=false and lists all properties as
    required. The pointwise/comparison schemas are flat (str/float/list), so
    this is a straightforward walk."""
    if isinstance(schema, dict):
        out = {k: _normalize_schema_for_strict(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [_normalize_schema_for_strict(v) for v in schema]
    return schema


class _OAIShapedUsage:
    def __init__(self, in_t, out_t):
        self.input_tokens = in_t
        self.prompt_tokens = in_t
        self.completion_tokens = out_t
        self.total_tokens = in_t + out_t


class _OAIShapedResponse:
    """Just enough of the OpenAI chat-completion response shape for the sorting
    code: response.choices[0].message.content and response.usage.*"""
    def __init__(self, content, in_t, out_t):
        message = type("_Msg", (), {"content": content})()
        choice = type("_Choice", (), {"message": message})()
        self.choices = [choice]
        self.usage = _OAIShapedUsage(in_t, out_t)


class _OAIShapedParsedResponse:
    """Shape for beta.chat.completions.parse(): choices[0].message.parsed."""
    def __init__(self, parsed, in_t, out_t):
        message = type("_Msg", (), {"parsed": parsed})()
        choice = type("_Choice", (), {"message": message})()
        self.choices = [choice]
        self.usage = _OAIShapedUsage(in_t, out_t)


class _AnthropicCompletionsAdapter:
    def __init__(self, client, model_map, sem):
        self._client = client
        self._model_map = model_map
        self._sem = sem

    async def create(self, *, model, messages, response_format=None,
                     max_completion_tokens=None, max_tokens=None,
                     temperature=None, extra_body=None, **_ignored):
        model = self._model_map.get(model, model)

        # OpenAI system/user messages -> Anthropic system param + user/assistant turns.
        system_parts, turns = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role in ("system", "developer"):
                system_parts.append(content if isinstance(content, str) else str(content))
            else:
                turns.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content,
                })
        system_text = "\n".join(p for p in system_parts if p)

        kwargs = {
            "model": model,
            "max_tokens": max_tokens or max_completion_tokens or 4096,
            "messages": turns,
        }
        if system_text:
            kwargs["system"] = system_text
        if temperature is not None:
            kwargs["temperature"] = temperature

        # response_format(json_schema) -> forced strict tool call (native
        # structured output). Guarantees schema-conformant JSON we can parse.
        tool_name = None
        if response_format and response_format.get("type") == "json_schema":
            js = response_format["json_schema"]
            tool_name = js.get("name") or "structured_output"
            kwargs["tools"] = [{
                "name": tool_name,
                "description": "Return the result as JSON matching the schema.",
                "input_schema": _normalize_schema_for_strict(js["schema"]),
                "strict": True,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        if self._sem is not None:
            async with self._sem:
                resp = await self._client.messages.create(**kwargs)
        else:
            resp = await self._client.messages.create(**kwargs)

        in_t = getattr(resp.usage, "input_tokens", 0) or 0
        out_t = getattr(resp.usage, "output_tokens", 0) or 0

        content = None
        if getattr(resp, "stop_reason", None) != "refusal":
            if tool_name is not None:
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        content = json.dumps(block.input)
                        break
            else:
                texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                content = "".join(texts) if texts else None
        return _OAIShapedResponse(content, in_t, out_t)

    async def parse(self, *, model, messages, response_format,
                    max_completion_tokens=None, max_tokens=None,
                    temperature=None, **_ignored):
        """beta.chat.completions.parse() equivalent: response_format is a
        pydantic class. Reuses create()'s forced-tool structured output, then
        parses the JSON into the class and returns .choices[0].message.parsed."""
        rf = pydantic_to_response_format(response_format)
        resp = await self.create(
            model=model, messages=messages, response_format=rf,
            max_completion_tokens=max_completion_tokens, max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        parsed = response_format(**json.loads(content)) if content else None
        return _OAIShapedParsedResponse(parsed, resp.usage.input_tokens, resp.usage.completion_tokens)


class AnthropicOpenAIAdapter:
    """Backs the subset of the AsyncOpenAI surface the sorting code uses
    (`.chat.completions.create`) with the native Anthropic Messages API, so the
    algorithms, sort_cache, and pricing stay provider-agnostic. Anthropic's own
    OpenAI-compat endpoint can't be used here because it ignores response_format."""
    def __init__(self, anthropic_client, model_map=None, max_concurrency=None):
        self._anthropic = anthropic_client
        sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        completions = _AnthropicCompletionsAdapter(anthropic_client, model_map or {}, sem)
        # .chat.completions.create (sorting) and .beta.chat.completions.parse
        # (optimizer inquiry/judge + web_search) both use the same adapter object.
        self.chat = type("_Chat", (), {"completions": completions})()
        self.beta = type("_Beta", (), {
            "chat": type("_BChat", (), {"completions": completions})(),
        })()


def build_anthropic_client() -> "AnthropicOpenAIAdapter":
    """Native Anthropic Messages API, wrapped in AnthropicOpenAIAdapter so the
    sorting code is unchanged. Structured output is done via a forced strict
    tool call (response_format is translated on the fly).

    Env vars : ANTHROPIC_API_KEY
               ANTHROPIC_MAX_RETRIES (default 4)
               ANTHROPIC_MAX_CONCURRENCY (default 128; set 0 to disable the gate)

    The concurrency gate defaults ON: sorting algorithms fan out huge numbers of
    concurrent calls (e.g. pooled HellaSwag = 800 candidates/query × several
    queries), which without a cap opens thousands of sockets and exhausts file
    descriptors — SQLite then fails with "unable to open database file" and 429s
    spike. 128 keeps it bounded; raise/lower via the env var.
    """
    from anthropic import AsyncAnthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required in environment or .env")
    max_retries = int(os.getenv("ANTHROPIC_MAX_RETRIES", "4"))
    client = AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=900.0)
    mc = int(os.getenv("ANTHROPIC_MAX_CONCURRENCY", "128"))
    return AnthropicOpenAIAdapter(
        client, model_map=ANTHROPIC_MODEL_MAP,
        max_concurrency=mc if mc > 0 else None,
    )


def build_cortex_client(connection_name: str | None = None) -> AsyncOpenAI:
    """AsyncOpenAI client for the Snowflake Cortex REST API (OpenAI-compatible).

    Two ways to authenticate:

    * Session token -- pass `connection_name` (or set SNOWFLAKE_CONNECTION) to a
      profile in ~/.snowflake/connections.toml. A custom httpx auth handler reads
      the live session token on every request, so a token refreshed by the
      Snowflake connector is picked up automatically; on 401 it reconnects and
      retries once. Needs `snowflake-connector-python`.
    * Programmatic access token -- with no connection name, OPENAI_API_KEY is
      sent as the bearer token.

    Env vars : OPENAI_BASE_URL (e.g. .../api/v2/cortex/v1)
               OPENAI_API_KEY (access-token authentication only)
               SNOWFLAKE_CONNECTION (session-token authentication only)
    """
    connection_name = connection_name or os.getenv("SNOWFLAKE_CONNECTION")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    if not connection_name:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in environment or .env")
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    import snowflake.connector

    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required in environment or .env")

    conn = snowflake.connector.connect(
        connection_name=connection_name,
        client_session_keep_alive=True,
    )
    token = conn.rest.token
    if not token:
        conn.close()
        raise RuntimeError(
            f"Snowflake connection '{connection_name}' did not yield an access token."
        )

    auth = _SnowflakeAuth(conn, connection_name)
    http_client = httpx.AsyncClient(
        auth=auth,
        timeout=httpx.Timeout(connect=60.0, read=900.0, write=60.0, pool=60.0),
    )
    client = AsyncOpenAI(
        api_key="not-used",
        base_url=base_url,
        http_client=http_client,
    )
    client._snowflake_auth = auth  # type: ignore[attr-defined]
    return client


class RealCall(BaseException):
    """Raised when an algorithm attempts a real API call through a
    NoNetworkClient, i.e. on a cache miss. A BaseException so the retry loops'
    `except Exception` cannot swallow it."""


class _RefusingCompletions:
    def __init__(self, client):
        self._client = client

    async def create(self, *args, **kwargs):
        self._client.misses += 1
        raise RealCall(f"response not cached (model={kwargs.get('model')})")

    parse = create


class NoNetworkClient:
    """Stands in for an LLM client and refuses every call. Cached responses are
    returned before a client is ever consulted, so with this client a run either
    comes entirely from the response cache or stops at the first miss — it can
    never spend money. `misses` counts the refused calls."""

    def __init__(self):
        self.misses = 0
        completions = _RefusingCompletions(self)
        self.chat = type("_Chat", (), {"completions": completions})()
        self.beta = type("_Beta", (), {"chat": self.chat})()


def run_async(coro, client):
    """asyncio.run(coro), turning a cache miss under the `cache` provider into a
    short explanation instead of a traceback."""
    try:
        return asyncio.run(coro)
    except BaseException:
        if getattr(client, "misses", 0):
            raise SystemExit(
                "Stopped: running with --provider cache, but this run needs LLM "
                "responses that are not in the response cache. No results were "
                "written. Import the cache (python cache_tools.py import ...) or "
                "use a real provider."
            ) from None
        raise


_BUILDERS = {
    "cache": NoNetworkClient,
    "cortex": build_cortex_client,
    "fireworks": build_fireworks_client,
    "hf": build_hf_client,
    "anthropic": build_anthropic_client,
    "openai": build_openai_client,
}

# Providers selectable with --provider. Model names stay the provider-agnostic
# short names everywhere, so the response cache is shared across providers
# (which is why `cache`, the no-network client, can stand in for any of them).
PROVIDERS = tuple(_BUILDERS)
PROVIDER_HELP = (
    "LLM provider/client (default: cortex). 'cache' answers only from the "
    "response cache and never calls an API; 'cortex', 'openai', 'anthropic', "
    "'fireworks' and 'hf' call that service. Model names stay the same across "
    "providers, so the response cache is shared."
)


def build_client(provider: str = "cortex"):
    """Build the LLM client for `provider` (one of PROVIDERS). Every client
    exposes the AsyncOpenAI surface the sorting code uses."""
    if provider not in _BUILDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")
    return _BUILDERS[provider]()


def reasoning_tokens(response) -> int:
    """Reasoning tokens for OpenAI reasoning models (gpt-5*, o-series). This is a
    SUBSET of output_tokens (already counted in cost); cached separately so the
    reasoning overhead can be reported. 0 when the provider doesn't report it."""
    try:
        ctd = getattr(response.usage, "completion_tokens_details", None)
        return int(getattr(ctd, "reasoning_tokens", 0) or 0)
    except Exception:
        return 0
