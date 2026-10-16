"""Small OpenRouter REST helpers: model-catalog check and account spend.

Local context: two GET endpoints, called with httpx.
- `GET /api/v1/models` lists every routable model id (https://openrouter.ai/docs/api-reference/list-available-models)
- `GET /api/v1/key` returns the API key's cumulative `usage` in USD (https://openrouter.ai/docs/api-reference/get-current-api-key)

Global context: phase 5 (`check_models_available`) fails fast on a mistyped or retired model id before any
money is spent; phases 7-8 read `usage` before/after runs to enforce MAX_BUDGET_USD.
Catalog pattern adapted from the sibling project llms-take-autism-test (openrouter.fetch_model_catalog).
"""

from __future__ import annotations

import os

import httpx  # HTTP client already used by inspect_ai and openai

BASE_URL = "https://openrouter.ai/api/v1"


def fetch_catalog() -> dict[str, dict]:
    """{model id: model record} for every model OpenRouter lists (no authentication needed)."""
    response = httpx.get(f"{BASE_URL}/models", timeout=60.0)
    response.raise_for_status()  # turn HTTP errors into exceptions
    return {model["id"]: model for model in response.json()["data"]}


def fetch_model_ids() -> set[str]:
    """All model ids currently listed by OpenRouter."""
    return set(fetch_catalog())


def fetch_prices() -> dict[str, tuple[float, float]]:
    """{model id: (USD per input token, USD per output token)} from the catalog's `pricing` field."""
    return {
        model_id: (float(m["pricing"].get("prompt") or 0), float(m["pricing"].get("completion") or 0))
        for model_id, m in fetch_catalog().items()
        if m.get("pricing")
    }


def missing_models(wanted: list[str], available: set[str]) -> list[str]:
    """Ids from `wanted` that OpenRouter does not list."""
    return [model for model in wanted if model not in available]


def check_models_available(models: list[str]) -> None:
    """Raise if any model id is not routable on OpenRouter right now."""
    missing = missing_models(models, fetch_model_ids())
    if missing:
        raise RuntimeError(f"Not available on OpenRouter: {missing}")


def key_usage_usd() -> float:
    """Cumulative spend (USD) of the configured OPENROUTER_API_KEY, from `GET /api/v1/key` -> data.usage."""
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    response = httpx.get(f"{BASE_URL}/key", headers=headers, timeout=60.0)
    response.raise_for_status()
    return float(response.json()["data"]["usage"])
