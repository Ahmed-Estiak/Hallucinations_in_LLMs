import os
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load local environment variables before reading provider credentials.
# This lets developers keep API keys in a .env file during local runs.
load_dotenv()


def _is_blackhole_proxy(value: str | None) -> bool:
    """Return true for the localhost:9 proxy used by some sandboxed shells."""
    if not value:
        return False
    parsed = urlparse(value if "://" in value else f"http://{value}")
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9


def _clear_blackhole_proxies() -> None:
    """Prevent provider SDKs from inheriting a known-bad local proxy.

    The user's normal terminal may not have this proxy, but automated/tool
    shells can inject HTTP(S)_PROXY=http://127.0.0.1:9. OpenAI/Gemini SDKs then
    fail with APIConnectionError before reaching the network. Only this exact
    blackhole proxy is removed; any real proxy setting is preserved.
    """
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        if _is_blackhole_proxy(os.getenv(name)):
            os.environ.pop(name, None)


_clear_blackhole_proxies()

# API keys are read once at import time and reused by modules that import config.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Keep startup non-blocking when keys are missing; downstream code can decide
# whether a missing provider credential should fail a specific operation.
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found")
