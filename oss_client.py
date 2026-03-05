# oss_client.py
import os
import asyncio
import aiohttp


# -------- Defaults (local cluster fallback) --------
DEFAULT_BASE_URL = "http://phi-4-predictor.tyf-ai-chatbot.svc.cluster.local:8080/v1"
DEFAULT_MODEL = "phi-4"


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default


class MistralClient:
    def __init__(self):
        self.base_url = _env("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model_name = _env("LLM_MODEL", DEFAULT_MODEL)
        self.api_key = _env("LLM_API_KEY", "")

        # Optional OpenRouter headers (safe for others)
        self.extra_headers = {}
        app_name = _env("OPENROUTER_APP_NAME", "")
        app_url = _env("OPENROUTER_APP_URL", "")
        if app_name:
            self.extra_headers["X-Title"] = app_name
        if app_url:
            self.extra_headers["HTTP-Referer"] = app_url

    async def generate(
        self,
        messages,
        max_tokens: int = 700,
        temperature: float = 0.5,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        # OpenRouter / hosted providers
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=120)

        max_retries = 5
        for attempt in range(max_retries):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        raw_text = await resp.text()

                        if resp.status == 429 and attempt < max_retries - 1:
                            wait = 2 ** attempt + 1
                            await asyncio.sleep(wait)
                            continue

                        if resp.status != 200:
                            raise RuntimeError(
                                f"LLM error {resp.status}: {raw_text[:600]}"
                            )

                        try:
                            data = await resp.json()
                        except Exception:
                            raise RuntimeError(
                                f"LLM response parse error: {raw_text[:600]}"
                            )

                    content = data["choices"][0]["message"]["content"]
                    # Some models don't support json_object response_format and return null
                    if content is None and "response_format" in payload:
                        del payload["response_format"]
                        continue
                    return content

                except aiohttp.ClientError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt + 1)
                        continue
                    raise RuntimeError(f"LLM connection error: {e}")

        raise RuntimeError("LLM max retries exceeded (429 rate limit)")
