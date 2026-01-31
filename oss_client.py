# oss_client.py
import os
import aiohttp

DEFAULT_BASE_URL = "http://phi-4-predictor.tyf-ai-chatbot.svc.cluster.local:8080/v1"
DEFAULT_MODEL = "phi-4"

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

class LLMClient:
    """
    OpenAI-compatible /v1/chat/completions endpoint client.
    Works with vLLM (OpenAI-compatible), Groq, OpenRouter, etc.
    """

    def __init__(self):
        self.base_url = _env("VLLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model_name = _env("MODEL_NAME", DEFAULT_MODEL)
        self.api_key = _env("LLM_API_KEY", "")  # optional

        # Some providers (e.g., OpenRouter) like extra headers (optional)
        self.extra_headers = {}
        app_name = _env("OPENROUTER_APP_NAME", "")
        app_url = _env("OPENROUTER_APP_URL", "")
        if app_name:
            self.extra_headers["X-Title"] = app_name
        if app_url:
            self.extra_headers["HTTP-Referer"] = app_url

    async def generate(self, messages, max_tokens=400, temperature=0.5):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=120)  # 120 sn genelde yeterli

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        # UI'da “kırmızı error” basmamak için burada RuntimeError atıyoruz,
                        # ama streamlit tarafında bunu yakalayıp kullanıcıya yumuşak mesaj verebilirsin.
                        raise RuntimeError(f"LLM error {resp.status}: {text[:600]}")
                    data = await resp.json()

                # OpenAI-style
                return data["choices"][0]["message"]["content"]

            except aiohttp.ClientError as e:
                raise RuntimeError(f"LLM connection error: {e}")
