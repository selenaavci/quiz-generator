import os
import aiohttp


VLLM_BASE_URL = "http://phi-4-predictor.tyf-ai-chatbot.svc.cluster.local:8080/v1"
MODEL_NAME = "phi-4"


class MistralClient:

    def __init__(self, base_url=None, model_name=None, api_key=None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or VLLM_BASE_URL).rstrip("/")
        self.model_name = model_name or os.environ.get("LLM_MODEL") or MODEL_NAME
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""

    async def generate(self, messages, max_tokens=400, temperature=0.5):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=12000)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"LLM API Error {resp.status}: {error}")

                data = await resp.json()
                return data["choices"][0]["message"]["content"]
