import aiohttp


VLLM_BASE_URL = "http://phi-4-predictor.tyf-ai-chatbot.svc.cluster.local:8080/v1"
MODEL_NAME = "phi-4"


class MistralClient:

    def __init__(self, base_url=VLLM_BASE_URL, model_name=MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def generate(self, messages, max_tokens=400, temperature=0.5):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        timeout = aiohttp.ClientTimeout(total=12000)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Mistral API Error {resp.status}: {error}")
                
                data = await resp.json()
                return data["choices"][0]["message"]["content"]