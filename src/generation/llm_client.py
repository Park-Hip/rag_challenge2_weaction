import json
import httpx
from httpx import AsyncClient
from langfuse import observe, get_client

from src.core.config import settings
from src.core.logger import logger

langfuse = get_client()

class LLMClient():
    """
    """

    def __init__(self):
        self.endpoint = f"{settings.OLLAMA_BASE_URL}/chat/completions"
        self.temperature = settings.config_yaml.get('ollama').get("temperature", 0.1)
        self.model = settings.config_yaml.get('ollama').get("model", "qwen2.5:1.5b")
        self.stream = settings.config_yaml.get('ollama').get("stream", False)

    @observe(as_type="generation")
    async def generate_response(self, prompt: str) -> str:
        """
        """

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": self.stream,
                "temperature": self.temperature
            }

            async with AsyncClient(timeout=30) as client:
                r = await client.post(url=self.endpoint, json=payload) 
                r.raise_for_status()
                response_data = r.json()
                
                usage = response_data.get("usage", {})
                langfuse.update_current_generation(
                    model=self.model,
                    usage_details={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0)
                    },
                    model_parameters={
                        "temperature": self.temperature
                    }
                )

                return response_data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error("Ollama failed to generate response", error=str(e))
            raise e

    