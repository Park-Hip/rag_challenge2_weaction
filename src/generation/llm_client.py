from src.core.langfuse import langfuse, observe
from langfuse.openai import OpenAI

from src.core.config import settings
from src.core.logger import logger

class LLMClient():
    """
    """

    def __init__(self):
        self.model = settings.config_yaml.get('ollama').get("model", "qwen2.5:1.5b")
        self.client = OpenAI(
            base_url = settings.OLLAMA_BASE_URL,
            api_key='ollama', # required, but unused
        )

    @observe(as_type="generation")
    async def generate_response(self, prompt: str) -> str:
        """
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            langfuse.flush()
            return response.choices[0].message.content


        except Exception as e:
            logger.error("Ollama failed to generate response", error=str(e))
            raise e

    