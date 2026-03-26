from ragas import evaluate
import json
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import os

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import JinaEmbeddings

from src.core.logger import logger
from src.core.config import settings

class RagasEvaluator:
    """
    Executes the Evaluation Loop against a Ground Truth dataset.
    """

    def __init__(self):
        groq_chat = ChatGroq(
            api_key=settings.GROQ_API_KEY, 
            model_name=settings.config_yaml.get('groq').get("model", "llama-3.3-70b-versatile"), 
            temperature=0.0
        )
        self.llm = LangchainLLMWrapper(groq_chat)

        jina_embedder = JinaEmbeddings(
            jina_api_key=settings.JINA_API_KEY, 
            model_name="jina-embeddings-v3"
        )
        self.embedder = LangchainEmbeddingsWrapper(jina_embedder)
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    async def evaluate_test_set(self, input_path: str, output_path: str):
        """
        Loads the dataset, evaluates it using Groq, and saves the dataframe scores.
        """
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                dataset_dict = json.load(f)

            dataset = Dataset.from_list(dataset_dict)

            logger.info("Starting RAGAS evaluation loop using Groq Judge...")
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embedder
            )
            
            result_df = result.to_pandas()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_json(output_path, orient="records", indent=4)
            
            logger.info(f"Successfully saved evaluation to output path: {output_path}")

        except Exception as e:
            logger.error(f"Failed to evaluate test set: {str(e)}")
            raise e
