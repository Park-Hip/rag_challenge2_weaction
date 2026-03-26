import asyncio
import json

from src.generation.response_builder import ResponseBuilder
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.core.logger import logger

async def run_evaluation():
    logger.info("Starting Full RAG Evaluation Pipeline...")
    
    with open("eval/dataset.json", "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    rag_pipeline = ResponseBuilder()
    
    evaluation_data = []
    for item in raw_dataset:
        logger.info(f"Evaluating Question: {item['question']}")
        
        response = await rag_pipeline.build_response(item["question"])

        evaluation_data.append({
            "question": item["question"],
            "answer": response["answer"],
            "contexts": [source["text"] for source in response["sources"]],
            "ground_truth": item["ground_truth"]
        })

    temp_path = "eval/completed_dataset.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4)

    evaluator = RagasEvaluator()
    await evaluator.evaluate_test_set(
        input_path=temp_path,
        output_path="eval/results/ragas_scores.json"
    )

if __name__ == "__main__":
    asyncio.run(run_evaluation())
