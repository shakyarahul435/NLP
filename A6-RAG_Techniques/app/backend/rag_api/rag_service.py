from pathlib import Path
import sys
from typing import Any, Dict, List


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.rag_pipeline import RAGPipeline, evaluate_rouge, load_qa_pairs


_pipeline = None
_evaluation = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        chapter_pdf = PROJECT_ROOT / "2.pdf"
        text = RAGPipeline.load_pdf_text(chapter_pdf)
        pipeline = RAGPipeline(generator_mode="extractive")
        pipeline.fit(chapter_text=text, source_name=chapter_pdf.name, title="Chapter 2")
        _pipeline = pipeline
    return _pipeline


def get_evaluation_summary() -> Dict[str, Any]:
    global _evaluation
    if _evaluation is None:
        pipeline = get_pipeline()
        qa_file = BACKEND_ROOT / "data" / "qa_pairs_chapter_2.json"
        qa_pairs = load_qa_pairs(qa_file)

        naive_preds: List[str] = []
        contextual_preds: List[str] = []
        references: List[str] = []

        for item in qa_pairs:
            question = item["question"]
            ground_truth = item["ground_truth_answer"]

            naive_answer, _ = pipeline.answer(question, mode="naive", top_k=3)
            contextual_answer, _ = pipeline.answer(question, mode="contextual", top_k=3)

            naive_preds.append(naive_answer)
            contextual_preds.append(contextual_answer)
            references.append(ground_truth)

        naive_scores = evaluate_rouge(naive_preds, references)
        contextual_scores = evaluate_rouge(contextual_preds, references)

        _evaluation = {
            "rows": [
                {
                    "Method": "Naive RAG",
                    "ROUGE-1": round(naive_scores["rouge1"], 4),
                    "ROUGE-2": round(naive_scores["rouge2"], 4),
                    "ROUGE-L": round(naive_scores["rougeL"], 4),
                },
                {
                    "Method": "Contextual Retrieval",
                    "ROUGE-1": round(contextual_scores["rouge1"], 4),
                    "ROUGE-2": round(contextual_scores["rouge2"], 4),
                    "ROUGE-L": round(contextual_scores["rougeL"], 4),
                },
            ]
        }

    return _evaluation


def get_example_before_after() -> Dict[str, Any]:
    pipeline = get_pipeline()
    first_record = pipeline.records[0]

    return {
        "document": first_record.source,
        "chunk_id": first_record.chunk_id,
        "before": first_record.text,
        "after": first_record.contextual_text,
    }
