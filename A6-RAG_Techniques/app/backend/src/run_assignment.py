import argparse
from pathlib import Path
from typing import Dict, List

from rag_pipeline import RAGPipeline, evaluate_rouge, load_qa_pairs, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A6 RAG Techniques Assignment Runner")
    parser.add_argument("--student-id", type=str, required=True, help="Example: st125982")
    parser.add_argument(
        "--chapter-pdf",
        type=str,
        default="2.pdf",
        help="Path to the chapter PDF assigned by student ID digit",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default="data/qa_pairs_chapter_2.json",
        help="Path to the 20 QA pair JSON file",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["extractive", "openai"],
        default="extractive",
        help="Generator model mode",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of retrieved chunks for each question"
    )
    return parser.parse_args()


def chapter_from_student_id(student_id: str) -> int:
    digits = [ch for ch in student_id if ch.isdigit()]
    if not digits:
        raise ValueError("student_id must include at least one digit")
    last_digit = int(digits[-1])
    return 10 if last_digit == 0 else last_digit


def main() -> None:
    args = parse_args()

    backend_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]
    chapter_pdf = (project_root / args.chapter_pdf).resolve()
    qa_file = (backend_root / args.qa_file).resolve()

    chapter_no = chapter_from_student_id(args.student_id)
    pipeline = RAGPipeline(generator_mode=args.generator)

    chapter_text = pipeline.load_pdf_text(chapter_pdf)
    source_name = chapter_pdf.name
    title = f"Chapter {chapter_no}"
    pipeline.fit(chapter_text=chapter_text, source_name=source_name, title=title)

    qa_pairs = load_qa_pairs(qa_file)

    naive_preds: List[str] = []
    contextual_preds: List[str] = []
    refs: List[str] = []
    output_rows: List[Dict[str, str]] = []

    for item in qa_pairs:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        naive_answer, naive_chunks = pipeline.answer(question, mode="naive", top_k=args.top_k)
        contextual_answer, contextual_chunks = pipeline.answer(
            question, mode="contextual", top_k=args.top_k
        )

        naive_preds.append(naive_answer)
        contextual_preds.append(contextual_answer)
        refs.append(ground_truth)

        output_rows.append(
            {
                "question": question,
                "ground_truth_answer": ground_truth,
                "naive_rag_answer": naive_answer,
                "contextual_retrieval_answer": contextual_answer,
                "naive_source_chunk_id": str(naive_chunks[0].chunk_id),
                "contextual_source_chunk_id": str(contextual_chunks[0].chunk_id),
                "source_document": source_name,
            }
        )

    naive_scores = evaluate_rouge(naive_preds, refs)
    contextual_scores = evaluate_rouge(contextual_preds, refs)

    eval_table = {
        "student_id": args.student_id,
        "chapter": chapter_no,
        "retriever_model": pipeline.embedding_model_name,
        "generator_model": "gpt-4o-mini" if args.generator == "openai" else "extractive sentence selector",
        "scores": {
            "Naive RAG": {
                "ROUGE-1": round(naive_scores["rouge1"], 4),
                "ROUGE-2": round(naive_scores["rouge2"], 4),
                "ROUGE-L": round(naive_scores["rougeL"], 4),
            },
            "Contextual Retrieval": {
                "ROUGE-1": round(contextual_scores["rouge1"], 4),
                "ROUGE-2": round(contextual_scores["rouge2"], 4),
                "ROUGE-L": round(contextual_scores["rougeL"], 4),
            },
        },
    }

    answer_dir = project_root / "answer"
    response_path = answer_dir / f"response-{args.student_id}-chapter-{chapter_no}.json"
    metrics_path = answer_dir / f"metrics-{args.student_id}-chapter-{chapter_no}.json"

    save_json(response_path, output_rows)
    save_json(metrics_path, eval_table)

    print("Saved:", response_path)
    print("Saved:", metrics_path)
    print("\nEvaluation Table")
    print("Method                  ROUGE-1   ROUGE-2   ROUGE-L")
    print(
        "Naive RAG               "
        f"{eval_table['scores']['Naive RAG']['ROUGE-1']:.4f}    "
        f"{eval_table['scores']['Naive RAG']['ROUGE-2']:.4f}    "
        f"{eval_table['scores']['Naive RAG']['ROUGE-L']:.4f}"
    )
    print(
        "Contextual Retrieval     "
        f"{eval_table['scores']['Contextual Retrieval']['ROUGE-1']:.4f}    "
        f"{eval_table['scores']['Contextual Retrieval']['ROUGE-2']:.4f}    "
        f"{eval_table['scores']['Contextual Retrieval']['ROUGE-L']:.4f}"
    )


if __name__ == "__main__":
    main()
