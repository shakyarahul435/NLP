import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: int
    text: str
    contextual_text: str
    source: str


class RAGPipeline:
    """Naive RAG and Contextual Retrieval pipeline for the A6 assignment."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_mode: str = "extractive",
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.generator_mode = generator_mode
        self.embedder = SentenceTransformer(embedding_model_name)
        self.records: List[ChunkRecord] = []
        self.naive_embeddings: Optional[np.ndarray] = None
        self.contextual_embeddings: Optional[np.ndarray] = None
        self._doc_excerpt: str = ""
        self._title: str = ""

    @staticmethod
    def load_pdf_text(pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\u00ad", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
            i += chunk_size - overlap
        return chunks

    def _heuristic_context_prefix(self, chunk: str) -> str:
        chunk_lower = chunk.lower()
        if "token" in chunk_lower:
            topic = "tokenization and word segmentation"
        elif "unicode" in chunk_lower or "utf" in chunk_lower:
            topic = "Unicode and character encoding"
        elif "regex" in chunk_lower or "regular expression" in chunk_lower:
            topic = "regular expressions for text processing"
        elif "edit distance" in chunk_lower:
            topic = "edit distance and string similarity"
        elif "byte-pair" in chunk_lower or "bpe" in chunk_lower:
            topic = "Byte-Pair Encoding (BPE)"
        elif "word type" in chunk_lower or "word instance" in chunk_lower:
            topic = "word types vs. word instances"
        else:
            topic = "core concepts from the chapter"

        return (
            f"This chunk from {self._title} discusses {topic} and connects this part "
            "to the larger chapter context."
        )

    def _llm_context_prefix(self, chunk: str) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                f"Title: {self._title}\n"
                f"Document excerpt: {self._doc_excerpt[:3500]}\n"
                f"Chunk: {chunk}\n\n"
                "Provide 1-2 concise sentences describing how this chunk relates "
                "to the full document. Start with 'This chunk from'."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=120,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def enrich_chunk(self, chunk: str) -> str:
        llm_prefix = self._llm_context_prefix(chunk)
        prefix = llm_prefix if llm_prefix else self._heuristic_context_prefix(chunk)
        return f"{prefix}\n\n{chunk}"

    def fit(self, chapter_text: str, source_name: str, title: str) -> None:
        cleaned = self.clean_text(chapter_text)
        self._doc_excerpt = cleaned
        self._title = title
        chunks = self.chunk_text(cleaned)

        self.records = []
        naive_texts = []
        contextual_texts = []

        for idx, chunk in enumerate(chunks):
            contextual_chunk = self.enrich_chunk(chunk)
            record = ChunkRecord(
                chunk_id=idx,
                text=chunk,
                contextual_text=contextual_chunk,
                source=source_name,
            )
            self.records.append(record)
            naive_texts.append(chunk)
            contextual_texts.append(contextual_chunk)

        self.naive_embeddings = self.embedder.encode(naive_texts, normalize_embeddings=True)
        self.contextual_embeddings = self.embedder.encode(
            contextual_texts, normalize_embeddings=True
        )

    def _retrieve(self, question: str, mode: str = "naive", top_k: int = 3) -> List[ChunkRecord]:
        if self.naive_embeddings is None or self.contextual_embeddings is None:
            raise RuntimeError("Pipeline is not fit yet. Run fit() first.")

        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        emb_matrix = self.naive_embeddings if mode == "naive" else self.contextual_embeddings
        scores = emb_matrix @ q_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.records[i] for i in top_indices]

    @staticmethod
    def _extractive_generate(question: str, chunks: List[ChunkRecord]) -> str:
        # Simple extractive generator for deterministic local runs.
        context = " ".join(chunk.text for chunk in chunks)
        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))

        best_sentence = ""
        best_score = -1
        for sentence in sentences:
            s_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            overlap = len(q_terms & s_terms)
            length_bonus = min(len(sentence.split()), 35) / 35
            score = overlap + 0.3 * length_bonus
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence.strip() if best_sentence else chunks[0].text[:260]

    def _openai_generate(self, question: str, chunks: List[ChunkRecord]) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            context_text = "\n\n".join(chunk.contextual_text for chunk in chunks)
            prompt = (
                "Answer only from the provided context. If the answer is not in the "
                "context, say so clearly.\n\n"
                f"Question: {question}\n\nContext:\n{context_text}"
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=220,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def answer(self, question: str, mode: str = "naive", top_k: int = 3) -> Tuple[str, List[ChunkRecord]]:
        retrieved = self._retrieve(question, mode=mode, top_k=top_k)
        if self.generator_mode == "openai":
            llm_answer = self._openai_generate(question, retrieved)
            if llm_answer:
                return llm_answer, retrieved

        return self._extractive_generate(question, retrieved), retrieved


def evaluate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in scores.items()}


def load_qa_pairs(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
