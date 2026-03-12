# NLP 2026 A6: Naive RAG vs Contextual Retrieval

This project implements the full assignment pipeline for **A6 RAG Techniques** using:
- Assignment spec: `NLP_2026_A6_RAG_Techniques.pdf`
- Chapter paper: `2.pdf` (Chapter 2: Words and Tokens)
- Student ID used in example: `st125982` (last digit = 2, so Chapter 2)

## Chapter Source Justification

- Student ID: `st125982` -> last digit `2`
- Assignment rule: last digit `2` maps to **Chapter 2**
- Source used: `2.pdf` in this repository (Chapter 2 content used for processing and QA generation)

## What is implemented

1. **Task 1 (Data preparation)**
- Load chapter text from PDF
- Clean and chunk text
- Provide 20 QA pairs in `app/backend/data/qa_pairs_chapter_2.json`

2. **Task 2 (Technique comparison)**
- Naive RAG (plain chunks)
- Contextual Retrieval (context prefix prepended to each chunk)
- Evaluation with ROUGE-1, ROUGE-2, ROUGE-L
- Output files in `answer/`

3. **Task 3 (Web app)**
- Django backend API in `app/backend/`
- React frontend in `app/frontend/`
- Uses **Contextual Retrieval**
- Shows generated answer and source chunk citation

## Retriever and Generator models

- Retriever model: `sentence-transformers/all-MiniLM-L6-v2`
- Generator model:
  - default: `extractive sentence selector` (local, deterministic)
  - optional: `gpt-4o-mini` if `OPENAI_API_KEY` is set and runner is called with `--generator openai`

## Setup

```bash
cd A6-RAG_Techniques
pip install -r requirements.txt
```

## Run assignment evaluation

```bash
python app/backend/src/run_assignment.py --student-id st125982 --chapter-pdf 2.pdf --qa-file data/qa_pairs_chapter_2.json --generator extractive
```

This generates:
- `answer/response-st-125982-chapter-2.json`
- `answer/metrics-st-125982-chapter-2.json`

## Run chatbot app (React + Django)

```bash
cd app/backend
python manage.py runserver 127.0.0.1:8000
```

```bash
cd app/frontend
npm install
npm start
```

## Notes for submission

- Required JSON naming is implemented as `response-st-xxxxxx-chapter-x.json`.
- If you want strict chapter matching from a different ID, pass your own `--student-id` and corresponding chapter PDF.

## Experiment Notes (Personal)

In my run, Naive RAG scored slightly better than Contextual Retrieval. I think this happened because my contextual prefixes are heuristic and sometimes too generic for this chapter.

Examples where retrieval/answering failed:
- Filled-pause question: answer drifted to general word-type text.
- Utterance question: retrieval returned ELIZA-related text instead of the definition.
- `N` in corpus statistics: output was a long BPE snippet instead of a short direct definition.

What I would improve next:
- Tune chunk size and overlap.
- Replace heuristic enrichment with LLM-based chunk context.
- Add reranking/short-answer generation after retrieval.

## Demo-Ready Check

- Backend check passed with `python manage.py check`.
- Frontend dependencies install and audit run successfully.
- I ran backend and frontend once to make sure both start normally.
