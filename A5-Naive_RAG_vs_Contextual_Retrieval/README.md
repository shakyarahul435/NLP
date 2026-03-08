# A5: Human Preference Optimization and LLM-as-a-Judge

This repository contains my work for A5 in AT82.05 NLU (2026).

**Hugging Face Model:** https://huggingface.co/shakyarahul/LLM-as-a-Judge

## Quick Repository Map

- `A5.ipynb`: main notebook with dataset prep, DPO training, evaluation, and discussion.
- `a5_outputs/`: saved artifacts from executed runs.
- `app/`: Django + React app that serves saved A5 outputs and generation endpoint.
- `NLP_2026_A5_Human_Preference.pdf`: assignment instructions.

## Assignment Completion Summary

1. Task 1 - Dataset Preparation 
- Implemented in `A5.ipynb` using `jondurbin/truthy-dpo-v0.1`.
- Prompt/chosen/rejected schema is loaded and explained in notebook markdown.
- Data version note: the assignment PDF text references `v0.11`, while the linked Hugging Face dataset and this implementation use `jondurbin/truthy-dpo-v0.1`.

2. Task 2 - DPO Training
- DPO implemented with `trl.DPOTrainer` in `A5.ipynb`.
- Base model used: `Qwen/Qwen2.5-1.5B-Instruct`.
- Training outputs are saved in `a5_outputs/`.
- Loss history and figure are saved as:
  - `a5_outputs/report_assets/dpo_loss_logs.csv`
  - `a5_outputs/report_assets/dpo_loss_curve.png`

3. Task 3 - Push Model to Hugging Face
- Local trained model is saved at `a5_outputs/dpo_truthful_model/`.
- Upload scripts are ready in `app/scripts/`.
- Model uploaded successfully:
  - `Model Hub URL: https://huggingface.co/shakyarahul/LLM-as-a-Judge`

4. Task 4 - LLM-as-a-Judge on AlpacaEval
- Notebook uses direct JSON loading for AlpacaEval and filters `helpful_base`.
- Evaluation sample table is exported to:
  - `a5_outputs/report_assets/judge_results.csv`
- Paired model outputs are exported to:
  - `a5_outputs/report_assets/alpacaeval_model_outputs.csv`
- Final metrics are exported to:
  - `a5_outputs/report_assets/final_metrics.json`

## Final Metrics Snapshot

From `a5_outputs/report_assets/final_metrics.json`:
- `model_b_wins`: `0`
- `ties`: `15`
- `total_valid`: `15`
- `win_rate_percent`: `50.0`

- Backend: Django REST API in `app/backend`
- Frontend: React (npm/react-scripts) in `app/frontend`

### Future Add-ons 
- Compose: `app/docker-compose.yml`
- Hugging Face Space assets: `app/huggingface-space`

---
## Author
Rahul Shakya <br/>
st125982 <br/>
Asian Institute of Technology - AIT
