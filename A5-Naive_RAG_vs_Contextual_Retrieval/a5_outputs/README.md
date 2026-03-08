# A5 Saved Outputs

This folder stores artifacts produced by the executed notebook run, so the assignment can be reviewed without retraining.

## Main Folders

- `checkpoint-115/`: intermediate checkpoint from training.
- `dpo_truthful_model/`: exported DPO model files used for inference.
- `report_assets/`: compact files used for reporting and app visualization.

## Report Assets

- `report_assets/dpo_loss_logs.csv`: training loss per step.
- `report_assets/dpo_loss_curve.png`: rendered loss curve.
- `report_assets/alpacaeval_model_outputs.csv`: base vs DPO generations for sampled prompts.
- `report_assets/judge_results.csv`: judge verdict table (`Model A`, `Model B`, `Tie`).
- `report_assets/final_metrics.json`: final win-rate summary.

## Why this folder exists

A5 includes model training and evaluation steps that are time-consuming to rerun. Keeping these outputs in versioned files makes grading and app deployment straightforward.
