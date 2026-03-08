import csv
import json
import math
import re
from numbers import Integral, Real
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "a5_outputs"
REPORT_DIR = OUTPUTS_DIR / "report_assets"
DPO_MODEL_DIR = OUTPUTS_DIR / "dpo_truthful_model"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    except Exception:
        # Some locally saved tokenizer configs are not portable across versions.
        # This fallback keeps inference usable by loading tokenizer.json directly.
        tokenizer_file = model_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        tokenizer_config = _read_json(model_dir / "tokenizer_config.json")

        for field in ("eos_token", "pad_token", "bos_token", "unk_token"):
            token_value = tokenizer_config.get(field)
            if isinstance(token_value, str) and token_value:
                setattr(tokenizer, field, token_value)

        chat_template_path = model_dir / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")

        return tokenizer


def _parse_scalar(value: str):
    if value is None:
        return None

    text = value.strip()
    if text == "":
        return None

    lowered = text.lower()
    if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return None

    try:
        if any(char in lowered for char in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _to_json_safe(value):
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    return str(value)


def _read_csv(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit and index >= limit:
                break
            rows.append({key: _parse_scalar(value) for key, value in row.items()})
    return rows


def get_report_payload() -> dict:
    metrics = _read_json(REPORT_DIR / "final_metrics.json")
    judge_results = _read_csv(REPORT_DIR / "judge_results.csv", limit=500)
    outputs = _read_csv(REPORT_DIR / "alpacaeval_model_outputs.csv", limit=200)
    loss_logs = _read_csv(REPORT_DIR / "dpo_loss_logs.csv", limit=1000)

    payload = {
        "metrics": metrics,
        "judge_results": judge_results,
        "outputs": outputs,
        "loss_logs": loss_logs,
        "loss_curve_path": str((REPORT_DIR / "dpo_loss_curve.png").relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    # DRF JSON renderer rejects NaN/Inf, so sanitize once before returning.
    return _to_json_safe(payload)


@lru_cache(maxsize=1)
def get_text_generator():
    import torch
    from transformers import AutoModelForCausalLM

    if not DPO_MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model directory: {DPO_MODEL_DIR}")

    tokenizer = _load_tokenizer(DPO_MODEL_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    model = AutoModelForCausalLM.from_pretrained(str(DPO_MODEL_DIR), **model_kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    return tokenizer, model


def _clean_generated_text(text: str) -> str:
    text = (text or "").replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_low_quality(text: str) -> bool:
    if not text:
        return True

    tokens = text.split()
    if len(tokens) < 4:
        return True

    # Repeated junk like "xxx xxx xxx" is a common decoding failure mode.
    if len(tokens) >= 12:
        repeated_ratio = 1.0 - (len(set(tokens)) / len(tokens))
        if repeated_ratio > 0.62:
            return True

    # Too many symbols/non-word fragments relative to normal words.
    symbol_chunks = len(re.findall(r"[^\w\s]{2,}", text, flags=re.UNICODE))
    return symbol_chunks >= 6


def generate_text(prompt: str, max_new_tokens: int) -> str:
    import torch

    tokenizer, model = get_text_generator()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly in English, briefly and without repetition.",
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        built_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        built_prompt = prompt

    model_device = next(model.parameters()).device
    encoded = tokenizer(built_prompt, return_tensors="pt")
    encoded.pop("token_type_ids", None)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    # Qwen-style chat templates often terminate with <|im_end|>.
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_ids.append(im_end_id)
    except Exception:
        pass

    eos_token_id = sorted(set(eos_ids)) if eos_ids else None

    prompt_len = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        # Pass 1: deterministic decoding for stable, less noisy answers.
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][prompt_len:]
    answer = _clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))

    if _looks_low_quality(answer):
        with torch.inference_mode():
            # Pass 2: conservative sampling fallback if deterministic output is still poor.
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.85,
                repetition_penalty=1.25,
                no_repeat_ngram_size=5,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][prompt_len:]
        answer = _clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return answer
