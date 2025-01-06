#!/usr/bin/env python3
"""Run a small STT benchmark on public AMI segments.

This script intentionally keeps the first benchmark small. It loads a few
segmented AMI samples from Hugging Face, runs WhisperX ASR, and records
accuracy plus speed/resource metrics. A Gemma audio runner can be added once
the concrete audio API contract is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "stt_eval_results.csv"


def normalize_text(text: str) -> str:
    text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)
    return " ".join(text.split())


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


def entity_tokens(text: str) -> set[str]:
    # Lightweight proxy: title-cased tokens only. AMI transcripts are often
    # all-caps, so treating every all-caps word as an entity is misleading.
    return set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))


def decode_audio(audio_obj: Any) -> tuple[Any, int]:
    """Return mono waveform array and sampling rate from datasets Audio variants."""
    if isinstance(audio_obj, dict):
        return audio_obj["array"], int(audio_obj["sampling_rate"])

    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        if getattr(data, "ndim", 1) == 2:
            data = data.mean(axis=0)
        return data, int(samples.sample_rate)

    raise TypeError(f"Unsupported audio object type: {type(audio_obj)!r}")


def recall(ref_items: set[str], hyp_items: set[str]) -> float | None:
    if not ref_items:
        return None
    return len(ref_items & hyp_items) / len(ref_items)


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_ami_samples(limit: int, split: str, config: str, min_duration_sec: float) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    ds = load_dataset("edinburghcstr/ami", config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    samples: list[dict[str, Any]] = []
    for row in ds:
        audio = row.get("audio")
        duration = 0.0
        if audio:
            audio_array, sampling_rate = decode_audio(audio)
            duration = len(audio_array) / float(sampling_rate)
        if row.get("text") and audio and duration >= min_duration_sec:
            samples.append(row)
        if len(samples) >= limit:
            break
    return samples


def run_whisperx(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import jiwer
    import psutil
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    compute_type = args.compute_type
    if device == "cpu" and compute_type in {"float16", "int8_float16"}:
        compute_type = "int8"

    model = whisperx.load_model(args.whisper_model, device, compute_type=compute_type)
    process = psutil.Process(os.getpid())
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(samples, start=1):
        audio = row["audio"]
        audio_array, sampling_rate = decode_audio(audio)
        duration = len(audio_array) / float(sampling_rate)
        ref = row["text"]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        failure = False
        error = ""
        hyp = ""
        try:
            transcribed = model.transcribe(audio_array, batch_size=args.batch_size)
            hyp = " ".join(seg.get("text", "").strip() for seg in transcribed.get("segments", []))
        except Exception as exc:  # noqa: BLE001 - benchmark records failures.
            failure = True
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start

        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        if failure or not hyp_norm:
            wer = cer = mer = wil = None
        else:
            wer = jiwer.wer(ref_norm, hyp_norm)
            cer = jiwer.cer(ref_norm, hyp_norm)
            mer = jiwer.mer(ref_norm, hyp_norm)
            wil = jiwer.wil(ref_norm, hyp_norm)

        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        ref_nums = numeric_tokens(ref)
        hyp_nums = numeric_tokens(hyp)
        ref_entities = entity_tokens(ref)
        hyp_entities = entity_tokens(hyp)

        results.append(
            {
                "dataset": "AMI",
                "meeting_id": row.get("meeting_id", ""),
                "condition": "B1_whisperx_asr",
                "model_or_pipeline": f"whisperx-{args.whisper_model}",
                "audio_minutes": duration / 60.0,
                "wer": wer,
                "cer": cer,
                "mer": mer,
                "wil": wil,
                "segment_wer": wer,
                "numeric_accuracy": recall(ref_nums, hyp_nums),
                "entity_recall": recall(ref_entities, hyp_entities),
                "der": None,
                "jer": None,
                "cpwer": None,
                "speaker_count_ref": None,
                "speaker_count_pred": None,
                "speaker_count_error": None,
                "speaker_attribution_accuracy": None,
                "rtf": elapsed / duration if duration else None,
                "wall_clock_sec": elapsed,
                "time_to_first_text_sec": None,
                "peak_gpu_vram_gb": peak_vram_gb,
                "peak_ram_gb": process.memory_info().rss / 1024**3,
                "failure": failure,
                "timeout": False,
                "cost_per_audio_hour_usd": None,
                "notes": error,
                "reference": ref,
                "hypothesis": hyp,
                "audio_id": row.get("audio_id", f"sample_{idx}"),
                "speaker_id": row.get("speaker_id", ""),
                "begin_time": row.get("begin_time", ""),
                "end_time": row.get("end_time", ""),
            }
        )

    del model
    return results


def run_gemma_audio(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import jiwer
    import psutil
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    process = psutil.Process(os.getpid())
    results: list[dict[str, Any]] = []

    quantization_config = None
    torch_dtype: Any = "auto"
    device_map: Any = "auto"
    model_label = args.gemma_model
    if args.gemma_quant == "4bit":
        model_label = f"{model_label}-4bit"
        torch_dtype = torch.float16
        if not torch.cuda.is_available():
            return [
                failure_row(
                    row,
                    "A1_gemma_audio_4bit",
                    model_label,
                    "bitsandbytes 4bit quantization requires CUDA, but torch.cuda.is_available() is false",
                )
                for row in samples
            ]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if args.force_single_gpu:
            device_map = {"": 0}
    elif args.gemma_quant == "8bit":
        model_label = f"{model_label}-8bit"
        torch_dtype = torch.float16
        if not torch.cuda.is_available():
            return [
                failure_row(
                    row,
                    "A1_gemma_audio_8bit",
                    model_label,
                    "bitsandbytes 8bit quantization requires CUDA, but torch.cuda.is_available() is false",
                )
                for row in samples
            ]
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        if args.force_single_gpu:
            device_map = {"": 0}
    elif args.gemma_quant == "none":
        torch_dtype = "auto"
    else:
        raise ValueError(f"Unsupported gemma quant mode: {args.gemma_quant}")

    try:
        processor = AutoProcessor.from_pretrained(args.gemma_model, token=hf_token)
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText")
        model = model_cls.from_pretrained(
            args.gemma_model,
            token=hf_token,
            dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark records failures.
        return [
            failure_row(
                row,
                f"A1_gemma_audio_{args.gemma_quant}",
                model_label,
                f"model_load_failed: {type(exc).__name__}: {exc}",
            )
            for row in samples
        ]

    for idx, row in enumerate(samples, start=1):
        audio_array, sampling_rate = decode_audio(row["audio"])
        duration = len(audio_array) / float(sampling_rate)
        ref = row["text"]
        tmp_path = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        failure = False
        error = ""
        hyp = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_array, sampling_rate)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.gemma_prompt},
                        {"type": "audio", "audio": tmp_path},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[-1]
            decoded = processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            hyp = decoded[0].strip() if decoded else ""
        except Exception as exc:  # noqa: BLE001 - benchmark records failures.
            failure = True
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        elapsed = time.perf_counter() - start
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        if failure or not hyp_norm:
            wer = cer = mer = wil = None
        else:
            wer = jiwer.wer(ref_norm, hyp_norm)
            cer = jiwer.cer(ref_norm, hyp_norm)
            mer = jiwer.mer(ref_norm, hyp_norm)
            wil = jiwer.wil(ref_norm, hyp_norm)

        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        ref_nums = numeric_tokens(ref)
        hyp_nums = numeric_tokens(hyp)
        ref_entities = entity_tokens(ref)
        hyp_entities = entity_tokens(hyp)
        results.append(
            {
                "dataset": "AMI",
                "meeting_id": row.get("meeting_id", ""),
                "condition": f"A1_gemma_audio_{args.gemma_quant}",
                "model_or_pipeline": model_label,
                "audio_minutes": duration / 60.0,
                "wer": wer,
                "cer": cer,
                "mer": mer,
                "wil": wil,
                "segment_wer": wer,
                "numeric_accuracy": recall(ref_nums, hyp_nums),
                "entity_recall": recall(ref_entities, hyp_entities),
                "der": None,
                "jer": None,
                "cpwer": None,
                "speaker_count_ref": None,
                "speaker_count_pred": None,
                "speaker_count_error": None,
                "speaker_attribution_accuracy": None,
                "rtf": elapsed / duration if duration else None,
                "wall_clock_sec": elapsed,
                "time_to_first_text_sec": None,
                "peak_gpu_vram_gb": peak_vram_gb,
                "peak_ram_gb": process.memory_info().rss / 1024**3,
                "failure": failure,
                "timeout": False,
                "cost_per_audio_hour_usd": None,
                "notes": error,
                "reference": ref,
                "hypothesis": hyp,
                "audio_id": row.get("audio_id", f"sample_{idx}"),
                "speaker_id": row.get("speaker_id", ""),
                "begin_time": row.get("begin_time", ""),
                "end_time": row.get("end_time", ""),
            }
        )

    del model
    return results


def failure_row(row: dict[str, Any], condition: str, model_label: str, error: str) -> dict[str, Any]:
    audio_minutes = None
    if row.get("audio"):
        try:
            audio_array, sampling_rate = decode_audio(row["audio"])
            audio_minutes = len(audio_array) / float(sampling_rate) / 60.0
        except Exception:
            audio_minutes = None
    return {
        "dataset": "AMI",
        "meeting_id": row.get("meeting_id", ""),
        "condition": condition,
        "model_or_pipeline": model_label,
        "audio_minutes": audio_minutes,
        "wer": None,
        "cer": None,
        "mer": None,
        "wil": None,
        "segment_wer": None,
        "numeric_accuracy": None,
        "entity_recall": None,
        "der": None,
        "jer": None,
        "cpwer": None,
        "speaker_count_ref": None,
        "speaker_count_pred": None,
        "speaker_count_error": None,
        "speaker_attribution_accuracy": None,
        "rtf": None,
        "wall_clock_sec": None,
        "time_to_first_text_sec": None,
        "peak_gpu_vram_gb": None,
        "peak_ram_gb": None,
        "failure": True,
        "timeout": False,
        "cost_per_audio_hour_usd": None,
        "notes": error,
        "reference": row.get("text", ""),
        "hypothesis": "",
        "audio_id": row.get("audio_id", ""),
        "speaker_id": row.get("speaker_id", ""),
        "begin_time": row.get("begin_time", ""),
        "end_time": row.get("end_time", ""),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "meeting_id",
        "condition",
        "model_or_pipeline",
        "audio_minutes",
        "wer",
        "cer",
        "mer",
        "wil",
        "segment_wer",
        "numeric_accuracy",
        "entity_recall",
        "der",
        "jer",
        "cpwer",
        "speaker_count_ref",
        "speaker_count_pred",
        "speaker_count_error",
        "speaker_attribution_accuracy",
        "rtf",
        "wall_clock_sec",
        "time_to_first_text_sec",
        "peak_gpu_vram_gb",
        "peak_ram_gb",
        "failure",
        "timeout",
        "cost_per_audio_hour_usd",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(rows),
        "failures": sum(1 for r in rows if r.get("failure")),
        "mean_wer": avg("wer"),
        "mean_cer": avg("cer"),
        "mean_mer": avg("mer"),
        "mean_wil": avg("wil"),
        "mean_numeric_accuracy": avg("numeric_accuracy"),
        "mean_entity_recall": avg("entity_recall"),
        "mean_rtf": avg("rtf"),
        "mean_wall_clock_sec": avg("wall_clock_sec"),
        "mean_peak_gpu_vram_gb": avg("peak_gpu_vram_gb"),
        "mean_peak_ram_gb": avg("peak_ram_gb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="ihm")
    parser.add_argument("--runner", choices=["whisperx", "gemma"], default="whisperx")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--min-duration-sec", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gemma-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--gemma-quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--force-single-gpu", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--gemma-prompt",
        default=(
            "Transcribe the following speech segment in its original language. "
            "Only output the transcription, with no newlines. "
            "When transcribing numbers, write digits."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    load_dotenv()
    samples = load_ami_samples(args.limit, args.split, args.config, args.min_duration_sec)
    if args.runner == "whisperx":
        rows = run_whisperx(samples, args)
    else:
        rows = run_gemma_audio(samples, args)
    write_csv(rows, args.out)
    detail_path = args.out.with_suffix(".jsonl")
    write_jsonl(rows, detail_path)
    summary = summarize(rows)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"csv={args.out}")
    print(f"details={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
