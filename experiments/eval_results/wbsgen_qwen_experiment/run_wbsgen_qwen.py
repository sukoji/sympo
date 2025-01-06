"""Run C3 with only the WBS generation agent routed to the Qwen endpoint.

Default backend remains qwen-api pointed at Gemma-4-26B on localhost:8081.
This runner temporarily switches QWEN_API_URL/QWEN_API_MODEL only when
agents.wbs_gen_agent asks for model_config["wbs_gen"] == "qwen-wbs-api".

Usage:
  python eval_results/wbsgen_qwen_experiment/run_wbsgen_qwen.py [n_runs=3]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

from agents import llm_config
from eval import experiment_runner as runner


BASE_BACKEND = "qwen-api"
BASE_CONDITION = "C3_3rounds"
CONDITION_KEY = "H_wbsgen_qwen4b"

GEMMA_URL = os.getenv("GEMMA26_API_URL", "http://127.0.0.1:8081")
GEMMA_MODEL = os.getenv("GEMMA26_API_MODEL", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
QWEN_URL = os.getenv("WBS_QWEN_API_URL", "http://127.0.0.1:8082")
QWEN_MODEL = os.getenv("WBS_QWEN_API_MODEL", "Qwen3.5-4B-Q4_K_M.gguf")

_orig_get_llm = llm_config.get_llm


def _with_qwen_wbs_endpoint(temperature=0.7, max_tokens=1024):
    old_url = os.environ.get("QWEN_API_URL")
    old_model = os.environ.get("QWEN_API_MODEL")
    old_backend = os.environ.get("LLM_BACKEND")
    os.environ["LLM_BACKEND"] = BASE_BACKEND
    os.environ["QWEN_API_URL"] = QWEN_URL
    os.environ["QWEN_API_MODEL"] = QWEN_MODEL
    llm_config._qwen_api_instance = None
    try:
        return _orig_get_llm(temperature=temperature, max_tokens=max_tokens, model_id=BASE_BACKEND)
    finally:
        if old_url is None:
            os.environ.pop("QWEN_API_URL", None)
        else:
            os.environ["QWEN_API_URL"] = old_url
        if old_model is None:
            os.environ.pop("QWEN_API_MODEL", None)
        else:
            os.environ["QWEN_API_MODEL"] = old_model
        if old_backend is None:
            os.environ.pop("LLM_BACKEND", None)
        else:
            os.environ["LLM_BACKEND"] = old_backend


def patched_get_llm(temperature=0.7, max_tokens=1024, model_id=None):
    if model_id == "qwen-wbs-api":
        print(f"DEBUG: WBS Gen routed to Qwen endpoint {QWEN_URL} ({QWEN_MODEL})")
        return _with_qwen_wbs_endpoint(temperature=temperature, max_tokens=max_tokens)
    return _orig_get_llm(temperature=temperature, max_tokens=max_tokens, model_id=model_id)


def register_condition() -> str:
    cond = dict(runner.CONDITIONS[BASE_CONDITION])
    cond.update(
        label="H: WBS Gen -> Qwen 4B, others -> Gemma 26B",
        description="C3 with only initial WBS generation using Qwen3.5-4B; all later agents use Gemma-4-26B.",
        model_config={"wbs_gen": "qwen-wbs-api"},
    )
    runner.CONDITIONS[CONDITION_KEY] = cond
    return CONDITION_KEY


def main() -> None:
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    os.environ["RUNNER_ID"] = "wbsgen_qwen4b"
    os.environ["LLM_BACKEND"] = BASE_BACKEND
    os.environ["QWEN_API_URL"] = GEMMA_URL
    os.environ["QWEN_API_MODEL"] = GEMMA_MODEL

    llm_config.get_llm = patched_get_llm
    cond_key = register_condition()

    print("WBS Gen Qwen experiment")
    print(f"Default agents: {GEMMA_URL} ({GEMMA_MODEL})")
    print(f"WBS Gen only:  {QWEN_URL} ({QWEN_MODEL})")
    print(f"Condition: {cond_key}, runs={n_runs}")

    runner.run_all_experiments(
        backend=BASE_BACKEND,
        runs_per_condition=n_runs,
        conditions=[cond_key],
        harness_settings=None,
        cross_judge=False,
    )


if __name__ == "__main__":
    main()
