#!/usr/bin/env python3
"""
UIBenchKit API server entrypoint.
"""

import datetime
import os

from flask import Flask
from dotenv import load_dotenv

from config import (
    API_VERSION,
    DATA_DIR,
    DEFAULT_API_KEY,
    MODEL_FAMILIES,
    OPENAI_BASE_URL,
    PROMPT_DCGEN,
    PROMPT_DIRECT,
    RESULTS_DIR,
    SEG_PARAMS_DEFAULT,
    SUPPORTED_METHODS,
    SUPPORTED_MODELS,
    get_model_info,
)
from dataset_manager import DATASETS_CONFIG, get_dataset_manager
from routes import create_auth_blueprint, create_datasets_blueprint, create_runs_blueprint
from run_model import Run
from services.auth import create_require_api_key
from services.evaluation_runner import run_evaluation_for_run
from services.fs_utils import create_path_resolver, ensure_dir, get_image_files, sanitize_for_filename
from services.generation import create_dcgen_generator, generate_single, take_screenshots_task
from services.model_factory import create_bot_factory, get_provider_env_status
from services.run_tasks import create_run_experiment_task, load_existing_runs

try:
    from methods.latcoder import generate_latcoder
except Exception as error:
    _LATCODER_IMPORT_ERROR = error

    def generate_latcoder(*args, **kwargs):
        raise RuntimeError(f"LatCoder method unavailable: {_LATCODER_IMPORT_ERROR}")


try:
    from methods.uicopilot import generate_uicopilot
except Exception as error:
    _UICOPILOT_IMPORT_ERROR = error

    def generate_uicopilot(*args, **kwargs):
        raise RuntimeError(f"UICoPilot method unavailable: {_UICOPILOT_IMPORT_ERROR}")


try:
    from methods.layoutcoder import generate_layoutcoder
except Exception as error:
    _LAYOUTCODER_IMPORT_ERROR = error

    def generate_layoutcoder(*args, **kwargs):
        raise RuntimeError(f"LayoutCoder method unavailable: {_LAYOUTCODER_IMPORT_ERROR}")

load_dotenv()

app = Flask(__name__)

# In-memory state
RUNS_DB = {}  # {run_id: Run}
API_KEYS = {}  # {api_key: metadata}
API_KEYS[DEFAULT_API_KEY] = {
    "email": "dev@localhost",
    "verified": True,
    "created_at": datetime.datetime.now().isoformat(),
}

SUPPORTED_DATASETS = list(DATASETS_CONFIG.keys())

# Service wiring
require_api_key = create_require_api_key(API_KEYS)
resolve_path = create_path_resolver(DATA_DIR)
get_bot = create_bot_factory(
    get_model_info=get_model_info,
    supported_models=SUPPORTED_MODELS,
    default_openai_base_url=OPENAI_BASE_URL,
)
generate_dcgen = create_dcgen_generator(
    prompt_dcgen=PROMPT_DCGEN,
    seg_params_default=SEG_PARAMS_DEFAULT,
)
run_experiment_task = create_run_experiment_task(
    ensure_dir=ensure_dir,
    get_image_files=get_image_files,
    get_bot=get_bot,
    generate_dcgen=generate_dcgen,
    generate_latcoder=generate_latcoder,
    generate_uicopilot=generate_uicopilot,
    generate_layoutcoder=generate_layoutcoder,
    generate_single=generate_single,
    prompt_direct=PROMPT_DIRECT,
    seg_params_default=SEG_PARAMS_DEFAULT,
    take_screenshots_task=take_screenshots_task,
    run_evaluation_for_run=run_evaluation_for_run,
)
provider_env_status = get_provider_env_status(OPENAI_BASE_URL)

# Load persisted runs (for gunicorn / restart resilience)
loaded_runs = load_existing_runs(results_dir=RESULTS_DIR, runs_db=RUNS_DB, run_cls=Run)
print(f"[UIBenchKit] Loaded {loaded_runs} existing runs from disk")

# Route registration
auth_bp = create_auth_blueprint(
    require_api_key=require_api_key,
    api_keys=API_KEYS,
    api_version=API_VERSION,
    supported_datasets=SUPPORTED_DATASETS,
    supported_models=SUPPORTED_MODELS,
    model_families=MODEL_FAMILIES,
    supported_methods=SUPPORTED_METHODS,
    provider_env_status=provider_env_status,
)
datasets_bp = create_datasets_blueprint(
    supported_datasets=SUPPORTED_DATASETS,
    get_dataset_manager=get_dataset_manager,
)
runs_bp = create_runs_blueprint(
    require_api_key=require_api_key,
    runs_db=RUNS_DB,
    run_cls=Run,
    supported_models=SUPPORTED_MODELS,
    supported_methods=SUPPORTED_METHODS,
    supported_datasets=SUPPORTED_DATASETS,
    datasets_config=DATASETS_CONFIG,
    get_model_info=get_model_info,
    get_dataset_manager=get_dataset_manager,
    resolve_path=resolve_path,
    sanitize_for_filename=sanitize_for_filename,
    run_experiment_task=run_experiment_task,
    run_evaluation_for_run=run_evaluation_for_run,
    get_bot=get_bot,
    generate_dcgen=generate_dcgen,
    generate_latcoder=generate_latcoder,
    generate_uicopilot=generate_uicopilot,
    generate_layoutcoder=generate_layoutcoder,
    generate_single=generate_single,
    seg_params_default=SEG_PARAMS_DEFAULT,
    prompt_direct=PROMPT_DIRECT,
    take_screenshots_task=take_screenshots_task,
)

app.register_blueprint(auth_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(runs_bp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UIBenchKit API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    ensure_dir(DATA_DIR)
    ensure_dir(RESULTS_DIR)

    print(
        f"[UIBenchKit] API v{API_VERSION} | host={args.host}:{args.port} | "
        f"datasets={','.join(SUPPORTED_DATASETS)} | methods={','.join(SUPPORTED_METHODS)}"
    )
    print(f"[UIBenchKit] OpenAI base URL: {OPENAI_BASE_URL}")
    print(f"[UIBenchKit] Default dev API key: {DEFAULT_API_KEY}")
    print(f"[UIBenchKit] Provider env status: {provider_env_status}")

    app.run(host=args.host, port=args.port, debug=args.debug)
