#!/usr/bin/env python3
"""
UIBenchKit local runner (no API server required).

This script mirrors API-side execution logic for local development:
- Same model/env resolution via services.model_factory
- Same generation methods: direct, dcgen, latcoder, uicopilot, layoutcoder
- Same persisted run artifacts via run_model.Run

Examples:
  python run.py run --input ./data/demo --method direct --model gpt4
  python run.py run --input ./data/demo --method uicopilot --model claude --no-eval
  python run.py quick --image ./data/demo/0.png --method dcgen --model gemini
  python run.py preflight
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from config import (
    OPENAI_BASE_URL,
    PROMPT_DCGEN,
    PROMPT_DIRECT,
    RESULTS_DIR,
    SEG_PARAMS_DEFAULT,
    SUPPORTED_METHODS,
    SUPPORTED_MODELS,
    get_model_info,
)
from run_model import Run
from services.evaluation_runner import run_evaluation_for_run
from services.fs_utils import ensure_dir, get_image_files, sanitize_for_filename
from services.generation import create_dcgen_generator, generate_single, take_screenshots_task
from services.model_factory import create_bot_factory, get_provider_env_status

load_dotenv()


METHOD_IMPORT_ERRORS = {}

try:
    from methods.latcoder import generate_latcoder
except Exception as error:
    METHOD_IMPORT_ERRORS["latcoder"] = error

    def generate_latcoder(*args, **kwargs):
        raise RuntimeError(f"LatCoder unavailable: {METHOD_IMPORT_ERRORS['latcoder']}")


try:
    from methods.uicopilot import generate_uicopilot
except Exception as error:
    METHOD_IMPORT_ERRORS["uicopilot"] = error

    def generate_uicopilot(*args, **kwargs):
        raise RuntimeError(f"UICoPilot unavailable: {METHOD_IMPORT_ERRORS['uicopilot']}")


try:
    from methods.layoutcoder import generate_layoutcoder
except Exception as error:
    METHOD_IMPORT_ERRORS["layoutcoder"] = error

    def generate_layoutcoder(*args, **kwargs):
        raise RuntimeError(f"LayoutCoder unavailable: {METHOD_IMPORT_ERRORS['layoutcoder']}")


get_bot = create_bot_factory(
    get_model_info=get_model_info,
    supported_models=SUPPORTED_MODELS,
    default_openai_base_url=OPENAI_BASE_URL,
)
generate_dcgen = create_dcgen_generator(
    prompt_dcgen=PROMPT_DCGEN,
    seg_params_default=SEG_PARAMS_DEFAULT,
)


@dataclass
class LocalRunOptions:
    input_path: str
    method: str
    model: str
    run_id: str | None
    output_root: str
    force: bool
    no_screenshot: bool
    no_eval: bool
    max_instances: int | None
    user_api_key: str | None
    user_base_url: str | None


def _build_run_id(method: str, model_input: str, explicit_run_id: str | None) -> str:
    if explicit_run_id:
        return explicit_run_id
    _, model_version = get_model_info(model_input)
    version_for_name = sanitize_for_filename(model_version if model_version else model_input)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{method}_{version_for_name}_{timestamp}"


def _prepare_input_dir(input_path: str) -> tuple[str, str | None]:
    """
    Return (input_dir, temp_dir_to_cleanup).
    If input_path is a file, create a temp dir with input.png and optional placeholder.png.
    """
    abs_path = os.path.abspath(input_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Input path not found: {abs_path}")

    if os.path.isdir(abs_path):
        return abs_path, None

    ext = Path(abs_path).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        raise ValueError(f"Unsupported input image extension: {ext}")

    temp_dir = tempfile.mkdtemp(prefix="uibenchkit-local-")
    target_img = os.path.join(temp_dir, "input.png")
    shutil.copy(abs_path, target_img)

    input_placeholder = os.path.join(os.path.dirname(abs_path), "placeholder.png")
    project_placeholder = os.path.join(os.path.dirname(__file__), "placeholder.png")
    if os.path.exists(input_placeholder):
        shutil.copy(input_placeholder, os.path.join(temp_dir, "placeholder.png"))
    elif os.path.exists(project_placeholder):
        shutil.copy(project_placeholder, os.path.join(temp_dir, "placeholder.png"))

    return temp_dir, temp_dir


def _method_generate(method: str, bot, img_path: str, save_path: str):
    if method == "dcgen":
        return generate_dcgen(bot, img_path, save_path, SEG_PARAMS_DEFAULT)
    if method == "latcoder":
        return generate_latcoder(bot, img_path, save_path)
    if method == "uicopilot":
        return generate_uicopilot(bot, img_path, save_path)
    if method == "layoutcoder":
        return generate_layoutcoder(bot, img_path, save_path)
    return generate_single(PROMPT_DIRECT, bot, img_path, save_path)


def _copy_placeholder(input_dir: str, output_dir: str):
    src = os.path.join(input_dir, "placeholder.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(output_dir, "placeholder.png"))


def execute_local_run(options: LocalRunOptions) -> int:
    method = options.method.lower()
    if method not in SUPPORTED_METHODS:
        print(f"Unsupported method: {method}. Supported: {', '.join(SUPPORTED_METHODS)}")
        return 2

    if method in METHOD_IMPORT_ERRORS:
        print(f"Method '{method}' is unavailable in this environment: {METHOD_IMPORT_ERRORS[method]}")
        return 2

    model_family, model_version = get_model_info(options.model)
    if not model_family:
        print(f"Unsupported model: {options.model}. Supported families: {', '.join(SUPPORTED_MODELS)}")
        return 2

    input_dir, temp_dir = None, None
    try:
        input_dir, temp_dir = _prepare_input_dir(options.input_path)
        run_id = _build_run_id(method, options.model, options.run_id)
        run = Run(
            run_id=run_id,
            model=model_version,
            method=method,
            input_dir=input_dir,
            api_key="local-cli",
            user_api_key=options.user_api_key,
            user_base_url=options.user_base_url,
        )

        output_root = os.path.abspath(options.output_root)
        ensure_dir(output_root)
        run.output_dir = os.path.join(output_root, run_id)
        ensure_dir(run.output_dir)

        files = get_image_files(run.input_dir)
        if options.max_instances is not None:
            files = files[: options.max_instances]

        if not files:
            run.status = "failed"
            run.error = "No valid input PNG files found"
            run.completed_at = datetime.datetime.now().isoformat()
            run.save_to_disk()
            print("No valid input images found.")
            return 1

        for img_path in files:
            instance_id = os.path.basename(img_path).replace(".png", "")
            run.instances[instance_id] = {"status": "pending", "result": None}

        run.status = "running"
        run.total_instances = len(files)
        run.save_to_disk()

        print(f"\n{'=' * 60}")
        print("UIBENCHKIT LOCAL RUN")
        print(f"{'=' * 60}")
        print(f"Run ID:    {run_id}")
        print(f"Method:    {method}")
        print(f"Model:     {model_version} ({model_family})")
        print(f"Input:     {run.input_dir}")
        print(f"Output:    {run.output_dir}")
        print(f"Instances: {len(files)}")
        print(f"{'=' * 60}")

        bot = get_bot(
            run.model,
            user_api_key=run.user_api_key,
            user_base_url=run.user_base_url,
        )
        if hasattr(bot, "reset_token_usage"):
            bot.reset_token_usage()

        for index, img_path in enumerate(files, 1):
            instance_id = os.path.basename(img_path).replace(".png", "")
            save_path = os.path.join(run.output_dir, f"{instance_id}.html")
            run.instances[instance_id]["status"] = "running"

            if os.path.exists(save_path) and not options.force:
                run.instances[instance_id]["status"] = "completed"
                run.instances[instance_id]["result"] = save_path
                print(f"[{index}/{len(files)}] skip {instance_id} (exists)")
                continue

            try:
                _method_generate(method, bot, img_path, save_path)
                run.instances[instance_id]["status"] = "completed"
                run.instances[instance_id]["result"] = save_path
                print(f"[{index}/{len(files)}] done {instance_id}")
            except Exception as error:
                run.instances[instance_id]["status"] = "failed"
                run.instances[instance_id]["error"] = str(error)
                print(f"[{index}/{len(files)}] fail {instance_id}: {error}")

            if hasattr(bot, "print_token_usage"):
                run.token_usage = bot.print_token_usage(method)
            run.save_to_disk()

        _copy_placeholder(run.input_dir, run.output_dir)

        if not options.no_screenshot:
            try:
                take_screenshots_task(run.output_dir, replace=options.force)
            except Exception as error:
                print(f"Screenshot generation warning: {error}")

        if hasattr(bot, "print_token_usage"):
            run.token_usage = bot.print_token_usage(method)

        if not options.no_eval:
            try:
                run.evaluation = run_evaluation_for_run(run)
            except Exception as error:
                run.error = f"Evaluation warning: {error}"

        run.status = "completed"
        run.completed_at = datetime.datetime.now().isoformat()
        run.save_to_disk()

        completed = len([x for x in run.instances.values() if x.get("status") == "completed"])
        failed = len([x for x in run.instances.values() if x.get("status") == "failed"])
        print(f"\nCompleted: {completed}/{len(files)} | Failed: {failed}/{len(files)}")
        print(f"Artifacts: {run.output_dir}")
        return 0 if completed > 0 else 1

    except Exception as error:
        print(f"Fatal error: {error}")
        return 1
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def cmd_preflight(_args) -> int:
    status = get_provider_env_status(OPENAI_BASE_URL)
    print("UIBenchKit local preflight")
    print(f"Python: {sys.executable}")
    print(f"OpenAI base URL: {OPENAI_BASE_URL}")
    print("Provider env status:")
    print(json.dumps(status, indent=2))
    print("Method import status:")
    for method in SUPPORTED_METHODS:
        if method in METHOD_IMPORT_ERRORS:
            print(f"  - {method}: unavailable ({METHOD_IMPORT_ERRORS[method]})")
        else:
            print(f"  - {method}: available")
    return 0


def cmd_run(args) -> int:
    options = LocalRunOptions(
        input_path=args.input,
        method=args.method,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        force=args.force,
        no_screenshot=args.no_screenshot,
        no_eval=args.no_eval,
        max_instances=args.max_instances,
        user_api_key=args.user_api_key,
        user_base_url=args.user_base_url,
    )
    return execute_local_run(options)


def cmd_quick(args) -> int:
    options = LocalRunOptions(
        input_path=args.image,
        method=args.method,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        force=args.force,
        no_screenshot=not args.with_screenshot,
        no_eval=not args.with_eval,
        max_instances=1,
        user_api_key=args.user_api_key,
        user_base_url=args.user_base_url,
    )
    return execute_local_run(options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UIBenchKit local runner (no API server required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run local benchmark/inference")
    run_parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    run_parser.add_argument("--method", choices=SUPPORTED_METHODS, default="direct", help="Generation method")
    run_parser.add_argument("--model", "-m", default="gpt4", help="Model family or version")
    run_parser.add_argument("--run-id", help="Explicit run ID")
    run_parser.add_argument("--output-root", default=RESULTS_DIR, help="Root directory for run artifacts")
    run_parser.add_argument("--max-instances", type=int, help="Limit number of input images")
    run_parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    run_parser.add_argument("--no-screenshot", action="store_true", help="Skip screenshot generation")
    run_parser.add_argument("--no-eval", action="store_true", help="Skip evaluation stage")
    run_parser.add_argument("--user-api-key", help="Override provider API key for this run")
    run_parser.add_argument("--user-base-url", help="Override OpenAI-compatible base URL for this run")
    run_parser.set_defaults(func=cmd_run)

    quick_parser = subparsers.add_parser("quick", help="Quick single-image local run")
    quick_parser.add_argument("--image", "-i", required=True, help="Input image path")
    quick_parser.add_argument("--method", choices=SUPPORTED_METHODS, default="direct", help="Generation method")
    quick_parser.add_argument("--model", "-m", default="gpt4", help="Model family or version")
    quick_parser.add_argument("--run-id", help="Explicit run ID")
    quick_parser.add_argument("--output-root", default=RESULTS_DIR, help="Root directory for run artifacts")
    quick_parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    quick_parser.add_argument("--with-screenshot", action="store_true", help="Enable screenshot generation")
    quick_parser.add_argument("--with-eval", action="store_true", help="Enable evaluation")
    quick_parser.add_argument("--user-api-key", help="Override provider API key for this run")
    quick_parser.add_argument("--user-base-url", help="Override OpenAI-compatible base URL for this run")
    quick_parser.set_defaults(func=cmd_quick)

    preflight_parser = subparsers.add_parser("preflight", help="Check env + method readiness")
    preflight_parser.set_defaults(func=cmd_preflight)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
