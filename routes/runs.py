#!/usr/bin/env python3
"""
Run management routes.
"""

import os
import shutil
import datetime
import tempfile
import uuid
from threading import Thread
from flask import Blueprint, jsonify, request, g, send_file, after_this_request


def create_runs_blueprint(
    *,
    require_api_key,
    runs_db: dict,
    run_cls,
    supported_models: list,
    supported_methods: list,
    supported_datasets: list,
    datasets_config: dict,
    get_model_info,
    get_dataset_manager,
    resolve_path,
    sanitize_for_filename,
    run_experiment_task,
    run_evaluation_for_run,
    get_bot,
    generate_dcgen,
    generate_latcoder,
    generate_uicopilot,
    generate_layoutcoder,
    generate_single,
    seg_params_default: dict,
    prompt_direct: str,
    take_screenshots_task,
):
    """Create runs blueprint with injected dependencies."""
    bp = Blueprint("runs", __name__)

    @bp.route("/submit", methods=["POST"])
    @require_api_key
    def submit():
        data = request.get_json()

        if not data:
            return jsonify({"message": "Request body is required"}), 400
        if "model" not in data:
            return jsonify({"message": "Missing required field: model"}), 400
        if "method" not in data:
            return jsonify({"message": "Missing required field: method"}), 400

        model_input = data["model"]
        method = data["method"].lower()
        dataset_name = data.get("dataset")
        user_api_key = data.get("user_api_key")
        user_base_url = data.get("user_base_url")
        sample_ids = data.get("sample_ids")

        model_family, model_version = get_model_info(model_input)
        if not model_family:
            return (
                jsonify(
                    {
                        "message": (
                            f"Unsupported model: {model_input}. Supported families: "
                            f"{', '.join(supported_models)}. Use GET /health to see all supported versions."
                        )
                    }
                ),
                400,
            )

        if method not in supported_methods:
            return jsonify({"message": f"Unsupported method: {method}. Supported: {', '.join(supported_methods)}"}), 400

        if dataset_name:
            if dataset_name not in supported_datasets:
                return (
                    jsonify(
                        {
                            "message": (
                                f"Unsupported dataset: {dataset_name}. Supported: {', '.join(supported_datasets)}"
                            )
                        }
                    ),
                    400,
                )

            dm = get_dataset_manager()
            dataset_info = dm.get_dataset_info(dataset_name)
            if not dataset_info:
                return (
                    jsonify(
                        {
                            "message": (
                                f"Dataset {dataset_name} not downloaded. "
                                f"Use POST /datasets/{dataset_name}/download first."
                            )
                        }
                    ),
                    400,
                )

            input_dir = dm.prepare_benchmark_dir(dataset_name, sample_ids)
            run_id = data.get("run_id") or (
                f"{dataset_name}_{method}_{sanitize_for_filename(model_version)}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif "input_dir" in data:
            input_dir = resolve_path(data["input_dir"])
            if not os.path.exists(input_dir):
                return jsonify({"message": f"Input directory not found: {input_dir}"}), 400
            run_id = data.get("run_id") or (
                f"{method}_{sanitize_for_filename(model_version)}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            dataset_name = None
        else:
            return jsonify({"message": "Either 'dataset' or 'input_dir' must be provided"}), 400

        if run_id in runs_db:
            existing = runs_db[run_id]
            if existing.status in ["running", "pending"]:
                return jsonify({"message": f"Run {run_id} already exists and is {existing.status}", "launched": False, "run_id": run_id})
            return jsonify({"message": f"Run {run_id} already completed", "launched": False, "run_id": run_id})

        run = run_cls(
            run_id,
            model_version,
            method,
            input_dir,
            g.api_key,
            dataset=dataset_name,
            sample_ids=sample_ids,
            user_api_key=user_api_key,
            user_base_url=user_base_url,
        )
        runs_db[run_id] = run

        thread = Thread(target=run_experiment_task, args=(run,))
        thread.start()

        return jsonify(
            {
                "message": f"Run {run_id} submitted successfully",
                "launched": True,
                "run_id": run_id,
                "model": model_version,
                "model_family": model_family,
                "method": method,
                "dataset": dataset_name,
            }
        )

    @bp.route("/poll-jobs", methods=["GET"])
    @require_api_key
    def poll_jobs():
        data = request.get_json(silent=True) or {}
        run_id = data.get("run_id") or request.args.get("run_id")

        if not run_id:
            return jsonify({"message": "run_id is required"}), 400
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        return jsonify(run.get_poll_status())

    @bp.route("/get-report", methods=["POST"])
    @require_api_key
    def get_report():
        data = request.get_json()

        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403
        if run.status != "completed":
            return jsonify({"message": f"Run {run_id} is not completed yet. Status: {run.status}", "status": run.status}), 400

        report = {
            "run_id": run.run_id,
            "model": run.model,
            "method": run.method,
            "dataset": run.dataset,
            "dataset_info": datasets_config.get(run.dataset, {}) if run.dataset else None,
            "total_instances": run.total_instances,
            "completed_instances": len([i for i in run.instances.values() if i.get("status") == "completed"]),
            "failed_instances": len([i for i in run.instances.values() if i.get("status") == "failed"]),
            "resolved_instances": len([i for i in run.instances.values() if i.get("status") == "completed"]),
            "pending_instances": 0,
            "error_instances": len([i for i in run.instances.values() if i.get("status") == "failed"]),
            "submitted_instances": run.total_instances,
            "token_usage": run.token_usage,
            "cost_estimate": run.cost_estimate,
            "evaluation": run.evaluation,
            "results": {
                "dataset": run.dataset,
                "method": run.method,
                "model": run.model,
                "output_dir": run.output_dir,
                "instances": {
                    instance_id: {
                        "status": inst.get("status"),
                        "output_file": inst.get("result"),
                        "error": inst.get("error"),
                        "dataset": run.dataset,
                        "method": run.method,
                        "model": run.model,
                    }
                    for instance_id, inst in run.instances.items()
                },
            },
            "created_at": run.created_at,
            "completed_at": run.completed_at,
        }

        return jsonify({"report": report})

    @bp.route("/download-artifacts", methods=["GET"])
    @require_api_key
    def download_artifacts():
        run_id = request.args.get("run_id")
        if not run_id:
            return jsonify({"message": "run_id is required"}), 400

        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        output_dir = run.output_dir
        if not output_dir or not os.path.isdir(output_dir):
            return jsonify({"message": f"Artifacts directory not found for run {run_id}"}), 404

        archive_base = os.path.join(
            tempfile.gettempdir(),
            f"dcgen_artifacts_{run_id}_{uuid.uuid4().hex}",
        )
        archive_path = shutil.make_archive(
            base_name=archive_base,
            format="zip",
            root_dir=output_dir,
        )

        @after_this_request
        def cleanup_archive(response):
            try:
                os.remove(archive_path)
            except OSError:
                pass
            return response

        return send_file(
            archive_path,
            as_attachment=True,
            download_name=f"{run_id}_artifacts.zip",
            mimetype="application/zip",
        )

    @bp.route("/list-runs", methods=["POST"])
    @require_api_key
    def list_runs():
        data = request.get_json() or {}
        model_filter = data.get("model")
        method_filter = data.get("method")
        dataset_filter = data.get("dataset")

        runs_list = []
        for run_id, run in runs_db.items():
            if run.api_key != g.api_key:
                continue
            if model_filter and run.model != model_filter:
                continue
            if method_filter and run.method != method_filter:
                continue
            if dataset_filter and run.dataset != dataset_filter:
                continue
            runs_list.append(
                {
                    "run_id": run_id,
                    "model": run.model,
                    "method": run.method,
                    "dataset": run.dataset,
                    "status": run.status,
                    "created_at": run.created_at,
                }
            )

        return jsonify({"runs": runs_list, "run_ids": [r["run_id"] for r in runs_list]})

    @bp.route("/delete-run", methods=["DELETE", "POST"])
    @require_api_key
    def delete_run():
        data = request.get_json()
        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        del runs_db[run_id]
        if os.path.exists(run.output_dir):
            shutil.rmtree(run.output_dir, ignore_errors=True)

        return jsonify({"message": f"Run {run_id} deleted successfully"})

    @bp.route("/stop-run", methods=["POST"])
    @require_api_key
    def stop_run():
        data = request.get_json()
        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        run_eval = data.get("run_evaluation", True)

        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        completed_count = len([i for i in run.instances.values() if i.get("status") == "completed"])
        running_count = len([i for i in run.instances.values() if i.get("status") == "running"])
        pending_count = len([i for i in run.instances.values() if i.get("status") == "pending"])

        for _, instance in run.instances.items():
            if instance.get("status") == "pending":
                instance["status"] = "skipped"
                instance["error"] = "Run stopped by user"
            elif instance.get("status") == "running":
                instance["status"] = "stopped"
                instance["error"] = "Run stopped by user"

        if run.status in ["pending", "running"]:
            run.status = "stopped"

        if completed_count == 0:
            run.completed_at = datetime.datetime.now().isoformat()
            run.save_to_disk()
            return jsonify(
                {
                    "message": f"Run {run_id} stopped. No completed instances to evaluate.",
                    "run_id": run_id,
                    "completed_instances": 0,
                    "stopped_instances": running_count,
                    "skipped_instances": pending_count,
                }
            )

        if run_eval:
            def run_evaluation_task():
                try:
                    run.evaluation = run_evaluation_for_run(run)
                    run.status = "completed"
                    run.completed_at = datetime.datetime.now().isoformat()
                    run.save_to_disk()
                except Exception as e:
                    run.error = f"Evaluation failed: {str(e)}"
                    run.save_to_disk()

            thread = Thread(target=run_evaluation_task)
            thread.start()

            return jsonify(
                {
                    "message": f"Run {run_id} stopped. Running evaluation on {completed_count} completed instances.",
                    "run_id": run_id,
                    "completed_instances": completed_count,
                    "stopped_instances": running_count,
                    "skipped_instances": pending_count,
                    "evaluation_started": True,
                }
            )

        run.completed_at = datetime.datetime.now().isoformat()
        run.save_to_disk()
        return jsonify(
            {
                "message": f"Run {run_id} stopped. Use rerun-evaluation to evaluate later.",
                "run_id": run_id,
                "completed_instances": completed_count,
                "stopped_instances": running_count,
                "skipped_instances": pending_count,
                "evaluation_started": False,
            }
        )

    @bp.route("/rerun-evaluation", methods=["POST"])
    @require_api_key
    def rerun_evaluation():
        data = request.get_json()
        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        completed_instances = len([i for i in run.instances.values() if i.get("status") == "completed"])
        if completed_instances == 0:
            return jsonify({"message": f"Run {run_id} has no completed instances to evaluate", "status": run.status}), 400

        run.status = "evaluating"
        run.evaluation = None
        run.save_to_disk()

        def run_evaluation_task():
            try:
                missing_screenshots = False
                for _, instance in run.instances.items():
                    if instance.get("status") == "completed":
                        result_file = instance.get("result", "")
                        png_file = result_file.replace(".html", ".png") if result_file else ""
                        if not png_file or not os.path.exists(png_file):
                            missing_screenshots = True
                            break

                if missing_screenshots:
                    take_screenshots_task(run.output_dir, replace=False)

                run.evaluation = run_evaluation_for_run(run)
                run.status = "completed"
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
            except Exception as e:
                run.status = "failed"
                run.error = f"Evaluation failed: {str(e)}"
                run.save_to_disk()

        thread = Thread(target=run_evaluation_task)
        thread.start()

        return jsonify(
            {
                "message": f"Re-running evaluation for run {run_id}",
                "run_id": run_id,
                "completed_instances": completed_instances,
                "status": "evaluating",
            }
        )

    @bp.route("/resume-run", methods=["POST"])
    @require_api_key
    def resume_run():
        data = request.get_json()
        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403
        if run.status == "running":
            return jsonify({"message": f"Run {run_id} is already running", "status": run.status}), 400

        pending_instances = {k: v for k, v in run.instances.items() if v.get("status") == "pending"}
        if not pending_instances:
            return jsonify({"message": f"Run {run_id} has no pending instances to process", "status": run.status}), 400

        def resume_run_task():
            try:
                run.status = "running"
                bot = get_bot(run.model, user_api_key=run.user_api_key, user_base_url=run.user_base_url)
                if hasattr(bot, "reset_token_usage"):
                    bot.reset_token_usage()

                for instance_id in sorted(pending_instances.keys(), key=lambda x: int(x)):
                    img_path = os.path.join(run.input_dir, f"{instance_id}.png")
                    if not os.path.exists(img_path):
                        run.instances[instance_id]["status"] = "failed"
                        run.instances[instance_id]["error"] = "Image file not found"
                        continue

                    filename = f"{instance_id}.html"
                    save_path = os.path.join(run.output_dir, filename)
                    run.instances[instance_id]["status"] = "running"

                    try:
                        if run.method == "dcgen":
                            generate_dcgen(bot, img_path, save_path, seg_params_default)
                        elif run.method == "latcoder":
                            generate_latcoder(bot, img_path, save_path)
                        elif run.method == "uicopilot":
                            generate_uicopilot(bot, img_path, save_path)
                        elif run.method == "layoutcoder":
                            generate_layoutcoder(bot, img_path, save_path)
                        else:
                            generate_single(prompt_direct, bot, img_path, save_path)

                        run.instances[instance_id]["status"] = "completed"
                        run.instances[instance_id]["result"] = save_path
                    except Exception as e:
                        run.instances[instance_id]["status"] = "failed"
                        run.instances[instance_id]["error"] = str(e)

                placeholder_src = os.path.join(run.input_dir, "placeholder.png")
                placeholder_dst = os.path.join(run.output_dir, "placeholder.png")
                if os.path.exists(placeholder_src) and not os.path.exists(placeholder_dst):
                    shutil.copy(placeholder_src, placeholder_dst)

                take_screenshots_task(run.output_dir, replace=False)

                if hasattr(bot, "print_token_usage"):
                    new_usage = bot.print_token_usage(f"{run.method}_resume")
                    if run.token_usage and new_usage:
                        run.token_usage["total_prompt_tokens"] += new_usage.get("total_prompt_tokens", 0)
                        run.token_usage["total_response_tokens"] += new_usage.get("total_response_tokens", 0)
                        run.token_usage["call_count"] += new_usage.get("call_count", 0)
                    elif new_usage:
                        run.token_usage = new_usage

                run.evaluation = run_evaluation_for_run(run)
                run.status = "completed"
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
            except Exception as e:
                run.status = "failed"
                run.error = f"Resume failed: {str(e)}"
                run.save_to_disk()

        thread = Thread(target=resume_run_task)
        thread.start()

        return jsonify(
            {
                "message": f"Resuming run {run_id} with {len(pending_instances)} pending instances",
                "run_id": run_id,
                "pending_instances": list(pending_instances.keys()),
                "total_pending": len(pending_instances),
            }
        )

    @bp.route("/retry-failed", methods=["POST"])
    @require_api_key
    def retry_failed():
        data = request.get_json()
        if not data or "run_id" not in data:
            return jsonify({"message": "run_id is required"}), 400

        run_id = data["run_id"]
        max_retries = data.get("max_retries", 3)
        if run_id not in runs_db:
            return jsonify({"message": f"Run {run_id} not found"}), 404

        run = runs_db[run_id]
        if run.api_key != g.api_key:
            return jsonify({"message": "Access denied"}), 403

        failed_instances = {k: v for k, v in run.instances.items() if v.get("status") in ["failed", "stopped"]}
        if not failed_instances:
            return jsonify({"message": f"Run {run_id} has no failed instances to retry", "status": run.status}), 400

        def retry_failed_task():
            try:
                run.status = "running"
                bot = get_bot(run.model, user_api_key=run.user_api_key, user_base_url=run.user_base_url)
                if hasattr(bot, "reset_token_usage"):
                    bot.reset_token_usage()

                for instance_id, instance in failed_instances.items():
                    img_path = os.path.join(run.input_dir, f"{instance_id}.png")
                    if not os.path.exists(img_path):
                        continue

                    filename = f"{instance_id}.html"
                    save_path = os.path.join(run.output_dir, filename)
                    run.instances[instance_id]["status"] = "running"
                    run.instances[instance_id]["retry_count"] = instance.get("retry_count", 0) + 1

                    if run.instances[instance_id]["retry_count"] > max_retries:
                        run.instances[instance_id]["status"] = "failed"
                        run.instances[instance_id]["error"] = f"Max retries ({max_retries}) exceeded"
                        continue

                    try:
                        if run.method == "dcgen":
                            generate_dcgen(bot, img_path, save_path, seg_params_default)
                        elif run.method == "latcoder":
                            generate_latcoder(bot, img_path, save_path)
                        elif run.method == "uicopilot":
                            generate_uicopilot(bot, img_path, save_path)
                        elif run.method == "layoutcoder":
                            generate_layoutcoder(bot, img_path, save_path)
                        else:
                            generate_single(prompt_direct, bot, img_path, save_path)

                        run.instances[instance_id]["status"] = "completed"
                        run.instances[instance_id]["result"] = save_path
                        run.instances[instance_id]["error"] = None
                    except Exception as e:
                        run.instances[instance_id]["status"] = "failed"
                        run.instances[instance_id]["error"] = str(e)

                placeholder_src = os.path.join(run.input_dir, "placeholder.png")
                placeholder_dst = os.path.join(run.output_dir, "placeholder.png")
                if os.path.exists(placeholder_src) and not os.path.exists(placeholder_dst):
                    shutil.copy(placeholder_src, placeholder_dst)

                take_screenshots_task(run.output_dir, replace=False)

                if hasattr(bot, "print_token_usage"):
                    new_usage = bot.print_token_usage(f"{run.method}_retry")
                    if run.token_usage and new_usage:
                        run.token_usage["total_prompt_tokens"] += new_usage.get("total_prompt_tokens", 0)
                        run.token_usage["total_response_tokens"] += new_usage.get("total_response_tokens", 0)
                        run.token_usage["call_count"] += new_usage.get("call_count", 0)
                    elif new_usage:
                        run.token_usage = new_usage

                run.evaluation = run_evaluation_for_run(run)
                run.status = "completed"
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
            except Exception as e:
                run.status = "failed"
                run.error = f"Retry failed: {str(e)}"
                run.save_to_disk()

        thread = Thread(target=retry_failed_task)
        thread.start()

        return jsonify(
            {
                "message": f"Retrying {len(failed_instances)} failed instances for run {run_id}",
                "run_id": run_id,
                "failed_instances": list(failed_instances.keys()),
                "max_retries": max_retries,
            }
        )

    @bp.route("/run-all", methods=["POST"])
    @require_api_key
    def run_all():
        data = request.get_json()
        if not data:
            return jsonify({"message": "Request body is required"}), 400
        if "model" not in data:
            return jsonify({"message": "Missing required field: model"}), 400

        model_input = data["model"]
        dataset_name = data.get("dataset")
        sample_ids = data.get("sample_ids")

        model_family, model_version = get_model_info(model_input)
        if not model_family:
            return jsonify({"message": f"Unsupported model: {model_input}. Supported families: {', '.join(supported_models)}"}), 400

        if dataset_name:
            if dataset_name not in supported_datasets:
                return jsonify({"message": f"Unsupported dataset: {dataset_name}. Supported: {', '.join(supported_datasets)}"}), 400

            dm = get_dataset_manager()
            dataset_info = dm.get_dataset_info(dataset_name)
            if not dataset_info:
                return (
                    jsonify(
                        {
                            "message": (
                                f"Dataset {dataset_name} not downloaded. "
                                f"Use POST /datasets/{dataset_name}/download first."
                            )
                        }
                    ),
                    400,
                )

            input_dir = dm.prepare_benchmark_dir(dataset_name, sample_ids)
            base_run_id = data.get("run_id") or (
                f"{dataset_name}_{sanitize_for_filename(model_version)}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif "input_dir" in data:
            input_dir = resolve_path(data["input_dir"])
            if not os.path.exists(input_dir):
                return jsonify({"message": f"Input directory not found: {input_dir}"}), 400
            base_run_id = data.get("run_id") or (
                f"exp_{sanitize_for_filename(model_version)}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            dataset_name = None
        else:
            return jsonify({"message": "Either 'dataset' or 'input_dir' must be provided"}), 400

        runs = []
        for method in ["dcgen", "direct"]:
            run_id = f"{base_run_id}_{method}"
            run = run_cls(run_id, model_version, method, input_dir, g.api_key, dataset=dataset_name, sample_ids=sample_ids)
            runs_db[run_id] = run
            runs.append(run_id)
            thread = Thread(target=run_experiment_task, args=(run,))
            thread.start()

        return jsonify(
            {
                "message": "Full pipeline started",
                "run_ids": runs,
                "dataset": dataset_name,
                "model": model_version,
                "model_family": model_family,
            }
        )

    return bp
