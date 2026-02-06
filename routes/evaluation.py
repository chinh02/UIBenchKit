#!/usr/bin/env python3
"""
Evaluation Routes
=================

Routes for running and retrieving evaluations, including MLLM Judge.
"""

import os
import json
import threading
import time
from flask import Blueprint, request, jsonify, g
from functools import wraps

evaluation_bp = Blueprint('evaluation', __name__)

# Track running MLLM judge jobs
MLLM_JUDGE_JOBS = {}


def require_api_key(f):
    """Decorator to require API key authentication (permissive for evaluation routes)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key') or request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            try:
                json_data = request.get_json(silent=True)
                if json_data:
                    api_key = json_data.get("api_key")
            except:
                pass
        
        if not api_key:
            return jsonify({"error": "API key required. Set x-api-key header."}), 401
        
        g.api_key = api_key
        return f(*args, **kwargs)
    
    return decorated_function


# ============================================================
# MLLM Judge Endpoint (New Feature)
# ============================================================

@evaluation_bp.route("/evaluate/mllm-judge", methods=["POST"])
@require_api_key
def mllm_judge_evaluation():
    """
    Run MLLM-as-a-Judge evaluation on a sample or run.
    
    Request body:
    {
        "mode": "single_score",       # single_score, pairwise, criteria_check
        "judge_model": "gemini-2.0-flash",  # Model to use as judge
        "reference_image": "/path/to/ref.png",  # Reference image path
        "generated_screenshot": "/path/to/gen.png",  # Generated screenshot
        "run_id": "my_run",           # Optional: evaluate entire run
        "sample_id": "sample1"        # Optional: specific sample
    }
    
    For pairwise comparison:
    {
        "mode": "pairwise",
        "judge_model": "gpt-4o",
        "reference_image": "/path/to/ref.png",
        "model_a_screenshot": "/path/to/model_a.png",
        "model_b_screenshot": "/path/to/model_b.png",
        "model_a_name": "GPT-4o",
        "model_b_name": "Gemini-2.0"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body required"}), 400
    
    mode = data.get("mode", "single_score")
    judge_model = data.get("judge_model", "gemini-2.0-flash")
    
    # Parse model family and version
    from config import get_model_info
    model_family, model_version = get_model_info(judge_model)
    
    if not model_family:
        return jsonify({
            "error": f"Unsupported judge model: {judge_model}"
        }), 400
    
    try:
        from evaluation.mllm_judge import MLLMJudgeEvaluator, JudgeMode
        
        evaluator = MLLMJudgeEvaluator({
            "model_family": model_family,
            "model_version": model_version,
            "mode": mode,
            "temperature": data.get("temperature", 0.1)
        })
        
        if mode == "pairwise":
            # Pairwise comparison
            reference_image = data.get("reference_image")
            model_a_screenshot = data.get("model_a_screenshot")
            model_b_screenshot = data.get("model_b_screenshot")
            
            if not all([reference_image, model_a_screenshot, model_b_screenshot]):
                return jsonify({
                    "error": "reference_image, model_a_screenshot, and model_b_screenshot required for pairwise mode"
                }), 400
            
            result = evaluator.evaluate_pairwise(
                reference_image,
                model_a_screenshot,
                model_b_screenshot,
                model_a_name=data.get("model_a_name", "Model A"),
                model_b_name=data.get("model_b_name", "Model B")
            )
            
        else:
            # Single score or criteria check
            reference_image = data.get("reference_image")
            generated_screenshot = data.get("generated_screenshot")
            
            if not reference_image or not generated_screenshot:
                return jsonify({
                    "error": "reference_image and generated_screenshot required"
                }), 400
            
            result = evaluator.evaluate_sample(
                generated_html_path=generated_screenshot,  # Used for sample_id
                reference_image_path=reference_image,
                generated_screenshot_path=generated_screenshot
            )
            result = result.to_dict()
        
        result["judge_model"] = f"{model_family}/{model_version}"
        result["token_usage"] = evaluator.get_token_usage()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "judge_model": judge_model
        }), 500


def _run_mllm_judge_background(job_id, run_id, run, judge_model, model_family, model_version, mode, sample_ids, save_interval, output_file):
    """Background worker for MLLM Judge evaluation."""
    from evaluation.mllm_judge import MLLMJudgeEvaluator
    from evaluation.base import EvaluationResult
    
    try:
        # Load existing results if resuming
        existing_results = {}
        existing_errors = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("per_sample", {})
                    existing_errors = existing_data.get("errors", [])
            except Exception:
                pass
        
        evaluator = MLLMJudgeEvaluator({
            "model_family": model_family,
            "model_version": model_version,
            "mode": mode
        })
        
        results = dict(existing_results)
        errors = list(existing_errors)
        
        instances_to_evaluate = sample_ids or list(run.instances.keys())
        
        # Skip already evaluated samples
        samples_to_process = [
            sid for sid in instances_to_evaluate 
            if sid not in results
        ]
        
        total_to_evaluate = len(samples_to_process)
        MLLM_JUDGE_JOBS[job_id]["total"] = total_to_evaluate
        MLLM_JUDGE_JOBS[job_id]["status"] = "running"
        
        evaluated_count = 0
        
        for instance_id in samples_to_process:
            instance = run.instances.get(instance_id)
            if not instance or instance.get("status") != "completed":
                continue
            
            ref_image = os.path.join(run.input_dir, f"{instance_id}.png")
            if not os.path.exists(ref_image):
                errors.append(f"{instance_id}: reference image not found")
                continue
            
            gen_html = instance.get("result")
            if not gen_html:
                errors.append(f"{instance_id}: no result HTML")
                continue
            
            gen_screenshot = gen_html.replace('.html', '.png')
            if not os.path.exists(gen_screenshot):
                gen_screenshot = gen_html.replace('.html', '_p.png')
            
            if not os.path.exists(gen_screenshot):
                errors.append(f"{instance_id}: generated screenshot not found")
                continue
            
            try:
                result = evaluator.evaluate_sample(
                    generated_html_path=gen_html,
                    reference_image_path=ref_image,
                    generated_screenshot_path=gen_screenshot
                )
                results[instance_id] = result.to_dict()
                evaluated_count += 1
                MLLM_JUDGE_JOBS[job_id]["evaluated"] = evaluated_count
                
                if evaluated_count % save_interval == 0:
                    _save_mllm_results(
                        output_file, run_id, judge_model, mode, 
                        results, errors, evaluator.get_token_usage(),
                        evaluated_count, total_to_evaluate
                    )
                    
            except Exception as e:
                errors.append(f"{instance_id}: {str(e)}")
        
        # Calculate final aggregates
        successful_results = []
        for sample_id, r in results.items():
            if r.get("success", True):
                eval_result = EvaluationResult(
                    sample_id=sample_id,
                    metric_name="mllm_judge",
                    scores=r.get("scores", {}),
                    success=True,
                    error=None,
                    metadata=r.get("metadata", {})
                )
                successful_results.append(eval_result)
        
        aggregated = evaluator.aggregate_results(successful_results)
        
        final_data = {
            "run_id": run_id,
            "judge_model": f"{model_family}/{model_version}",
            "mode": mode,
            "per_sample": results,
            "aggregate": aggregated,
            "errors": errors,
            "token_usage": evaluator.get_token_usage(),
            "status": "completed"
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        MLLM_JUDGE_JOBS[job_id]["status"] = "completed"
        MLLM_JUDGE_JOBS[job_id]["result"] = aggregated
        
    except Exception as e:
        MLLM_JUDGE_JOBS[job_id]["status"] = "failed"
        MLLM_JUDGE_JOBS[job_id]["error"] = str(e)


def _save_mllm_results(output_file, run_id, judge_model, mode, results, errors, token_usage, evaluated, total):
    """Helper to save incremental MLLM Judge results."""
    data = {
        "run_id": run_id,
        "judge_model": judge_model,
        "mode": mode,
        "per_sample": results,
        "errors": errors,
        "token_usage": token_usage,
        "status": "in_progress",
        "progress": f"{evaluated}/{total}"
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


@evaluation_bp.route("/evaluate/mllm-judge/run", methods=["POST"])
@require_api_key
def mllm_judge_run():
    """
    Submit MLLM Judge evaluation on a run. Returns immediately, runs in background.
    
    Request body:
    {
        "run_id": "my_run",
        "judge_model": "gemini-2.0-flash",
        "mode": "single_score",
        "sample_ids": ["sample1", "sample2"],  # Optional: specific samples
        "save_interval": 5  # Optional: save every N samples (default: 5)
    }
    
    Returns job_id to check status via GET /evaluate/mllm-judge/status?job_id=xxx
    """
    from api import RUNS_DB
    from config import get_model_info
    
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"error": "run_id required"}), 400
    
    run_id = data["run_id"]
    judge_model = data.get("judge_model", "gemini-2.0-flash")
    mode = data.get("mode", "single_score")
    sample_ids = data.get("sample_ids")
    save_interval = data.get("save_interval", 5)
    
    if run_id not in RUNS_DB:
        return jsonify({"error": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    if run.api_key != g.api_key:
        return jsonify({"error": "Access denied"}), 403
    
    model_family, model_version = get_model_info(judge_model)
    if not model_family:
        return jsonify({"error": f"Unsupported judge model: {judge_model}"}), 400
    
    # Determine samples to evaluate
    instances_to_evaluate = sample_ids or list(run.instances.keys())
    completed_samples = [
        sid for sid in instances_to_evaluate
        if run.instances.get(sid, {}).get("status") == "completed"
    ]
    
    # Check for existing results
    judge_model_safe = judge_model.replace("/", "_").replace("-", "_")
    output_file = os.path.join(run.output_dir, f"mllm_judge_{judge_model_safe}.json")
    
    already_evaluated = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                already_evaluated = len(existing_data.get("per_sample", {}))
        except Exception:
            pass
    
    remaining = len(completed_samples) - already_evaluated
    
    # Create job ID
    job_id = f"mllm_{run_id}_{judge_model_safe}_{int(time.time())}"
    
    MLLM_JUDGE_JOBS[job_id] = {
        "run_id": run_id,
        "judge_model": judge_model,
        "status": "starting",
        "total": remaining,
        "evaluated": 0,
        "output_file": output_file
    }
    
    # Start background thread
    thread = threading.Thread(
        target=_run_mllm_judge_background,
        args=(job_id, run_id, run, judge_model, model_family, model_version, mode, sample_ids, save_interval, output_file),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "message": "MLLM Judge evaluation submitted",
        "job_id": job_id,
        "run_id": run_id,
        "judge_model": judge_model,
        "total_samples": len(completed_samples),
        "already_evaluated": already_evaluated,
        "remaining": remaining,
        "output_file": output_file,
        "status_endpoint": f"/evaluate/mllm-judge/status?job_id={job_id}"
    })


@evaluation_bp.route("/evaluate/mllm-judge/status", methods=["GET"])
@require_api_key
def mllm_judge_status():
    """Check status of an MLLM Judge evaluation job."""
    job_id = request.args.get("job_id")
    run_id = request.args.get("run_id")
    
    if job_id and job_id in MLLM_JUDGE_JOBS:
        job = MLLM_JUDGE_JOBS[job_id]
        return jsonify({
            "job_id": job_id,
            "run_id": job.get("run_id"),
            "judge_model": job.get("judge_model"),
            "status": job.get("status"),
            "evaluated": job.get("evaluated", 0),
            "total": job.get("total", 0),
            "progress": f"{job.get('evaluated', 0)}/{job.get('total', 0)}",
            "output_file": job.get("output_file"),
            "error": job.get("error"),
            "result": job.get("result")
        })
    
    # Try to read status from file if no job_id or job not found
    if run_id:
        from api import RUNS_DB
        if run_id in RUNS_DB:
            run = RUNS_DB[run_id]
            import glob
            pattern = os.path.join(run.output_dir, "mllm_judge_*.json")
            files = glob.glob(pattern)
            
            if files:
                results = []
                for f in files:
                    try:
                        with open(f, 'r') as fp:
                            data = json.load(fp)
                            results.append({
                                "file": os.path.basename(f),
                                "status": data.get("status", "unknown"),
                                "progress": data.get("progress"),
                                "judge_model": data.get("judge_model"),
                                "samples_evaluated": len(data.get("per_sample", {}))
                            })
                    except Exception:
                        pass
                return jsonify({"run_id": run_id, "evaluations": results})
    
    return jsonify({"error": "Job not found. Provide job_id or run_id."}), 404


@evaluation_bp.route("/evaluate/compare-models", methods=["POST"])
@require_api_key
def compare_models():
    """
    Compare two model runs using MLLM Judge pairwise comparison.
    
    Request body:
    {
        "run_id_a": "run_gpt4o",
        "run_id_b": "run_gemini",
        "judge_model": "claude-3-5-sonnet",
        "sample_ids": ["sample1", "sample2"]  # Optional
    }
    """
    from api import RUNS_DB
    from config import get_model_info
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body required"}), 400
    
    run_id_a = data.get("run_id_a")
    run_id_b = data.get("run_id_b")
    judge_model = data.get("judge_model", "gemini-2.0-flash")
    sample_ids = data.get("sample_ids")
    
    if not run_id_a or not run_id_b:
        return jsonify({"error": "run_id_a and run_id_b required"}), 400
    
    if run_id_a not in RUNS_DB:
        return jsonify({"error": f"Run {run_id_a} not found"}), 404
    if run_id_b not in RUNS_DB:
        return jsonify({"error": f"Run {run_id_b} not found"}), 404
    
    run_a = RUNS_DB[run_id_a]
    run_b = RUNS_DB[run_id_b]
    
    # Verify ownership
    if run_a.api_key != g.api_key or run_b.api_key != g.api_key:
        return jsonify({"error": "Access denied"}), 403
    
    # Parse judge model
    model_family, model_version = get_model_info(judge_model)
    if not model_family:
        return jsonify({"error": f"Unsupported judge model: {judge_model}"}), 400
    
    try:
        from evaluation.mllm_judge import MLLMJudgeEvaluator
        
        evaluator = MLLMJudgeEvaluator({
            "model_family": model_family,
            "model_version": model_version,
            "mode": "pairwise"
        })
        
        # Find common samples
        common_samples = set(run_a.instances.keys()) & set(run_b.instances.keys())
        if sample_ids:
            common_samples = common_samples & set(sample_ids)
        
        results = []
        errors = []
        wins = {"model_a": 0, "model_b": 0, "tie": 0}
        
        for sample_id in common_samples:
            inst_a = run_a.instances.get(sample_id)
            inst_b = run_b.instances.get(sample_id)
            
            if not inst_a or inst_a.get("status") != "completed":
                continue
            if not inst_b or inst_b.get("status") != "completed":
                continue
            
            # Find images
            ref_image = os.path.join(run_a.input_dir, f"{sample_id}.png")
            if not os.path.exists(ref_image):
                continue
            
            screenshot_a = inst_a.get("result", "").replace('.html', '.png')
            screenshot_b = inst_b.get("result", "").replace('.html', '.png')
            
            if not os.path.exists(screenshot_a) or not os.path.exists(screenshot_b):
                errors.append(f"{sample_id}: screenshots not found")
                continue
            
            try:
                result = evaluator.evaluate_pairwise(
                    ref_image,
                    screenshot_a,
                    screenshot_b,
                    model_a_name=run_a.model,
                    model_b_name=run_b.model
                )
                
                result["sample_id"] = sample_id
                results.append(result)
                
                # Count wins
                winner = result.get("comparison", {}).get("winner", "tie").lower()
                if winner == "a":
                    wins["model_a"] += 1
                elif winner == "b":
                    wins["model_b"] += 1
                else:
                    wins["tie"] += 1
                    
            except Exception as e:
                errors.append(f"{sample_id}: {str(e)}")
        
        total_comparisons = len(results)
        
        return jsonify({
            "model_a": {
                "run_id": run_id_a,
                "model": run_a.model,
                "wins": wins["model_a"],
                "win_rate": wins["model_a"] / total_comparisons if total_comparisons else 0
            },
            "model_b": {
                "run_id": run_id_b,
                "model": run_b.model,
                "wins": wins["model_b"],
                "win_rate": wins["model_b"] / total_comparisons if total_comparisons else 0
            },
            "ties": wins["tie"],
            "total_comparisons": total_comparisons,
            "judge_model": f"{model_family}/{model_version}",
            "comparisons": results,
            "errors": errors,
            "token_usage": evaluator.get_token_usage()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
