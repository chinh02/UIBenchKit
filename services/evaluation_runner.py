#!/usr/bin/env python3
"""
Evaluation orchestration for completed runs.
"""

import os
import platform
import sys

from evaluation import FineGrainedEvaluator, CLIPScoreEvaluator, CodeSimilarityEvaluator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(PROJECT_ROOT, "scripts", "metric")
if METRICS_PATH not in sys.path:
    sys.path.insert(0, METRICS_PATH)

FINE_GRAINED_METRICS_AVAILABLE = platform.system() in ["Linux", "Darwin"]
if FINE_GRAINED_METRICS_AVAILABLE:
    try:
        from Design2Code.metrics.visual_score import visual_eval_v3_multi  # noqa: F401
        print("[UIBenchKit] Fine-grained metrics (Design2Code) loaded successfully")
    except (ImportError, RuntimeError, Exception) as error:
        FINE_GRAINED_METRICS_AVAILABLE = False
        print(f"[UIBenchKit] Fine-grained metrics not available: {error}")


def run_evaluation_for_run(run) -> dict:
    """Run evaluation for a single run using modular evaluators."""
    results = {
        "dataset": run.dataset,
        "model": run.model,
        "method": run.method,
        "run_id": run.run_id,
        "metrics": {},
    }

    completed_sample_ids = []
    for instance_id, instance in run.instances.items():
        if instance.get("status") == "completed":
            test_file = instance.get("result")
            if test_file and os.path.exists(test_file):
                completed_sample_ids.append(instance_id)

    if not completed_sample_ids:
        return results

    try:
        code_sim_evaluator = CodeSimilarityEvaluator()
        code_sim_scores = {}
        for instance_id in completed_sample_ids:
            ref_file = os.path.join(run.input_dir, f"{instance_id}.html")
            test_file = run.instances[instance_id].get("result")
            if os.path.exists(ref_file) and os.path.exists(test_file):
                eval_result = code_sim_evaluator.evaluate_sample(
                    generated_html_path=test_file,
                    reference_html_path=ref_file,
                )
                if eval_result.success and "overall" in eval_result.scores:
                    code_sim_scores[instance_id] = eval_result.scores["overall"]

        if code_sim_scores:
            results["metrics"]["code_similarity"] = {
                "scores": code_sim_scores,
                "average": sum(code_sim_scores.values()) / len(code_sim_scores),
            }
    except Exception as error:
        results["metrics"]["code_similarity"] = {"error": str(error)}

    try:
        clip_evaluator = CLIPScoreEvaluator()
        clip_evaluator.initialize()
        clip_scores = {}
        for instance_id in completed_sample_ids:
            ref_file = os.path.join(run.input_dir, f"{instance_id}.png")
            generated_html = run.instances[instance_id].get("result")
            test_file = generated_html.replace(".html", ".png") if generated_html else ""
            if os.path.exists(ref_file) and os.path.exists(test_file):
                eval_result = clip_evaluator.evaluate_sample(
                    generated_html_path=generated_html,
                    reference_image_path=ref_file,
                    generated_screenshot_path=test_file,
                )
                if eval_result.success and "clip_score" in eval_result.scores:
                    clip_scores[instance_id] = eval_result.scores["clip_score"]
        clip_evaluator.cleanup()

        if clip_scores:
            results["metrics"]["clip"] = {
                "scores": clip_scores,
                "average": sum(clip_scores.values()) / len(clip_scores),
            }
    except ImportError:
        results["metrics"]["clip"] = {"error": "CLIP dependencies not installed (transformers, torch)"}
    except Exception as error:
        results["metrics"]["clip"] = {"error": str(error)}

    if FINE_GRAINED_METRICS_AVAILABLE:
        try:
            fine_grained_results = run_fine_grained_evaluation(run)
            if fine_grained_results:
                results["metrics"]["fine_grained"] = fine_grained_results
        except Exception as error:
            results["metrics"]["fine_grained"] = {"error": str(error)}
    else:
        results["metrics"]["fine_grained"] = {
            "error": f"Fine-grained metrics only available on Linux/Mac. Current platform: {platform.system()}"
        }

    return results


def run_fine_grained_evaluation(run) -> dict:
    """Run Design2Code fine-grained visual evaluation for completed samples."""
    if not FINE_GRAINED_METRICS_AVAILABLE:
        return None

    completed_sample_ids = []
    for instance_id, instance in run.instances.items():
        if instance.get("status") == "completed":
            ref_html = os.path.join(run.input_dir, f"{instance_id}.html")
            test_html = instance.get("result")
            if ref_html and test_html and os.path.exists(ref_html) and os.path.exists(test_html):
                completed_sample_ids.append(instance_id)

    if not completed_sample_ids:
        return None

    try:
        evaluator = FineGrainedEvaluator({"metrics_path": METRICS_PATH})
        print(
            f"[UIBenchKit] Running fine-grained evaluation on {len(completed_sample_ids)} "
            "instances using FineGrainedEvaluator"
        )
        eval_results = evaluator.evaluate_run(
            run_dir=run.output_dir,
            reference_dir=run.input_dir,
            sample_ids=completed_sample_ids,
        )
        evaluator.cleanup()

        if not eval_results or "aggregate" not in eval_results:
            return None

        aggregate = eval_results.get("aggregate", {})
        per_sample = eval_results.get("per_sample", {})
        failed_samples = eval_results.get("failed_samples", [])

        results = {}
        metric_names = ["block_match", "text", "position", "color", "clip", "overall"]
        for metric_name in metric_names:
            scores = {}
            for sample_id, sample_result in per_sample.items():
                if sample_result.get("success") and "scores" in sample_result:
                    score_value = sample_result["scores"].get(metric_name)
                    if score_value is not None:
                        scores[sample_id] = float(score_value)

            if scores:
                results[metric_name] = {
                    "scores": scores,
                    "average": sum(scores.values()) / len(scores),
                }

        if failed_samples:
            results["_metadata"] = {
                "failed_samples": failed_samples,
                "failed_count": len(failed_samples),
                "successful_count": aggregate.get(
                    "successful_count",
                    len(completed_sample_ids) - len(failed_samples),
                ),
                "total_attempted": len(completed_sample_ids),
            }
            print(f"[UIBenchKit] Fine-grained evaluation: {len(failed_samples)} samples failed evaluation")

        return results if results else None

    except Exception as error:
        import traceback

        print(f"[UIBenchKit] Fine-grained evaluation error: {error}")
        print(f"[UIBenchKit] Traceback:\n{traceback.format_exc()}")
        return {"error": str(error)}
