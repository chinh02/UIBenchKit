#!/usr/bin/env python3
"""
Fine-Grained Evaluator
======================

Wrapper for Design2Code fine-grained visual metrics.
Provides Block-Match, Text, Position, Color, and CLIP scores.
"""

import os
import sys
import gc
import traceback
from typing import Dict, Any, Optional, List
import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import BaseEvaluator, EvaluationResult


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage info in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual = psutil.virtual_memory()
        return {
            "process_rss_mb": mem_info.rss / 1024 / 1024,
            "process_vms_mb": mem_info.vms / 1024 / 1024,
            "system_available_mb": virtual.available / 1024 / 1024,
            "system_percent_used": virtual.percent
        }
    except ImportError:
        return {"error": "psutil not installed"}
    except Exception as e:
        return {"error": str(e)}


def log_memory(prefix: str = ""):
    """Log current memory usage."""
    mem = get_memory_info()
    if "error" not in mem:
        print(f"[FineGrained] {prefix}Memory: process={mem['process_rss_mb']:.0f}MB, "
              f"available={mem['system_available_mb']:.0f}MB ({mem['system_percent_used']:.1f}% used)")
    return mem


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class FineGrainedEvaluator(BaseEvaluator):
    """
    Evaluator using Design2Code fine-grained visual metrics.
    
    Computes:
    - Block-Match: How well blocks/elements match
    - Text: Text content accuracy
    - Position: Element positioning accuracy
    - Color: Color matching accuracy
    - CLIP: Visual similarity using CLIP
    """
    
    metric_name = "fine_grained"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default to scripts/metric path relative to DCGen root
        default_metrics_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "metric"
        )
        
        self.metrics_path = config.get('metrics_path', default_metrics_path) if config else default_metrics_path
        self._visual_eval_func = None
    
    def initialize(self) -> None:
        """Add Design2Code to path and import evaluation function."""
        if self.metrics_path not in sys.path:
            sys.path.insert(0, self.metrics_path)
        
        try:
            from Design2Code.metrics.visual_score import visual_eval_v3_multi
            self._visual_eval_func = visual_eval_v3_multi
            self._initialized = True
        except ImportError as e:
            raise ImportError(
                f"Could not import Design2Code from {self.metrics_path}. "
                f"Ensure Design2Code is installed: {e}"
            )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._visual_eval_func = None
        self._initialized = False
    
    def evaluate_sample(
        self,
        generated_html_path: str,
        reference_html_path: str,
        generated_screenshot_path: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single sample using Design2Code metrics.
        
        Args:
            generated_html_path: Path to generated HTML file
            reference_html_path: Path to reference HTML file (visual_eval_v3_multi generates screenshots internally)
            generated_screenshot_path: Optional, not used (screenshots generated internally)
        
        Returns:
            EvaluationResult with fine-grained scores
        """
        sample_id = os.path.splitext(os.path.basename(generated_html_path))[0]
        
        if not self._initialized:
            self.initialize()
        
        if not os.path.exists(generated_html_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Generated HTML not found"
            )
        
        if not os.path.exists(reference_html_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Reference HTML not found"
            )
        
        try:
            # visual_eval_v3_multi expects [[list of predict htmls], reference_html]
            # It internally generates screenshots from both HTML files
            input_list = [[generated_html_path], reference_html_path]
            
            # Change CWD to Design2Code directory so it can find its scripts (metrics/screenshot_single.py)
            d2c_path = os.path.join(self.metrics_path, "Design2Code")
            cwd = os.getcwd()
            
            try:
                if os.path.exists(d2c_path):
                    os.chdir(d2c_path)
                
                # Call the Design2Code evaluation function
                # returns list of results for each prediction
                results = self._visual_eval_func(input_list)
            finally:
                os.chdir(cwd)
                # Force garbage collection after each sample to reduce memory pressure
                gc.collect()
            
            if not results or len(results) == 0:
                 return EvaluationResult(sample_id=sample_id, metric_name=self.metric_name, success=False, error="No results returned from visual_eval_v3_multi")
            
            # We only had one prediction
            result_stats = results[0]
            # result_stats structure: [sum_areas, final_score, (block_match, text, position, color, clip)]
            
            if len(result_stats) > 2 and isinstance(result_stats[2], (list, tuple)):
                scores = result_stats[2]
                score_dict = {
                    "block_match": float(scores[0]),
                    "text": float(scores[1]),
                    "position": float(scores[2]),
                    "color": float(scores[3]),
                    "clip": float(scores[4]),
                }
                # overall score
                score_dict["overall"] = float(result_stats[1])
            else:
                return EvaluationResult(
                    sample_id=sample_id,
                    metric_name=self.metric_name,
                    success=False,
                    error=f"Unexpected score format: {type(result_stats)}"
                )
            
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                scores=score_dict,
                metadata={
                    "reference_html": os.path.basename(reference_html_path),
                    "generated_html": os.path.basename(generated_html_path)
                }
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[FineGrained] Error evaluating sample {sample_id}: {error_msg}")
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error=error_msg
            )
    
    def evaluate_run(
        self,
        run_dir: str,
        reference_dir: str,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all samples in a run directory.
        
        Args:
            run_dir: Directory containing generated HTML files
            reference_dir: Directory containing reference images
            sample_ids: Optional list of sample IDs to evaluate (without extension)
        
        Returns:
            Dictionary with per-sample and aggregate results
        """
        if not self._initialized:
            self.initialize()
        
        # Log initial memory state
        print(f"[FineGrained] Starting evaluation for {run_dir}")
        log_memory("Initial - ")
        
        # Find HTML files
        if sample_ids is None:
            sample_ids = []
            for f in os.listdir(run_dir):
                if f.endswith('.html') and not f.startswith('_'):
                    sample_ids.append(os.path.splitext(f)[0])
        
        results = {}
        all_results = []
        failed_samples = []  # Track failed samples separately
        
        # Prepare evaluation tasks
        eval_tasks = []
        
        for sample_id in sample_ids:
            html_path = os.path.join(run_dir, f"{sample_id}.html")
            
            # Reference is an HTML file (visual_eval_v3_multi generates screenshots internally)
            ref_html_path = os.path.join(reference_dir, f"{sample_id}.html")
            
            if not os.path.exists(ref_html_path):
                results[sample_id] = {
                    "success": False,
                    "error": "Reference HTML not found"
                }
                failed_samples.append(sample_id)
                continue
            
            if not os.path.exists(html_path):
                results[sample_id] = {
                    "success": False,
                    "error": "Generated HTML not found"
                }
                failed_samples.append(sample_id)
                continue
            
            eval_tasks.append((html_path, ref_html_path, sample_id))
        
        print(f"[FineGrained] Prepared {len(eval_tasks)} evaluation tasks, {len(failed_samples)} pre-failed")
        
        # Run evaluations
        if eval_tasks:
            # Default to 1 worker for stability (each worker loads CLIP + spawns browser)
            # Can be overridden with DCGEN_FINE_GRAINED_WORKERS env var
            # Setting to 1 avoids OOM issues from multiple CLIP models + browsers
            default_workers = 1
            n_jobs = int(os.environ.get('DCGEN_FINE_GRAINED_WORKERS', default_workers))
            n_jobs = min(n_jobs, len(eval_tasks))
            
            # Check available memory - if low, force sequential
            mem_info = get_memory_info()
            if "system_available_mb" in mem_info and mem_info["system_available_mb"] < 2000:
                print(f"[FineGrained] Low memory ({mem_info['system_available_mb']:.0f}MB available), forcing sequential processing")
                n_jobs = 1
            
            parallel_failed = False
            parallel_error_msg = None
            
            if n_jobs > 1:
                try:
                    print(f"[FineGrained] Running parallel evaluation with {n_jobs} workers...")
                    with tqdm_joblib(tqdm(total=len(eval_tasks), desc=f"Fine-grained evaluation (workers={n_jobs})")) as progress_bar:
                        parallel_results = Parallel(n_jobs=n_jobs, timeout=300)(
                            delayed(self.evaluate_sample)(html_path, ref_path) 
                            for html_path, ref_path, _ in eval_tasks
                        )
                except Exception as parallel_error:
                    parallel_failed = True
                    parallel_error_msg = str(parallel_error)
                    print(f"[FineGrained] Parallel processing failed: {parallel_error}")
                    print(f"[FineGrained] Full traceback:\n{traceback.format_exc()}")
                    log_memory("After parallel failure - ")
            else:
                parallel_failed = True  # Skip parallel, go directly to sequential
                parallel_error_msg = "Using sequential mode (n_jobs=1)"
            
            if parallel_failed:
                # Sequential fallback with better error handling
                print(f"[FineGrained] Running sequential evaluation ({parallel_error_msg})...")
                log_memory("Before sequential - ")
                
                parallel_results = []
                for idx, (html_path, ref_path, sample_id) in enumerate(tqdm(eval_tasks, desc="Fine-grained evaluation (sequential)")):
                    try:
                        res = self.evaluate_sample(html_path, ref_path)
                        parallel_results.append(res)
                        
                        if not res.success:
                            print(f"[FineGrained] Sample {sample_id} failed: {res.error}")
                            failed_samples.append(sample_id)
                        
                        # Periodic memory logging and GC
                        if (idx + 1) % 20 == 0:
                            gc.collect()
                            log_memory(f"After {idx + 1}/{len(eval_tasks)} samples - ")
                            
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        print(f"[FineGrained] Exception evaluating sample {sample_id}: {error_msg}")
                        print(f"[FineGrained] Traceback:\n{traceback.format_exc()}")
                        
                        parallel_results.append(EvaluationResult(
                            sample_id=sample_id,
                            metric_name=self.metric_name,
                            success=False,
                            error=error_msg
                        ))
                        failed_samples.append(sample_id)
                        
                        # Force GC after errors
                        gc.collect()
            
            # Process results
            for res in parallel_results:
                results[res.sample_id] = res.to_dict()
                all_results.append(res)
                
                # Track failures
                if not res.success and res.sample_id not in failed_samples:
                    failed_samples.append(res.sample_id)
        
        # Log final stats
        successful_count = len([r for r in all_results if r.success])
        print(f"[FineGrained] Completed: {successful_count} successful, {len(failed_samples)} failed")
        log_memory("Final - ")
        
        # Aggregate (only successful results)
        aggregated = self.aggregate_results(all_results)
        
        # Add failure tracking to output
        aggregated["failed_samples"] = failed_samples
        aggregated["failed_count"] = len(failed_samples)
        aggregated["successful_count"] = successful_count
        aggregated["total_count"] = len(eval_tasks) + len([s for s in failed_samples if s not in [t[2] for t in eval_tasks]])
        
        return {
            "per_sample": results,
            "aggregate": aggregated,
            "failed_samples": failed_samples
        }
