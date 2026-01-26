#!/usr/bin/env python3
"""
DCGen API Server
================

A REST API server for running image-to-HTML experiments and evaluations.
Designed to work with a CLI client similar to sb-cli.

Usage:
    python api.py                    # Start server on default port 5000
    python api.py --port 8080        # Start server on custom port
    python api.py --host 0.0.0.0     # Listen on all interfaces

Authentication:
    Set DCGEN_API_KEY environment variable or pass via x-api-key header.

Endpoints:
    POST /submit          - Submit an experiment run
    GET  /poll-jobs       - Poll job status
    POST /get-report      - Get evaluation report for a run
    POST /list-runs       - List all runs
    POST /delete-run      - Delete a run
    GET  /get-quotas      - Get API usage quotas
    POST /verify-api-key  - Verify API key
    GET  /health          - Health check
"""

import os
import sys
import json
import uuid
import datetime
import threading
import shutil
import hashlib
import platform
import numpy as np
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, g

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add path for Design2Code metrics (fine-grained evaluation)
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "metric")
if METRICS_PATH not in sys.path:
    sys.path.insert(0, METRICS_PATH)

# Fine-grained metrics are only available on Linux/Mac due to dependencies
FINE_GRAINED_METRICS_AVAILABLE = platform.system() in ["Linux", "Darwin"]
if FINE_GRAINED_METRICS_AVAILABLE:
    try:
        from Design2Code.metrics.visual_score import visual_eval_v3_multi
        print("[DCGen] Fine-grained metrics (Design2Code) loaded successfully")
    except ImportError as e:
        FINE_GRAINED_METRICS_AVAILABLE = False
        print(f"[DCGen] Fine-grained metrics not available: {e}")

from utils import (
    encode_image, get_driver, take_screenshot,
    GPT4, Gemini, Claude, QwenVL,
    ImgSegmentation, DCGenGrid
)
from dataset_manager import DatasetManager, DATASETS_CONFIG, get_dataset_manager
from dotenv import load_dotenv
from tqdm import tqdm
from threading import Thread
from rapidfuzz import fuzz
import re
import time

load_dotenv()

# Import modular components (new architecture)
from config import (
    MODEL_PRICING, DEFAULT_PRICING, MLLM_JUDGE_PROMPTS, MODEL_FAMILIES, SUPPORTED_MODELS,
    calculate_cost as config_calculate_cost, 
    get_model_info as config_get_model_info
)
from evaluation import (
    MLLMJudgeEvaluator,
    FineGrainedEvaluator,
    CLIPScoreEvaluator,
    CodeSimilarityEvaluator
)
from routes.evaluation import evaluation_bp

# Flag for modular evaluation (can use new evaluator classes)
USE_MODULAR_EVALUATION = True

app = Flask(__name__)

# Register modular route blueprints
app.register_blueprint(evaluation_bp)
print("[DCGen] Registered evaluation blueprint (/evaluate/* routes)")

# ============================================================
# Configuration
# ============================================================
API_VERSION = "1.0.0"
RUNS_DB = {}  # In-memory runs storage {run_id: Run}
API_KEYS = {}  # API key -> user info
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Default API key for development (set via env var in production)
DEFAULT_API_KEY = os.getenv("DCGEN_API_KEY", "dev-api-key-12345")
API_KEYS[DEFAULT_API_KEY] = {"email": "dev@localhost", "verified": True, "created_at": datetime.datetime.now().isoformat()}

# Model families and their supported versions are imported from config.py

SUPPORTED_METHODS = ["dcgen", "direct"]
SUPPORTED_DATASETS = list(DATASETS_CONFIG.keys())  # ["design2code", "dcgen"]

# ============================================================
# Prompts
# ============================================================
PROMPT_DIRECT = """Here is a prototype image of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use "placeholder.png" to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""

PROMPT_DCGEN = {
    "prompt_leaf": """Here is a prototype image of a container. Please fill a single piece of HTML and tail-wind CSS code to reproduce exactly the given container. Use 'placeholder.png' to replace the images. Pay attention to things like size, text, and color of all the elements, as well as the background color and layout. Here is the code for you to fill in:
    <div>
    You code here
    </div>
    Respond with only the code inside the <div> tags.""",

    "prompt_root": """Here is a prototype image of a webpage. I have an draft HTML file that contains most of the elements and their correct positions, but it has *inaccurate background*, and some missing or wrong elements. Please compare the draft and the prototype image, then revise the draft implementation. Return a single piece of accurate HTML+tail-wind CSS code to reproduce the website. Use "placeholder.png" to replace the images. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is: \n\n [CODE]"""
}

SEG_PARAMS_DEFAULT = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50
}

# Model pricing and default pricing are imported from config.py

calculate_cost = config_calculate_cost
get_model_info = config_get_model_info


# ============================================================
# Run Model
# ============================================================
class Run:
    """Represents an experiment run."""
    
    # File names for persistence
    RUN_METADATA_FILE = "run_metadata.json"
    EVALUATION_FILE = "evaluation.json"
    RESULTS_FILE = "results.json"
    COST_REPORT_FILE = "cost_report.json"
    
    def __init__(self, run_id: str, model: str, method: str, input_dir: str, api_key: str, 
                 dataset: str = None, sample_ids: list = None):
        self.run_id = run_id
        self.model = model
        self.method = method
        self.input_dir = input_dir
        self.output_dir = os.path.join(RESULTS_DIR, run_id)
        self.api_key = api_key
        self.dataset = dataset  # Dataset name (design2code, dcgen, or None for custom)
        self.sample_ids = sample_ids  # Specific sample IDs to run (None = all)
        self.status = "pending"  # pending, running, completed, failed
        self.created_at = datetime.datetime.now().isoformat()
        self.completed_at = None
        self.instances = {}  # instance_id -> {status, result}
        self.total_instances = 0
        self.error = None
        self.token_usage = None
        self.cost_estimate = None
        self.evaluation = None
    
    def save_to_disk(self):
        """Save run metadata, evaluation, results, and cost report to disk."""
        ensure_dir(self.output_dir)
        
        # Calculate cost estimate if we have token usage
        if self.token_usage:
            self.cost_estimate = calculate_cost(self.model, self.token_usage)
        
        # Save run metadata
        metadata = {
            "run_id": self.run_id,
            "model": self.model,
            "method": self.method,
            "dataset": self.dataset,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "api_key": self.api_key,
            "sample_ids": self.sample_ids,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_instances": self.total_instances,
            "error": self.error,
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate
        }
        
        metadata_path = os.path.join(self.output_dir, self.RUN_METADATA_FILE)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save results (instances)
        results = {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "run_id": self.run_id,
            "instances": self.instances
        }
        
        results_path = os.path.join(self.output_dir, self.RESULTS_FILE)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save evaluation separately for easy lookup
        if self.evaluation:
            eval_path = os.path.join(self.output_dir, self.EVALUATION_FILE)
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation, f, indent=2, ensure_ascii=False)
        
        # Save cost report separately for easy lookup
        if self.cost_estimate:
            cost_report = {
                "run_id": self.run_id,
                "model": self.model,
                "method": self.method,
                "dataset": self.dataset,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "total_instances": self.total_instances,
                "token_usage": self.token_usage,
                "cost_estimate": self.cost_estimate,
                "summary": {
                    "total_tokens": self.cost_estimate.get("total_tokens", 0),
                    "total_cost_usd": self.cost_estimate.get("total_cost_usd", 0),
                    "cost_per_instance_usd": round(
                        self.cost_estimate.get("total_cost_usd", 0) / max(self.total_instances, 1), 6
                    )
                }
            }
            cost_path = os.path.join(self.output_dir, self.COST_REPORT_FILE)
            with open(cost_path, 'w', encoding='utf-8') as f:
                json.dump(cost_report, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_disk(cls, run_dir: str) -> 'Run':
        """Load a run from disk."""
        metadata_path = os.path.join(run_dir, cls.RUN_METADATA_FILE)
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Create run instance
            run = cls(
                run_id=metadata["run_id"],
                model=metadata["model"],
                method=metadata["method"],
                input_dir=metadata.get("input_dir", ""),
                api_key=metadata.get("api_key", ""),
                dataset=metadata.get("dataset"),
                sample_ids=metadata.get("sample_ids")
            )
            
            # Restore metadata
            run.status = metadata.get("status", "completed")
            run.created_at = metadata.get("created_at")
            run.completed_at = metadata.get("completed_at")
            run.total_instances = metadata.get("total_instances", 0)
            run.error = metadata.get("error")
            run.token_usage = metadata.get("token_usage")
            run.cost_estimate = metadata.get("cost_estimate")
            run.output_dir = run_dir
            
            # Load results
            results_path = os.path.join(run_dir, cls.RESULTS_FILE)
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                run.instances = results.get("instances", {})
            
            # Load evaluation
            eval_path = os.path.join(run_dir, cls.EVALUATION_FILE)
            if os.path.exists(eval_path):
                with open(eval_path, 'r', encoding='utf-8') as f:
                    run.evaluation = json.load(f)
            
            # Load cost estimate from cost report if not in metadata
            if not run.cost_estimate:
                cost_path = os.path.join(run_dir, cls.COST_REPORT_FILE)
                if os.path.exists(cost_path):
                    with open(cost_path, 'r', encoding='utf-8') as f:
                        cost_report = json.load(f)
                    run.cost_estimate = cost_report.get("cost_estimate")
            
            return run
            
        except Exception as e:
            print(f"Error loading run from {run_dir}: {e}")
            return None
    
    def to_dict(self, include_details: bool = False):
        result = {
            "run_id": self.run_id,
            "model": self.model,
            "method": self.method,
            "dataset": self.dataset,
            "status": self.status,
            "total_instances": self.total_instances,
            "completed_instances": len([i for i in self.instances.values() if i.get("status") == "completed"]),
            "pending_instances": len([i for i in self.instances.values() if i.get("status") == "pending"]),
            "running_instances": len([i for i in self.instances.values() if i.get("status") == "running"]),
            "failed_instances": len([i for i in self.instances.values() if i.get("status") == "failed"]),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
        if include_details:
            result["input_dir"] = self.input_dir
            result["output_dir"] = self.output_dir
            result["instances"] = self.instances
            result["token_usage"] = self.token_usage
            result["cost_estimate"] = self.cost_estimate
            result["evaluation"] = self.evaluation
            result["error"] = self.error
            result["sample_ids"] = self.sample_ids
        return result
    
    def get_poll_status(self):
        """Get status for polling."""
        running = [k for k, v in self.instances.items() if v.get("status") == "running"]
        completed = [k for k, v in self.instances.items() if v.get("status") == "completed"]
        pending = [k for k, v in self.instances.items() if v.get("status") == "pending"]
        failed = [k for k, v in self.instances.items() if v.get("status") == "failed"]
        result = {
            "run_id": self.run_id,
            "status": self.status,
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "running": running,
            "completed": completed,
            "pending": pending,
            "failed": failed
        }
        # Include evaluation and cost when run is completed
        if self.status == "completed":
            if self.evaluation:
                result["evaluation"] = self.evaluation
            if self.cost_estimate:
                result["cost_estimate"] = self.cost_estimate
        return result


# ============================================================
# Authentication
# ============================================================
def verify_api_key_check(api_key: str) -> bool:
    """Verify if API key is valid."""
    return api_key in API_KEYS and API_KEYS[api_key].get("verified", False)


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key:
            try:
                json_data = request.get_json(silent=True)
                if json_data:
                    api_key = json_data.get("api_key")
            except:
                pass
        
        if not api_key:
            return jsonify({"message": "API key is required. Set x-api-key header or DCGEN_API_KEY environment variable."}), 401
        
        # if not verify_api_key_check(api_key):
        #     return jsonify({"message": "Invalid or unverified API key."}), 401
        
        g.api_key = api_key
        return f(*args, **kwargs)
    return decorated


# ============================================================
# Model Factory
# ============================================================
# Custom OpenAI-compatible API base URL (set via env var or default)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")

def get_bot(model_name: str):
    """
    Initialize the specified model.
    
    Accepts both family names (gemini, gpt4, claude, qwen, deepseek, etc.) and specific versions.
    """
    family, version = get_model_info(model_name)
    
    if not family:
        raise ValueError(f"Unknown model: {model_name}. Supported families: {', '.join(SUPPORTED_MODELS)}")
    
    if family == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        return Gemini(api_key, model=version)
    
    elif family in ["gpt4", "deepseek", "grok", "doubao", "kimi"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
             # Try DCGEN_API_KEY as fallback (OpenKey often uses one key)
             api_key = os.getenv("DCGEN_API_KEY")
        
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY (for {family}) not found in environment")
        
        # Use OpenKey compatible setup
        return GPT4(api_key, model=version, base_url=OPENAI_BASE_URL)
    
    elif family == "claude":
        # Check native key first
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return Claude(api_key, model=version)
        
        # Fallback to OpenKey
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DCGEN_API_KEY")
        if api_key and ("openkey" in OPENAI_BASE_URL or "api.openai.com" not in OPENAI_BASE_URL):
            print(f"Initializing {family} model ({version}) via OpenKey API")
            return GPT4(api_key, model=version, base_url=OPENAI_BASE_URL)
            
        raise ValueError("ANTHROPIC_API_KEY not found (and fallback to OpenKey failed)")
    
    elif family == "qwen":
        # Check native key first
        api_key = os.getenv("QWEN_API_KEY")
        if api_key:
            return QwenVL(api_key, model=version)
        
        # Fallback to OpenKey
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DCGEN_API_KEY")
        if api_key and ("openkey" in OPENAI_BASE_URL or "api.openai.com" not in OPENAI_BASE_URL):
            print(f"Initializing {family} model ({version}) via OpenKey API")
            return GPT4(api_key, model=version, base_url=OPENAI_BASE_URL)
            
        raise ValueError("QWEN_API_KEY not found (and fallback to OpenKey failed)")
    
    else:
        raise ValueError(f"Unknown model family: {family}")


# ============================================================
# Utility Functions
# ============================================================
def get_image_files(directory: str, exclude=["placeholder", "bbox"]):
    """Get list of PNG files in directory."""
    files = []
    for f in os.listdir(directory):
        if f.endswith(".png"):
            if not any(ex in f for ex in exclude):
                files.append(os.path.join(directory, f))
    return sorted(files)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str) -> str:
    """Resolve relative path to absolute, using DATA_DIR as base."""
    if os.path.isabs(path):
        return path
    return os.path.join(DATA_DIR, path)


# ============================================================
# Generation Functions
# ============================================================
def generate_single(prompt: str, bot, img_path: str, save_path: str = None, max_retries: int = 5):
    """Generate HTML from a single image using direct prompting."""
    last_error = None
    for i in range(max_retries):
        try:
            html = bot.ask(prompt, encode_image(img_path))
            code = re.findall(r"```html([^`]+)```", html)
            if code:
                html = code[0]
            if len(html) < 10:
                raise Exception(f"No HTML code found in response. Response was: {html[:200] if html else 'empty'}")
            if save_path:
                with open(save_path, 'w', encoding="utf-8") as f:
                    f.write(html)
            return html
        except Exception as e:
            last_error = e
            # Exponential backoff: 2, 4, 8, 16, 32 seconds
            wait_time = 2 ** (i + 1)
            print(f"Retry {i+1}/{max_retries} for {os.path.basename(img_path)}: {str(e)[:100]}... waiting {wait_time}s")
            time.sleep(wait_time)
    raise Exception(f"Failed to generate HTML for {img_path} after {max_retries} retries. Last error: {last_error}")


def generate_dcgen(bot, img_path: str, save_path: str = None, seg_params: dict = None):
    """Generate HTML from a single image using DCGen method."""
    params = seg_params or SEG_PARAMS_DEFAULT
    img_seg = ImgSegmentation(img_path, **params)
    
    dcgen_grid = DCGenGrid(
        img_seg, 
        prompt_seg=PROMPT_DCGEN["prompt_leaf"], 
        prompt_refine=PROMPT_DCGEN["prompt_root"]
    )
    dcgen_grid.generate_code(bot, multi_thread=True)
    
    if save_path:
        with open(save_path, 'w', encoding="utf-8", errors="ignore") as f:
            f.write(dcgen_grid.code)
    
    return dcgen_grid.code


def take_screenshots_task(directory: str, replace: bool = False):
    """Generate screenshots for HTML files using Playwright for full-page capture."""
    from playwright.sync_api import sync_playwright
    
    html_files = [f for f in os.listdir(directory) if f.endswith('.html')]
    
    if not html_files:
        return
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})
        
        for filename in html_files:
            html_path = os.path.join(directory, filename)
            png_path = html_path.replace('.html', '.png')
            
            if os.path.exists(png_path) and not replace:
                continue
            
            try:
                page.goto("file://" + os.path.abspath(html_path))
                page.wait_for_load_state("networkidle")
                page.screenshot(path=png_path, full_page=True)
            except:
                pass
        
        browser.close()


def run_experiment_task(run: Run):
    """Run experiment task in background."""
    try:
        run.status = "running"
        ensure_dir(run.output_dir)
        
        # Get image files
        filelist = get_image_files(run.input_dir)
        run.total_instances = len(filelist)
        
        # Initialize instances
        for img_path in filelist:
            instance_id = os.path.basename(img_path).replace('.png', '')
            run.instances[instance_id] = {"status": "pending", "result": None}
        
        # Save initial state (fault tolerance - saves the plan)
        run.save_to_disk()
        
        # Initialize bot
        bot = get_bot(run.model)
        if hasattr(bot, 'reset_token_usage'):
            bot.reset_token_usage()
        
        # Process each image
        save_interval = max(1, len(filelist) // 20)  # Save every 5% progress
        for idx, img_path in enumerate(filelist):
            instance_id = os.path.basename(img_path).replace('.png', '')
            filename = f"{instance_id}.html"
            save_path = os.path.join(run.output_dir, filename)
            
            run.instances[instance_id]["status"] = "running"
            
            try:
                if run.method == "dcgen":
                    generate_dcgen(bot, img_path, save_path, SEG_PARAMS_DEFAULT)
                else:
                    generate_single(PROMPT_DIRECT, bot, img_path, save_path)
                
                run.instances[instance_id]["status"] = "completed"
                run.instances[instance_id]["result"] = save_path
                
            except Exception as e:
                run.instances[instance_id]["status"] = "failed"
                run.instances[instance_id]["error"] = str(e)
            
            # Periodic save for fault tolerance (every 5% or so)
            if (idx + 1) % save_interval == 0:
                if hasattr(bot, 'print_token_usage'):
                    run.token_usage = bot.print_token_usage(run.method)
                run.save_to_disk()
        
        # Copy placeholder
        placeholder_src = os.path.join(run.input_dir, "placeholder.png")
        if os.path.exists(placeholder_src):
            shutil.copy(placeholder_src, os.path.join(run.output_dir, "placeholder.png"))
        
        # Take screenshots
        take_screenshots_task(run.output_dir, replace=True)
        
        # Get token usage
        if hasattr(bot, 'print_token_usage'):
            run.token_usage = bot.print_token_usage(run.method)
        
        # Run evaluation
        run.evaluation = run_evaluation_for_run(run)
        
        run.status = "completed"
        run.completed_at = datetime.datetime.now().isoformat()
        
        # Save run to disk for persistence
        run.save_to_disk()
        
    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.completed_at = datetime.datetime.now().isoformat()
        
        # Save failed run to disk too
        run.save_to_disk()


def load_existing_runs():
    """Load existing runs from disk on startup."""
    if not os.path.exists(RESULTS_DIR):
        return 0
    
    loaded = 0
    for run_dir_name in os.listdir(RESULTS_DIR):
        run_dir = os.path.join(RESULTS_DIR, run_dir_name)
        if not os.path.isdir(run_dir):
            continue
        
        # Skip if already in memory
        if run_dir_name in RUNS_DB:
            continue
        
        run = Run.load_from_disk(run_dir)
        if run:
            RUNS_DB[run.run_id] = run
            loaded += 1
    
    return loaded


# Load runs on module import (for gunicorn)
# This ensures runs persist across restarts
_loaded_count = load_existing_runs()
print(f"[DCGen] Loaded {_loaded_count} existing runs from disk")


def run_evaluation_for_run(run: Run) -> dict:
    """Run evaluation for a single run using modular evaluators."""
    results = {
        "dataset": run.dataset,
        "model": run.model,
        "method": run.method,
        "run_id": run.run_id,
        "metrics": {}
    }
    
    # Collect completed sample IDs
    completed_sample_ids = []
    for instance_id, instance in run.instances.items():
        if instance.get("status") == "completed":
            test_file = instance.get("result")
            if test_file and os.path.exists(test_file):
                completed_sample_ids.append(instance_id)
    
    if not completed_sample_ids:
        return results
    
    # Code similarity using modular evaluator
    try:
        code_sim_evaluator = CodeSimilarityEvaluator()
        code_sim_scores = {}
        
        for instance_id in completed_sample_ids:
            ref_file = os.path.join(run.input_dir, f"{instance_id}.html")
            test_file = run.instances[instance_id].get("result")
            
            if os.path.exists(ref_file) and os.path.exists(test_file):
                result = code_sim_evaluator.evaluate_sample(
                    generated_html_path=test_file,
                    reference_html_path=ref_file
                )
                if result.success and "overall" in result.scores:
                    # Use overall score (fuzz_ratio, 0-100 scale)
                    code_sim_scores[instance_id] = result.scores["overall"]
        
        if code_sim_scores:
            results["metrics"]["code_similarity"] = {
                "scores": code_sim_scores,
                "average": sum(code_sim_scores.values()) / len(code_sim_scores)
            }
    except Exception as e:
        results["metrics"]["code_similarity"] = {"error": str(e)}
    
    # CLIP scores using modular evaluator
    try:
        clip_evaluator = CLIPScoreEvaluator()
        clip_evaluator.initialize()
        
        clip_scores = {}
        for instance_id in completed_sample_ids:
            ref_file = os.path.join(run.input_dir, f"{instance_id}.png")
            test_file = run.instances[instance_id].get("result", "").replace('.html', '.png')
            
            if os.path.exists(ref_file) and os.path.exists(test_file):
                result = clip_evaluator.evaluate_sample(
                    generated_html_path=run.instances[instance_id].get("result"),
                    reference_image_path=ref_file,
                    generated_screenshot_path=test_file
                )
                if result.success and "clip_score" in result.scores:
                    clip_scores[instance_id] = result.scores["clip_score"]
        
        clip_evaluator.cleanup()
        
        if clip_scores:
            results["metrics"]["clip"] = {
                "scores": clip_scores,
                "average": sum(clip_scores.values()) / len(clip_scores)
            }
    except ImportError:
        results["metrics"]["clip"] = {"error": "CLIP dependencies not installed (transformers, torch)"}
    except Exception as e:
        results["metrics"]["clip"] = {"error": str(e)}
    
    # Fine-grained metrics (Block-Match, Text, Position, Color, CLIP)
    # Only available on Linux/Mac
    if FINE_GRAINED_METRICS_AVAILABLE:
        try:
            fine_grained_results = run_fine_grained_evaluation(run)
            if fine_grained_results:
                results["metrics"]["fine_grained"] = fine_grained_results
        except Exception as e:
            results["metrics"]["fine_grained"] = {"error": str(e)}
    else:
        results["metrics"]["fine_grained"] = {
            "error": f"Fine-grained metrics only available on Linux/Mac. Current platform: {platform.system()}"
        }
    
    return results


def run_fine_grained_evaluation(run: Run) -> dict:
    """
    Run fine-grained visual evaluation using Design2Code metrics.
    
    Uses the FineGrainedEvaluator class which supports multiprocessing via joblib.
    
    This evaluates:
    - Block-Match: How well the generated blocks match the reference
    - Text: Text content similarity
    - Position: Positional accuracy of elements
    - Color: Color accuracy of text elements
    - CLIP: Visual similarity using CLIP embeddings
    
    Returns:
        Dictionary with per-instance scores and averages
    """
    if not FINE_GRAINED_METRICS_AVAILABLE:
        return None
    
    # Collect completed instances - build list of sample IDs
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
        # Use the modular FineGrainedEvaluator which has multiprocessing support
        evaluator = FineGrainedEvaluator({
            'metrics_path': METRICS_PATH
        })
        
        print(f"[DCGen] Running fine-grained evaluation on {len(completed_sample_ids)} instances using FineGrainedEvaluator")
        
        # Use evaluate_run which handles parallel evaluation internally
        eval_results = evaluator.evaluate_run(
            run_dir=run.output_dir,
            reference_dir=run.input_dir,
            sample_ids=completed_sample_ids
        )
        
        # Clean up
        evaluator.cleanup()
        
        # Convert from per_sample format to our expected format
        if not eval_results or "aggregate" not in eval_results:
            return None
        
        aggregate = eval_results.get("aggregate", {})
        per_sample = eval_results.get("per_sample", {})
        failed_samples = eval_results.get("failed_samples", [])
        
        # Build results in the expected format
        results = {}
        metric_names = ["block_match", "text", "position", "color", "clip", "overall"]
        
        for metric_name in metric_names:
            scores_dict = {}
            for sample_id, sample_result in per_sample.items():
                # Only include successful evaluations with actual scores
                if sample_result.get("success") and "scores" in sample_result:
                    score_value = sample_result["scores"].get(metric_name)
                    if score_value is not None:
                        scores_dict[sample_id] = float(score_value)
            
            if scores_dict:
                results[metric_name] = {
                    "scores": scores_dict,
                    "average": sum(scores_dict.values()) / len(scores_dict)
                }
        
        # Add failure tracking metadata
        if failed_samples:
            results["_metadata"] = {
                "failed_samples": failed_samples,
                "failed_count": len(failed_samples),
                "successful_count": aggregate.get("successful_count", len(completed_sample_ids) - len(failed_samples)),
                "total_attempted": len(completed_sample_ids)
            }
            print(f"[DCGen] Fine-grained evaluation: {len(failed_samples)} samples failed evaluation")
        
        return results if results else None
        
    except Exception as e:
        import traceback
        print(f"[DCGen] Fine-grained evaluation error: {e}")
        print(f"[DCGen] Traceback:\n{traceback.format_exc()}")
        return {"error": str(e)}


# ============================================================
# API Endpoints
# ============================================================

# ------------------------------------------------------------
# Dataset Management Endpoints
# ------------------------------------------------------------

@app.route("/datasets", methods=["GET"])
def list_datasets():
    """
    List all available datasets and their download status.
    
    Returns:
        List of datasets with name, description, size, and download status
    """
    dm = get_dataset_manager()
    datasets = dm.list_available_datasets()
    return jsonify({
        "datasets": datasets,
        "supported": SUPPORTED_DATASETS
    })


@app.route("/datasets/<dataset_name>", methods=["GET"])
def get_dataset_info(dataset_name: str):
    """
    Get detailed info about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (design2code, dcgen)
    
    Returns:
        Dataset metadata and sample information
    """
    if dataset_name not in SUPPORTED_DATASETS:
        return jsonify({
            "message": f"Unknown dataset: {dataset_name}. Available: {', '.join(SUPPORTED_DATASETS)}"
        }), 400
    
    dm = get_dataset_manager()
    info = dm.get_dataset_info(dataset_name)
    
    if not info:
        return jsonify({
            "message": f"Dataset {dataset_name} not downloaded. Please contact administrator.",
            "downloaded": False
        })
    
    return jsonify({
        "downloaded": True,
        "info": info
    })



@app.route("/datasets/<dataset_name>/samples", methods=["GET"])
def get_dataset_samples(dataset_name: str):
    """
    Get samples from a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Query params:
        limit: Maximum number of samples to return
        offset: Starting offset
    """
    if dataset_name not in SUPPORTED_DATASETS:
        return jsonify({
            "message": f"Unknown dataset: {dataset_name}"
        }), 400
    
    dm = get_dataset_manager()
    
    try:
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", 0, type=int)
        samples = dm.get_samples(dataset_name, limit=limit, offset=offset)
        total = len(dm.get_sample_ids(dataset_name))
        
        return jsonify({
            "samples": samples,
            "total": total,
            "offset": offset,
            "limit": limit
        })
    except ValueError as e:
        return jsonify({"message": str(e)}), 400


# ------------------------------------------------------------
# Health and Auth Endpoints
# ------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.datetime.now().isoformat(),
        "supported_datasets": SUPPORTED_DATASETS,
        "supported_model_families": SUPPORTED_MODELS,
        "supported_model_versions": MODEL_FAMILIES,
        "supported_methods": SUPPORTED_METHODS
    })


@app.route("/gen-api-key", methods=["POST"])
def gen_api_key():
    """
    Generate a new API key.
    
    Request body:
    {
        "email": "user@example.com"
    }
    """
    data = request.get_json()
    if not data or "email" not in data:
        return jsonify({"message": "Email is required"}), 400
    
    email = data["email"]
    
    # Generate API key
    key_raw = f"{email}-{datetime.datetime.now().isoformat()}-{uuid.uuid4()}"
    api_key = hashlib.sha256(key_raw.encode()).hexdigest()[:32]
    
    # For development, auto-verify
    API_KEYS[api_key] = {
        "email": email,
        "verified": True,  # Auto-verify for dev
        "created_at": datetime.datetime.now().isoformat()
    }
    
    return jsonify({
        "message": f"API key generated for {email}. Key is auto-verified for development.",
        "api_key": api_key
    })


@app.route("/verify-api-key", methods=["POST"])
@require_api_key
def verify_api_key_endpoint():
    """Verify API key."""
    return jsonify({
        "message": "API key is valid and verified.",
        "email": API_KEYS[g.api_key].get("email")
    })


@app.route("/get-quotas", methods=["GET"])
@require_api_key
def get_quotas():
    """Get API usage quotas."""
    # For now, return unlimited quotas
    return jsonify({
        "remaining_quotas": {
            "gemini": {"runs": 1000},
            "gpt4": {"runs": 1000},
            "claude": {"runs": 1000},
            "qwen": {"runs": 1000}
        }
    })


@app.route("/submit", methods=["POST"])
@require_api_key
def submit():
    """
    Submit an experiment run.
    
    Request body:
    {
        "model": "gemini",           # Required: gemini, gpt4, claude, qwen
        "method": "dcgen",           # Required: dcgen or direct
        "dataset": "design2code",    # Optional: dataset name (design2code, dcgen)
        "input_dir": "./demo",       # Required if no dataset: input directory with PNG images
        "sample_ids": ["0", "1"],    # Optional: specific sample IDs to run (for datasets)
        "run_id": "my_run"           # Optional: run ID (auto-generated if not provided)
    }
    
    Either 'dataset' or 'input_dir' must be provided. If both are provided,
    'dataset' takes precedence.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"message": "Request body is required"}), 400
    
    # Validate required fields
    if "model" not in data:
        return jsonify({"message": "Missing required field: model"}), 400
    if "method" not in data:
        return jsonify({"message": "Missing required field: method"}), 400
    
    model_input = data["model"]
    method = data["method"].lower()
    dataset_name = data.get("dataset")
    sample_ids = data.get("sample_ids")
    
    # Parse model family and version
    model_family, model_version = get_model_info(model_input)
    
    # Validate model
    if not model_family:
        return jsonify({
            "message": f"Unsupported model: {model_input}. Supported families: {', '.join(SUPPORTED_MODELS)}. Use GET /health to see all supported versions."
        }), 400
    
    # Validate method
    if method not in SUPPORTED_METHODS:
        return jsonify({"message": f"Unsupported method: {method}. Supported: {', '.join(SUPPORTED_METHODS)}"}), 400
    
    # Determine input directory
    if dataset_name:
        # Using a HuggingFace dataset
        if dataset_name not in SUPPORTED_DATASETS:
            return jsonify({
                "message": f"Unsupported dataset: {dataset_name}. Supported: {', '.join(SUPPORTED_DATASETS)}"
            }), 400
        
        dm = get_dataset_manager()
        dataset_info = dm.get_dataset_info(dataset_name)
        
        if not dataset_info:
            return jsonify({
                "message": f"Dataset {dataset_name} not downloaded. Use POST /datasets/{dataset_name}/download first."
            }), 400
        
        # Prepare benchmark directory
        input_dir = dm.prepare_benchmark_dir(dataset_name, sample_ids)
        # Use model version in run_id for clarity
        run_id = data.get("run_id") or f"{dataset_name}_{method}_{model_version.replace('.', '-')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    elif "input_dir" in data:
        # Using a custom directory
        input_dir = resolve_path(data["input_dir"])
        if not os.path.exists(input_dir):
            return jsonify({"message": f"Input directory not found: {input_dir}"}), 400
        run_id = data.get("run_id") or f"{method}_{model_version.replace('.', '-')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_name = None
        
    else:
        return jsonify({"message": "Either 'dataset' or 'input_dir' must be provided"}), 400
    
    # Check if run_id already exists
    if run_id in RUNS_DB:
        existing = RUNS_DB[run_id]
        if existing.status in ["running", "pending"]:
            return jsonify({
                "message": f"Run {run_id} already exists and is {existing.status}",
                "launched": False,
                "run_id": run_id
            })
        else:
            return jsonify({
                "message": f"Run {run_id} already completed",
                "launched": False,
                "run_id": run_id
            })
    
    # Create run with model version
    run = Run(run_id, model_version, method, input_dir, g.api_key, dataset=dataset_name, sample_ids=sample_ids)
    RUNS_DB[run_id] = run
    
    # Start background task
    thread = Thread(target=run_experiment_task, args=(run,))
    thread.start()
    
    return jsonify({
        "message": f"Run {run_id} submitted successfully",
        "launched": True,
        "run_id": run_id,
        "model": model_version,
        "model_family": model_family,
        "method": method,
        "dataset": dataset_name
    })


@app.route("/poll-jobs", methods=["GET"])
@require_api_key
def poll_jobs():
    """
    Poll job status.
    
    Query params or JSON body:
    {
        "run_id": "my_run"           # Required: run ID to poll
    }
    """
    data = request.get_json(silent=True) or {}
    run_id = data.get("run_id") or request.args.get("run_id")
    
    if not run_id:
        return jsonify({"message": "run_id is required"}), 400
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    return jsonify(run.get_poll_status())


@app.route("/get-report", methods=["POST"])
@require_api_key
def get_report():
    """
    Get evaluation report for a run.
    
    Request body:
    {
        "run_id": "my_run"           # Required: run ID
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    if run.status != "completed":
        return jsonify({
            "message": f"Run {run_id} is not completed yet. Status: {run.status}",
            "status": run.status
        }), 400
    
    # Build report with dataset labeling
    report = {
        "run_id": run.run_id,
        "model": run.model,
        "method": run.method,
        "dataset": run.dataset,
        "dataset_info": DATASETS_CONFIG.get(run.dataset, {}) if run.dataset else None,
        "total_instances": run.total_instances,
        "completed_instances": len([i for i in run.instances.values() if i.get("status") == "completed"]),
        "failed_instances": len([i for i in run.instances.values() if i.get("status") == "failed"]),
        "resolved_instances": len([i for i in run.instances.values() if i.get("status") == "completed"]),
        "pending_instances": 0,
        "error_instances": len([i for i in run.instances.values() if i.get("status") == "failed"]),
        "submitted_instances": run.total_instances,
        "token_usage": run.token_usage,
        "cost_estimate": run.cost_estimate,
        "evaluation": run.evaluation,  # Already contains dataset, model, method, metrics
        "results": {
            "dataset": run.dataset,
            "method": run.method,
            "model": run.model,
            "output_dir": run.output_dir,
            "instances": {
                instance_id: {
                    "status": inst.get("status"),
                    "output_file": inst.get("result"),
                    "dataset": run.dataset,
                    "method": run.method,
                    "model": run.model
                }
                for instance_id, inst in run.instances.items()
            }
        },
        "created_at": run.created_at,
        "completed_at": run.completed_at
    }
    
    return jsonify({"report": report})


@app.route("/list-runs", methods=["POST"])
@require_api_key
def list_runs():
    """
    List all runs for the authenticated user.
    
    Request body:
    {
        "model": "gemini",           # Optional: filter by model
        "method": "dcgen",           # Optional: filter by method
        "dataset": "design2code"     # Optional: filter by dataset
    }
    """
    data = request.get_json() or {}
    model_filter = data.get("model")
    method_filter = data.get("method")
    dataset_filter = data.get("dataset")
    
    runs_list = []
    for run_id, run in RUNS_DB.items():
        if run.api_key != g.api_key:
            continue
        if model_filter and run.model != model_filter:
            continue
        if method_filter and run.method != method_filter:
            continue
        if dataset_filter and run.dataset != dataset_filter:
            continue
        runs_list.append({
            "run_id": run_id,
            "model": run.model,
            "method": run.method,
            "dataset": run.dataset,
            "status": run.status,
            "created_at": run.created_at
        })
    
    return jsonify({"runs": runs_list, "run_ids": [r["run_id"] for r in runs_list]})


@app.route("/delete-run", methods=["DELETE", "POST"])
@require_api_key
def delete_run():
    """
    Delete a run.
    
    Request body:
    {
        "run_id": "my_run"           # Required: run ID to delete
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    # Delete run
    del RUNS_DB[run_id]
    
    # Optionally delete output directory
    if os.path.exists(run.output_dir):
        shutil.rmtree(run.output_dir, ignore_errors=True)
    
    return jsonify({"message": f"Run {run_id} deleted successfully"})


@app.route("/stop-run", methods=["POST"])
@require_api_key
def stop_run():
    """
    Stop a running job and optionally run evaluation on completed instances.
    
    This is useful when you want to stop a long-running job early and
    evaluate only the instances that have completed so far.
    
    Request body:
    {
        "run_id": "my_run",           # Required: run ID to stop
        "run_evaluation": true        # Optional: run evaluation on completed instances (default: true)
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    run_eval = data.get("run_evaluation", True)
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    # Get counts before stopping
    completed_count = len([i for i in run.instances.values() if i.get("status") == "completed"])
    running_count = len([i for i in run.instances.values() if i.get("status") == "running"])
    pending_count = len([i for i in run.instances.values() if i.get("status") == "pending"])
    
    # Mark pending instances as skipped and running as stopped
    for instance_id, instance in run.instances.items():
        if instance.get("status") == "pending":
            instance["status"] = "skipped"
            instance["error"] = "Run stopped by user"
        elif instance.get("status") == "running":
            instance["status"] = "stopped"
            instance["error"] = "Run stopped by user"
    
    # Update run status
    if run.status == "running":
        run.status = "stopped"
    
    if completed_count == 0:
        run.completed_at = datetime.datetime.now().isoformat()
        run.save_to_disk()
        return jsonify({
            "message": f"Run {run_id} stopped. No completed instances to evaluate.",
            "run_id": run_id,
            "completed_instances": 0,
            "stopped_instances": running_count,
            "skipped_instances": pending_count
        })
    
    if run_eval:
        # Run evaluation in background
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
        
        return jsonify({
            "message": f"Run {run_id} stopped. Running evaluation on {completed_count} completed instances.",
            "run_id": run_id,
            "completed_instances": completed_count,
            "stopped_instances": running_count,
            "skipped_instances": pending_count,
            "evaluation_started": True
        })
    else:
        run.completed_at = datetime.datetime.now().isoformat()
        run.save_to_disk()
        return jsonify({
            "message": f"Run {run_id} stopped. Use rerun-evaluation to evaluate later.",
            "run_id": run_id,
            "completed_instances": completed_count,
            "stopped_instances": running_count,
            "skipped_instances": pending_count,
            "evaluation_started": False
        })


@app.route("/rerun-evaluation", methods=["POST"])
@require_api_key
def rerun_evaluation():
    """
    Re-run evaluation for a completed or failed run.
    
    This is useful when the initial evaluation timed out or failed,
    but the generation was successful.
    
    Request body:
    {
        "run_id": "my_run"           # Required: run ID to re-evaluate
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    # Check if run has completed instances
    completed_instances = len([i for i in run.instances.values() if i.get("status") == "completed"])
    if completed_instances == 0:
        return jsonify({
            "message": f"Run {run_id} has no completed instances to evaluate",
            "status": run.status
        }), 400
    
    # Set status to evaluating so the CLI knows to wait
    run.status = "evaluating"
    run.evaluation = None  # Clear previous evaluation
    run.save_to_disk()
    
    # Run evaluation in background
    def run_evaluation_task():
        try:
            # Check if PNG screenshots already exist for all completed instances
            missing_screenshots = False
            for instance_id, instance in run.instances.items():
                if instance.get("status") == "completed":
                    result_file = instance.get("result", "")
                    png_file = result_file.replace('.html', '.png') if result_file else ""
                    if not png_file or not os.path.exists(png_file):
                        missing_screenshots = True
                        break
            
            # Only regenerate screenshots if some are missing
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
    
    return jsonify({
        "message": f"Re-running evaluation for run {run_id}",
        "run_id": run_id,
        "completed_instances": completed_instances,
        "status": "evaluating"
    })


@app.route("/resume-run", methods=["POST"])
@require_api_key
def resume_run():
    """
    Resume a stopped/interrupted run by processing remaining pending instances.
    
    This is useful when a run was interrupted (server stopped, network issues, etc.)
    and you want to continue processing the remaining instances.
    
    Request body:
    {
        "run_id": "my_run"           # Required: run ID to resume
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    # Check if run is already running
    if run.status == "running":
        return jsonify({
            "message": f"Run {run_id} is already running",
            "status": run.status
        }), 400
    
    # Get pending instances
    pending_instances = {k: v for k, v in run.instances.items() 
                        if v.get("status") == "pending"}
    
    if not pending_instances:
        return jsonify({
            "message": f"Run {run_id} has no pending instances to process",
            "status": run.status
        }), 400
    
    # Start resume task in background
    def resume_run_task():
        try:
            run.status = "running"
            
            # Initialize bot
            bot = get_bot(run.model)
            if hasattr(bot, 'reset_token_usage'):
                bot.reset_token_usage()
            
            # Process pending instances
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
                        generate_dcgen(bot, img_path, save_path, SEG_PARAMS_DEFAULT)
                    else:
                        generate_single(PROMPT_DIRECT, bot, img_path, save_path)
                    
                    run.instances[instance_id]["status"] = "completed"
                    run.instances[instance_id]["result"] = save_path
                    
                except Exception as e:
                    run.instances[instance_id]["status"] = "failed"
                    run.instances[instance_id]["error"] = str(e)
            
            # Copy placeholder if not exists
            placeholder_src = os.path.join(run.input_dir, "placeholder.png")
            placeholder_dst = os.path.join(run.output_dir, "placeholder.png")
            if os.path.exists(placeholder_src) and not os.path.exists(placeholder_dst):
                shutil.copy(placeholder_src, placeholder_dst)
            
            # Take screenshots
            take_screenshots_task(run.output_dir, replace=False)
            
            # Update token usage
            if hasattr(bot, 'print_token_usage'):
                new_usage = bot.print_token_usage(f"{run.method}_resume")
                if run.token_usage and new_usage:
                    run.token_usage["total_prompt_tokens"] += new_usage.get("total_prompt_tokens", 0)
                    run.token_usage["total_response_tokens"] += new_usage.get("total_response_tokens", 0)
                    run.token_usage["call_count"] += new_usage.get("call_count", 0)
                elif new_usage:
                    run.token_usage = new_usage
            
            # Run evaluation
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
    
    return jsonify({
        "message": f"Resuming run {run_id} with {len(pending_instances)} pending instances",
        "run_id": run_id,
        "pending_instances": list(pending_instances.keys()),
        "total_pending": len(pending_instances)
    })


@app.route("/retry-failed", methods=["POST"])
@require_api_key
def retry_failed():
    """
    Retry failed instances in a run.
    
    This will re-attempt generation for instances that failed due to temporary
    errors (network issues, API rate limits, etc.) while keeping completed instances.
    
    Request body:
    {
        "run_id": "my_run",           # Required: run ID to retry failed instances
        "max_retries": 3              # Optional: max retry attempts per instance (default: 3)
    }
    """
    data = request.get_json()
    
    if not data or "run_id" not in data:
        return jsonify({"message": "run_id is required"}), 400
    
    run_id = data["run_id"]
    max_retries = data.get("max_retries", 3)
    
    if run_id not in RUNS_DB:
        return jsonify({"message": f"Run {run_id} not found"}), 404
    
    run = RUNS_DB[run_id]
    
    # Verify ownership
    if run.api_key != g.api_key:
        return jsonify({"message": "Access denied"}), 403
    
    # Get failed instances
    failed_instances = {k: v for k, v in run.instances.items() 
                       if v.get("status") in ["failed", "stopped"]}
    
    if not failed_instances:
        return jsonify({
            "message": f"Run {run_id} has no failed instances to retry",
            "status": run.status
        }), 400
    
    # Start retry task in background
    def retry_failed_task():
        try:
            # Reset run status to running
            run.status = "running"
            
            # Initialize bot
            bot = get_bot(run.model)
            if hasattr(bot, 'reset_token_usage'):
                bot.reset_token_usage()
            
            # Retry failed instances
            retry_count = 0
            for instance_id, instance in failed_instances.items():
                # Find the original image file
                img_path = os.path.join(run.input_dir, f"{instance_id}.png")
                if not os.path.exists(img_path):
                    continue
                
                filename = f"{instance_id}.html"
                save_path = os.path.join(run.output_dir, filename)
                
                # Mark as running
                run.instances[instance_id]["status"] = "running"
                run.instances[instance_id]["retry_count"] = instance.get("retry_count", 0) + 1
                
                # Skip if max retries exceeded
                if run.instances[instance_id]["retry_count"] > max_retries:
                    run.instances[instance_id]["status"] = "failed"
                    run.instances[instance_id]["error"] = f"Max retries ({max_retries}) exceeded"
                    continue
                
                try:
                    if run.method == "dcgen":
                        generate_dcgen(bot, img_path, save_path, SEG_PARAMS_DEFAULT)
                    else:
                        generate_single(PROMPT_DIRECT, bot, img_path, save_path)
                    
                    run.instances[instance_id]["status"] = "completed"
                    run.instances[instance_id]["result"] = save_path
                    run.instances[instance_id]["error"] = None
                    retry_count += 1
                    
                except Exception as e:
                    run.instances[instance_id]["status"] = "failed"
                    run.instances[instance_id]["error"] = str(e)
            
            # Copy placeholder if not exists
            placeholder_src = os.path.join(run.input_dir, "placeholder.png")
            placeholder_dst = os.path.join(run.output_dir, "placeholder.png")
            if os.path.exists(placeholder_src) and not os.path.exists(placeholder_dst):
                shutil.copy(placeholder_src, placeholder_dst)
            
            # Take screenshots for newly completed instances
            take_screenshots_task(run.output_dir, replace=False)
            
            # Update token usage
            if hasattr(bot, 'print_token_usage'):
                new_usage = bot.print_token_usage(f"{run.method}_retry")
                if run.token_usage and new_usage:
                    # Accumulate token usage
                    run.token_usage["total_prompt_tokens"] += new_usage.get("total_prompt_tokens", 0)
                    run.token_usage["total_response_tokens"] += new_usage.get("total_response_tokens", 0)
                    run.token_usage["call_count"] += new_usage.get("call_count", 0)
                elif new_usage:
                    run.token_usage = new_usage
            
            # Re-run evaluation
            run.evaluation = run_evaluation_for_run(run)
            
            # Update status
            still_failed = len([i for i in run.instances.values() if i.get("status") == "failed"])
            if still_failed == 0:
                run.status = "completed"
            else:
                run.status = "completed"  # Mark as completed even if some failed after retries
            
            run.completed_at = datetime.datetime.now().isoformat()
            run.save_to_disk()
            
        except Exception as e:
            run.status = "failed"
            run.error = f"Retry failed: {str(e)}"
            run.save_to_disk()
    
    thread = Thread(target=retry_failed_task)
    thread.start()
    
    return jsonify({
        "message": f"Retrying {len(failed_instances)} failed instances for run {run_id}",
        "run_id": run_id,
        "failed_instances": list(failed_instances.keys()),
        "max_retries": max_retries
    })


@app.route("/run-all", methods=["POST"])
@require_api_key
def run_all():
    """
    Run full pipeline: both DCGen and Direct methods with evaluation.
    
    Request body:
    {
        "model": "gemini",           # Required: model family or specific version
        "dataset": "design2code",    # Optional: dataset name (design2code, dcgen)
        "input_dir": "./demo",       # Required if no dataset: input directory with PNG images
        "sample_ids": ["0", "1"],    # Optional: specific sample IDs to run (for datasets)
        "run_id": "my_experiment"    # Optional: base run ID
    }
    
    Either 'dataset' or 'input_dir' must be provided.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"message": "Request body is required"}), 400
    
    if "model" not in data:
        return jsonify({"message": "Missing required field: model"}), 400
    
    model_input = data["model"]
    dataset_name = data.get("dataset")
    sample_ids = data.get("sample_ids")
    
    # Parse model family and version
    model_family, model_version = get_model_info(model_input)
    
    if not model_family:
        return jsonify({
            "message": f"Unsupported model: {model_input}. Supported families: {', '.join(SUPPORTED_MODELS)}"
        }), 400
    
    # Determine input directory
    if dataset_name:
        # Using a HuggingFace dataset
        if dataset_name not in SUPPORTED_DATASETS:
            return jsonify({
                "message": f"Unsupported dataset: {dataset_name}. Supported: {', '.join(SUPPORTED_DATASETS)}"
            }), 400
        
        dm = get_dataset_manager()
        dataset_info = dm.get_dataset_info(dataset_name)
        
        if not dataset_info:
            return jsonify({
                "message": f"Dataset {dataset_name} not downloaded. Use POST /datasets/{dataset_name}/download first."
            }), 400
        
        # Prepare benchmark directory
        input_dir = dm.prepare_benchmark_dir(dataset_name, sample_ids)
        base_run_id = data.get("run_id") or f"{dataset_name}_{model_version.replace('.', '-')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    elif "input_dir" in data:
        # Using a custom directory
        input_dir = resolve_path(data["input_dir"])
        if not os.path.exists(input_dir):
            return jsonify({"message": f"Input directory not found: {input_dir}"}), 400
        base_run_id = data.get("run_id") or f"exp_{model_version.replace('.', '-')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_name = None
        
    else:
        return jsonify({"message": "Either 'dataset' or 'input_dir' must be provided"}), 400
    
    # Create runs for both methods
    runs = []
    for method in ["dcgen", "direct"]:
        run_id = f"{base_run_id}_{method}"
        run = Run(run_id, model_version, method, input_dir, g.api_key, dataset=dataset_name, sample_ids=sample_ids)
        RUNS_DB[run_id] = run
        runs.append(run_id)
        
        thread = Thread(target=run_experiment_task, args=(run,))
        thread.start()
    
    return jsonify({
        "message": "Full pipeline started",
        "run_ids": runs,
        "dataset": dataset_name,
        "model": model_version,
        "model_family": model_family
    })


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DCGen API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_dir(DATA_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Load existing runs from disk
    loaded_runs = load_existing_runs()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Loaded {loaded_runs} existing runs from disk                                     ║
║                        DCGen API Server v{API_VERSION}                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Authentication:                                                         ║
║    Set DCGEN_API_KEY env var or use x-api-key header                     ║
║    Default dev key: {DEFAULT_API_KEY}                                ║
║                                                                          ║
║  OpenAI Base URL: {OPENAI_BASE_URL:50}║
║                                                                          ║
║  Datasets: {', '.join(SUPPORTED_DATASETS):52}║
║  Models:   {', '.join(SUPPORTED_MODELS):52}║
║  Methods:  {', '.join(SUPPORTED_METHODS):52}║
║                                                                          ║
║  Dataset Endpoints (read-only, pre-downloaded):                          ║
║    GET  /datasets                    - List available datasets           ║
║    GET  /datasets/<name>             - Get dataset info                  ║
║    GET  /datasets/<name>/samples     - Get dataset samples               ║
║                                                                          ║
║  Experiment Endpoints:                                                   ║
║    POST /submit         - Submit experiment run                          ║
║    GET  /poll-jobs      - Poll job status                                ║
║    POST /get-report     - Get evaluation report                          ║
║    POST /list-runs      - List all runs                                  ║
║    POST /delete-run     - Delete a run                                   ║
║    POST /run-all        - Run full pipeline (DCGen + Direct)             ║
║                                                                          ║
║  Other Endpoints:                                                        ║
║    POST /gen-api-key    - Generate new API key                           ║
║    GET  /get-quotas     - Get API quotas                                 ║
║    GET  /health         - Health check                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug)
