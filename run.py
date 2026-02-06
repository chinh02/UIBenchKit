#!/usr/bin/env python3
"""
DCGen End-to-End Pipeline
=========================

A unified CLI for running image-to-HTML experiments and evaluation.

Usage:
    python run.py experiment --model gemini --input ./data/demo --output ./data/output
    python run.py evaluate --reference ./data/demo --dcgen ./data/dcgen_demo --direct ./data/direct_demo
    python run.py run-all --model gemini --input ./data/demo --name my_experiment
    python run.py screenshot --dir ./data/dcgen_demo

Examples:
    # Run full experiment with Gemini
    python run.py run-all --model gemini --input ./data/demo --name test_run

    # Run only DCGen method
    python run.py experiment --model gpt4 --input ./data/demo --output ./data/gpt4_dcgen --method dcgen

    # Run only evaluation
    python run.py evaluate --reference ./data/demo --dcgen ./data/dcgen_demo --direct ./data/direct_demo

    # Force regeneration (ignore existing files)
    python run.py run-all --model gemini --input ./data/demo --name test --force
"""

import argparse
import os
import sys
import json
import datetime
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    encode_image, get_driver, take_screenshot,
    ImgSegmentation, DCGenGrid
)
from models import GPT4, Gemini, Claude, QwenVL
from dotenv import load_dotenv
from tqdm import tqdm
from threading import Thread
from rapidfuzz import fuzz
import re
import time

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


# ============================================================
# Model Factory
# ============================================================
def get_bot(model_name: str):
    """Initialize the specified model."""
    load_dotenv()
    
    model_name = model_name.lower()
    
    if model_name in ["gemini", "gemini-pro"]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        return Gemini(api_key)
    
    elif model_name in ["gpt4", "gpt-4", "gpt4o", "gpt-4o"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return GPT4(api_key, model="gpt-4o")
    
    elif model_name in ["claude", "claude-sonnet"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return Claude(api_key)
    
    elif model_name in ["qwen", "qwen-vl"]:
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY not found in environment")
        return QwenVL(api_key)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: gemini, gpt4, claude, qwen")


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


# ============================================================
# Generation Methods
# ============================================================
def generate_single(prompt: str, bot, img_path: str, save_path: str = None, max_retries: int = 3):
    """Generate HTML from a single image using direct prompting."""
    for i in range(max_retries):
        try:
            html = bot.ask(prompt, encode_image(img_path))
            code = re.findall(r"```html([^`]+)```", html)
            if code:
                html = code[0]
            if len(html) < 10:
                raise Exception("No HTML code found in response")
            if save_path:
                with open(save_path, 'w', encoding="utf-8") as f:
                    f.write(html)
            return html
        except Exception as e:
            print(f"  Attempt {i+1} failed: {e}")
            time.sleep(1)
    raise Exception(f"Failed to generate HTML for {img_path}")


def generate_dcgen(bot, img_path: str, save_path: str = None, seg_params: dict = None):
    """Generate HTML from a single image using DCGen method."""
    print(f"  DCGen: {os.path.basename(img_path)}")
    
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


def run_experiment(bot, input_dir: str, output_dir: str, method: str = "dcgen", 
                   force: bool = False, multi_thread: bool = True, seg_params: dict = None):
    """Run experiment on a directory of images."""
    ensure_dir(output_dir)
    filelist = get_image_files(input_dir)
    
    print(f"\nRunning {method.upper()} on {len(filelist)} images...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    
    def process_file(img_path):
        filename = os.path.basename(img_path).replace('.png', '.html')
        save_path = os.path.join(output_dir, filename)
        
        if os.path.exists(save_path) and not force:
            print(f"  Skipping {filename} (exists)")
            return
        
        try:
            if method == "dcgen":
                generate_dcgen(bot, img_path, save_path, seg_params)
            else:  # direct
                generate_single(PROMPT_DIRECT, bot, img_path, save_path)
                print(f"  Direct: {filename}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    if multi_thread:
        threads = []
        for img_path in tqdm(filelist, desc=method.upper()):
            t = Thread(target=process_file, args=(img_path,))
            t.start()
            threads.append(t)
            if len(threads) >= 5:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()
    else:
        for img_path in tqdm(filelist, desc=method.upper()):
            process_file(img_path)


# ============================================================
# Screenshot Generation
# ============================================================
def take_screenshots(directory: str, replace: bool = False):
    """Generate screenshots for all HTML files in directory."""
    html_files = [f for f in os.listdir(directory) if f.endswith('.html')]
    
    if not html_files:
        print(f"No HTML files found in {directory}")
        return
    
    print(f"\nGenerating screenshots for {len(html_files)} files in {directory}...")
    
    driver = get_driver(string="<html></html>")
    
    for filename in tqdm(html_files, desc="Screenshots"):
        html_path = os.path.join(directory, filename)
        png_path = html_path.replace('.html', '.png')
        
        if os.path.exists(png_path) and not replace:
            continue
        
        try:
            driver.get("file://" + os.path.abspath(html_path))
            take_screenshot(driver, png_path)
        except Exception as e:
            print(f"  Error: {filename}: {e}")
    
    driver.quit()


def copy_placeholder(src_dir: str, dst_dir: str):
    """Copy placeholder.png to destination directory."""
    src = os.path.join(src_dir, "placeholder.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(dst_dir, "placeholder.png"))


# ============================================================
# Evaluation
# ============================================================
def compute_code_similarity(file1: str, file2: str) -> float:
    """Compute fuzzy string similarity between two HTML files."""
    with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
        html1 = f.read()
    with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
        html2 = f.read()
    return fuzz.ratio(html1, html2)


def compute_clip_score(img1_path: str, img2_path: str, scorer) -> float:
    """Compute CLIP similarity score between two images."""
    from PIL import Image
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    return scorer.score(img1, img2)


def evaluate_directory(ref_dir: str, test_dir: str, metric: str = "clip", scorer=None) -> dict:
    """Evaluate a test directory against reference."""
    results = {}
    test_files = os.listdir(test_dir)
    
    for filename in tqdm(test_files, desc=f"Eval {os.path.basename(test_dir)}"):
        if metric == "clip":
            if not filename.endswith('.png') or 'placeholder' in filename:
                continue
            ref_file = os.path.join(ref_dir, filename)
            test_file = os.path.join(test_dir, filename)
            if os.path.exists(ref_file):
                try:
                    results[filename] = compute_clip_score(ref_file, test_file, scorer)
                except Exception as e:
                    print(f"  Error: {filename}: {e}")
        else:  # code similarity
            if not filename.endswith('.html'):
                continue
            ref_file = os.path.join(ref_dir, filename)
            test_file = os.path.join(test_dir, filename)
            if os.path.exists(ref_file):
                try:
                    results[filename] = compute_code_similarity(ref_file, test_file)
                except Exception as e:
                    print(f"  Error: {filename}: {e}")
    
    return results


def run_evaluation(ref_dir: str, test_dirs: dict, output_name: str = "evaluation"):
    """Run full evaluation suite."""
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    print(f"Reference: {ref_dir}")
    for name, path in test_dirs.items():
        print(f"  {name}: {path}")
    
    results = {"timestamp": datetime.datetime.now().isoformat()}
    
    # CLIP evaluation
    print("\n--- CLIP Score (Visual Similarity) ---")
    try:
        import open_clip
        import torch
        from PIL import Image
        
        class CLIPScorer:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32-quickgelu', pretrained='openai'
                )
                self.model.to(self.device)
            
            def score(self, img1, img2):
                image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
                image2 = self.preprocess(img2).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat1 = self.model.encode_image(image1)
                    feat2 = self.model.encode_image(image2)
                feat1 /= feat1.norm(dim=-1, keepdim=True)
                feat2 /= feat2.norm(dim=-1, keepdim=True)
                return torch.nn.functional.cosine_similarity(feat1, feat2).item()
        
        scorer = CLIPScorer()
        results["clip"] = {}
        
        for name, test_dir in test_dirs.items():
            scores = evaluate_directory(ref_dir, test_dir, metric="clip", scorer=scorer)
            if scores:
                avg = sum(scores.values()) / len(scores)
                results["clip"][name] = {"scores": scores, "average": avg}
                print(f"  {name}: {avg:.4f}")
    except ImportError:
        print("  Skipped (open_clip not installed)")
    
    # Code similarity evaluation
    print("\n--- Code Similarity ---")
    results["code_sim"] = {}
    
    for name, test_dir in test_dirs.items():
        scores = evaluate_directory(ref_dir, test_dir, metric="code")
        if scores:
            avg = sum(scores.values()) / len(scores)
            results["code_sim"][name] = {"scores": scores, "average": avg}
            print(f"  {name}: {avg:.2f}%")
    
    # Save results
    output_file = f"{output_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'CLIP Score':>15} {'Code Sim':>15}")
    print("-" * 45)
    
    for name in test_dirs.keys():
        clip_avg = results.get("clip", {}).get(name, {}).get("average", 0)
        code_avg = results.get("code_sim", {}).get(name, {}).get("average", 0)
        print(f"{name:<15} {clip_avg:>15.4f} {code_avg:>14.2f}%")
    
    return results


# ============================================================
# CLI Commands
# ============================================================
def cmd_experiment(args):
    """Run generation experiment."""
    bot = get_bot(args.model)
    
    if hasattr(bot, 'reset_token_usage'):
        bot.reset_token_usage()
    
    run_experiment(
        bot, 
        args.input, 
        args.output, 
        method=args.method,
        force=args.force,
        multi_thread=not args.no_parallel,
        seg_params=SEG_PARAMS_DEFAULT
    )
    
    # Copy placeholder and take screenshots
    copy_placeholder(args.input, args.output)
    take_screenshots(args.output, replace=True)
    
    if hasattr(bot, 'print_token_usage'):
        bot.print_token_usage(f"{args.method.upper()} Experiment")


def cmd_evaluate(args):
    """Run evaluation."""
    test_dirs = {}
    if args.dcgen:
        test_dirs["dcgen"] = args.dcgen
    if args.direct:
        test_dirs["direct"] = args.direct
    
    if not test_dirs:
        print("Error: At least one of --dcgen or --direct must be specified")
        sys.exit(1)
    
    run_evaluation(args.reference, test_dirs, args.output)


def cmd_screenshot(args):
    """Generate screenshots for HTML files."""
    take_screenshots(args.dir, replace=args.force)


def cmd_run_all(args):
    """Run full end-to-end pipeline."""
    print(f"\n{'='*60}")
    print(f"DCGen End-to-End Pipeline")
    print(f"{'='*60}")
    print(f"Model:  {args.model}")
    print(f"Input:  {args.input}")
    print(f"Name:   {args.name}")
    print(f"{'='*60}")
    
    # Setup directories
    base_dir = os.path.dirname(args.input.rstrip('/'))
    dcgen_dir = os.path.join(base_dir, f"{args.name}_dcgen")
    direct_dir = os.path.join(base_dir, f"{args.name}_direct")
    
    bot = get_bot(args.model)
    
    # Run DCGen
    print(f"\n{'='*60}")
    print("STEP 1: DCGen Generation")
    print(f"{'='*60}")
    if hasattr(bot, 'reset_token_usage'):
        bot.reset_token_usage()
    
    run_experiment(bot, args.input, dcgen_dir, method="dcgen", 
                   force=args.force, seg_params=SEG_PARAMS_DEFAULT)
    copy_placeholder(args.input, dcgen_dir)
    take_screenshots(dcgen_dir, replace=True)
    
    dcgen_tokens = None
    if hasattr(bot, 'print_token_usage'):
        dcgen_tokens = bot.print_token_usage("DCGen")
    
    # Run Direct
    print(f"\n{'='*60}")
    print("STEP 2: Direct Prompting Generation")
    print(f"{'='*60}")
    if hasattr(bot, 'reset_token_usage'):
        bot.reset_token_usage()
    
    run_experiment(bot, args.input, direct_dir, method="direct", 
                   force=args.force)
    copy_placeholder(args.input, direct_dir)
    take_screenshots(direct_dir, replace=True)
    
    direct_tokens = None
    if hasattr(bot, 'print_token_usage'):
        direct_tokens = bot.print_token_usage("Direct")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("STEP 3: Evaluation")
    print(f"{'='*60}")
    test_dirs = {"dcgen": dcgen_dir, "direct": direct_dir}
    results = run_evaluation(args.input, test_dirs, args.name)
    
    # Token comparison
    if dcgen_tokens and direct_tokens:
        print(f"\n{'='*60}")
        print("TOKEN USAGE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'DCGen':>12} {'Direct':>12}")
        print("-" * 45)
        print(f"{'API Calls':<20} {dcgen_tokens.get('call_count', 0):>12,} {direct_tokens.get('call_count', 0):>12,}")
        print(f"{'Total Tokens':<20} {dcgen_tokens.get('total_tokens', 0):>12,} {direct_tokens.get('total_tokens', 0):>12,}")
        
        # Save token comparison
        token_comparison = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dcgen": dcgen_tokens,
            "direct": direct_tokens
        }
        with open(f"{args.name}_tokens.json", 'w') as f:
            json.dump(token_comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  DCGen output:  {dcgen_dir}")
    print(f"  Direct output: {direct_dir}")
    print(f"  Results:       {args.name}_results.json")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="DCGen End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run generation experiment")
    exp_parser.add_argument("--model", "-m", required=True, 
                           choices=["gemini", "gpt4", "claude", "qwen"],
                           help="Model to use")
    exp_parser.add_argument("--input", "-i", required=True, help="Input directory with PNG images")
    exp_parser.add_argument("--output", "-o", required=True, help="Output directory for HTML files")
    exp_parser.add_argument("--method", choices=["dcgen", "direct"], default="dcgen",
                           help="Generation method (default: dcgen)")
    exp_parser.add_argument("--force", "-f", action="store_true", 
                           help="Force regeneration of existing files")
    exp_parser.add_argument("--no-parallel", action="store_true",
                           help="Disable parallel processing")
    exp_parser.set_defaults(func=cmd_experiment)
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--reference", "-r", required=True, 
                            help="Reference directory with original HTML/PNG")
    eval_parser.add_argument("--dcgen", help="DCGen output directory")
    eval_parser.add_argument("--direct", help="Direct prompting output directory")
    eval_parser.add_argument("--output", "-o", default="evaluation",
                            help="Output name prefix for results")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # screenshot command
    ss_parser = subparsers.add_parser("screenshot", help="Generate screenshots for HTML files")
    ss_parser.add_argument("--dir", "-d", required=True, help="Directory with HTML files")
    ss_parser.add_argument("--force", "-f", action="store_true", 
                          help="Replace existing screenshots")
    ss_parser.set_defaults(func=cmd_screenshot)
    
    # run-all command
    all_parser = subparsers.add_parser("run-all", help="Run full end-to-end pipeline")
    all_parser.add_argument("--model", "-m", required=True,
                           choices=["gemini", "gpt4", "claude", "qwen"],
                           help="Model to use")
    all_parser.add_argument("--input", "-i", required=True, 
                           help="Input directory with PNG images")
    all_parser.add_argument("--name", "-n", required=True,
                           help="Experiment name (used for output directories)")
    all_parser.add_argument("--force", "-f", action="store_true",
                           help="Force regeneration of existing files")
    all_parser.set_defaults(func=cmd_run_all)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
