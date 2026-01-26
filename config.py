#!/usr/bin/env python3
"""
DCGen Configuration Module
==========================

Central configuration for the DCGen API server.
Contains constants, model configurations, pricing, and prompts.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Server Configuration
# ============================================================
API_VERSION = "1.0.0"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Default API key for development (set via env var in production)
DEFAULT_API_KEY = os.getenv("DCGEN_API_KEY", "dev-api-key-12345")

# Custom OpenAI-compatible API base URL
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")

# ============================================================
# Model Configuration
# ============================================================
MODEL_FAMILIES = {
    "gemini": {
        "default": "gemini-2.0-flash",
        "versions": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
            "gemini-exp-1206",
        ]
    },
    "gpt4": {
        "default": "gpt-4o",
        "versions": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-4-1",
            "gpt-4-vision-preview",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-2024-12-17",
            "o1-mini-2024-09-12",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o3-2025-04-16",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1",
            "gpt-5.2"
        ]
    },
    "claude": {
        "default": "claude-3-5-sonnet-20241022",
        "versions": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001"
        ]
    },
    "qwen": {
        "default": "qwen2.5-vl-72b-instruct",
        "versions": [
            "qwen2.5-vl-72b-instruct",
            "qwen2.5-vl-7b-instruct",
            "qwen2-vl-72b-instruct",
            "qwen2-vl-7b-instruct",
            "qwen-vl-max",
            "qwen-vl-plus",
            "qwen-turbo",
            "qwen-plus", 
            "qwen-max",
            "qwen-long",
            "qwen-max-latest",
            "qwen-plus-latest",
            "qwen-turbo-latest",
            "qwq-plus",
            "qwen3-32b",
            "qwen3-235b-a22b-instruct-2507"
        ]
    },
    "deepseek": {
        "default": "deepseek-chat",
        "versions": [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
            "deepseek-v3",
            "deepseek-r1"
        ]
    },
    "grok": {
        "default": "grok-vision-beta", 
        "versions": [
            "grok-2-vision-1212",
            "grok-2-1212",
            "grok-beta",
            "grok-vision-beta",
            "grok-3",
            "grok-4",
            "grok-3-mini-fast"
        ]
    },
    "doubao": {
        "default": "doubao-vision-pro-32k",
        "versions": [
             "doubao-vision-pro-32k",
             "doubao-pro-4k",
             "doubao-pro-32k",
             "doubao-lite-4k",
             "doubao-lite-32k",
             "doubao-1-5-vision-pro-32k-250115",
             "doubao-1-5-pro-32k-250115",
             "doubao-1.5-vision-pro-250328"
        ]
    },
    "kimi": {
         "default": "moonshot-v1-8k",
         "versions": [
             "moonshot-v1-8k",
             "moonshot-v1-32k",
             "moonshot-v1-128k",
             "kimi-k2"
         ]
    },
    "mistral": {
        "default": "us.mistral.mistral-large-2407-v1:0",
        "versions": [
            # AWS Bedrock Mistral models - Cross-region inference profiles (recommended)
            "us.mistral.mistral-large-2407-v1:0",
            "us.mistral.mistral-large-2402-v1:0",
            "us.mistral.mistral-small-2402-v1:0",
            "us.mistral.mixtral-8x7b-instruct-v0:1",
            "us.mistral.mistral-7b-instruct-v0:2",
            "us.mistral.pixtral-large-2411-v1:0",
            "us.mistral.pixtral-12b-2409-v1:0",
            # Direct model IDs (on-demand)
            "mistral.mistral-large-2407-v1:0",
            "mistral.mistral-large-2402-v1:0",
            "mistral.mistral-small-2402-v1:0",
            "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.pixtral-large-2411-v1:0",
            "mistral.pixtral-12b-2409-v1:0",
        ]
    },
    "llama": {
        "default": "us.meta.llama3-2-90b-instruct-v1:0",
        "versions": [
            # AWS Bedrock Llama models - Cross-region inference profiles (recommended)
            # Use us.*, eu.*, or apac.* prefix for cross-region inference
            "us.meta.llama3-2-90b-instruct-v1:0",
            "us.meta.llama3-2-11b-instruct-v1:0",
            "us.meta.llama3-2-3b-instruct-v1:0",
            "us.meta.llama3-2-1b-instruct-v1:0",
            "us.meta.llama3-1-405b-instruct-v1:0",
            "us.meta.llama3-1-70b-instruct-v1:0",
            "us.meta.llama3-1-8b-instruct-v1:0",
            "us.meta.llama3-70b-instruct-v1:0",
            "us.meta.llama3-8b-instruct-v1:0",
            # Llama 4 models
            "us.meta.llama4-scout-17b-instruct-v1:0",
            "us.meta.llama4-maverick-17b-128e-instruct-v1:0",
            # Direct model IDs (on-demand, may not be available in all regions)
            "meta.llama3-2-90b-instruct-v1:0",
            "meta.llama3-2-11b-instruct-v1:0",
            "meta.llama3-2-3b-instruct-v1:0",
            "meta.llama3-2-1b-instruct-v1:0",
            "meta.llama3-1-405b-instruct-v1:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "meta.llama3-1-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-128e-instruct-v1:0",
        ]
    }
}

SUPPORTED_MODELS = list(MODEL_FAMILIES.keys())
SUPPORTED_METHODS = ["dcgen", "direct"]

# ============================================================
# Model Pricing (per 1M tokens, in USD)
# ============================================================
MODEL_PRICING = {
    # OpenAI GPT-4 family
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-1": {"input": 2.00, "output": 8.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-vision-preview": {"input": 10.00, "output": 30.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    
    # Google Gemini family
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "gemini-exp-1206": {"input": 0.10, "output": 0.40},
    
    # Anthropic Claude family
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    
    # Alibaba Qwen family
    "qwen2.5-vl-72b-instruct": {"input": 0.80, "output": 0.80},
    "qwen2.5-vl-7b-instruct": {"input": 0.20, "output": 0.20},
    "qwen2-vl-72b-instruct": {"input": 0.80, "output": 0.80},
    "qwen2-vl-7b-instruct": {"input": 0.20, "output": 0.20},
    "qwen-vl-max": {"input": 2.00, "output": 2.00},
    "qwen-vl-plus": {"input": 0.80, "output": 0.80},
    
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    
    # xAI Grok
    "grok-beta": {"input": 5.00, "output": 15.00},
    "grok-vision-beta": {"input": 5.00, "output": 15.00},
    
    # Moonshot Kimi
    "moonshot-v1-8k": {"input": 0.012, "output": 0.012},
    "moonshot-v1-32k": {"input": 0.024, "output": 0.024},
    
    # Doubao
    "doubao-pro-4k": {"input": 0.012, "output": 0.015},
    "doubao-pro-32k": {"input": 0.012, "output": 0.015},
    
    # AWS Bedrock - Mistral models (per 1M tokens)
    "mistral.mistral-large-2407-v1:0": {"input": 3.00, "output": 9.00},
    "mistral.mistral-large-2402-v1:0": {"input": 4.00, "output": 12.00},
    "mistral.mistral-small-2402-v1:0": {"input": 0.10, "output": 0.30},
    "mistral.mixtral-8x7b-instruct-v0:1": {"input": 0.45, "output": 0.70},
    "mistral.mistral-7b-instruct-v0:2": {"input": 0.15, "output": 0.20},
    "mistral.pixtral-large-2411-v1:0": {"input": 3.00, "output": 9.00},
    "mistral.pixtral-12b-2409-v1:0": {"input": 0.15, "output": 0.15},
    
    # AWS Bedrock - Meta Llama models (per 1M tokens)
    "meta.llama3-2-90b-instruct-v1:0": {"input": 2.00, "output": 2.00},
    "meta.llama3-2-11b-instruct-v1:0": {"input": 0.35, "output": 0.40},
    "meta.llama3-2-3b-instruct-v1:0": {"input": 0.15, "output": 0.15},
    "meta.llama3-2-1b-instruct-v1:0": {"input": 0.10, "output": 0.10},
    "meta.llama3-1-405b-instruct-v1:0": {"input": 5.32, "output": 16.00},
    "meta.llama3-1-70b-instruct-v1:0": {"input": 2.65, "output": 3.50},
    "meta.llama3-1-8b-instruct-v1:0": {"input": 0.30, "output": 0.60},
    "meta.llama3-70b-instruct-v1:0": {"input": 2.65, "output": 3.50},
    "meta.llama3-8b-instruct-v1:0": {"input": 0.30, "output": 0.60},
    "meta.llama4-scout-17b-instruct-v1:0": {"input": 0.17, "output": 0.68},
    "meta.llama4-maverick-17b-128e-instruct-v1:0": {"input": 0.20, "output": 0.80},
    
    # Cross-region inference profiles have same pricing as base models
    # The calculate_cost function will strip the region prefix when looking up prices
}

DEFAULT_PRICING = {"input": 5.00, "output": 15.00}

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

# ============================================================
# DCGen Segmentation Parameters
# ============================================================
SEG_PARAMS_DEFAULT = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50
}

# ============================================================
# MLLM Judge Prompts
# ============================================================
MLLM_JUDGE_PROMPTS = {
    "single_score": """You are an expert web designer and UI/UX evaluator. Compare the two images below:

IMAGE 1 (Reference): The original target webpage design
IMAGE 2 (Generated): A generated webpage attempting to reproduce the reference

Evaluate how well the generated webpage reproduces the reference on these criteria:
1. **Layout Accuracy** (0-10): Overall structure, positioning of elements, spacing
2. **Visual Fidelity** (0-10): Colors, fonts, visual style matching
3. **Content Completeness** (0-10): All text, images, and elements present
4. **Responsiveness/Polish** (0-10): Professional appearance, no obvious bugs

Provide your evaluation in the following JSON format:
```json
{
    "layout_accuracy": <score>,
    "visual_fidelity": <score>,
    "content_completeness": <score>,
    "responsiveness_polish": <score>,
    "overall_score": <weighted average>,
    "strengths": ["list of things done well"],
    "weaknesses": ["list of issues or missing elements"],
    "summary": "Brief 1-2 sentence overall assessment"
}
```""",

    "pairwise_comparison": """You are an expert web designer and UI/UX evaluator. You will see THREE images:

IMAGE 1 (Reference): The original target webpage design
IMAGE 2 (Model A): A generated webpage from Model A
IMAGE 3 (Model B): A generated webpage from Model B

Compare both generated webpages against the reference and determine which one better reproduces the original design.

Consider:
- Layout accuracy and element positioning
- Visual fidelity (colors, fonts, spacing)
- Content completeness
- Overall polish and professionalism

Provide your evaluation in the following JSON format:
```json
{
    "winner": "A" or "B" or "tie",
    "model_a_score": <0-10>,
    "model_b_score": <0-10>,
    "model_a_strengths": ["list"],
    "model_a_weaknesses": ["list"],
    "model_b_strengths": ["list"],
    "model_b_weaknesses": ["list"],
    "reasoning": "Explain why you chose the winner"
}
```""",

    "criteria_check": """You are an expert web designer. Compare the generated webpage (IMAGE 2) against the reference (IMAGE 1).

For each criterion, answer YES or NO and provide a brief explanation:

1. Does the layout match the reference structure?
2. Are all major text elements present and readable?
3. Do the colors approximately match?
4. Are navigation elements correctly positioned?
5. Is the overall visual hierarchy preserved?

Provide your evaluation in the following JSON format:
```json
{
    "layout_match": {"pass": true/false, "explanation": "..."},
    "text_elements": {"pass": true/false, "explanation": "..."},
    "color_match": {"pass": true/false, "explanation": "..."},
    "navigation": {"pass": true/false, "explanation": "..."},
    "visual_hierarchy": {"pass": true/false, "explanation": "..."},
    "pass_count": <number of passed criteria>,
    "total_criteria": 5
}
```"""
}


def calculate_cost(model: str, token_usage: dict) -> dict:
    """
    Calculate the estimated cost based on model and token usage.
    
    Args:
        model: Model name/version
        token_usage: Dict with total_prompt_tokens and total_response_tokens
    
    Returns:
        Dict with cost breakdown and total
    """
    if not token_usage:
        return None
    
    # Strip cross-region inference profile prefix (us., eu., apac.) for pricing lookup
    model_for_pricing = model
    if model.lower().startswith(("us.", "eu.", "apac.")):
        model_for_pricing = model.split(".", 1)[1] if "." in model else model
    
    pricing = MODEL_PRICING.get(model_for_pricing, MODEL_PRICING.get(model, DEFAULT_PRICING))
    
    input_tokens = token_usage.get("total_prompt_tokens", 0)
    output_tokens = token_usage.get("total_response_tokens", 0)
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "pricing_per_1m_tokens": pricing,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "call_count": token_usage.get("call_count", 0)
    }


def get_model_info(model_name: str) -> tuple:
    """
    Parse model name and return (family, version).
    
    Examples:
        "gemini" -> ("gemini", "gemini-2.0-flash")
        "gemini-1.5-pro" -> ("gemini", "gemini-1.5-pro")
        "gpt4" -> ("gpt4", "gpt-4o")
    """
    model_name_lower = model_name.lower()
    
    # Check if it's a family name (use default version)
    if model_name_lower in MODEL_FAMILIES:
        return model_name_lower, MODEL_FAMILIES[model_name_lower]["default"]
    
    # Check if it's a specific version
    for family, config in MODEL_FAMILIES.items():
        for version in config["versions"]:
            if model_name_lower == version.lower():
                return family, version
    
    # Try to match by prefix
    if model_name_lower.startswith("gemini"):
        return "gemini", model_name
    elif model_name_lower.startswith("gpt") or model_name_lower.startswith("chatgpt") or model_name_lower.startswith("o1") or model_name_lower.startswith("o3"):
        return "gpt4", model_name
    elif model_name_lower.startswith("claude"):
        return "claude", model_name
    elif model_name_lower.startswith("qwen"):
        return "qwen", model_name
    elif model_name_lower.startswith("deepseek"):
        return "deepseek", model_name
    elif model_name_lower.startswith("grok"):
        return "grok", model_name
    elif model_name_lower.startswith("doubao"):
        return "doubao", model_name
    elif model_name_lower.startswith("moonshot") or model_name_lower.startswith("kimi"):
        return "kimi", model_name
    elif model_name_lower.startswith("mistral") or model_name_lower.startswith("pixtral"):
        return "mistral", model_name
    elif model_name_lower.startswith("meta.llama") or model_name_lower.startswith("llama"):
        return "llama", model_name
    # AWS Bedrock cross-region inference profiles (us.*, eu.*, apac.*)
    elif model_name_lower.startswith(("us.", "eu.", "apac.")):
        # Extract the actual model identifier after the region prefix
        # e.g., "us.meta.llama4-scout-17b-instruct-v1:0" -> check "meta.llama4..."
        model_without_region = model_name_lower.split(".", 1)[1] if "." in model_name_lower else model_name_lower
        if "mistral" in model_without_region or "pixtral" in model_without_region:
            return "mistral", model_name
        elif "meta" in model_without_region or "llama" in model_without_region:
            return "llama", model_name
        # Default to llama for unrecognized bedrock models
        return "llama", model_name
    
    return None, None
