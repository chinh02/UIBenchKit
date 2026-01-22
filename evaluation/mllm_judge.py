#!/usr/bin/env python3
"""
MLLM Judge Evaluator
====================

Uses Multimodal LLMs (GPT-4V, Gemini, Claude) as judges to evaluate
generated webpages against reference designs.

This evaluator provides qualitative feedback and structured scores
by having an MLLM compare screenshots of generated vs reference pages.
"""

import os
import re
import json
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .base import BaseEvaluator, EvaluationResult


class JudgeMode(Enum):
    """Evaluation modes for the MLLM Judge."""
    SINGLE_SCORE = "single_score"        # Rate a single generated webpage
    PAIRWISE = "pairwise"                # Compare two models
    CRITERIA_CHECK = "criteria_check"    # Yes/No on specific criteria


@dataclass
class JudgeConfig:
    """Configuration for MLLM Judge."""
    model_family: str = "gemini"
    model_version: str = "gemini-2.0-flash"
    mode: JudgeMode = JudgeMode.SINGLE_SCORE
    temperature: float = 0.1
    max_tokens: int = 2000
    custom_prompt: Optional[str] = None
    retry_count: int = 2


# Default prompts for different modes
DEFAULT_PROMPTS = {
    JudgeMode.SINGLE_SCORE: """You are an expert web designer and UI/UX evaluator. Compare the two images below:

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
    "overall_score": <weighted average 0-10>,
    "strengths": ["list of things done well"],
    "weaknesses": ["list of issues or missing elements"],
    "summary": "Brief 1-2 sentence overall assessment"
}
```""",

    JudgeMode.PAIRWISE: """You are an expert web designer and UI/UX evaluator. You will see THREE images:

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

    JudgeMode.CRITERIA_CHECK: """You are an expert web designer. Compare the generated webpage (IMAGE 2) against the reference (IMAGE 1).

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


class MLLMJudgeEvaluator(BaseEvaluator):
    """
    Multimodal LLM Judge for evaluating generated webpages.
    
    Uses vision-capable LLMs to provide qualitative and quantitative
    evaluation of generated webpages compared to reference designs.
    """
    
    metric_name = "mllm_judge"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Parse config
        cfg = config or {}
        self.judge_config = JudgeConfig(
            model_family=cfg.get('model_family', 'gemini'),
            model_version=cfg.get('model_version', 'gemini-2.0-flash'),
            mode=JudgeMode(cfg.get('mode', 'single_score')),
            temperature=cfg.get('temperature', 0.1),
            max_tokens=cfg.get('max_tokens', 2000),
            custom_prompt=cfg.get('custom_prompt'),
            retry_count=cfg.get('retry_count', 2)
        )
        
        # API clients
        self._gemini_client = None
        self._openai_client = None
        self._anthropic_client = None
        
        # Token tracking
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "call_count": 0
        }
    
    def initialize(self) -> None:
        """Initialize the appropriate API client based on model family."""
        family = self.judge_config.model_family
        
        if family == "gemini":
            self._init_gemini()
        elif family == "gpt4":
            self._init_openai()
        elif family == "claude":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported model family: {family}")
        
        self._initialized = True
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
            
            genai.configure(api_key=api_key)
            self._gemini_client = genai
        except ImportError:
            raise ImportError("Google Generative AI not installed. Run: pip install google-generativeai")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            
            self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._gemini_client = None
        self._openai_client = None
        self._anthropic_client = None
        self._initialized = False
    
    def _encode_image(self, image_path: str) -> tuple:
        """
        Encode an image to base64.
        
        Returns:
            Tuple of (base64_data, mime_type)
        """
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/png')
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return image_data, mime_type
    
    def _get_prompt(self) -> str:
        """Get the appropriate prompt for the current mode."""
        if self.judge_config.custom_prompt:
            return self.judge_config.custom_prompt
        return DEFAULT_PROMPTS[self.judge_config.mode]
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {"raw_response": response_text, "parse_error": "No JSON found"}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"raw_response": response_text, "parse_error": str(e)}
    
    def _call_gemini(
        self,
        prompt: str,
        images: List[str]
    ) -> Dict[str, Any]:
        """Call Gemini API with images."""
        from PIL import Image
        
        model = self._gemini_client.GenerativeModel(self.judge_config.model_version)
        
        # Prepare content with images
        content = []
        for i, img_path in enumerate(images):
            img = Image.open(img_path)
            content.append(img)
        content.append(prompt)
        
        response = model.generate_content(
            content,
            generation_config={
                "temperature": self.judge_config.temperature,
                "max_output_tokens": self.judge_config.max_tokens,
            }
        )
        
        # Track tokens
        if hasattr(response, 'usage_metadata'):
            self.token_usage["total_prompt_tokens"] += response.usage_metadata.prompt_token_count
            self.token_usage["total_response_tokens"] += response.usage_metadata.candidates_token_count
        self.token_usage["call_count"] += 1
        
        return self._parse_json_response(response.text)
    
    def _call_openai(
        self,
        prompt: str,
        images: List[str]
    ) -> Dict[str, Any]:
        """Call OpenAI API with images."""
        # Build message content with images
        content = []
        
        for i, img_path in enumerate(images):
            img_data, mime_type = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_data}",
                    "detail": "high"
                }
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = self._openai_client.chat.completions.create(
            model=self.judge_config.model_version,
            messages=[{"role": "user", "content": content}],
            temperature=self.judge_config.temperature,
            max_tokens=self.judge_config.max_tokens
        )
        
        # Track tokens
        if response.usage:
            self.token_usage["total_prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["total_response_tokens"] += response.usage.completion_tokens
        self.token_usage["call_count"] += 1
        
        return self._parse_json_response(response.choices[0].message.content)
    
    def _call_anthropic(
        self,
        prompt: str,
        images: List[str]
    ) -> Dict[str, Any]:
        """Call Anthropic API with images."""
        # Build message content with images
        content = []
        
        for i, img_path in enumerate(images):
            img_data, mime_type = self._encode_image(img_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": img_data
                }
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = self._anthropic_client.messages.create(
            model=self.judge_config.model_version,
            max_tokens=self.judge_config.max_tokens,
            messages=[{"role": "user", "content": content}]
        )
        
        # Track tokens
        self.token_usage["total_prompt_tokens"] += response.usage.input_tokens
        self.token_usage["total_response_tokens"] += response.usage.output_tokens
        self.token_usage["call_count"] += 1
        
        return self._parse_json_response(response.content[0].text)
    
    def _call_llm(
        self,
        prompt: str,
        images: List[str]
    ) -> Dict[str, Any]:
        """Route to appropriate LLM API."""
        family = self.judge_config.model_family
        
        for attempt in range(self.judge_config.retry_count + 1):
            try:
                if family == "gemini":
                    return self._call_gemini(prompt, images)
                elif family == "gpt4":
                    return self._call_openai(prompt, images)
                elif family == "claude":
                    return self._call_anthropic(prompt, images)
                else:
                    raise ValueError(f"Unsupported model family: {family}")
            except Exception as e:
                if attempt == self.judge_config.retry_count:
                    raise
                # Brief delay before retry
                import time
                time.sleep(1)
    
    def evaluate_sample(
        self,
        generated_html_path: str,
        reference_image_path: str,
        generated_screenshot_path: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single sample using MLLM as judge.
        
        Args:
            generated_html_path: Path to generated HTML (used for sample ID)
            reference_image_path: Path to reference screenshot
            generated_screenshot_path: Path to generated screenshot
        
        Returns:
            EvaluationResult with MLLM judge scores
        """
        sample_id = os.path.splitext(os.path.basename(generated_html_path))[0]
        
        if not self._initialized:
            self.initialize()
        
        # Find generated screenshot if not provided
        if not generated_screenshot_path:
            html_dir = os.path.dirname(generated_html_path)
            possible_paths = [
                os.path.join(html_dir, f"{sample_id}.png"),
                os.path.join(html_dir, f"{sample_id}_p.png"),  # Design2Code format
                generated_html_path.replace('.html', '.png')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    generated_screenshot_path = path
                    break
        
        if not generated_screenshot_path or not os.path.exists(generated_screenshot_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Generated screenshot not found"
            )
        
        if not os.path.exists(reference_image_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Reference image not found"
            )
        
        try:
            prompt = self._get_prompt()
            images = [reference_image_path, generated_screenshot_path]
            
            result = self._call_llm(prompt, images)
            
            # Extract scores based on mode
            scores = {}
            if self.judge_config.mode == JudgeMode.SINGLE_SCORE:
                for key in ['layout_accuracy', 'visual_fidelity', 'content_completeness', 
                           'responsiveness_polish', 'overall_score']:
                    if key in result:
                        scores[key] = float(result[key])
                # Normalize to 0-1 scale
                scores = {k: round(v / 10.0, 4) for k, v in scores.items()}
                
            elif self.judge_config.mode == JudgeMode.CRITERIA_CHECK:
                if 'pass_count' in result:
                    scores['pass_rate'] = result['pass_count'] / result.get('total_criteria', 5)
                
            elif self.judge_config.mode == JudgeMode.PAIRWISE:
                if 'model_a_score' in result:
                    scores['model_a_score'] = float(result['model_a_score']) / 10.0
                if 'model_b_score' in result:
                    scores['model_b_score'] = float(result['model_b_score']) / 10.0
            
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                scores=scores,
                metadata={
                    "judge_model": f"{self.judge_config.model_family}/{self.judge_config.model_version}",
                    "mode": self.judge_config.mode.value,
                    "full_response": result
                }
            )
            
        except Exception as e:
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error=str(e)
            )
    
    def evaluate_pairwise(
        self,
        reference_image_path: str,
        model_a_screenshot: str,
        model_b_screenshot: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Compare two model outputs against a reference.
        
        Args:
            reference_image_path: Path to reference screenshot
            model_a_screenshot: Path to Model A's screenshot
            model_b_screenshot: Path to Model B's screenshot
            model_a_name: Display name for Model A
            model_b_name: Display name for Model B
        
        Returns:
            Dictionary with comparison results
        """
        if not self._initialized:
            self.initialize()
        
        # Temporarily switch to pairwise mode
        original_mode = self.judge_config.mode
        self.judge_config.mode = JudgeMode.PAIRWISE
        
        try:
            prompt = self._get_prompt()
            images = [reference_image_path, model_a_screenshot, model_b_screenshot]
            
            result = self._call_llm(prompt, images)
            
            # Add model names for clarity
            result["model_a_name"] = model_a_name
            result["model_b_name"] = model_b_name
            
            return {
                "success": True,
                "comparison": result,
                "judge_model": f"{self.judge_config.model_family}/{self.judge_config.model_version}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.judge_config.mode = original_mode
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return self.token_usage.copy()
    
    def reset_token_usage(self):
        """Reset token usage counters."""
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "call_count": 0
        }
