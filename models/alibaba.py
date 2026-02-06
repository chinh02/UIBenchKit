"""
Alibaba Cloud Models (Qwen-VL)
==============================

Bot implementation for Alibaba's Qwen-VL models via DashScope API.
"""

from openai import OpenAI

from .openai_models import GPT4


class QwenVL(GPT4):
    """
    Alibaba Qwen-VL model bot.
    
    Uses OpenAI-compatible API via DashScope.
    Inherits from GPT4 for API compatibility.
    
    Args:
        key_path: Path to API key file or the API key itself
        model: Qwen model version (default: "qwen2.5-vl-72b-instruct")
        patience: Number of retry attempts
    """
    
    def __init__(self, key_path, model="qwen2.5-vl-72b-instruct", patience=3) -> None:
        super().__init__(key_path, patience, model)
        self.name = "qwenvl"
        self.model_version = model
        self.client = OpenAI(
            api_key=self.key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.max_tokens = 8192
