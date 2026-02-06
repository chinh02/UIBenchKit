"""
DCGen Models Package
====================

This package provides AI model bot implementations for various vendors:

- Google (Gemini)
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude)
- Alibaba Cloud (Qwen-VL)
- AWS Bedrock (Mistral, Llama)

All bots inherit from the base Bot class and provide a unified interface
for sending prompts and receiving responses, with optional image inputs.

Usage:
    from models import GPT4, Gemini, Claude, QwenVL, BedrockBot

    # Initialize a bot
    bot = GPT4("your-api-key", model="gpt-4o")
    
    # Ask a question
    response = bot.ask("What is this image?", image_encoding=base64_image)
    
    # Get token usage
    usage = bot.get_token_usage()
"""

from .base import Bot, FakeBot
from .google import Gemini
from .openai_models import GPT4
from .anthropic_models import Claude
from .alibaba import QwenVL
from .aws import BedrockBot

__all__ = [
    # Base classes
    "Bot",
    "FakeBot",
    # Vendor implementations
    "Gemini",
    "GPT4", 
    "Claude",
    "QwenVL",
    "BedrockBot",
]
