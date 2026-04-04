"""
Base Bot Class
==============

Abstract base class for all AI model bots with common functionality
for token tracking, retry logic, and response optimization.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from playwright.sync_api import sync_playwright
import base64
import io
import numpy as np
import re
import time


class Bot(ABC):
    """
    Base class for AI model bots.
    
    Provides:
    - API key management
    - Token usage tracking
    - Retry logic with exponential backoff
    - Response optimization via screenshot comparison
    """
    
    def __init__(self, key_path, patience=3) -> None:
        import os
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                self.key = f.read().replace("\n", "")
        else:
            self.key = key_path
        self.patience = patience
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.call_count = 0
        self.token_log = []
    
    @abstractmethod
    def ask(self, question, image_encoding=None, verbose=False, system_prompt=None):
        """Send a question to the model and return the response.

        Args:
            question: The text prompt (user message content)
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            system_prompt: Optional system message sent before the user message.
                          When provided, question is sent as user text alongside the image,
                          and system_prompt is sent as a separate system role message.
        """
        raise NotImplementedError
    
    def get_token_usage(self):
        """Return token usage statistics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_response_tokens,
            "call_count": self.call_count
        }
    
    def reset_token_usage(self):
        """Reset token counters."""
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.call_count = 0
        self.token_log = []
    
    def print_token_usage(self, label=""):
        """Print token usage summary and return usage dict."""
        usage = self.get_token_usage()
        print(f"\n{'=' * 50}")
        print(f"TOKEN USAGE{' - ' + label if label else ''}")
        print(f"{'=' * 50}")
        print(f"  API Calls:       {usage['call_count']}")
        print(f"  Prompt Tokens:   {usage['total_prompt_tokens']:,}")
        print(f"  Response Tokens: {usage['total_response_tokens']:,}")
        print(f"  Total Tokens:    {usage['total_tokens']:,}")
        print(f"{'=' * 50}")
        return usage
    
    def attempt_ask_with_retries(self, question, image_encoding, verbose):
        """Attempt to ask a question with retry logic."""
        for attempt in range(self.patience):
            try:
                return self.ask(question, image_encoding, verbose)
            except Exception as e:
                if attempt < self.patience - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"All attempts failed for this generation: {e}")
                    return None
    
    def try_ask(self, question, image_encoding=None, verbose=False, num_generations=1, multithread=True):
        """
        Try to ask a question with retry logic and optional multiple generations.
        
        Args:
            question: The question to ask
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug info
            num_generations: Number of response generations to create
            multithread: Whether to use multithreading for multiple generations
            
        Returns:
            The best response (optimized if multiple generations)
        """
        assert num_generations > 0, "num_generations must be greater than 0"
        
        if num_generations == 1:
            for i in range(self.patience):
                try:
                    return self.ask(question, image_encoding, verbose)
                except Exception as e:
                    print(e, "waiting for 5 seconds")
                    time.sleep(5)
            return None
        elif multithread:
            responses = []
            
            with ThreadPoolExecutor() as executor:
                futures = []
                
                for i in range(num_generations):
                    futures.append(executor.submit(
                        self.attempt_ask_with_retries, 
                        question, 
                        image_encoding, 
                        verbose
                    ))
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        responses.append(result)
                    else:
                        print(f"Generation {futures.index(future)} failed after {self.patience} attempts.")
        else:
            responses = []
            for i in range(num_generations):
                for j in range(self.patience):
                    try:
                        responses.append(self.ask(question, image_encoding, verbose))
                        break
                    except Exception as e:
                        print(e, "waiting for 5 seconds")
                        time.sleep(5)
        
        return self.optimize(responses, image_encoding)

    def optimize(self, candidates, img, window_size=(1920, 1080), showimg=False):
        """
        Optimize candidates by comparing screenshots to the original image.
        
        Selects the candidate whose rendered output has the lowest
        mean absolute error (MAE) compared to the original image.
        """
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import get_placeholder, take_screenshot_pw
        
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tailwind CSS Template</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body>
            [CODE]
        </body>
        </html>
        """
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            min_mae = float('inf')
            
            if type(img) == str:
                img = Image.open(io.BytesIO(base64.b64decode(img)))
            img = img.convert("RGB")
            page.set_viewport_size({"width": img.size[0], "height": img.size[1]})
            
            for candidate in candidates:
                code = re.findall(r"```html([^`]+)```", candidate)
                if code:
                    candidate = code[0]
                candidate = html_template.replace("[CODE]", candidate)
                page.set_content(get_placeholder(candidate))
                
                screenshot_data = take_screenshot_pw(page)
                screenshot_img = Image.open(io.BytesIO(screenshot_data)).convert("RGB").resize(img.size)
                
                mae = np.mean(np.abs(np.array(screenshot_img) - np.array(img)))
                
                if mae < min_mae:
                    min_mae = mae
                    best_response = candidate
            
            browser.close()
            return best_response


class FakeBot(Bot):
    """
    A fake bot for testing purposes.
    
    Always returns a placeholder HTML response.
    """
    
    def __init__(self, key_path=None, patience=1) -> None:
        self.name = "FakeBot"
        self.patience = patience
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.call_count = 0
        self.token_log = []
        
    def ask(self, question, image_encoding=None, verbose=False, system_prompt=None):
        print(question)
        return f"```html \nxxxxxxxxxxxxxxxxxxx\n```"
