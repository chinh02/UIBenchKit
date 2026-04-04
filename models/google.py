"""
Google AI Models (Gemini)
=========================

Bot implementation for Google's Gemini models.
"""

import base64
import io
from PIL import Image
import google.generativeai as genai

from .base import Bot


class Gemini(Bot):
    """
    Google Gemini model bot.
    
    Supports vision capabilities with image inputs.
    
    Args:
        key_path: Path to API key file or the API key itself
        patience: Number of retry attempts
        model: Gemini model version (default: "gemini-2.0-flash")
    """
    
    def __init__(self, key_path, patience=3, model="gemini-2.0-flash") -> None:
        super().__init__(key_path, patience)
        genai.configure(api_key=self.key)
        self.name = "gemini"
        self.model = model
        self.model_version = model
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False, system_prompt=None):
        """
        Send a question to Gemini and return the response.
        
        Args:
            question: The text prompt
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            
        Returns:
            The model's text response
        """
        model = genai.GenerativeModel(self.model)
        config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=10000)

        if verbose:
            print(f"##################{self.file_count}##################")
            print("question:\n", question)

        if image_encoding:
            img = base64.b64decode(image_encoding)
            img = Image.open(io.BytesIO(img))
            response = model.generate_content(
                [question, img], 
                request_options={"timeout": 3000}, 
                generation_config=config
            )
        else:
            response = model.generate_content(
                question, 
                request_options={"timeout": 3000}, 
                generation_config=config
            )
        response.resolve()

        # Track token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            response_tokens = response.usage_metadata.candidates_token_count
            self.total_prompt_tokens += prompt_tokens
            self.total_response_tokens += response_tokens
            self.call_count += 1
            self.token_log.append({
                "call_id": self.call_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total": prompt_tokens + response_tokens
            })

        if verbose:
            print("####################################")
            print("response:\n", response.text)
            self.file_count += 1

        return response.text
