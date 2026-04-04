"""
OpenAI Models (GPT-4, GPT-4o, etc.)
===================================

Bot implementation for OpenAI's GPT models.
"""

from openai import OpenAI

from .base import Bot


class GPT4(Bot):
    """
    OpenAI GPT-4/GPT-4o model bot.
    
    Supports vision capabilities with image inputs.
    
    Args:
        key_path: Path to API key file or the API key itself
        patience: Number of retry attempts
        model: OpenAI model version (default: "gpt-4o")
        base_url: Custom base URL for API (optional, for compatible APIs)
    """
    
    def __init__(self, key_path, patience=3, model="gpt-4o", base_url=None) -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key, base_url=base_url)
        self.name = "gpt4"
        self.model = model
        self.model_version = model
        self.max_tokens = 10000
        
    def ask(self, question, image_encoding=None, verbose=False, system_prompt=None):
        """
        Send a question to OpenAI and return the response.

        Args:
            question: The text prompt
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            system_prompt: Optional system message (sent as a separate system role)

        Returns:
            The model's text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_encoding:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_encoding}",
                    },
                },
                {"type": "text", "text": question},
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
            seed=42,
        )
        
        # Track token usage from OpenAI response
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            self.total_prompt_tokens += prompt_tokens
            self.total_response_tokens += completion_tokens
            self.call_count += 1
            self.token_log.append({
                "call_id": self.call_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "total": prompt_tokens + completion_tokens
            })
        
        response_text = response.choices[0].message.content
        
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response_text)
            print("seed used: 42")
        
        return response_text


class GPT5(Bot):
    """
    OpenAI GPT-5 model bot.
    
    GPT-5 models have enhanced capabilities including:
    - Higher max output tokens (32K)
    - Improved reasoning and code generation
    - Enhanced vision capabilities
    
    Args:
        key_path: Path to API key file or the API key itself
        patience: Number of retry attempts
        model: OpenAI model version (default: "gpt-5")
        base_url: Custom base URL for API (optional, for compatible APIs)
    """
    
    def __init__(self, key_path, patience=3, model="gpt-5", base_url=None) -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key, base_url=base_url)
        self.name = "gpt5"
        self.model = model
        self.model_version = model
        self.max_completion_tokens = 32000  # GPT-5 supports higher output tokens
        
    def ask(self, question, image_encoding=None, verbose=False, system_prompt=None):
        """
        Send a question to OpenAI GPT-5 and return the response.

        Args:
            question: The text prompt
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            system_prompt: Optional system message (sent as a separate system role)

        Returns:
            The model's text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_encoding:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_encoding}",
                    },
                },
                {"type": "text", "text": question},
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens
        )
        
        # Track token usage from OpenAI response
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            self.total_prompt_tokens += prompt_tokens
            self.total_response_tokens += completion_tokens
            self.call_count += 1
            self.token_log.append({
                "call_id": self.call_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "total": prompt_tokens + completion_tokens
            })
        
        response_text = response.choices[0].message.content
        
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response_text)
            print("seed used: 42")
        
        return response_text
