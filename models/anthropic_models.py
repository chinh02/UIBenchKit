"""
Anthropic Models (Claude)
=========================

Bot implementation for Anthropic's Claude models.
"""

import anthropic

from .base import Bot


class Claude(Bot):
    """
    Anthropic Claude model bot.
    
    Supports vision capabilities with image inputs.
    
    Args:
        key_path: Path to API key file or the API key itself
        patience: Number of retry attempts
        model: Claude model version (default: "claude-3-5-sonnet-20241022")
    """
    
    def __init__(self, key_path, patience=3, model="claude-3-5-sonnet-20241022") -> None:
        super().__init__(key_path, patience)
        self.client = anthropic.Anthropic(api_key=self.key)
        self.name = "claude"
        self.model = model
        self.model_version = model
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):
        """
        Send a question to Claude and return the response.
        
        Args:
            question: The text prompt
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            
        Returns:
            The model's text response
        """
        if image_encoding:
            content = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_encoding,
                        },
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ],
            }
        else:
            content = {"role": "user", "content": question}

        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=0.2,
            messages=[content],
        )
        
        # Track token usage from Anthropic response
        if hasattr(message, 'usage') and message.usage:
            prompt_tokens = message.usage.input_tokens or 0
            completion_tokens = message.usage.output_tokens or 0
            self.total_prompt_tokens += prompt_tokens
            self.total_response_tokens += completion_tokens
            self.call_count += 1
            self.token_log.append({
                "call_id": self.call_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "total": prompt_tokens + completion_tokens
            })
        
        response = message.content[0].text
        
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)

        return response
