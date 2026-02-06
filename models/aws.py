"""
AWS Bedrock Models (Mistral, Llama)
===================================

Bot implementation for AWS Bedrock models including Mistral and Llama.
"""

import base64
import os

from .base import Bot


class BedrockBot(Bot):
    """
    AWS Bedrock bot for Mistral and Llama models.
    
    Requires AWS credentials to be configured via:
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    - Or AWS CLI configuration (~/.aws/credentials)
    - Or IAM role (when running on AWS infrastructure)
    
    Args:
        model_id: The Bedrock model ID (e.g., "mistral.mistral-large-2407-v1:0")
        patience: Number of retry attempts
        region_name: AWS region (default: from AWS_REGION env var or "us-east-1")
    """
    
    def __init__(self, model_id, patience=3, region_name=None) -> None:
        # Don't call super().__init__ with key_path since we use AWS credentials
        self.patience = patience
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.call_count = 0
        self.token_log = []
        
        self.model_id = model_id
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        
        # Determine model family from model_id
        if "mistral" in model_id.lower():
            self.name = "mistral"
        elif "llama" in model_id.lower() or "meta" in model_id.lower():
            self.name = "llama"
        else:
            self.name = "bedrock"
        
        self.model_version = model_id
        
        # Initialize Bedrock client
        try:
            import boto3
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {e}")
    
    def ask(self, question, image_encoding=None, verbose=False):
        """
        Send a request to AWS Bedrock.
        
        Supports both Mistral and Llama models with vision capabilities.
        Uses the Converse API for better compatibility across models.
        
        Args:
            question: The text prompt
            image_encoding: Base64 encoded image (optional)
            verbose: Whether to print debug information
            
        Returns:
            The model's text response
        """
        try:
            # Build content blocks for Converse API
            content = []
            
            if image_encoding:
                content.append({
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": base64.b64decode(image_encoding)
                        }
                    }
                })
            
            content.append({"text": question})
            
            messages = [{
                "role": "user",
                "content": content
            }]
            
            inference_config = {
                "maxTokens": 8192,
                "temperature": 0.2
            }
            
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
            
            # Extract response text
            response_text = ""
            if "output" in response and "message" in response["output"]:
                output_content = response["output"]["message"].get("content", [])
                for block in output_content:
                    if "text" in block:
                        response_text += block["text"]
            
            # Extract and track token usage
            usage = response.get("usage", {})
            prompt_tokens = usage.get("inputTokens", 0)
            completion_tokens = usage.get("outputTokens", 0)
            
            self.total_prompt_tokens += prompt_tokens
            self.total_response_tokens += completion_tokens
            self.call_count += 1
            self.token_log.append({
                "call_id": self.call_count,
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "total": prompt_tokens + completion_tokens
            })
            
            if verbose:
                print("####################################")
                print("question:\n", question)
                print("####################################")
                print("response:\n", response_text)
            
            return response_text
            
        except Exception as e:
            raise Exception(f"Bedrock API error: {e}")
