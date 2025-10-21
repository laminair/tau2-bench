import httpx
import torch
import litellm
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from litellm import CustomLLM, completion, get_llm_provider
from litellm.utils import ModelResponse, Choices, Message, Usage
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from typing import Optional, Union, List, Dict, Any


class TransformersHandler(CustomLLM):
    """Handler for local transformers models"""

    def __init__(self):
        super().__init__()
        self.models = {}
        self.tokenizers = {}

    def load_model(self, model_name: str, device_map: str = "auto", model_tokenizer_kwargs: dict = {}, **kwargs):
        """Load a model and tokenizer if not already loaded"""

        if model_name not in self.models:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name, **model_tokenizer_kwargs
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # or "fp4"
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )

            # We add cost information
            litellm.register_model({
                model_name: {
                    "max_tokens": 8192,
                    "max_input_tokens": 8192,
                    "max_output_tokens": 4096,
                    "input_cost_per_token": 0,  # Cost per input token in USD
                    "output_cost_per_token": 0,  # Cost per output token in USD
                    "litellm_provider": "hf_local",
                    "mode": "chat"  # or "completion" for completion models
                }
            })

            print(f"Model {model_name} loaded successfully!")

        return self.models[model_name], self.tokenizers[model_name]

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: Optional[str] = None,
        custom_prompt_dict: dict = None,
        optional_params: dict = None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        custom_llm_provider: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        ** kwargs
        ) -> ModelResponse:
        """
        Custom completion handler for HuggingFace transformers models.
        This function is called by LiteLLM when using 'hf_local/' prefix.

        Args:
            model: The full model path (e.g., "hf_local/Qwen/Qwen2-14B-AWQ")
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream responses (not implemented)
            **kwargs: Additional arguments (device, trust_remote_code, etc.)
        """
        if stream:
            raise NotImplementedError("Streaming not yet implemented")

        # Extract the actual model name by removing the 'huggingface/' prefix
        if model.startswith('hf_local/'):
            actual_model_name = model.replace('hf_local/', '', 1)
        else:
            actual_model_name = model

        # Extract device from kwargs or use auto
        device = kwargs.pop('device', 'auto')

        # Load model and tokenizer, if not present. Fetch from cache otherwise.
        if actual_model_name not in self.models.keys():
            hf_model, tokenizer = self.load_model(actual_model_name, device, **kwargs)
        else:
            hf_model, tokenizer = self.models[actual_model_name], self.tokenizers[actual_model_name]

        # Format messages into prompt
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant:"

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate token counts
        prompt_tokens = input_length
        completion_tokens = len(generated_tokens)

        # Create LiteLLM-compatible response
        message = Message(
            content=response_text,
            role="assistant"
        )

        choice = Choices(
            finish_reason="stop",
            index=0,
            message=message
        )

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

        response = ModelResponse(
            id=f"chatcmpl-hf-{int(time.time())}",
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion",
            usage=usage
        )

        return response


def setup_huggingface_local_provider():
    """
    Register HuggingFace transformers as a custom provider with LiteLLM.
    Call this once at the start of your application.
    """
    local_llm_provider = TransformersHandler()
    litellm.custom_provider_map = [
        {"provider": "hf_local", "custom_handler": local_llm_provider}
    ]


# Example usage
if __name__ == "__main__":
    # Setup the HuggingFace provider (do this once)
    setup_huggingface_local_provider()

    # Now you can use litellm.completion() with HuggingFace models!
    # Use the format: huggingface/owner/model-name
    response = litellm.completion(
        model="hf_local/Qwen/Qwen3-14B",
        messages=[
            {"role": "user", "content": "Who are you?"}
        ],
        max_tokens=100,
        temperature=0.7
    )

    print(response.choices[0].message.content)
    print(f"\nTokens used: {response.usage.total_tokens}")
