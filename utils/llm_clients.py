class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError



class OpenAIClient(BaseLLMClient):
    def __init__(self, model_name="gpt-4o-mini", max_tokens=400, temperature=0.2):
        
        import os
        from openai import OpenAI
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content



class HFClient(BaseLLMClient):
    """
    HuggingFace-based LLM client.

    Note:
    - CPU path prioritizes simplicity and compatibility.
    - GPU path optionally supports quantization for large models.
    """

    def __init__(
        self,
        model_name="google/flan-t5-base",
        max_tokens=400,
        temperature=0.2,
        device=None,
        load_in_8bit=False,
        load_in_4bit=False,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Auto device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {}

        if device == "cuda":
            if load_in_4bit:
                model_kwargs.update(dict(load_in_4bit=True, device_map="auto"))
            elif load_in_8bit:
                model_kwargs.update(dict(load_in_8bit=True, device_map="auto"))
            else:
                model_kwargs.update(dict(torch_dtype=torch.float16))
        else:
            model_kwargs.update(dict(torch_dtype=torch.float32))

        # Auto-detect model type
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **model_kwargs
            )

        if device == "cpu":
            self.model.to(device)

        self.model.eval()


    def generate(self, prompt: str) -> str:

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)