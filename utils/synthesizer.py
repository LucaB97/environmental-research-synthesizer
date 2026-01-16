import os
import json


class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, model="gpt-4o-mini", max_tokens=400, temperature=0.2):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content



class ResearchSynthesisEngine:
    """
    Synthesizes research answers using a structured JSON contract.

    Expected output schema:
    {
        "in_scope": bool,
        "answer": list,
        "limitations": list
    }
    """
    def __init__(self, llm_client, prompt_template):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client
        self.prompt_template = prompt_template

    def build_prompt(self, question, chunks):
        sources_text = ""
        for c in chunks:
            sources_text += f"""
SOURCE:
chunk_id: {c['chunk_id']}
paper_id: {c['paper_id']}
authors: {c['authors']}
year: {c['year']}
title: {c['title']}
content:
{c['text']}
"""
        prompt = self.prompt_template.replace("{{SOURCES}}", sources_text).replace("{{QUESTION}}", question)
        return prompt



    def synthesize(self, question, chunks):
        prompt = self.build_prompt(question, chunks)
        raw_output = self.llm.generate(prompt)

        # Truncation heuristics
        if raw_output.count("{") != raw_output.count("}"):
            raise ValueError(
                "LLM output appears truncated.\n"
                f"Raw output:\n{raw_output}"
            )
        
        # JSON parsing
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise ValueError(
                "LLM returned malformed JSON.\n"
                f"Raw output:\n{raw_output}\n"
                f"JSON error: {e}"
            )

        # Structural checks (domain-level)
        if "reason" not in parsed:
            raise ValueError(f"Missing 'reason' field.\nOutput: {parsed}")

        if parsed["reason"] == "out_of_scope":
            parsed["answer"] = []

        if "answer" not in parsed or not isinstance(parsed["answer"], list):
            raise ValueError(f"Invalid or missing 'answer'.\nOutput: {parsed}")

        if "limitations" not in parsed or not isinstance(parsed["limitations"], list):
            raise ValueError(f"Invalid or missing 'limitations'.\nOutput: {parsed}")

        # Sentence-level checks
        for i, s in enumerate(parsed["answer"]):
            if "text" not in s or "citations" not in s:
                raise ValueError(
                    f"Malformed answer item at index {i}.\nItem: {s}"
                )

        return parsed
