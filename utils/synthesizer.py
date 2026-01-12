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

    # def synthesize(self, question, chunks):
    #     prompt = self.build_prompt(question, chunks)
    #     raw_output = self.llm.generate(prompt)

    #     try:
    #         parsed = json.loads(raw_output)

    #         # Structural checks (domain-level)
    #         assert "in_scope" in parsed
    #         assert isinstance(parsed["in_scope"], bool)

    #         if not parsed["in_scope"]:
    #             parsed["answer"] = []
                
    #         assert "answer" in parsed
    #         assert "limitations" in parsed

    #         return parsed  # ← plain dict

    #     except Exception as e:
    #         raise ValueError(
    #             f"Invalid LLM output.\nRaw output:\n{raw_output}\nError: {e}"
    #         )

    def synthesize(self, question, chunks):
        prompt = self.build_prompt(question, chunks)
        raw_output = self.llm.generate(prompt)

        try:
            parsed = json.loads(raw_output)

            # Structural checks (domain-level)
            assert "reason" in parsed

            if parsed["reason"] == "out_of_scope":
                parsed["answer"] = []
                
            assert "answer" in parsed
            assert "limitations" in parsed

            return parsed  # ← plain dict

        except Exception as e:
            raise ValueError(
                f"Invalid LLM output.\nRaw output:\n{raw_output}\nError: {e}"
            )
