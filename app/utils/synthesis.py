import json

class QueryScopeClassifier:
    def __init__(self, llm_client):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client

    def is_in_scope(self, question: str, prompt_template: str) -> bool:
        prompt = prompt_template.replace("{{QUESTION}}", question)
        output = self.llm.generate(prompt).strip().lower()
        return output.startswith("yes")



class ResearchSynthesisEngine:
    """
    Synthesizes research answers using a structured JSON contract.
    """

    def __init__(self, llm_client, max_attempts=3):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client
        self.max_attempts = max_attempts


    def build_prompt(self, question, chunks, prompt_template):
        sources_text = ""
        for c in chunks:
            raw_tags = [c.get("first_tag"), c.get("second_tag")]
            tags = [
                str(t) for t in raw_tags
                if t is not None and isinstance(t, str) and t.lower() != "nan"
            ]
            
            sources_text += f"""
SOURCE:
chunk_id: {c['chunk_id']}
tags: {", ".join(tags)}
content:
{c['text']}
                            """
        prompt = prompt_template.replace("{{SOURCES}}", sources_text).replace("{{QUESTION}}", question)
        return prompt



    def synthesize(self, question, chunks, prompt_template):
        
        prompt = self.build_prompt(question, chunks, prompt_template)
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            
            if attempt > 1:
                prompt = prompt + "\n\nREMINDER: Return ONLY valid JSON."
            
            raw_output = self.llm.generate(prompt)

            try:
                # --- Truncation heuristics ---
                if raw_output.count("{") != raw_output.count("}"):
                    raise ValueError(
                        "LLM output appears truncated."
                    )

                # --- JSON parsing ---
                parsed = json.loads(raw_output)

                # --- Structural checks ---
                if "answer" not in parsed or not isinstance(parsed["answer"], list):
                    raise ValueError("Invalid or missing 'answer'.")

                if "limitations" not in parsed or not isinstance(parsed["limitations"], list):
                    raise ValueError("Invalid or missing 'limitations'.")

                for i, s in enumerate(parsed["answer"]):
                    if "text" not in s or "citations" not in s:
                        raise ValueError(
                            f"Malformed answer item at index {i}."
                        )

                # ✅ Success
                return parsed

            except Exception as e:
                last_error = e
                # Optional: log attempt-level failure
                # logger.warning(f"Synthesis attempt {attempt} failed: {e}")

        # ❌ All attempts failed
        raise ValueError(
            "LLM failed to produce valid structured output "
            f"after {self.max_attempts} attempts.\n"
            f"Last error: {last_error}"
        )