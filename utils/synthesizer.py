import os

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

class Synthesizer:
    def __init__(self, llm_client):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client

    def build_prompt(self, question, chunks):
        sources_text = ""
        for c in chunks:
            sources_text += f"""
    --- SOURCE START ---
    Citation: [{c['authors']}, {c['year']}]
    Title: {c['title']}
    Paper ID: {c['paper_id']}
    Content:
    {c['text']}
    --- SOURCE END ---
    """

        return f"""
    You are an expert research assistant specialized in environmental and social impact analysis.
    You synthesize evidence from peer-reviewed academic sources only.

    STRICT RULES:
    - Use ONLY the information contained in the provided sources.
    - Do NOT use prior knowledge, assumptions, or general world knowledge.
    - Do NOT speculate or infer beyond what is explicitly supported.
    - Every factual statement MUST be supported by at least one cited source.
    - If the sources do NOT provide evidence relevant to the question, you MUST say so explicitly.
    - Do NOT fabricate citations or force citations where no evidence exists.

    {sources_text}

    TASK:
    Using ONLY the sources above, answer the following question:

    "{question}"

    OUTPUT FORMAT (follow exactly):

    Answer:
    - 3–5 bullet points synthesizing the key findings that directly address the question.
    - If sources report conflicting findings, explicitly describe the disagreement and cite the respective sources.
    - If NO relevant evidence exists, include a single bullet stating that the sources do not address the question.

    Citations:
    - List all citations used in the Answer section in the format [authors, year].
    - If no relevant evidence exists, write exactly:
    "No relevant citations available."

    Limitations:
    - Briefly describe gaps, uncertainties, or limitations in the available evidence.
    - If the sources do not address the question at all, state this clearly.
    """

    def synthesize(self, question, chunks):
        prompt = self.build_prompt(question, chunks)
        return self.llm.generate(prompt)
