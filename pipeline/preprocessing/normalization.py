# import re


# TERM_MAP = {
#     "renewables": "renewable energy",
#     "renewable energies": "renewable energy",
    
#     "wind farms": "wind energy",
# }


# def normalize_query_terms(q: str) -> str:
#     for k, v in TERM_MAP.items():
#         q = re.sub(rf"\b{k}\b", v, q)
#     return q


# def normalize_query(q: str) -> str:
#     q = q.lower()
#     q = q.strip()
#     q = re.sub(r'\s+', ' ', q)
#     return q

import re
import spacy

class Normalizer:
    def __init__(self, use_lemmatization=True):
        self.use_lemmatization = use_lemmatization
        self.nlp = None

        if self.use_lemmatization:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


    def normalize(self, text: str) -> str:
        # 1. lowercase
        text = text.lower()

        # 2. punctuation cleanup (keep words/numbers/spaces only)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3. optional lemmatization
        if self.use_lemmatization:
            doc = self.nlp(text)
            text = " ".join([t.lemma_ for t in doc if not t.is_space])

        return text