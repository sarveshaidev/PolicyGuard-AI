from openai import OpenAI
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, JUDGE_MODEL
import json

class LLMJudge:
    def __init__(self):
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    def evaluate(self, question: str, context: str, answer: str) -> dict:
        """Scores the answer on Faithfulness and Relevance (1-5)."""
        prompt = f"""
        You are an expert evaluator. Rate the AI's answer based on the context.
        Question: {question}
        Context: {context}
        Answer: {answer}

        Provide a JSON response with two scores (1-5):
        - faithfulness: Is the answer grounded in the context?
        - relevance: Does the answer directly address the question?
        """
        
        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)