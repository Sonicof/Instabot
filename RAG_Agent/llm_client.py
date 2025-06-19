from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.llm = OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=0.1,
            top_p=0.9,
            num_predict=500
        )
        self.model = model
    
    def generate_response(self, prompt: str, context: List[str]) -> str:
        """Generate response using Ollama via LangChain"""
        # Construct the full prompt with context
        context_text = "\n\n".join(context)
        full_prompt = f"""Based on the following context, answer the question. If the answer is not in the context, answer normally.

            Context:
            {context_text}

            Question: {prompt}

            Answer:"""
        
        try:
            response = self.llm.invoke(full_prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def evaluate_answer(self, question: str, expected: str, actual: str) -> bool:
        """Use LLM to evaluate if answers are equivalent"""
        eval_prompt = f"""You are an evaluation assistant. Compare two answers to the same question and determine if they convey the same information, even if worded differently.

            Question: {question}

            Expected Answer: {expected}

            Actual Answer: {actual}

            Are these answers equivalent in meaning? Respond with only "YES" or "NO"."""
        
        try:
            response = self.llm.invoke(eval_prompt)
            result = response.strip().upper()
            return "YES" in result
        except:
            return False