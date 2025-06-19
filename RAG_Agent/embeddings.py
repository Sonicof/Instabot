from typing import List
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
import numpy as np

class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class OllamaEmbeddingsWrapper:
    """Wrapper for Ollama embeddings - requires nomic-embed-text model"""
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)