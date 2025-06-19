import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database settings
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "document_chunks"
    
    # Embedding settings - Using Ollama embeddings
    EMBEDDING_TYPE = "ollama"  # Using Ollama for embeddings
    
    # Local embeddings (sentence-transformers) - fallback only
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Ollama embeddings (primary choice)
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # OpenAI embeddings (requires API key)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # LLM settings - Using Llama 3.2 3B
    OLLAMA_LLM_MODEL = "llama3.2:3b"  # Using Llama 3.2 3B model
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHUNKING_MODE = "spacy"
    
    # Retrieval settings
    TOP_K_CHUNKS = 5