from typing import List, Dict, Any
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import OllamaClient
from embeddings import LocalEmbeddings, OllamaEmbeddingsWrapper

class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            mode="spacy"
        )
        
        # Initialize embeddings based on config
        if self.config.EMBEDDING_TYPE == "local":
            self.embeddings = LocalEmbeddings(self.config.LOCAL_EMBEDDING_MODEL)
        elif self.config.EMBEDDING_TYPE == "ollama":
            self.embeddings = OllamaEmbeddingsWrapper(
                self.config.OLLAMA_EMBEDDING_MODEL,
                self.config.OLLAMA_BASE_URL
            )
        else:  # Default to local
            self.embeddings = LocalEmbeddings(self.config.LOCAL_EMBEDDING_MODEL)
        
        self.vector_store = VectorStore(
            self.config.CHROMA_DB_PATH,
            self.config.COLLECTION_NAME
        )
        self.llm_client = OllamaClient(
            self.config.OLLAMA_BASE_URL,
            self.config.OLLAMA_LLM_MODEL
        )
    
    def add_text(self, text: str, source: str, page: int = 0):
        """Add text to the RAG system"""
        print(f"Processing text from {source}...")
        
        # Create chunks
        documents = self.document_processor.create_chunks(text, source, page)
        print(f"Created {len(documents)} chunks")
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        try:
            embeddings = self.embeddings.embed_documents(texts)
            print("Generated embeddings")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            print("Falling back to local embeddings...")
            # Fallback to local embeddings
            local_embeddings = LocalEmbeddings()
            embeddings = local_embeddings.embed_documents(texts)
        
        # Add to vector store
        self.vector_store.add_documents(documents, embeddings)
        print("Added documents to vector store")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        print(f"Processing query: {question}")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(question)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            # Fallback to local embeddings
            local_embeddings = LocalEmbeddings()
            query_embedding = local_embeddings.embed_query(question)
        
        # Retrieve relevant chunks
        results = self.vector_store.similarity_search(
            query_embedding, 
            k=self.config.TOP_K_CHUNKS
        )
        
        if not results:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "sources": [],
                "context_chunks": []
            }
        
        # Extract context
        context_chunks = [result["content"] for result in results]
        
        # Generate response
        answer = self.llm_client.generate_response(question, context_chunks)
        
        # Extract sources
        sources = list(set([result["metadata"]["source"] for result in results]))
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_chunks": context_chunks,
            "similarity_scores": [result["distance"] for result in results]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.vector_store.get_collection_stats()