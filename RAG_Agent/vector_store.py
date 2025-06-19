import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from langchain.schema import Document

class VectorStore:
    def __init__(self, db_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to the vector store with incremental updates"""
        ids = []
        texts = []
        metadatas = []
        embeddings_to_add = []
        
        for doc, embedding in zip(documents, embeddings):
            chunk_id = doc.metadata["chunk_id"]
            
            # Check if chunk already exists
            try:
                existing = self.collection.get(ids=[chunk_id])
                if existing["ids"]:
                    print(f"Chunk {chunk_id} already exists, skipping...")
                    continue
            except:
                pass
            
            ids.append(chunk_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            embeddings_to_add.append(embedding)
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings_to_add
            )
            print(f"Added {len(ids)} new chunks to the database")
        else:
            print("No new chunks to add")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {"total_chunks": count}