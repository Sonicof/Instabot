"""
Example usage scenarios for the RAG system
"""
import pytest
from rag_system import RAGSystem

def example_multiple_documents():
    """Example: Adding multiple documents"""
    rag = RAGSystem()
    
    # Add multiple texts
    texts_and_sources = [
        ("Monopoly rules text here...", "monopoly_rules.pdf"),
        ("CodeNames rules text here...", "codenames_rules.pdf"),
        ("Chess rules text here...", "chess_rules.pdf")
    ]
    
    for text, source in texts_and_sources:
        rag.add_text(text, source)
    
    # Query across all documents
    result = rag.query("How do you win in board games?")
    print(result["answer"])

def example_incremental_updates():
    """Example: Adding new content incrementally"""
    rag = RAGSystem()
    
    # Add initial content
    rag.add_text("Initial game rules...", "game_v1.txt")
    
    # Add updated content (won't duplicate)
    rag.add_text("Updated game rules...", "game_v2.txt")
    
    # Add more content to existing document
    rag.add_text("Additional rules...", "game_v1.txt", page=2)

if __name__ == "__main__":
    # Run examples
    example_multiple_documents()
    example_incremental_updates()