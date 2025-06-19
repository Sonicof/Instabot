from rag_system import RAGSystem
import os

def load_text_from_file(file_path: str) -> str:
    """Load text from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_multiple_texts(text_directory: str, rag_system: RAGSystem):
    """Load multiple text files from a directory"""
    for filename in os.listdir(text_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(text_directory, filename)
            text = load_text_from_file(file_path)
            rag_system.add_text(text, filename)
            print(f"Loaded {filename}")

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Option 1: Load single text file
    if os.path.exists("your_extracted_text.txt"):
        text = load_text_from_file("your_extracted_text.txt")
        rag.add_text(text, "your_document.txt")
    
    # Option 2: Load multiple text files from directory
    text_dir = "extracted_texts"  # Directory containing your text files
    if os.path.exists(text_dir):
        load_multiple_texts(text_dir, rag)
    
    # Option 3: Add text directly in code
    sample_text = """
    Your extracted text content goes here.
    This could be from PDFs, web scraping, or any other source.
    The system will process this into searchable chunks.
    """
    rag.add_text(sample_text, "manual_input.txt")
    
    # Show stats
    stats = rag.get_stats()
    print(f"\nDatabase initialized with {stats['total_chunks']} chunks")
    
    # Interactive query session
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("Type your questions or 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\n❓ Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            print("🔍 Searching...")
            result = rag.query(question)
            
            print(f"\n💡 Answer: {result['answer']}")
            print(f"📚 Sources: {', '.join(result['sources'])}")
            
            # Optionally show similarity scores
            if result['similarity_scores']:
                avg_score = sum(result['similarity_scores']) / len(result['similarity_scores'])
                print(f"🎯 Confidence: {(1-avg_score)*100:.1f}%")

if __name__ == "__main__":
    main()