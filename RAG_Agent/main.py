from rag_system import RAGSystem

def read_file():
    with open("D:/pyenv312/RAG/Codes/extracted_texts/dr-arunkumar.txt", "r", encoding="utf-8") as file:
        return file.read()

text = read_file()
print(text)

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    file=read_file()
    # Example: Add your extracted text
    sample_text = f"""
    {file}
    """
    
    # Add text to the system
    rag.add_text(sample_text, "sample_document.txt")
    
    # Print stats
    stats = rag.get_stats()
    print(f"Database contains {stats['total_chunks']} chunks")
    
    # Interactive query loop
    print("\nRAG System Ready! Type 'quit' to exit.")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        
        if question:
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}")

if __name__ == "__main__":
    main()