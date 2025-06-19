# Retrieval-Augmented Generation (RAG) System

A modular Python framework for building advanced document and website Q&A chatbots using Retrieval-Augmented Generation. This system can ingest, chunk, embed, and search large volumes of text—including OCR-extracted website data—enabling instant, context-aware answers to user queries.

---

## Features

- **Flexible Document Ingestion:** Add text from files, web scraping, or OCR (e.g., Google Vision API).
- **ML-Based Chunking:** Uses spaCy for intelligent sentence segmentation and chunking, optimized for noisy OCR/web data.
- **Embeddings:** Supports both local (all-MiniLM-L6-v2) and Ollama-based embeddings.
- **Vector Search:** Stores and retrieves document chunks using ChromaDB for fast similarity search.
- **LLM Integration:** Uses Ollama (e.g., Llama 3.2 3B) for generating answers from retrieved context.
- **Interactive Q&A:** CLI for asking questions about your ingested data.
- **Extensible:** Easily adapt to new data sources, chunking strategies, or LLMs.

---

## Installation

1. **Clone the repository** and set up a Python 3.10+ environment (recommended: use `venv`).

2. **Install dependencies:**
   ```bash
   pip install chromadb langchain langchain_community langchain_ollama sentence-transformers ollama pytest python-dotenv spacy
   python -m spacy download en_core_web_sm
   ```

3. **(Optional) Set up Ollama** for LLM and embedding support:
   - [Install Ollama](https://ollama.com/)
   - Start the Ollama server:  
     `ollama serve`
   - Pull required models:  
     `ollama pull llama3.2:3b`  
     `ollama pull nomic-embed-text`

4. **(Optional) Set up Google Vision API** for OCR if extracting text from images.

---

## Usage

### **Quick Start**

```bash
python main.py
```
- This will load a sample text file, chunk and embed it, and start an interactive Q&A loop.

### **Adding Your Own Data**

- Place your extracted text files in the appropriate directory.
- Edit `main.py` or use `text_loader.py` to load your data.

### **Chunking Strategy**

- By default, the system uses spaCy-based sentence segmentation for robust chunking of OCR/web data.
- You can adjust chunking mode and parameters in `rag_system.py` or `document_processor.py`.

---

## Project Structure

```
new_try/
  main.py                # Entry point for demo/interactive Q&A
  rag_system.py          # Main RAG system logic
  document_processor.py  # Chunking (ML-based and default)
  embeddings.py          # Embedding logic (Ollama/local)
  vector_store.py        # ChromaDB vector storage
  llm_client.py          # LLM interface (Ollama)
  config.py              # Configuration
  ...
```

---

## Example Workflow

1. **Extract text** from a website using OCR (e.g., Google Vision API + Crawl4AI).
2. **Ingest text** into the RAG system.
3. **Ask questions** interactively or via API.
4. **Get answers** with sources and confidence scores.

---

## Extending

- **Change chunking:** Edit `DocumentProcessor` or its parameters.
- **Switch embedding/LLM:** Update `config.py` and ensure models are available.
- **Integrate with web UI:** Wrap the Q&A logic in a web server (e.g., FastAPI, Flask).

---

## Troubleshooting

- **Unicode errors:** Ensure files are read with `encoding='utf-8'`.
- **spaCy errors:** Run `python -m spacy download en_core_web_sm`.
- **Ollama errors:** Ensure Ollama is running and required models are pulled.

---

## License

MIT License (or specify your license here)

---

## Acknowledgements

- [spaCy](https://spacy.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)

---

**For questions or contributions, open an issue or pull request!** 