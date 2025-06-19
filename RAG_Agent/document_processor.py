import hashlib
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, mode: str = "default", spacy_sentences_per_chunk: int = 8):
        self.mode = mode
        self.spacy_sentences_per_chunk = spacy_sentences_per_chunk
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Load spaCy model if needed
        if self.mode == "spacy" and SPACY_AVAILABLE:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None
    
    def create_chunks(self, text: str, source: str, page: int = 0) -> List[Document]:
        """Create chunks from text with deterministic IDs"""
        if self.mode == "spacy" and self.nlp is not None:
            # ML-based chunking: split into sentences, then group
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            # Group sentences into chunks
            chunks = []
            for i in range(0, len(sentences), self.spacy_sentences_per_chunk):
                chunk = " ".join(sentences[i:i+self.spacy_sentences_per_chunk])
                if chunk:
                    chunks.append(chunk)
        else:
            # Default: character-based chunking
            chunks = self.text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            # Create deterministic ID based on source, page, and chunk index
            chunk_id = self._generate_chunk_id(source, page, i)
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "page": page,
                    "chunk_index": i,
                    "chunk_id": chunk_id
                }
            )
            documents.append(doc)
        return documents
    
    def _generate_chunk_id(self, source: str, page: int, chunk_index: int) -> str:
        """Generate deterministic chunk ID"""
        content = f"{source}_{page}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()