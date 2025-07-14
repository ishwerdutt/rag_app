import os
import gc
import json
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from contextlib import contextmanager
import logging

import torch
import numpy as np
from flask import Flask, request, render_template_string, jsonify, Response
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import glob
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ----------------------------
# Configuration & Data Classes
# ----------------------------
@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    qa_model: str = "distilbert-base-uncased-distilled-squad"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_in_memory: int = 1000
    cache_dir: str = "cache"
    data_dir: str = "data"
    top_k_retrieve: int = 5
    top_k_rerank: int = 3
    similarity_threshold: float = 0.3
    use_faiss: bool = True
    enable_hybrid_search: bool = True
    
@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: str
    metadata: Dict = None

# ----------------------------
# Memory-Efficient Embedding Manager
# ----------------------------
class EmbeddingManager:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    @contextmanager
    def load_model(self):
        """Context manager to load model only when needed"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.model.to(self.device)
        
        try:
            yield self.model
        finally:
            # Optionally unload model to save memory
            pass
    
    def encode_chunks(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """Memory-efficient batch encoding with caching"""
        cache_key = hashlib.md5(str(chunks).encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"embeddings_{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            logger.info("Loading cached embeddings")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        with self.load_model() as model:
            logger.info(f"Encoding {len(chunks)} chunks in batches of {batch_size}")
            embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                # Force garbage collection to free memory
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            embeddings = np.array(embeddings)
            
            # Cache embeddings
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            return embeddings

# ----------------------------
# Advanced Document Processor
# ----------------------------
class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with better error handling"""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue
                
                return " ".join(text_parts)
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""
    
    def smart_chunking(self, text: str, source: str) -> List[DocumentChunk]:
        """Advanced chunking with sentence boundary preservation"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.chunk_size and current_chunk:
                # Create chunk
                chunk_id = hashlib.md5(f"{source}_{len(chunks)}".encode()).hexdigest()[:8]
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    metadata={"chunk_index": len(chunks)}
                ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.config.chunk_overlap:] if len(current_chunk) > self.config.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += sentence + ". "
                current_size += sentence_size + 2
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = hashlib.md5(f"{source}_{len(chunks)}".encode()).hexdigest()[:8]
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                metadata={"chunk_index": len(chunks)}
            ))
        
        return chunks
    
    def process_documents(self, folder: str) -> List[DocumentChunk]:
        """Process all PDFs in folder with parallel processing"""
        pdf_files = glob.glob(f"{folder}/*.pdf")
        all_chunks = []
        
        def process_file(file_path):
            text = self.extract_text_from_pdf(file_path)
            if text:
                return self.smart_chunking(text, os.path.basename(file_path))
            return []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_file, pdf_files))
        
        for chunks in results:
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(pdf_files)} files into {len(all_chunks)} chunks")
        return all_chunks

# ----------------------------
# Hybrid Retrieval System
# ----------------------------
class HybridRetriever:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index and TF-IDF matrix"""
        self.chunks = chunks
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Build dense embeddings index
        logger.info("Building dense embeddings index...")
        embeddings = self.embedding_manager.encode_chunks(chunk_texts)
        
        if self.config.use_faiss:
            # Use FAISS for efficient similarity search
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype(np.float32))
        else:
            self.embeddings = embeddings
        
        # Build sparse TF-IDF index for hybrid search
        if self.config.enable_hybrid_search:
            logger.info("Building TF-IDF index...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
        
        logger.info("Index building complete")
    
    def dense_retrieval(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """Dense retrieval using embeddings"""
        with self.embedding_manager.load_model() as model:
            query_embedding = model.encode([query], convert_to_tensor=False)
        
        if self.config.use_faiss:
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for i, score in zip(indices[0], scores[0]):
                if score > self.config.similarity_threshold:
                    results.append((self.chunks[i], float(score)))
            
            return results
        else:
            # Fallback to manual similarity computation
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > self.config.similarity_threshold:
                    results.append((self.chunks[idx], similarities[idx]))
            
            return results
    
    def sparse_retrieval(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """Sparse retrieval using TF-IDF"""
        if not self.config.enable_hybrid_search or self.tfidf_vectorizer is None:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Lower threshold for sparse retrieval
                results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def hybrid_retrieve(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """Combine dense and sparse retrieval"""
        dense_results = self.dense_retrieval(query, k)
        sparse_results = self.sparse_retrieval(query, k) if self.config.enable_hybrid_search else []
        
        # Combine and deduplicate results
        combined_results = {}
        
        for chunk, score in dense_results:
            combined_results[chunk.chunk_id] = (chunk, score * 0.7)  # Weight dense results
        
        for chunk, score in sparse_results:
            if chunk.chunk_id in combined_results:
                # Boost score if found in both
                combined_results[chunk.chunk_id] = (chunk, combined_results[chunk.chunk_id][1] + score * 0.3)
            else:
                combined_results[chunk.chunk_id] = (chunk, score * 0.3)
        
        # Sort by combined score
        sorted_results = sorted(combined_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

# ----------------------------
# Advanced QA Pipeline
# ----------------------------
class AdvancedQA:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    @contextmanager
    def load_qa_model(self):
        """Context manager for QA model"""
        if self.pipeline is None:
            logger.info(f"Loading QA model: {self.config.qa_model}")
            self.pipeline = pipeline(
                "question-answering",
                model=self.config.qa_model,
                tokenizer=self.config.qa_model,
                device=0 if torch.cuda.is_available() else -1
            )
        
        try:
            yield self.pipeline
        finally:
            pass
    
    def answer_question(self, question: str, chunks: List[Tuple[DocumentChunk, float]]) -> Dict:
        """Generate answer from top chunks"""
        if not chunks:
            return {"answer": "No relevant information found.", "confidence": 0.0, "source": ""}
        
        best_answer = ""
        best_confidence = 0.0
        best_source = ""
        
        with self.load_qa_model() as qa_pipeline:
            for chunk, retrieval_score in chunks:
                try:
                    # Truncate context to avoid token limits
                    context = chunk.text[:2000]
                    
                    result = qa_pipeline(
                        question=question,
                        context=context,
                        max_answer_len=200,
                        handle_impossible_answer=True
                    )
                    
                    # Combine retrieval and QA confidence
                    combined_confidence = result["score"] * 0.7 + retrieval_score * 0.3
                    
                    if combined_confidence > best_confidence:
                        best_confidence = combined_confidence
                        best_answer = result["answer"]
                        best_source = chunk.text
                        
                except Exception as e:
                    logger.warning(f"Error in QA for chunk {chunk.chunk_id}: {e}")
                    continue
        
        return {
            "answer": best_answer,
            "confidence": best_confidence,
            "source": best_source
        }

# ----------------------------
# Main RAG System
# ----------------------------
class AdvancedRAGSystem:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.retriever = HybridRetriever(self.config)
        self.qa_system = AdvancedQA(self.config)
        self.is_initialized = False
        self._lock = threading.Lock()
    
    def initialize(self):
        """Initialize the RAG system"""
        with self._lock:
            if self.is_initialized:
                return
            
            logger.info("Initializing RAG system...")
            
            # Process documents
            chunks = self.doc_processor.process_documents(self.config.data_dir)
            
            if not chunks:
                logger.warning("No documents found to process!")
                return
            
            # Build retrieval index
            self.retriever.build_index(chunks)
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully!")
    
    def query(self, question: str) -> Dict:
        """Main query function"""
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_initialized:
            return {"error": "System not initialized - no documents found"}
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.hybrid_retrieve(
            question, 
            self.config.top_k_retrieve
        )
        
        # Get top chunks for QA
        top_chunks = retrieved_chunks[:self.config.top_k_rerank]
        
        # Generate answer
        result = self.qa_system.answer_question(question, top_chunks)
        
        # Add metadata
        result["retrieved_chunks"] = len(retrieved_chunks)
        result["sources"] = [chunk.source for chunk, _ in top_chunks]
        
        return result

# ----------------------------
# Flask Application
# ----------------------------
rag_system = AdvancedRAGSystem()

ADVANCED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced RAG System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .search-box { margin-bottom: 20px; }
        .search-box input { width: 70%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 5px; }
        .search-box button { padding: 12px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px; }
        .search-box button:hover { background: #0056b3; }
        .result { margin-top: 30px; }
        .answer { background: #e8f5e8; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #28a745; }
        .confidence { font-size: 14px; color: #666; margin-top: 10px; }
        .source { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #6c757d; font-family: monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
        .metadata { font-size: 12px; color: #888; margin-top: 10px; }
        .loading { text-align: center; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced RAG System</h1>
            <p>Memory-efficient retrieval-augmented generation with hybrid search</p>
        </div>
        
        <div class="search-box">
            <form method="post" id="searchForm">
                <input name="question" placeholder="Ask me anything about your documents..." required>
                <button type="submit">Search</button>
            </form>
        </div>
        
        {% if result %}
        <div class="result">
            {% if result.error %}
            <div class="status error">
                <strong>Error:</strong> {{ result.error }}
            </div>
            {% else %}
            <div class="answer">
                <h3>Answer:</h3>
                <div>{{ result.answer }}</div>
                <div class="confidence">
                    <strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%
                    <strong>Sources:</strong> {{ result.retrieved_chunks }} chunks retrieved
                </div>
            </div>
            
            <h4>Source Context:</h4>
            <div class="source">{{ result.source }}</div>
            
            <div class="metadata">
                <strong>Document Sources:</strong> {{ result.sources|join(', ') }}
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="status success" style="margin-top: 30px;">
            <strong>System Features:</strong>
            <ul>
                <li>🔍 Hybrid search (dense + sparse retrieval)</li>
                <li>💾 Memory-efficient processing with caching</li>
                <li>⚡ FAISS indexing for fast similarity search</li>
                <li>🎯 Smart chunking with sentence boundaries</li>
                <li>📊 Confidence scoring and source tracking</li>
            </ul>
        </div>
    </div>
    
    <script>
        document.getElementById('searchForm').addEventListener('submit', function() {
            document.querySelector('button[type="submit"]').textContent = 'Searching...';
            document.querySelector('button[type="submit"]').disabled = true;
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            try:
                result = rag_system.query(question)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                result = {"error": str(e)}
    
    return render_template_string(ADVANCED_HTML, result=result)

@app.route("/api/query", methods=["POST"])
def api_query():
    """REST API endpoint for programmatic access"""
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        result = rag_system.query(question)
        return jsonify(result)
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": rag_system.is_initialized,
        "config": {
            "embedding_model": rag_system.config.embedding_model,
            "qa_model": rag_system.config.qa_model,
            "hybrid_search": rag_system.config.enable_hybrid_search
        }
    })

if __name__ == "__main__":
    # Create necessary directories
    config = RAGConfig()
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    
    logger.info("Starting Advanced RAG System...")
    logger.info(f"Place your PDF files in the '{config.data_dir}' directory")
    
    app.run(debug=True, host="0.0.0.0", port=5000)