from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
from pathlib import Path
import logging

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Initialize FastAPI app
app = FastAPI(
    title="Domain-Specific RAG Chatbot",
    description="A chatbot using Llama-4, LangChain, and RAG for document-based queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    use_context: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    status: str

# Global variables for the RAG system
vectorstore = None
qa_chain = None
embeddings = None
llm = None

# Configuration
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = "path/to/your/llama-4-model.gguf"  # Update this path
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Custom prompt template for domain-specific responses
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

async def initialize_models():
    """Initialize the LLM and embeddings models"""
    global llm, embeddings
    
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Llama model
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=2000,
            n_ctx=4096,
            verbose=False
        )
        
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

async def process_documents(file_paths: List[str]):
    """Process uploaded documents and create vector store"""
    global vectorstore, qa_chain
    
    try:
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                continue
                
            docs = loader.load()
            documents.extend(docs)
        
        if not documents:
            raise ValueError("No valid documents found")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info(f"Processed {len(texts)} document chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await initialize_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Domain-Specific RAG Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    status = {
        "status": "healthy",
        "models_loaded": llm is not None and embeddings is not None,
        "vectorstore_ready": vectorstore is not None,
        "qa_chain_ready": qa_chain is not None
    }
    return status

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and TXT files are supported"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        success = await process_documents([str(file_path)])
        
        if success:
            return DocumentUploadResponse(
                message="Document uploaded and processed successfully",
                filename=file.filename,
                status="success"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to process document"
            )
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    try:
        file_paths = []
        uploaded_files = []
        
        for file in files:
            if not file.filename.endswith(('.pdf', '.txt')):
                continue
                
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_paths.append(str(file_path))
            uploaded_files.append(file.filename)
        
        if not file_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid files uploaded"
            )
        
        success = await process_documents(file_paths)
        
        if success:
            return {
                "message": "Documents uploaded and processed successfully",
                "files": uploaded_files,
                "status": "success"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to process documents"
            )
            
    except Exception as e:
        logger.error(f"Multiple upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for querying the RAG system"""
    try:
        if not qa_chain:
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded yet. Please upload documents first."
            )
        
        if request.use_context:
            # Use RAG with document context
            result = qa_chain({"query": request.query})
            response = result["result"]
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.append(doc.metadata['source'])
            
            return ChatResponse(response=response, sources=list(set(sources)))
        
        else:
            # Direct LLM response without RAG
            response = llm(request.query)
            return ChatResponse(response=response, sources=[])
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List uploaded documents"""
    try:
        documents = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                documents.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a specific document"""
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            return {"message": f"Document {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_all_documents():
    """Clear all uploaded documents and reset vector store"""
    global vectorstore, qa_chain
    
    try:
        # Remove all files
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
        
        # Reset vector store and QA chain
        vectorstore = None
        qa_chain = None
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)