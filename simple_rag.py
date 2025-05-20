import streamlit as st
import os
import datetime
import re
import io
import base64
import tempfile
from typing import List, Dict, Any, Optional

# Create directory for logging
os.makedirs("artifacts/responses", exist_ok=True)
qa_log_path = "artifacts/responses/qa_log.txt"

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Main header and description
st.title("RAG Chatbot for PDF Documents")
st.markdown("""
This application uses a Retrieval-Augmented Generation (RAG) framework to answer questions about PDF documents.
Upload a PDF file, process it, and then ask questions about its content.
""")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""

# Robust error handling wrapper
def safe_operation(func):
    """Decorator for safe operation with proper error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

# Function for ultra-safe text handling
def ultra_clean_text(text: Optional[str]) -> str:
    """Clean and normalize text to avoid any encoding or processing issues"""
    if not text:
        return ""
    
    try:
        # Handle different string types
        if not isinstance(text, str):
            text = str(text)
        
        # Replace problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(c if c.isprintable() or c in ['\n', '\t'] else ' ' for c in text)
        
        # Fix common encoding issues
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        return text
    except Exception:
        # Ultimate fallback - if all else fails, return empty string
        return ""

# Function to try multiple PDF extraction methods
@safe_operation
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using multiple methods for robustness"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_path = temp_file.name
    
    extracted_text = ""
    
    # Method 1: Try PyMuPDF (if available)
    try:
        import fitz
        doc = fitz.open(temp_path)
        for page in doc:
            extracted_text += page.get_text()
        doc.close()
    except Exception as e:
        st.warning(f"PyMuPDF extraction failed: {str(e)}. Trying alternative method...")
        
        # Method 2: Try PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(temp_path)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}. Trying final method...")
            
            # Method 3: Try pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(temp_path) as pdf:
                    for page in pdf.pages:
                        extracted_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"All PDF extraction methods failed: {str(e)}")
                extracted_text = "Failed to extract text from PDF."
    
    # Clean up the temporary file
    try:
        os.unlink(temp_path)
    except:
        pass
    
    # Clean and return the text
    return ultra_clean_text(extracted_text)

# Function to generate document summary
@safe_operation
def generate_document_summary(text: str) -> str:
    """Generate a summary of the document"""
    # Clean text first
    text = ultra_clean_text(text)
    
    if not text or len(text) < 50:
        return "# Document Summary\n\nInsufficient text extracted from document to generate a summary."
    
    # Get title (first 100 chars or first sentence)
    title = text[:100].strip()
    if '.' in title:
        title = title.split('.')[0].strip()
    
    # Extract paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # Create summary text
    summary = "# Document Summary\n\n"
    
    # Add document title
    summary += f"## Title\n{title}\n\n"
    
    # Add content overview
    summary += "## Content Overview\n"
    
    # Add first paragraph as abstract if it's of reasonable length
    if paragraphs and len(paragraphs[0]) > 50:
        summary += f"{paragraphs[0]}\n\n"
    else:
        # If first paragraph is too short, combine first few paragraphs
        combined = " ".join(paragraphs[:3])
        if combined:
            summary += f"{combined[:500]}...\n\n"
    
    # Add document stats
    summary += f"\n## Document Statistics\n"
    summary += f"- Total characters: {len(text)}\n"
    summary += f"- Estimated pages: {max(1, len(text) // 3000)}\n"
    
    return summary

# Function to split text into chunks
@safe_operation
def split_text(text: str) -> List[str]:
    """Split text into manageable chunks for embedding"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Clean text
    text = ultra_clean_text(text)
    
    # Create splitter with robust settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text
    chunks = text_splitter.split_text(text)
    
    # Ensure all chunks are properly cleaned
    chunks = [ultra_clean_text(chunk) for chunk in chunks]
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks

# Function to create vector store
@safe_operation
def create_vector_store(chunks: List[str]):
    """Create a vector store from text chunks"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    if not chunks:
        st.error("No valid text chunks to process")
        return None
    
    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None
    
    # Create vector store
    try:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

# Function to generate response
@safe_operation
def generate_response(question: str, context: str) -> str:
    """Generate a response based on retrieved context"""
    # Clean inputs
    question = ultra_clean_text(question)
    context = ultra_clean_text(context)
    
    if not question or not context or len(context) < 20:
        return "I don't have enough information to answer that question."
    
    # Split context into paragraphs
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() + "." for p in context.split(".") if p.strip()]
    
    if not paragraphs:
        return "I couldn't extract useful information from the document."
    
    # Extract key terms from the question (longer than 3 chars)
    question_terms = [w.lower() for w in re.findall(r'\b\w\w\w\w+\b', question.lower())]
    
    if not question_terms:
        # If no significant terms found, return first paragraph
        return "From the document, I found this information that might be relevant:\n\n" + paragraphs[0]
    
    # Find relevant paragraphs
    scored_paragraphs = []
    for p in paragraphs:
        # Count how many question terms appear in the paragraph
        score = sum(1 for term in question_terms if term in p.lower())
        if score > 0:
            scored_paragraphs.append((p, score))
    
    # Sort paragraphs by relevance score
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # If no relevant paragraphs found, return the first paragraph
    if not scored_paragraphs:
        return "From the document, I found this information that might be relevant:\n\n" + paragraphs[0]
    
    # Construct response from most relevant paragraphs
    top_paragraphs = [p[0] for p in scored_paragraphs[:2]]
    return "Based on the document:\n\n" + "\n\n".join(top_paragraphs)

# Function to safely write to log file
@safe_operation
def safe_write_log(question: str, response: str) -> None:
    """Write to log file with comprehensive error handling"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean text
    question = ultra_clean_text(question)
    response = ultra_clean_text(response)
    
    log_entry = f"Time: {timestamp}\nQ: {question}\nA: {response}\n\n"
    
    try:
        # Try UTF-8 encoding first
        with open(qa_log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(log_entry)
    except Exception as e:
        try:
            # Fall back to ASCII if UTF-8 fails
            with open(qa_log_path, "a", encoding="ascii", errors="replace") as f:
                f.write(log_entry)
        except Exception as e:
            # Last resort - try to create a new file
            try:
                with open(qa_log_path + ".new", "w", encoding="utf-8", errors="replace") as f:
                    f.write(log_entry)
            except:
                # If all logging fails, just continue without logging
                pass

# Function to process query
@safe_operation
def process_query(question: str, vector_store) -> str:
    """Process a query and generate a response"""
    # Handle summary request
    if any(term in question.lower() for term in ["summary", "overview", "what is this document about"]):
        if st.session_state.document_summary:
            summary = ultra_clean_text(st.session_state.document_summary)
            safe_write_log(question, summary)
            return summary
    
    # Search for relevant documents
    try:
        docs = vector_store.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        error_msg = f"Error during document retrieval: {str(e)}"
        st.error(error_msg)
        return error_msg
    
    # Generate response
    response = generate_response(question, context)
    
    # Log the interaction
    safe_write_log(question, response)
    
    return response

# Install required packages
@safe_operation
def ensure_dependencies() -> bool:
    """Ensure all required packages are installed"""
    required_packages = ["pymupdf", "pypdf2", "pdfplumber", "langchain", "langchain_community", "faiss-cpu", "sentence-transformers"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.info(f"Installing missing packages: {', '.join(missing_packages)}")
        import subprocess
        for package in missing_packages:
            subprocess.check_call(["pip", "install", package])
        st.success("All required packages installed!")
        # Force a rerun to use newly installed packages
        st.experimental_rerun()
    
    return True

# Main application
def main():
    # Ensure dependencies are installed
    ensure_dependencies()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text or len(text) < 50:
                        st.error("Could not extract sufficient text from the document. Please try another PDF.")
                    else:
                        st.session_state.document_text = text
                        st.success(f"Extracted {len(text)} characters from document.")
                        
                        # Generate document summary
                        summary = generate_document_summary(text)
                        st.session_state.document_summary = summary
                        
                        # Split text into chunks
                        chunks = split_text(text)
                        
                        if not chunks:
                            st.error("Could not create text chunks from the document.")
                        else:
                            st.success(f"Split text into {len(chunks)} chunks.")
                            
                            # Create vector store
                            vector_store = create_vector_store(chunks)
                            
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.session_state.processed = True
                                st.success("Document processed successfully! Ready to answer questions.")
                            else:
                                st.error("Failed to create vector store.")

    # Main area for Q&A
    if not st.session_state.processed:
        st.info("Please upload and process a document to begin.")
    else:
        # Display document summary
        if st.session_state.document_summary:
            st.markdown(st.session_state.document_summary)
        
        st.success("Document processed! You can now ask questions about it.")
        
        # Question input
        question = st.text_input("Ask a question about the document:")
        
        if question:
            with st.spinner("Generating answer..."):
                response = process_query(question, st.session_state.vector_store)
                st.markdown(f"**Answer:** {response}")
        
        # Display recent Q&A history
        st.subheader("Recent Q&A History")
        try:
            if os.path.exists(qa_log_path):
                try:
                    with open(qa_log_path, "r", encoding="utf-8", errors="replace") as f:
                        qa_text = f.read()
                except:
                    # Try alternative encoding if UTF-8 fails
                    with open(qa_log_path, "r", encoding="ascii", errors="replace") as f:
                        qa_text = f.read()
                        
                if qa_text:
                    st.text_area("Q&A Log", qa_text, height=300)
                else:
                    st.info("No questions asked yet.")
        except Exception as e:
            st.warning(f"Could not read log file: {str(e)}")

if __name__ == "__main__":
    main()