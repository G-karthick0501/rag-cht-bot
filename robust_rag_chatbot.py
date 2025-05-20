import streamlit as st
import os
import re
import datetime
import tempfile
import traceback
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging

os.makedirs("artifacts/responses", exist_ok=True)
os.makedirs("artifacts/logs", exist_ok=True)
qa_log_path = "artifacts/responses/qa_log.txt"

try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("artifacts/logs/app.log", encoding="utf-8", mode="a"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("rag_chatbot")
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("rag_chatbot")

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot for PDF Documents")
st.markdown("""
This application uses a Retrieval-Augmented Generation (RAG) framework to answer questions about PDF documents.
Upload a PDF file, process it, and then ask questions about its content.
""")

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""
if 'show_errors' not in st.session_state:
    st.session_state.show_errors = False
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = "auto"

def quiet_error(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            if st.session_state.show_errors:
                st.error(error_msg)
            return None
    return wrapper

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    
    try:
        if not isinstance(text, str):
            text = str(text)
        
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')
        
        text = re.sub(r'\s+', ' ', text)
        
        text = ''.join(c if c.isprintable() or c in ['\n', '\t'] else ' ' for c in text)
        
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '–')
        text = text.replace('â€"', '—')
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        if text:
            try:
                return str(text).encode('ascii', errors='replace').decode('ascii', errors='replace')
            except:
                return "Text cleaning failed."
        return ""

@quiet_error
def extract_text_with_pymupdf(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    
    doc.close()
    return text

@quiet_error
def extract_text_with_pypdf2(pdf_path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    
    return text

@quiet_error
def extract_text_with_pdfplumber(pdf_path: str) -> str:
    import pdfplumber
    text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    
    return text

@quiet_error
def extract_text_from_pdf(pdf_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_path = temp_file.name
    
    extracted_text = ""
    extraction_method = st.session_state.extraction_method
    
    try:
        if extraction_method == "pymupdf" or extraction_method == "auto":
            logger.info("Attempting extraction with PyMuPDF")
            try:
                import fitz
                extracted_text = extract_text_with_pymupdf(temp_path)
                if extracted_text and extraction_method == "auto":
                    logger.info("PyMuPDF extraction successful")
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {str(e)}")
                if extraction_method == "pymupdf":
                    raise

        if (not extracted_text and extraction_method == "auto") or extraction_method == "pypdf2":
            logger.info("Attempting extraction with PyPDF2")
            try:
                extracted_text = extract_text_with_pypdf2(temp_path)
                if extracted_text and extraction_method == "auto":
                    logger.info("PyPDF2 extraction successful")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {str(e)}")
                if extraction_method == "pypdf2":
                    raise

        if (not extracted_text and extraction_method == "auto") or extraction_method == "pdfplumber":
            logger.info("Attempting extraction with pdfplumber")
            try:
                extracted_text = extract_text_with_pdfplumber(temp_path)
                if extracted_text:
                    logger.info("pdfplumber extraction successful")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {str(e)}")
                if extraction_method == "pdfplumber":
                    raise

        if not extracted_text:
            logger.error("All extraction methods failed or returned empty text")
            if extraction_method == "auto":
                extracted_text = "Failed to extract text from the document using multiple methods."
            else:
                extracted_text = f"Failed to extract text using the selected method: {extraction_method}."
    
    finally:
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {str(e)}")
    
    cleaned_text = clean_text(extracted_text)
    logger.info(f"Extracted and cleaned {len(cleaned_text)} characters from PDF")
    
    return cleaned_text

@quiet_error
def generate_document_summary(text: str) -> str:
    text = clean_text(text)
    
    if not text or len(text) < 50:
        return "# Document Summary\n\nInsufficient text extracted from document."
    
    title = text[:100].strip()
    if '.' in title:
        title = title.split('.')[0].strip()
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() + "." for p in text.split(".") if p.strip()]
    
    summary = "# Document Summary\n\n"
    summary += f"## Title\n{title}\n\n"
    summary += "## Content Overview\n"
    
    if paragraphs:
        for p in paragraphs:
            if len(p) >= 50:
                summary += f"{p[:500]}...\n\n"
                break
        else:
            combined = " ".join(paragraphs[:3])
            summary += f"{combined[:500]}...\n\n"
    else:
        summary += f"{text[:500]}...\n\n"
    
    summary += f"\n## Document Statistics\n"
    summary += f"- Total characters: {len(text)}\n"
    summary += f"- Estimated pages: {max(1, len(text) // 3000)}\n"
    summary += f"- Extraction method: {st.session_state.extraction_method}\n"
    
    return summary

@quiet_error
def split_text(text: str) -> List[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text = clean_text(text)
    
    if not text:
        return []
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        cleaned_chunks = []
        for chunk in chunks:
            clean_chunk = clean_text(chunk)
            if clean_chunk and len(clean_chunk.strip()) > 20:
                cleaned_chunks.append(clean_chunk)
        
        if not cleaned_chunks and text:
            logger.warning("Normal text splitting failed, using manual chunking")
            text_length = len(text)
            chunk_size = 800
            overlap = 200
            
            cleaned_chunks = []
            for i in range(0, text_length, chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk and len(chunk.strip()) > 20:
                    cleaned_chunks.append(chunk)
        
        logger.info(f"Split text into {len(cleaned_chunks)} chunks")
        return cleaned_chunks
    
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        if text and len(text.strip()) > 20:
            return [text]
        return []

@quiet_error
def create_vector_store(chunks: List[str]):
    if not chunks:
        logger.error("No chunks provided to create vector store")
        return None
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    try:
        logger.info("Initializing HuggingFace embeddings")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        logger.info(f"Creating FAISS vector store with {len(chunks)} chunks")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        logger.info("Vector store created successfully")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@quiet_error
def generate_response(question: str, context: str) -> str:
    question = clean_text(question)
    context = clean_text(context)
    
    if not question or not context or len(context) < 20:
        return "I don't have enough information to answer that question."
    
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
    
    if len(paragraphs) <= 1:
        sentences = [s.strip() + "." for s in context.split(".") if s.strip()]
        paragraphs = []
        for i in range(0, len(sentences), 3):
            paragraph = " ".join(sentences[i:i+3])
            if paragraph.strip():
                paragraphs.append(paragraph)
    
    if len(paragraphs) <= 1 and len(context) > 100:
        for i in range(0, len(context), 100):
            chunk = context[i:i+100].strip()
            if chunk:
                paragraphs.append(chunk)
    
    if not paragraphs:
        return "I couldn't extract useful information from the document."
    
    question_terms = [w.lower() for w in re.findall(r'\b\w\w\w\w+\b', question.lower())]
    
    if not question_terms:
        return "From the document: " + paragraphs[0]
    
    scored_paragraphs = []
    for p in paragraphs:
        score = sum(1 for term in question_terms if term in p.lower())
        if score > 0:
            scored_paragraphs.append((p, score))
    
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_paragraphs:
        return "From the document: " + paragraphs[0]
    
    top_paragraphs = [p[0] for p in scored_paragraphs[:2]]
    return "Based on the document:\n\n" + "\n\n".join(top_paragraphs)

@quiet_error
def write_to_log(question: str, response: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    question = clean_text(question)
    response = clean_text(response)
    
    log_entry = f"Time: {timestamp}\nQ: {question}\nA: {response}\n\n"
    
    try:
        with open(qa_log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Error writing to log with UTF-8: {str(e)}")
        try:
            with open(qa_log_path, "a", encoding="ascii", errors="replace") as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing to log with ASCII: {str(e)}")
            try:
                with open(qa_log_path + ".new", "w", encoding="utf-8", errors="replace") as f:
                    f.write(log_entry)
            except:
                logger.error("All logging attempts failed")

@quiet_error
def save_responses_to_file() -> str:
    response_file_path = "artifacts/responses/qa_samples.txt"
    
    try:
        qa_text = "No interactions logged."
        if os.path.exists(qa_log_path):
            try:
                with open(qa_log_path, "r", encoding="utf-8", errors="replace") as f:
                    qa_text = f.read()
            except:
                try:
                    with open(qa_log_path, "r", encoding="ascii", errors="replace") as f:
                        qa_text = f.read()
                except Exception as e:
                    logger.error(f"Error reading log file: {str(e)}")
        
        output = "# RAG Chatbot Sample Questions and Responses\n\n"
        output += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        output += "## Interactions\n\n"
        output += qa_text
        
        try:
            with open(response_file_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(output)
        except:
            with open(response_file_path, "w", encoding="ascii", errors="replace") as f:
                f.write(output)
        
        logger.info(f"Responses saved to {response_file_path}")
        return response_file_path
    
    except Exception as e:
        error_msg = f"Error saving responses: {str(e)}"
        logger.error(error_msg)
        if st.session_state.show_errors:
            st.error(error_msg)
        return ""

@quiet_error
def process_query(question: str, vector_store) -> str:
    question = clean_text(question)
    if not question:
        return "Please provide a question."
    
    if any(term in question.lower() for term in ["summary", "overview", "what is this document about"]):
        if st.session_state.document_summary:
            summary = clean_text(st.session_state.document_summary)
            write_to_log(question, summary)
            return summary
    
    try:
        docs = vector_store.similarity_search(question, k=5)
        
        if not docs:
            logger.warning("No documents retrieved from vector store")
            response = "I couldn't find relevant information in the document."
            write_to_log(question, response)
            return response
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        response = generate_response(question, context)
        
        write_to_log(question, response)
        
        return response
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        response = "I'm having trouble searching the document. Please try a different question."
        write_to_log(question, response)
        return response

def main():
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        with st.expander("Advanced Settings", expanded=False):
            st.session_state.show_errors = st.checkbox("Show detailed error messages", value=False)
            
            st.session_state.extraction_method = st.radio(
                "PDF Extraction Method",
                options=["auto", "pymupdf", "pypdf2", "pdfplumber"],
                index=0,
                help="Select the method to extract text from PDFs. 'auto' tries multiple methods."
            )
            
            if st.button("Save Responses for Submission"):
                response_file = save_responses_to_file()
                if response_file:
                    st.success(f"Responses saved to {response_file}")
                    try:
                        with open(response_file, "r", encoding="utf-8", errors="replace") as f:
                            st.download_button(
                                label="Download Responses",
                                data=f.read(),
                                file_name="qa_samples.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        if st.session_state.show_errors:
                            st.error(f"Error creating download button: {str(e)}")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        text = extract_text_from_pdf(uploaded_file)
                        
                        if not text or len(text) < 50:
                            st.warning("Limited text extracted from the document. Results may not be accurate.")
                            if len(text) < 20:
                                st.info("Try a different extraction method in Advanced Settings.")
                            text = text or "Limited content available."
                        
                        st.session_state.document_text = text
                        
                        summary = generate_document_summary(text)
                        st.session_state.document_summary = summary
                        
                        chunks = split_text(text)
                        
                        if not chunks:
                            st.warning("Could not split the document text into chunks. Using the full text instead.")
                            chunks = [text]
                        
                        vector_store = create_vector_store(chunks)
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.processed = True
                            st.success("Document processed successfully!")
                        else:
                            st.error("Failed to create vector store. Please try again or use different settings.")
                    
                    except Exception as e:
                        error_msg = f"Error processing document: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.error("Failed to process the document. Please try again or use different settings.")
                        if st.session_state.show_errors:
                            st.error(error_msg)

    if not st.session_state.processed:
        st.info("Please upload and process a document to begin.")
    else:
        if st.session_state.document_summary:
            st.markdown(st.session_state.document_summary)
        
        st.success("Document processed! You can now ask questions about it.")
        
        question = st.text_input("Ask a question about the document:")
        
        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = process_query(question, st.session_state.vector_store)
                    if response:
                        st.markdown(f"**Answer:** {response}")
                    else:
                        st.error("Could not generate a response. Please try a different question.")
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    st.error("An error occurred while generating the response. Please try again.")
                    if st.session_state.show_errors:
                        st.error(error_msg)
        
        st.subheader("Recent Q&A History")
        try:
            if os.path.exists(qa_log_path):
                try:
                    with open(qa_log_path, "r", encoding="utf-8", errors="replace") as f:
                        qa_text = f.read()
                except:
                    try:
                        with open(qa_log_path, "r", encoding="ascii", errors="replace") as f:
                            qa_text = f.read()
                    except Exception as e:
                        logger.error(f"Error reading log file: {str(e)}")
                        qa_text = "Error reading Q&A history."
                
                if qa_text:
                    st.text_area("Q&A Log", qa_text, height=300)
                else:
                    st.info("No questions asked yet.")
        except Exception as e:
            logger.error(f"Error displaying Q&A history: {str(e)}")
            st.info("Could not display Q&A history.")

if __name__ == "__main__":
    main()