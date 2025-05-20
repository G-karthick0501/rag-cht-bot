import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import datetime

# Create directory for logging
os.makedirs("artifacts/responses", exist_ok=True)
qa_log_path = "artifacts/responses/qa_log.txt"

def load_pdf_with_pymupdf(file_path):
    """Load and extract text from a PDF file using PyMuPDF
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from PDF
    """
    print(f"Loading PDF from: {file_path}")
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
        text = ""
        
        # Extract text from each page
        for page in doc:
            text += page.get_text()
        
        print(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return None

def split_text(text):
    """Split text into smaller chunks with better overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger chunks for better context
        chunk_overlap=400,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting on paragraph breaks
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create a vector store from text chunks"""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Creating vector store...")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store created and saved")
    return vector_store

def generate_response(question, context):
    """Generate a better response based on retrieved context"""
    if not context:
        return "I don't have enough information to answer that question."
    
    # For detailed explanation requests, return more comprehensive context
    if any(word in question.lower() for word in ['explain', 'detail', 'summary', 'abstract']):
        # Find paragraphs instead of just sentences
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        # Find most relevant paragraphs
        question_words = [w.lower() for w in question.lower().split() if len(w) > 3]
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            if any(word in paragraph.lower() for word in question_words):
                relevant_paragraphs.append(paragraph)
        
        # Return more comprehensive response for explanations
        if relevant_paragraphs:
            return "Based on the document:\n\n" + "\n\n".join(relevant_paragraphs[:3])
        else:
            return "From the document:\n\n" + "\n\n".join(paragraphs[:2])
    else:
        # For regular questions, keep the original approach
        sentences = []
        for line in context.split('.'):
            if line.strip():
                sentences.append(line.strip() + '.')
        
        question_words = [w.lower() for w in question.lower().split() if len(w) > 3]
        relevant_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return "Based on the document: " + " ".join(relevant_sentences[:5])
        else:
            return "From the document: " + " ".join(sentences[:3])

def query_document(question, vector_store=None):
    """Process a query and generate a response"""
    try:
        # Load vector store if not provided
        if vector_store is None:
            print("Loading vector store from disk...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.load_local("faiss_index", embeddings)
        
        # Retrieve relevant documents (more for better context)
        docs = vector_store.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        response = generate_response(question, context)
        
        # Log the interaction
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(qa_log_path, "a") as f:
            f.write(f"Time: {timestamp}\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {response}\n\n")
        
        return response
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Main function to run the chatbot"""
    print("Enhanced RAG Chatbot")
    print("=" * 50)
    
    # Get the PDF path
    pdf_path = input("Enter the path to your PDF file: ")
    
    # Process the PDF
    print("\nProcessing document...")
    text = load_pdf_with_pymupdf(pdf_path)
    
    if text:
        chunks = split_text(text)
        vector_store = create_vector_store(chunks)
        
        # Interactive query loop
        print("\nChatbot is ready! Type 'exit' to quit.")
        print("-" * 50)
        
        while True:
            question = input("\nYou: ")
            if question.lower() == 'exit':
                break
            
            response = query_document(question, vector_store)
            print(f"Bot: {response}")
    
    print("\nThank you for using the RAG Chatbot!")

if __name__ == "__main__":
    main()