import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import datetime

# Set up page configuration
st.set_page_config(page_title="Document Genie", layout="wide")

# Create directory for logging
os.makedirs("artifacts/responses", exist_ok=True)
qa_log_path = "artifacts/responses/qa_log.txt"

# Main header and description
st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries based on the document content.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Process Documents**: Click "Submit & Process" to analyze the documents.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """Split the text into manageable chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    """Create a vector store from the text chunks using HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return "Vector store created successfully"

# Function to generate response based on retrieved context
def generate_response(question, context):
    """Generate a response based on the retrieved context"""
    if not context:
        return "I don't have enough information to answer that question."
    
    # Extract sentences from the context
    sentences = []
    for line in context.split('\n'):
        sentences.extend([s.strip() + '.' for s in line.split('.') if s.strip()])
    
    # Find sentences that contain keywords from the question
    question_words = [w.lower() for w in question.lower().split() if len(w) > 3]
    relevant_sentences = []
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in question_words):
            relevant_sentences.append(sentence)
    
    # Construct the response
    if relevant_sentences:
        return "Based on the document: " + " ".join(relevant_sentences[:3])
    else:
        return "From the document, I found this information that might be relevant: " + " ".join(sentences[:2])

# Function to process user input
def user_input(user_question):
    """Process user question and generate response"""
    try:
        # Load the vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings)
        
        # Search for relevant documents
        docs = vector_store.similarity_search(user_question, k=4)
        
        # Extract context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate response
        response = generate_response(user_question, context)
        
        # Log the interaction
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(qa_log_path, "a") as f:
            f.write(f"Time: {timestamp}\n")
            f.write(f"Q: {user_question}\n")
            f.write(f"A: {response}\n\n")
            
        return response
    except Exception as e:
        return f"Error processing your question: {str(e)}"

def main():
    """Main function to run the Streamlit app"""
    st.header("Document Genie: Your AI-powered Document Assistant")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    # Process question if entered
    if user_question:
        with st.spinner("Generating answer..."):
            response = user_input(user_question)
            st.write("Reply: ", response)

    # Sidebar for document upload and processing
    with st.sidebar:
        st.title("Document Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                   accept_multiple_files=True, 
                                   key="pdf_uploader")
        
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Split text into {len(text_chunks)} chunks")
                    
                    # Create vector store
                    result = get_vector_store(text_chunks)
                    st.success(result)
            else:
                st.warning("Please upload at least one PDF document")

if __name__ == "__main__":
    main()