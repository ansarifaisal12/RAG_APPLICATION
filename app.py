import streamlit as st
import warnings
import os
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")

GOOGLE_API_KEY = ""  # Add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY,
                                   temperature=0.4, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return model, embeddings

def get_data(pdf_file):
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    model, embeddings = load_model()
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=False  # Do not return source documents
    )
    return qa_chain

def main():
    st.set_page_config(page_title="AI-Powered PDF Query Application", layout="wide")
    st.title("AI-Powered PDF Query Application")
    st.write("Upload a PDF document and ask any question related to its content. The AI will extract the relevant information and provide an answer.")
    
    left_column, right_column = st.columns(2)
    
    with left_column:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    with right_column:
        st.header("Ask a Question")
        query = st.text_input("Enter your query here:")
        
        if pdf_file is not None and query:
            if st.button("Submit"):
                with st.spinner("Processing..."):
                    # Save the uploaded file temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    qa_chain = get_data("temp.pdf")
                    output = qa_chain({"query": query})
                    st.success("Query processed successfully!")
                    st.subheader("Result:")
                    st.write(output["result"])  # Display only the result

if __name__ == "__main__":
    main()