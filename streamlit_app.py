import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Make sure you have set HUGGINGFACEHUB_API_TOKEN in your environment variables
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_token:
    st.error("Please set your Hugging Face API token in the environment variables.")
    st.stop()

def load_vectorstore(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        st.stop()
    with open(filepath, "rb") as f:
        return pickle.load(f)

def setup_chatbot(vectorstore):
    # Use Hugging Face model for text generation
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Streamlit app
st.title("Conversational PDF Chatbot")

# Specify the path to the FAISS vector store
vectorstore_path = r"C:\Users\enrico\Desktop\Test\02_Model_CPU\faiss_vectorstore.pkl"

# Load the existing vector store
vectorstore = load_vectorstore(vectorstore_path)

# Initialize the chatbot
chatbot = setup_chatbot(vectorstore)

st.write("Chatbot is ready! You can now interact with the PDFs.")

# Chat loop
user_question = st.text_input("Ask a question (or type 'exit' to quit):")

if user_question:
    if user_question.lower() == "exit":
        st.write("Goodbye!")
    else:
        response = chatbot({"question": user_question, "chat_history": []})
        st.write(f"Answer: {response['answer']}")
        
        # Display the sources
        if 'source_documents' in response:
            st.write("Sources:")
            for doc in response['source_documents']:
                st.write(f"- {doc.metadata['source']}")
        
        st.write("---")
