import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Streamlit
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDF and chat with its content")
st.secrets["HF_API_KEY"]

# Input the API key
api_key = st.text_input("Enter the Groq API key:", type="password")

# Ensure HuggingFace API key is loaded correctly
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
if not os.getenv("HF_API_KEY"):
    st.error("HF_API_KEY is missing. Check your .env file.")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if API key is provided
if api_key:
    os.environ["GROQ_API_KEY"] = api_key  # Set the Groq API key dynamically
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # Statefully manage chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Use persistent storage for Chroma
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        
        retriever = vectorstore.as_retriever()

        # Contextualize the question
        contextualize_q_system_prompt = (
            '''
            Given a chat history and the latest user question,
            reformulate it to be a standalone question without history references.
            '''
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User interaction
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Display response and history
            st.success("Assistant: " + response["answer"])
            st.write("Chat history:", session_history.messages)
            logger.info(f"Response: {response['answer']}")

else:
    st.warning("Enter a valid API key")
