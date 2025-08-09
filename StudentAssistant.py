from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from essential import contextualize_q_system_prompt, chat_prompt_template
from dotenv import load_dotenv
import streamlit as st
import tempfile
import random
import string
import os
load_dotenv()
os.environ['HF_TOKEN'] = st.secrets['HF_TOKEN']
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class StudentAssistant:
    def __init__(self, contextual_prompt, prompt_templete, file):
        self.file = file
        self.contextual_prompt = contextual_prompt
        self.prompt_templete = prompt_templete
        self.embeddings = self.return_embeddings()
        self.docs = self.loading_and_chunking()
        self.llm = self.return_llm()
        self.retriever = self.return_vector_store()
        self.rag_chain = self.return_chain()

    def loading_and_chunking(self):
        loader = PyPDFLoader(self.file)
        documents = loader.load()
        if not documents:
            st.error('The PDF contains no readable text.')
        chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        docs = chunker.split_documents(documents)
        if not docs:
            st.error("Text was loaded but no chunks were created.")
        return docs
    
    def return_embeddings(self):
        return HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    
    def return_llm(self):
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    
    def return_vector_store(self):
        if not self.docs:
            st.error("No document chunks available for FAISS indexing.")
        vsdb = FAISS.from_documents(self.docs, self.embeddings)
        return vsdb.as_retriever()

    def create_chat_prompt_templetes(self):
        cp = ChatPromptTemplate.from_messages([
            ("system", self.contextual_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        sp = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templete),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        return cp, sp

    def get_session_history(self, session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    def return_chain(self):
        cp, sp = self.create_chat_prompt_templetes()
        history_retriever = create_history_aware_retriever(self.llm, self.retriever, cp)
        doc_chain = create_stuff_documents_chain(self.llm, sp)
        qa_chain = create_retrieval_chain(history_retriever, doc_chain)
        rag_chain = RunnableWithMessageHistory(
            qa_chain,
            self.get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )
        return rag_chain
    
    def return_response(self, query, session):
        resp = self.rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session}}
        )
        return resp['answer']

    



def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.store = {}
        st.session_state.bot = None
        st.session_state.path = None
        st.session_state.session_id = f"default_session-{''.join(random.choices(string.hexdigits, k=10))}"
        st.session_state.messages = []
        st.session_state.initialized = True

# Upload and save the PDF to a temporary path
def handle_file_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# Render chat history
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle user input and response generation
def handle_user_input(choice):
    if choice == "Summary":
        if prompt := "Summarize the entire document into a concise, well-structured overview that captures all key points, main ideas, and essential details, while maintaining the original meaning and context.":
            # Show user input
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
    
            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        session_history = st.session_state.session_id
                        response = st.session_state.bot.return_response(prompt, session_history)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    elif choice == "Important MCQ's":
        if prompt := "From the entire document, extract the most important multiple-choice questions (MCQs) that comprehensively cover the key concepts and critical details. Ensure each MCQ is clear, concise, and focuses on testing essential knowledge.":
            # Show user input
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
    
            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        session_history = st.session_state.session_id
                        response = st.session_state.bot.return_response(prompt, session_history)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    elif choice == "Important Answer Questions":
        if prompt := "From the entire document, create a list of important short-answer questions that focus on the key facts, concepts, and details, ensuring each question is concise and directly tests essential knowledge.":
            # Show user input
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
    
            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        session_history = st.session_state.session_id
                        response = st.session_state.bot.return_response(prompt, session_history)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    elif choice == "Chat With Document":
        if prompt := st.chat_input("Ask me anything from document..."):
            # Show user input
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        session_history = st.session_state.session_id
                        response = st.session_state.bot.return_response(prompt, session_history)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Main app
def main():
    initialize_session_state()

    st.title("🧠 Student Learning Assistant")
    st.markdown("Ask questions about your uploaded document or any general topic. I'm here to help you learn!")

    st.text_input("📌 Session ID", value=st.session_state.session_id, disabled=True)

    file_to_upload = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

    if file_to_upload and st.session_state.bot is None:
        st.session_state.path = handle_file_upload(file_to_upload)
        try:
            st.session_state.bot = StudentAssistant(
                contextual_prompt=contextualize_q_system_prompt,
                prompt_templete=chat_prompt_template,
                file=st.session_state.path
            )
            st.success("PDF loaded successfully. You can now ask questions.")
        except Exception as e:
            st.error(f"Failed to initialize assistant: {e}")
            return
    
    if st.session_state.bot:
        choice = st.radio("Select One of the Following Options: ", ["-- Select an option --", "Important MCQ's", "Important Answer Questions", "Summary", "Chat With Document"])
        if choice == "Chat With Document":
            render_chat_history()
            handle_user_input(choice)
        elif choice == "Important MCQ's" or choice == "Important Answer Questions" or choice == "Summary":
            handle_user_input(choice)
        else:
            pass
    else:
        st.info("Please upload a PDF file to begin.")


main()

