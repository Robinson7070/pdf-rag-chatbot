from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

def get_response(user_question, knowledge_base, chat_history):
    docs = knowledge_base.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Build messages with history
    messages = []
    
    # Add system context
    messages.append(HumanMessage(content=f"""You are a helpful assistant answering 
    questions about a document. Use the following context to answer questions.
    
    Context: {context}
    
    If the answer is not in the context, say so clearly."""))
    
    # Add conversation history
    for message in chat_history:
        messages.append(message)
    
    # Add current question
    messages.append(HumanMessage(content=user_question))
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(messages)
    return response.content

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Initialise chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Process PDF only once
        if st.session_state.knowledge_base is None:
            with st.spinner("Processing your PDF..."):
                text = get_pdf_text(pdf)
                chunks = get_text_chunks(text)
                st.session_state.knowledge_base = get_knowledge_base(chunks)
            st.success("PDF processed. Ask your questions!")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        # User input
        user_question = st.chat_input("Ask a question about your PDF:")
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append(
                HumanMessage(content=user_question)
            )
            
            # Get response
            with st.spinner("Thinking..."):
                response = get_response(
                    user_question,
                    st.session_state.knowledge_base,
                    st.session_state.chat_history
                )
            
            # Add AI response to history
            st.session_state.chat_history.append(
                AIMessage(content=response)
            )
            
            # Rerun to display updated chat
            st.rerun()

if __name__ == '__main__':
    main()