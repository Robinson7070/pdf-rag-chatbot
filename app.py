from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask you PDF 💭")

    # Upload file 
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract the text
    if pdf is  not None:
        pdf_reader = PdfReader(pdf)
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join([doc.page_content for doc in docs])
            response = llm.invoke([HumanMessage(content=f"Use this context to answer the question.\n\nContext: {context}\n\nQuestion: {user_question}")])

            st.write(response.content)

if __name__ == '__main__':
    main()