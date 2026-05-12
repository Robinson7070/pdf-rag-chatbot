# PDF RAG Chatbot

A conversational AI application that lets you ask questions about 
any PDF document using Retrieval-Augmented Generation (RAG).

## How it works
1. Upload any PDF document
2. Ask questions about its content
3. Get accurate answers powered by OpenAI and FAISS vector search

## Stack
- Python · Streamlit · LangChain · OpenAI GPT-3.5
- FAISS vector database · OpenAI Embeddings

## How to run
1. Clone the repo
2. Create a `.env` file with your OpenAI API key: `OPENAI_API_KEY=your_key`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
