import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import re
import os

load_dotenv()

def get_pdf_text(pdf_dir):
    text = ""
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            try:
                pdf_reader = PdfReader(pdf_path)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return text

def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def vector_db(chunks):
    embeddings = HuggingFaceEmbeddings()
    valid_chunks = [chunk for chunk in chunks if chunk]
    try:
        vectorstore = Chroma.from_texts(texts=valid_chunks, embedding=embeddings)
        return vectorstore
    except AttributeError as e:
        print(f"Error creating vectorstore: {e}")
        print("Chunks causing issues:", chunks)
        return None

def conversation_chain(vectorstore):
    prompt = ChatPromptTemplate.from_template("""
    Based on the provided context, please answer the question accurately. Answer the question completely:
    {context}
    
    Question: {input}
    """)
    
    repo_id = "mistralai/Mistral-7B-v0.1"
    llm = HuggingFaceHub(
        huggingfacehub_api_token= os.getenv('HF_TOKEN'),
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 100}
    )
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain

def get_answer(result, question):
    full_text = result['answer']
    pattern = rf"{re.escape(question)}\s*Answer:\s*(.*?)(?=\s*Question:|$)"
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    

def main():
    st.set_page_config(page_title="Chat with multiple pdfs")
    st.title("Chat with me")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    pdf_dir = r"C:\Users\Hemasri\Desktop\PDFs"
    text = get_pdf_text(pdf_dir)
    
    if not text:
        st.error("No text could be extracted from the PDFs. Please check the directory and PDF contents.")
        return

    chunks = split_text(text)
    vectorstore = vector_db(chunks)

    if vectorstore is None:
        st.error("Failed to create vector database. Please check your PDFs and try again.")
        return

    chain = conversation_chain(vectorstore)

    user_question = st.chat_input("Ask anything")

    if user_question is not None and user_question != "":
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        with st.chat_message("Human"):
            st.markdown(user_question)
        
        result = chain.invoke({"input": user_question})
        
        with st.chat_message("AI"):
            ai_response = get_answer(result, user_question)
            st.markdown(ai_response)
        st.session_state.chat_history.append(AIMessage(content=ai_response))

if __name__ == "__main__":
    main()