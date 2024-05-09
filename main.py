import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from htmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama3")  

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
 

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat com multiplos editais", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat com multiplos editais :books:")
    user_question = st.chat_input("Faça perguntas sobre seus editais")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Seus editais")
        pdf_docs = st.file_uploader(
            "Carregue seu editais aqui e pressione em 'processar'", accept_multiple_files=True)
        if st.button("Processar"):
            with st.spinner("Processing"):
                # extrair o texto do pdf
                raw_text = get_pdf_text(pdf_docs)

                # pegar o texto e separar em partes(chunks)
                text_chunks = get_text_chunks(raw_text)

                # criar uma base de conhecimento de vetores(vector store)
                vectorstore = get_vectorstore(text_chunks)

                # criando uma cadeia(chain) de conversação
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()