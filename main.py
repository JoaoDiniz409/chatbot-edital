import streamlit as st

import fitz

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf is not None:
            doc = fitz.open(stream=pdf.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            doc.close()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_context_retriever_chain(vectorstore):

    llm = ChatOllama(model="llama3", temperature=0.3) #temperature=0.3

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    contextualize_q_system_prompt = """Você é um assistente que é bom."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """Você é um assistente para tarefas de resposta a perguntas, limitando-se estritamente ao contexto fornecido. Utilize as partes do contexto recuperado para responder à pergunta de forma concisa, em até três frases. Se a resposta não estiver disponível no contexto, é aceitável declarar que não sabe. É importante ressaltar que qualquer resposta deve permanecer dentro dos limites do contexto fornecido, sem responder a perguntas fora desse escopo: 

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
        

def get_response(user_question): 
    retriever_chain = get_context_retriever_chain(st.session_state.vectorstore)

    response = retriever_chain.invoke({
        "input": user_question, 
        "chat_history": st.session_state.chat_history
    })
    return response['answer']


def main():
    st.set_page_config(page_title="A.L.E", page_icon=":page_with_curl:")

    st.header("Bem-vindo ao Assistente de Leitura de Editais (A.L.E)! :page_with_curl:") 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Olá! Sou A.L.E., aqui para ajudá-lo. Como posso ser útil hoje?")
        ]

    user_question = st.chat_input("Faça perguntas sobre seus editais")
    if user_question is not None and user_question !="":
        response = get_response(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))

  
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    with st.sidebar:
        st.subheader(":page_facing_up: Seus editais")
        pdf_docs = st.file_uploader("Carregue seu editais aqui e pressione em 'processar'", accept_multiple_files=True)


        if st.button(" :repeat: Processar"):
            with st.spinner("Processing"):
                # extrair o texto do pdf
                raw_text = get_pdf_text(pdf_docs)

                # pegar o texto e separar em partes(chunks)
                text_chunks = get_text_chunks(raw_text)

                # criar uma base de conhecimento de vetores(vector store)
                vectorstore = get_vectorstore(text_chunks)

                # criando uma cadeia(chain) de conversação
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever_chain = get_context_retriever_chain(vectorstore)
            st.success("Processamento concluído!")
        
            
if __name__ == '__main__':
    main()