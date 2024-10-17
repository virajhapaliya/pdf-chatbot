# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from chromadb.config import Settings

from htmlTemplates import css, bot_template, user_template
import os
# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
os.environ["OPENAI_API_KEY"] = "API KEY HERE"


CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=faiss.FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

def get_vectorstore_crhoma(chunks):
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    # Initialize ChromaDB as vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="my_collection",  # Specify a name for the collection
        persist_directory="chroma_db",  # Directory to persist the database (optional, can be removed if no persistence is needed)
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",  # Backend implementation for Chroma
            persist_directory="chroma_db"  # Ensure consistency with above
        )
    )

    # Add texts (chunks) to the vector store
    vectorstore.add_texts(texts=chunks)

    return vectorstore
# generating conversation chain  
def get_conversationchain(vectorstore):
    llm=ChatOpenAI(model='gpt-4o-mini',temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    
    st.header("Chat with multiple PDFs :books:")
    question=st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        #get the pdf
        # pdfs = [ 'PDFs\Annual-Report-for-the-Financial-Year-2023-2024.pdf',
        #         'PDFs\tata-motor-IAR-2023-24.pdf',
        #         'PDFs\tata-steel-limited-ir-2024.pdf']
        # raw_text=get_pdf_text(pdfs)
        
        # #get the text chunks
        # text_chunks=get_chunks(raw_text)
        
        # #create vectorstore
        # vectorstore=get_vectorstore(text_chunks)
        
        # #create conversation chain
        # st.session_state.conversation=get_conversationchain(vectorstore)
        print("Done")
        print("docs",docs)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                #get the pdf
                raw_text=get_pdf_text(docs)
                
                #get the text chunks
                text_chunks=get_chunks(raw_text)
                
                #create vectorstore
                # vectorstore=get_vectorstore(text_chunks)
                vectorstore=get_vectorstore_crhoma(text_chunks)
                
                #create conversation chain
                st.session_state.conversation=get_conversationchain(vectorstore)
                print("DONE")

if __name__ == '__main__':
    main()
