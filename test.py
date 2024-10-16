import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo-1106')
chain = llm 
print(chain.invoke( "Who is Virat Kohli?"))

