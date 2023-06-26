import streamlit as st
import pickle
import os
import re
import time
import locale
#from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import pipeline
from InstructorEmbedding import INSTRUCTOR
from langchain import HuggingFacePipeline

locale.getpreferredencoding = lambda: "UTF-8"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ['OPENAI_API_KEY'] = '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'


# Sidebar contents
with st.sidebar:
  st.title(' LLM Chat App')
  st.markdown('''
   ### About
   This app is an LMM-powered chatbot built using:
   - [Streamlit]
   - [LangChain]
   - [OpenAI] LLM Model

   ''')
  add_vertical_space(5)
  st.write ('Made with Passion')

#load_dotenv()
def main():
  llm = OpenAI()
  #model = pipeline(
            task="text2text-generation",
            model = "lmsys/fastchat-t5-3b-v1.0",
            model_kwargs={"device_map": "auto", "max_length": 512, "temperature": 0})

  hf_llm = HuggingFacePipeline(pipeline=model) 

  st.header('Chat with the PDF')
  #upload a PDF file
  pdf = st.file_uploader("Uplaod your PDF", type='pdf')

  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    st.write(pdf.name)
    
    raw_text =''
    for i, page in enumerate(pdf_reader.pages):
      text = page.extract_text()
      if text:
        raw_text += text
    # Let's split the text into smaller chunck so we dont hit the token size limits during information retreival
    text_splitter = CharacterTextSplitter(separator = '\n',chunk_size = 1000,chunk_overlap = 0,length_function = len)
    chunks = text_splitter.split_text(raw_text)

    # embeddings
    store_name = pdf.name[:-4]

    if os.path.exists(f'{store_name}.pkl'):
      with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
      #st.write('Embeddings Loaded from the Disk')
    else :
      embeddings = OpenAIEmbeddings()
      #Instructor = "hkunlp/instructor-xl"
      #hf_embeddings = HuggingFaceInstructEmbeddings(model_name=Instructor, model_kwargs={"device": "cuda"})
      VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
      with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)
        
      #st.write('Embeddings Cumputation Completed')

    # Accept user questions/query
    query = st.text_input('Ask questions about your PDF file:')
    
    if query:
      #docs = VectorStore.similarity_search(query=query)

        
      #chain = RetrievalQA.from_chain_type(llm=hf_llm, 
                                    chain_type="stuff", 
                                    retriever=VectorStore.as_retriever(), 
                                    input_key="question")
      
      chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=VectorStore.as_retriever(), 
                                    input_key="question")
      #with get_openai_callback() as cb:
      response = chain.run(query)
      #  print(cb)
      st.write(response)    
    
    #st.write(text)
  
if __name__ == '__main__':
  main()
