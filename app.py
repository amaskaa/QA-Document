import json
import os
import sys
import boto3
import streamlit as st
import pickle

from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

st.set_page_config(page_title="Chat PDF", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)




"""
Titan embedding model

"""
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


# Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock = boto3.client("bedrock-runtime", region_name='us-east-1',aws_access_key_id='AKIAZQ3DQVCULHVCJWYU',
                       aws_secret_access_key='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')
bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1",client = bedrock)

# Data Ingestion





def data_ingestion():

    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_spiltter = RecursiveCharacterTextSplitter(chunk_size = 10000,chunk_overlap=1000)

    docs = text_spiltter.split_documents(documents)

    return docs


# Vector Embedding

def get_vector_store(docs):
    vector_faiss = FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vector_faiss.save_local("faiss_index")


def get_claude_llm():
    ##create the Anthropic Model
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock,
                  model_kwargs={'maxTokens': 512})

    return llm


def get_llama2_llm():
    ##create the Anthropic Model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock,
                  model_kwargs={'max_gen_len': 512})

    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


#




def main():
    # st.set_page_config(page_title="Chat PDF", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # with st.sidebar:
    #     st.title("Update Or Create Vector Store:")
    #
    #     if st.button("Vectors Update"):
    #         with st.spinner("Processing..."):
    #             docs = data_ingestion()
    #             get_vector_store(docs)
    #             st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):

            docs = data_ingestion()
            get_vector_store(docs)

            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding,allow_dangerous_deserialization=True)

            llm = get_claude_llm()



            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):

            docs = data_ingestion()
            get_vector_store(docs)

            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding,allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":

    main()





