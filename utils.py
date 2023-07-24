from dotenv import load_dotenv

load_dotenv()

import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

import pandas as pd

__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join('/workspaces/contact-extraction', 'db.sqlite3'),
    }
}

def load_conversation(filename):

    with open(filename, 'r') as f:
        conversation = f.read()

    return conversation

def ask_gpt(prompt):
    llm = OpenAI()
    return llm.predict(prompt)

def qa(file_path, prompt):

    loader = TextLoader(file_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

    return qa.run(prompt)
