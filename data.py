from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
import os
import shutil
import tiktoken
from dotenv import load_dotenv


load_dotenv()

OPENAI_KEY = os.getenv("API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_KEY
)

tokenizer = tiktoken.get_encoding('cl100k_base')


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def load_docs():
    loader = PyPDFLoader("data/row_01.pdf")
    pages = loader.load_and_split()
    pages = pages[1:] # exlude first page
    print(f"loaded {len(pages)} pages.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    docs = text_splitter.split_documents(pages)
    print(f"loaded {len(docs)} docs.")
    db = FAISS.from_documents(docs, embeddings)
    if os.path.exists("local_vector_store/faiss_index"):
        shutil.rmtree("local_vector_store/faiss_index")
        print("Deleted existing vector store")
    db.save_local("local_vector_store/faiss_index")
    print("Saved vector store")


def view_docs():
    loader = PyPDFLoader("data/row_01.pdf")
    pages = loader.load_and_split()
    pages = pages[1:]
    with open("data/doc.txt", "w", encoding='utf-8') as file:
        file.write(pages[1].page_content)


def load_dict():
    loader = DirectoryLoader('data/manual', glob="**/*.txt", loader_cls=TextLoader)
    pages = loader.load()
    print(f"loaded {len(pages)} pages.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=30, length_function=tiktoken_len)
    docs = text_splitter.split_documents(pages)
    print(f"loaded {len(docs)} docs.")
    db = FAISS.from_documents(docs, embeddings)
    if os.path.exists("local_vector_store/faiss_index"):
        shutil.rmtree("local_vector_store/faiss_index")
        print("Deleted existing vector store")
    db.save_local("local_vector_store/faiss_index")
    print("Saved vector store")


#load_dict()