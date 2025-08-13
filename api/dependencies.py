import os
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_INDEX_PATH = os.getenv('VECTOR_INDEX_PATH', './vector_index')


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
