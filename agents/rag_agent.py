import os
from functools import lru_cache
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_INDEX_PATH = os.getenv('VECTOR_INDEX_PATH', './vector_index')
@lru_cache(maxsize=1)
def _get_vector_store():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

@lru_cache(maxsize=1)
def _get_qa_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = _get_vector_store().as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def answer_from_pdfs(question: str) -> dict:
    qa = _get_qa_chain()
    res = qa.invoke({"query": question})
    docs = res.get('source_documents', []) or []
    sources = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get('source') or meta.get('file_path') or 'unknown'
        page = meta.get('page')
        if page is not None:
            src = f"{src}#page={page+1}"
        sources.append(src)
    return {"answer": res.get('result', ''), "sources": sources, "hits": len(docs)}