import os
import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_pdfs(pdf_dir: str, index_path: str):
    pdf_dir = pathlib.Path(pdf_dir)
    index_path = pathlib.Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    docs = []
    if not pdf_dir.exists():
        raise FileNotFoundError("This directory does not exists")

    for fname in pdf_dir.iterdir():
        if fname.suffix.lower() != ".pdf":
            continue
        loader = PyPDFLoader(str(fname))
        pages = loader.load_and_split()
        docs.extend(pages)

    if not docs:
        raise RuntimeError("No PDF found")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(split_docs, embeddings)
    faiss_index.save_local(str(index_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ingest PDFs into vector store')
    parser.add_argument('--pdf_dir', type=str, default='./data', help='Directory with PDF files')
    parser.add_argument('--index_path', type=str, default='./vector_index', help='Output index path')
    args = parser.parse_args()
    ingest_pdfs(args.pdf_dir, args.index_path)
