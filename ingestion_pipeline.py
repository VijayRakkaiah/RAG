import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain_chroma import Chroma

load_dotenv()

def load_documents(docs_path = "docs"):
    """Load all text files from the docs directory"""
    print(f"loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} does not exist. please create it and add your company files.")

    # load all .txt files from the doc directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Content Length: {len(doc.page_content)} characters")
        print(f"    Content Preview: {doc.page_content[:100]}...")
        print(f"    metadata: {doc.metadata}")

    return documents


def main():
    print("main function")

    # 1. Load the files
    print(load_documents(docs_path="docs"))



if __name__ == "__main__":
    main()