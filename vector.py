import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def get_retriever():
    pdf_dir = "./pdfs"
    pdf_paths = [os.path.join(pdf_dir, f)
                 for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    all_documents = []

    for path in pdf_paths:
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(all_documents)

    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = "./chroma_langchain_db"

    vectorstore = Chroma(
        collection_name="pdf_docs",
        persist_directory=db_location,
        embedding_function=embedding
    )

    #  Only embed if we haven’t already
    if not os.path.exists(db_location + "/index.lock"):
        print("Building new vector index...")
        vectorstore.add_documents(split_docs)
        open(db_location + "/index.lock", "w").close()
    else:
        print("Using existing vector index...")

    #  DEBUG: Show some export-related chunks
    for i, doc in enumerate(split_docs):
        content = doc.page_content.lower()
        if "export" in content and "allocation" in content:
            print(f"\n MATCHED CHUNK {i+1}:\n{doc.page_content[:500]}...\n")

    #  Use MMR for more diverse, relevant results
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
