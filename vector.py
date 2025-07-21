def get_retriever():
    import os
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance

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

    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # Persistent local Qdrant instance (no SQLite used)
    client = QdrantClient(path="qdrant_data")

    # Ensure collection exists
    client.recreate_collection(
        collection_name="pdf_docs",
        vectors_config={"size": 1536, "distance": Distance.COSINE}
    )

    # Create vector store
    vectorstore = Qdrant(
        client=client,
        collection_name="pdf_docs",
        embeddings=embedding
    )

    # Add documents
    vectorstore.add_documents(split_docs)

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
