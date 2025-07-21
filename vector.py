def get_retriever():
    import os
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance

    pdf_dir = "./pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_paths = [os.path.join(pdf_dir, f)
                 for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    all_documents = []

    # for path in pdf_paths:
    #     loader = PDFPlumberLoader(path)
    #     docs = loader.load()
    #     all_documents.extend(docs)

    for path in pdf_paths:
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        for doc in docs:
              doc.metadata["source"] = os.path.basename(path)  # 🏷️ Add PDF filename
        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    split_docs = splitter.split_documents(all_documents)

    for doc in split_docs:
        if "Excel report contains" in doc.page_content or "What the Excel Report Contains" in doc.page_content:
            print(" Found correct chunk:", doc.page_content[:300])


    for chunk in split_docs:
        if "What the Excel Report Contains" in chunk.page_content:
            print("\n Found the Excel Report Section in Indexed Content:")
        print(chunk.page_content[:500])

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
