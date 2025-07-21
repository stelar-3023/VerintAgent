def get_retriever(filter_sources=None):
    import os
    from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue

    pdf_dir = "./pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_paths = [os.path.join(pdf_dir, f)
                 for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    all_documents = []

    for path in pdf_paths:
        loader = PDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(all_documents)

    # Debug: Check for key phrases
    for doc in split_docs:
        if "What the Excel Report Contains" in doc.page_content:
            print("Found Excel section:", doc.page_content[:300])

    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    client = QdrantClient(path="qdrant_data")

    client.recreate_collection(
        collection_name="pdf_docs",
        vectors_config={"size": 1536, "distance": Distance.COSINE}
    )

    vectorstore = Qdrant(
        client=client,
        collection_name="pdf_docs",
        embeddings=embedding
    )

    vectorstore.add_documents(split_docs)

    #  Optional filtering by source (PDF filename)
    retriever_filter = None
    if filter_sources:
        retriever_filter = Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source))
                for source in filter_sources
            ]
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15, "filter": retriever_filter}
    )

#  List all available sources for sidebar


def get_available_sources():
    pdf_dir = "./pdfs"
    return [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
