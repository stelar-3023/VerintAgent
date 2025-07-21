def get_retriever():
    import os
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

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

    vectorstore = Chroma(
        collection_name="pdf_docs",
        embedding_function=embedding
    )

    vectorstore.add_documents(split_docs)

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
