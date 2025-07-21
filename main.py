import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever, get_available_sources

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDFs")

# Sidebar PDF filter
available_sources = get_available_sources()
selected_sources = st.sidebar.multiselect(
    "Filter by PDF source:", available_sources, default=available_sources
)

# Keyword filter input
custom_keywords = st.sidebar.text_input(
    "Only index chunks containing these keywords (comma-separated):",
    value=""
)
keyword_list = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()]

# Load retriever with filters
retriever = get_retriever(
    filter_sources=selected_sources, keywords=keyword_list)

# LLM setup
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that strictly answers questions using the provided context only.

<context>
{answers}
</context>

If the answer is not in the context, respond: "The answer is not in the provided documents."

Answer this question:
{question}
""")

chain = prompt | model

question = st.text_input(
    "Ask a question about the documents:", key="question_input")

if question:
    with st.spinner("Searching and generating answer..."):
        answers = retriever.invoke(question)

        st.markdown("### Retrieved Context")
        for i, doc in enumerate(answers):
            st.markdown(
                f"**Chunk {i+1} — `{doc.metadata.get('source', 'unknown')}`**")
            st.text(doc.page_content[:700])
            

        context_text = "\n\n".join([doc.page_content for doc in answers])
        result = chain.invoke({"answers": answers, "question": question})
        st.markdown("### Answer")
        st.write(result)
