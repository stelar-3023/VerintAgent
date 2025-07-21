import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDFs")

retriever = get_retriever()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or gpt-4
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
        for doc in answers:
            source = doc.metadata.get("source", "unknown")
            st.markdown(f"**From:** `{source}`")
            st.markdown(f"> {doc.page_content[:500]}...")

        result = chain.invoke({"answers": answers, "question": question})
        st.markdown("### Answer")
        st.write(result)
