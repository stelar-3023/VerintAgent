import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever

# Load vector retriever and LLM
retriever = get_retriever()
model = OllamaLLM(model="mistral")

# Configure Streamlit
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDFs")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that strictly answers questions using the provided context only.

<context>
{answers}
</context>
                                          
Use only the above content. Do not make up any information.
Answer the following question:
{question}
""")

# Create chain
chain = prompt | model

# Question input (no while loop!)
question = st.text_input("Ask a question about Verint:",
                         key="unique_question_key")

# Only run inference if a question is entered
if question:
    with st.spinner("Searching PDFs and generating answer..."):
        answers = retriever.invoke(question)
        # DEBUG: Show raw context
        st.markdown("### 🔍 Retrieved Chunks")
        for doc in answers:
            # truncate for readability
            st.markdown(f"> {doc.page_content[:500]}...")
        result = chain.invoke({"answers": answers, "question": question})
        st.markdown("### Answer")
        st.write(result)
