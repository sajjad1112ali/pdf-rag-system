import streamlit as st

from rag import load_vector_db, retrieve_context, ask_llm_stream


st.title("📄 PDF Question Answering")

db = load_vector_db()

question = st.text_input("Ask a question about the PDF")

if question:

    context = retrieve_context(db, question)

    st.write("### Answer")

    response_area = st.empty()

    full_answer = ""

    for token in ask_llm_stream(question, context):

        full_answer += token

        response_area.markdown(full_answer)
