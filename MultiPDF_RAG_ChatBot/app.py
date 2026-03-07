import streamlit as st
import tempfile
import os

from rag_pipeline import MultiPDFRAG

# Set API key from Streamlit secrets
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

st.set_page_config(
    page_title="Multi PDF Chatbot",
    page_icon="📚"
)

st.title("📚 Chat with Multiple PDFs")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    if "chatbot" not in st.session_state:

        pdf_paths = []

        # Save uploaded PDFs temporarily
        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

                tmp_file.write(uploaded_file.read())

                pdf_paths.append(tmp_file.name)

        with st.spinner("Processing PDFs..."):

            st.session_state.chatbot = MultiPDFRAG(pdf_paths)

        st.session_state.messages = []


    chatbot = st.session_state.chatbot


    # Display previous messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.write(msg["content"])


    user_question = st.chat_input("Ask a question about the PDFs")

    if user_question:

        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking..."):
            answer = chatbot.ask(user_question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
