from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain

from langchain_core.prompts import PromptTemplate


class MultiPDFRAG:

    def __init__(self, pdf_paths):

        # Cohere embeddings
        self.embeddings = CohereEmbeddings(
            model="embed-english-v3.0"
        )

        # Cohere chat model
        self.llm = ChatCohere(
            model="command-a-03-2025",
            temperature=0
        )

        # Conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )

        # Build RAG chain
        self.chain = self.build_chain(pdf_paths)


    def build_chain(self, pdf_paths):

        documents = []

        # Load PDFs
        for pdf in pdf_paths:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        template = """
You are an AI assistant answering questions from documents.

Use ONLY the provided context to answer the question.

Rules:
- Always respond in English
- Never switch to another language.
- If the context is in another language, translate it to English in your answer.
- If the user asks for key points or a summary, provide the answer by summarizing the whole document and give key points.
- If the question asks for explanation, give a short explanation followed by key ideas.
- If the user asks to compare between the uploaded documents, give answers by analyzing the source documents provided.
- If the answer is not in the context, say "I don't know"

Context:
{context}

Question:
{question}
"""

        prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return chain


    def ask(self, question):

        result = self.chain.invoke({
            "question": f"Answer in English: {question}"
        })

        return result["answer"]

