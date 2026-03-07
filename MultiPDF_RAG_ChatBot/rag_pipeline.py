from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings

from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain


class MultiPDFRAG:

    def __init__(self, pdf_paths):

        # Cohere embeddings
        self.embeddings = CohereEmbeddings(
            model="embed-english-v3.0"
        )

        # Cohere chat model
        self.llm = ChatCohere(
            model="command-r",
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

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )

        return chain


    def ask(self, question):

        result = self.chain.invoke({
            "question": question
        })

        return result["answer"]
