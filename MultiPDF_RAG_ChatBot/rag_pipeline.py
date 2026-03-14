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

        # Load PDFs and add metadata
        for pdf in pdf_paths:

            loader = PyPDFLoader(pdf)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf

            documents.extend(pages)

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create Vector store
        vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 12}
        )

        # Prompt template
        template = """
You are an AI assistant answering questions from documents.
Guidelines:
- Always answer in English.
- Use ONLY the provided context.
- Never switch to another language.
- If the context is in another language, translate it to English in your answer.
- If the question asks for summarization, comparison, similarities, or differences,
  analyze information from all relevant documents.
Context:
{context}

Question:
{question}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Conversational RAG chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
            output_key="answer"
        )

        return chain


    def ask(self, question):

        result = self.chain.invoke({
            "question": question
        })

        answer = result["answer"]

        # Get source PDFs
        sources = list(set(
            doc.metadata["source"]
            for doc in result["source_documents"]
        ))

        source_text = "\n\nSources:\n" + "\n".join(sources)

        return answer + source_text
