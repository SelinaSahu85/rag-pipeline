# -------------------- OFFLINE SAFETY --------------------
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -------------------- IMPORTS --------------------
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import jq

# -------------------- ENV --------------------
load_dotenv()

# -------------------- RAG PIPELINE --------------------
def create_rag_pipeline():
    try:
        # ✅ Load embeddings from LOCAL MODEL (offline-safe)
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        #     model_kwargs = {"local_files_only": True}
        )

        # ✅ Load or CREATE FAISS vector store
        if os.path.exists("faiss_store"):
            print("✅ Loading existing FAISS vector store...")
            faiss_store = FAISS.load_local(
                "faiss_store",
                embeddings=hf_embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("✅ FAISS store not found. Creating new FAISS index...")

            jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"

            json_loader = JSONLoader(
                file_path="data/chatdoctor5k.json",
                jq_schema=jq_schema,
                text_content=False
            )
            json_docs = json_loader.load()

            csv_loader = CSVLoader(file_path="data/format_dataset.csv")
            csv_docs = csv_loader.load()

            documents = csv_docs + json_docs

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=60,
                separators=["\n\n", "\n", " ", ".", ",", ";"]
            )

            chunks = splitter.split_documents(documents)

            faiss_store = FAISS.from_documents(
                documents=chunks,
                embedding=hf_embeddings
            )

            faiss_store.save_local("faiss_store")
            print("✅ FAISS vector store created and saved")

        # ✅ Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7,
            max_output_tokens=4000
        )

        # ✅ Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # ✅ Retriever
        retriever = faiss_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2}
        )

        # ✅ Prompt
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""
You are a medical consultant.
Use ONLY the provided dataset context.
If the answer is not present, reply with:
"The answer is not available in provided context."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
        )

        # ✅ Conversational RAG Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            output_key="answer"
        )

        print("✅ Medical RAG Pipeline initialized successfully")
        return qa_chain

    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        raise
