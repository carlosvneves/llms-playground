import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

import faiss

from utils import ModelType
from utils import PROMPT_BR, PROMPT

from dotenv import load_dotenv

load_dotenv()

# Ensure the vector_db folder exists
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Load and convert PDF to markdown content
def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# Split markdown into chunks
def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return splitter.split_text(markdown_content)


# Create or load the vector store
def create_or_load_vector_store(filename, chunks, embeddings):
    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{filename}.faiss"

    if vector_db_path.exists():
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        single_vector = embeddings.embed_query("initialize")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(chunks)
        vector_store.save_local(str(vector_db_path))
    return vector_store


# Build RAG chain
def build_rag_chain(retriever, model_option):
 

    prompt = PROMPT
    # Handle different model backends
    if model_option in (ModelType.Deepseek_r1_8b_Distill_Llama.name, 
                       ModelType.Deepseek_r1_1dot5b_Distill_Qwen.name, 
                       ModelType.Mistral_7b.name):
        # Ollama backend
        llm = ChatOllama(
            model = str(ModelType[model_option].value),
            base_url="http://localhost:11434"
        )
        
    elif model_option == ModelType.MaritacaAI.name:
        prompt = PROMPT_BR
        
        # MaritacaAI backend
        llm = ChatOpenAI(
            api_key=os.environ['MARITACA_API_KEY'], # type: ignore
            base_url="https://chat.maritaca.ai/api",
            model = str(ModelType.MaritacaAI.value)
        )
    elif model_option in (ModelType.Gemini_2dot0_flash_lite.name, 
                          ModelType.Gemini_2dot0_pro.name):
        llm = ChatGoogleGenerativeAI(
            model = str(ModelType[model_option].value)
        )
    elif model_option in (ModelType.Mistral_large.name, ModelType.Mistral_small.name):
        llm = ChatMistralAI(
            model_name = str(ModelType[model_option].value)
        )
    else:
        # Groq backend
        llm = ChatGroq(
            #api_key= os.environ['GROQ_API_KEY'], # type: ignore
            model = str(ModelType[model_option].value) 
        )

    prompt_template = ChatPromptTemplate.from_template(prompt)
    return (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
        "question": RunnablePassthrough()}
        | prompt_template
        | llm 
        | StrOutputParser()
    ) 