from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from groq import Groq
from langchain_ollama import ChatOllama
import openai
from dotenv import load_dotenv
import os
import re
from typing import Generator
# imports for RAG
from pathlib import Path

from rag import load_and_convert_document, get_markdown_splits, create_or_load_vector_store, build_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path, exceptions
from PIL import Image

from utils import ModelType, BackendType, EmbeddingType


load_dotenv()


# Path to vector DB folder
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
    
def generate_chat_stream(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Function to display PDF content as images in the sidebar
def display_pdf_in_sidebar(pdf_path, file_name):
    try:
        images_folder = Path(VECTOR_DB_FOLDER) / file_name / "images"
        os.makedirs(images_folder, exist_ok=True)

        # Check if images already exist
        image_paths = list(images_folder.glob("*.png"))
        if image_paths:
            # If images exist, display them
            for img_path in image_paths:
                image = Image.open(img_path)
                st.sidebar.image(image, caption=f"Page {image_paths.index(img_path) + 1}", use_container_width=True)
        else:
            # Convert PDF to images (one per page)
            images = convert_from_path(pdf_path)  # This will render all pages by default
            for i, image in enumerate(images):
                img_path = images_folder / f"page_{i + 1}.png"
                image.save(img_path, "PNG")  # Save image to disk
                st.sidebar.image(image, caption=f"Page {i + 1}", use_container_width=True)

    except exceptions.PDFPageCountError:
        st.sidebar.error("Error: Unable to get page count. The PDF may be corrupted or empty.")
    except exceptions.PDFSyntaxError:
        st.sidebar.error("Error: PDF syntax is invalid or the document is corrupted.")
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")
        
def display_pdf_panel(file_name):
    pdf_viewer(file_name, 
               height= 800, 
               width=800, 
               resolution_boost=5,
               )

def model_opts_component(backend_option):
    # pull down menu to select the model
    if backend_option == BackendType.LocalOllama.name:
        opts = (ModelType.Deepseek_r1_1dot5b_Distill_Qwen.name,
                ModelType.Deepseek_r1_8b_Distill_Llama.name,
                ModelType.Mistral_7b.name)
    elif backend_option == BackendType.OnlineGroq.name:
        opts = (ModelType.Deepseek_r1_70b_Distill_Llama.name,
                ModelType.Llama_3dot3_70b_versatile.name)
    else:
        opts = (ModelType.MaritacaAI.name)
    
    model_option = st.selectbox(
        "Selecione o modelo",
        opts,
        index=0
    )
    return model_option

def model_opts_backend():
    # pull down menu to select the model
    backend_option = st.selectbox(
        "Selecione o backend",
        (BackendType.LocalOllama.name,
         BackendType.OnlineMaritacaAI.name,
         BackendType.OnlineGroq.name),  
        index=0
        
    )
    return backend_option
def model_opts_embedding():
    embedding_option = st.selectbox(
        "Selecione o embedding",
        (EmbeddingType.OllamaEmbeddings.name,
         EmbeddingType.MistralEmbeddings.name),  
        index=0
    )
    return embedding_option
    
    
def write_simple_response(text):
    
    st.markdown("""
    :robot_face: ...resposta:

    """)

    if type(text) is str:
        st.write(text)
    else:
        st.write_stream(text)

def write_thought_response(text):

    with st.spinner("Pensando..."):
        match = re.search(r"<think>(.*?)</think>\s*", text, flags=re.DOTALL)

        if match:
            think = "\n" + match.group(1).strip() + "\n"  # Store the content inside <think> in a variable
            remaining_text = text[match.end():].strip()  # Store the rest of the text
        else:
            think = ""
            remaining_text = "\n" + text + "\n" 

    st.markdown(f"""
    :brain: ...pensamento:

    {think}                              
    
    """)
    
    st.divider()

    st.markdown(f"""

    :robot_face: ...resposta: 

    {remaining_text}
    
    """)


# ----------------------------------------------------------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_icon="💬", 
                   page_title="Chat...",
                   layout= "centered"
                   )
st.title("🧠 Exemplos de Aplicações de LLMs")

backend_option = model_opts_backend()
model_option = model_opts_component(backend_option)

st.divider()

tab_simple_chatbot, tab_chat_rag = st.tabs(["🤖 Chatbot Simples", "🤖 :book: Chatbot com RAG"])

st.session_state.display_pdf_sidebar = False

with tab_simple_chatbot:
    
    with st.form("llm-form"):
        text = st.text_area("Em que posso ajudar?", placeholder="Qual era a capital do Brasil em 1921?")
        submit = st.form_submit_button("Enviar")

    def generate_response(input_text):
        if model_option == ModelType.MaritacaAI.name:
            client = openai.OpenAI(
            api_key=os.environ['MARITACA_API_KEY'],
            base_url="https://chat.maritaca.ai/api",
            )
            response = client.chat.completions.create(
            model=ModelType.MaritacaAI.value,
            messages=[
                {"role": "user", "content": input_text},
            ] ,
                max_tokens=1000,
                stream=True,
            
            )
            return response 
        else:
            if backend_option == BackendType.OnlineGroq.name:
                client = Groq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                )
                if model_option == ModelType.Deepseek_r1_70b_Distill_Llama.name:
                    model = ModelType.Deepseek_r1_70b_Distill_Llama.value
                    stream = False 
                else:
                    model = ModelType.Llama_3dot3_70b_versatile.value
                    stream = True
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user","content": input_text,

                        }
                    ],
                    max_tokens=1000,
                    stream=stream,
                    model=model,
                )
                return response
            else:
                base_url = "http://localhost:11434/"
                if model_option == ModelType.Deepseek_r1_1dot5b_Distill_Qwen.name:
                    model = ChatOllama(model=ModelType.Deepseek_r1_1dot5b_Distill_Qwen.value, base_url=base_url)
                elif model_option == ModelType.Deepseek_r1_8b_Distill_Llama.name: 
                    model = ChatOllama(model=ModelType.Deepseek_r1_8b_Distill_Llama.value, base_url=base_url)
                else:
                    model = ChatOllama(model=ModelType.Mistral_7b.value, base_url=base_url)

                response = model.invoke(input_text)

            return response.content

    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []

    if submit and text:
        with st.spinner("Gerando resposta..."):
            
            if model_option == ModelType.MaritacaAI.name:
                response = generate_response(text)
                st.session_state['chat_history'].append({"user": text, "assistant": response})
                write_simple_response(response) 

            elif backend_option == BackendType.OnlineGroq.name:

                if model_option == ModelType.Deepseek_r1_70b_Distill_Llama.name:
                    response = generate_response(text)

                    response = response.choices[0].message.content # type: ignore
                    write_thought_response(response) 
                else:    
                    response = generate_response(text)
                    response = generate_chat_stream(response)
                    write_simple_response(response)

                st.session_state['chat_history'].append({"user": text, "assistant": response})
            else:
                response = generate_response(text)
                
                st.session_state['chat_history'].append({"user": text, "assistant": response})

                if model_option in (ModelType.Deepseek_r1_1dot5b_Distill_Qwen.name, ModelType.Deepseek_r1_8b_Distill_Llama.name):
                    write_thought_response(response)
                else:
                    write_simple_response(response)

    st.write("## Histórico das converas")
    with st.expander("Expandir"):
        for chat in reversed(st.session_state['chat_history']):
            st.write(f"**🧑 Usuário**: {chat['user']}")
            st.write(f"**:robot_face: Assistente**: {chat['assistant']}")
            st.write("---")

with tab_chat_rag:
    embedding_option = model_opts_embedding()
    # Dropdown to select vector DB or upload a new document
    vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]
    vector_db_options.append("Carregar Novo Documento")  # Add option to upload a new document
    selected_vector_db = st.selectbox("Selecione a Base Vetorial ou Carregue Novo Documento", vector_db_options, index=0)
    
    check_display_pdf = st.checkbox("Mostrar PDF na barra lateral", value=False)
    st.session_state.display_pdf_sidebar = check_display_pdf

    # If 'Upload New Document' is selected, show the file uploader
    if selected_vector_db == "Carregar Novo Documento":
        uploaded_file = st.file_uploader("Carregar arquivo PDF para análise", type=["pdf"])
        

        # Process the uploaded PDF
        if uploaded_file:
            # st.sidebar.subheader("PDF Carregado")
            # st.sidebar.write(uploaded_file.name)
            st.success(f"PDF Carregado\n {uploaded_file.name}")

            # Save the PDF file temporarily and display it
            temp_path = f"temp_{uploaded_file.name}"
            document_binary = uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(document_binary)

            # Display PDF in the sidebar (show all pages)
            #display_pdf_in_sidebar(temp_path, uploaded_file.name.split('.')[0])
            display_pdf_panel(document_binary)

            # PDF processing button
            if st.button("Processar o PDF e Armazenar na Base Vetorial"):
                with st.spinner("Processando documento..."):
                    # Convert PDF to markdown directly
                    markdown_content = load_and_convert_document(temp_path)
                    chunks = get_markdown_splits(markdown_content)

                    # Initialize embeddings
                    if embedding_option == EmbeddingType.MistralEmbeddings.name:
                        embeddings = MistralAIEmbeddings(                    
                            model=EmbeddingType.MistralEmbeddings.value
                        )
                    else:    
                        embeddings = OllamaEmbeddings(
                            model=EmbeddingType.OllamaEmbeddings.value, 
                            base_url="http://localhost:11434")

                    # Create or load vector DB and store PDF along with it
                    vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings)

                    # Ensure vector DB and PDF are stored correctly
                    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name.split('.')[0]}.faiss"
                    vector_store.save_local(str(vector_db_path))  # Save FAISS vector store

                    # Store the PDF file alongside the vector DB
                    pdf_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(document_binary)

                    st.success("PDF processado e armazenado no banco de dados vetorial.")

                    # Clean up the temporary file
                    Path(temp_path).unlink()

    elif selected_vector_db != "Carregar Novo Documento":
        # Load the selected vector DB
        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
        if vector_db_path.exists():


            # Initialize embeddings
            if embedding_option == EmbeddingType.MistralEmbeddings.name:
                embeddings = MistralAIEmbeddings(                    
                    model=EmbeddingType.MistralEmbeddings.value
                )
            else:    
                embeddings = OllamaEmbeddings(
                    model=EmbeddingType.OllamaEmbeddings.value, 
                    base_url="http://localhost:11434")
            
            vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

            # Display PDF in the sidebar
            pdf_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.pdf"
            if pdf_path.exists():
                if st.session_state.display_pdf_sidebar:                
                    display_pdf_in_sidebar(pdf_path, selected_vector_db)
                    #display_pdf_panel(selected_vector_db)
            else:
                st.sidebar.warning("Arquivo PDF não encontrado na Base Vetorial selecionada.")
        else:
            st.sidebar.warning(f"Base Vetorial '{selected_vector_db}' não encontrada.")

    # Question input section
    question = st.text_input("Entre sua pergunta:", placeholder="por exemplo, O que é o cartel em licitação de acordo com a  Lei de Concorrência?")
    
    if "chat_history_rag" not in st.session_state:
        st.session_state['chat_history_rag'] = []
        
    # Button to process and generate answers
    if st.button("Enviar") and question and selected_vector_db != "Carregar Novo Documento":
        with st.spinner("Respondendo sua questão..."):
            
            # Build retriever from the selected vector store
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5}) # type: ignore

            # Build and run the RAG chain
            rag_chain = build_rag_chain(retriever, model_option)

            # Create a placeholder for streaming response
            response_placeholder = st.empty()  # Create an empty placeholder for the answer

            # Stream the response as it is generated
            response = ""
            for chunk in rag_chain.stream(question):
                response += chunk  # Append each chunk of the response
                response_placeholder.markdown(response.replace('$', '\\$').replace('<think>', ':green[\\<pensando\\>]\n').replace('</think>', ':green[\\</pensando\\>]\n'))  # Update the placeholder with the new response

            st.session_state['chat_history_rag'].append({"user": question, "assistant": response})
    
    st.write("## Histórico das converas")
    with st.expander("Expandir"):
        for chat in reversed(st.session_state['chat_history_rag']):
            st.write(f"**🧑 Usuário**: {chat['user']}")
            st.write(f"**:robot_face: Assistente**: {chat['assistant']}")
            st.write("---")