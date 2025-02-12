from langchain_core.vectorstores.base import VectorStoreRetriever
# from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.schema import SystemMessage
from dotenv import load_dotenv
import os
# imports for RAG
from pathlib import Path

from rag import load_and_convert_document, get_markdown_splits, create_or_load_vector_store, build_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path, exceptions
from PIL import Image

from utils import ModelType, BackendType, EmbeddingType
# from utils import SYSTEM_TEMPLATE, SYSTEM_TEMPLATE_BR
from utils import Chatbot, parse_stream

load_dotenv()


# Path to vector DB folder
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
    
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
               height= 750, 
               width=800, 
               resolution_boost=5,
               )

def model_opts_component(backend_option):
    # pull down menu to select the model
    if backend_option == BackendType.LocalOllama.name:
        opts = (ModelType.Deepseek_r1_1dot5b_Distill_Qwen.name,
                ModelType.Deepseek_r1_8b_Distill_Llama.name,
                ModelType.Mistral_7b.name,
                ModelType.Phi4_14b.name,
                ModelType.Gemma2_9b.name,
                ModelType.TinyLlama_r1_limo.name)
    elif backend_option == BackendType.OnlineGroq.name:
        opts = (ModelType.Deepseek_r1_70b_Distill_Llama.name,
                ModelType.Llama_3dot3_70b_versatile.name)
    elif backend_option == BackendType.OnlineGoogle.name:
        opts = (ModelType.Gemini_2dot0_flash_lite.name, 
                ModelType.Gemini_2dot0_pro.name)
    elif backend_option == BackendType.OnlineMistral.name:
        opts = (
                ModelType.Mistral_small.name,
                ModelType.Mistral_nemo.name)
    else:
        opts = (ModelType.Sabia3_small.name, 
                ModelType.Sabia3_large.name)
    
    model_option = st.selectbox(
        "Selecione o modelo :bulb:",
        opts,
        index=0
    )
    return model_option

def model_opts_backend():
    # pull down menu to select the model
    backend_option = st.selectbox(
        "Selecione o backend :rocket:",
        (BackendType.LocalOllama.name,
         BackendType.OnlineMaritacaAI.name,
         BackendType.OnlineGroq.name,
         BackendType.OnlineGoogle.name,
         BackendType.OnlineMistral.name),  
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

# def generate_response(input_text):
#     
#     if backend_option == BackendType.OnlineMaritacaAI.name:
#         system_template = SYSTEM_TEMPLATE_BR
#     else:
#         system_template = SYSTEM_TEMPLATE
#
#     # Define the system and human message templates
#     human_template = "{input_text}"
#     
#     # Create the chat prompt template
#     chat_prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content=system_template),
#         HumanMessagePromptTemplate.from_template(human_template)
#     ])
#     
#     # Format the messages with the input text
#     messages = chat_prompt.format_messages(input_text=input_text)
#     
#
#     if backend_option == BackendType.OnlineMaritacaAI.name:
#         model = ChatOpenAI(
#             api_key=os.environ['MARITACA_API_KEY'], # type: ignore
#             base_url="https://chat.maritaca.ai/api",
#             model = str(ModelType.MaritacaAI.value)
#         )
#     
#     elif backend_option == BackendType.OnlineGroq.name:
#         model =ChatGroq(
#             #api_key=os.environ['GROQ_API_KEY'],
#             model = str(ModelType[model_option].value)
#         ) 
#     elif backend_option == BackendType.OnlineGoogle.name:
#         model = ChatGoogleGenerativeAI(
#             model = str(ModelType[model_option].value)
#         )
#     elif backend_option == BackendType.OnlineMistral:
#         model = ChatMistralAI(
#             model_name = str(ModelType[model_option].value)
#         )
#     else:
#         base_url = "http://localhost:11434/"
#         model = ChatOllama(model=ModelType[model_option].value, 
#                             base_url=base_url,
#                             num_thread=8)
#             
#     response = model.stream(messages)
#
#     return response
# def parse_stream(stream):
#     for chunk in stream:
#         yield (chunk.content.
#                 replace('$', '\\$').
#                 replace('<think>', '\n:brain:\n\n:green[\\<pensando\\>]\n').
#                 replace('</think>', '\n\n:green[\\</pensando\\>]\n\n---')
#                 )

def process_rag(selected_vector_db, embedding_option):

    vector_store = None

    # If 'Upload New Document' is selected, show the file uploader
    if selected_vector_db == "Carregar Novo Documento":
        uploaded_file = st.file_uploader("Carregar arquivo PDF para análise", type=["pdf"])
        

        # Process the uploaded PDF
        if uploaded_file:
            st.sidebar.subheader("PDF Carregado")
            st.sidebar.write(uploaded_file.name)
            
            # Save the PDF file temporarily and display it
            temp_path = f"temp_{uploaded_file.name}"
            document_binary = uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(document_binary)

            # Display PDF in the sidebar (show all pages)
            #display_pdf_in_sidebar(temp_path, uploaded_file.name.split('.')[0])
            #display_pdf_panel(document_binary)
            
            with st.sidebar:

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
    
    return vector_store
# ----------------------------------------------------------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------------------------------------------------------


#Session state variables
st.session_state.display_pdf_sidebar = False
st.session_state.display_pdf = False
st.session_state.show_rag_opts = False 
st.session_state.selected_vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []
if "vector_db_options" not in st.session_state:
    st.session_state.vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]    
    st.session_state.vector_db_options.append("Carregar Novo Documento")

st.set_page_config(page_icon="💬", 
                   page_title="Synapsis...",
                   layout= "wide"
                   )

with st.sidebar:
    st.header("🧠 SinapsisChat")    
    st.markdown("#### Desenvolvido por Carlos E. V. Neves")
    st.markdown("""
    ## Apresentação:
                
    - O app foi desenvolvido para fins didáticos...
    - O app faz ...
    - Os modelos disponíveis são...
    """)
    st.markdown("## Opções para chat:")
    backend_option = model_opts_backend()
    model_option = model_opts_component(backend_option)
    if st.button("Limpar Conversa", icon="🗑️"):
        st.session_state.messages = []
        
    st.divider()
    st.session_state.show_rag_opts = st.checkbox(":book: Conversar com a base de dados vetorial (RAG)", 
                                                 value=False)
    
    if st.session_state.show_rag_opts:

        st.markdown("## Opções para RAG:")
    
        #embedding_option = model_opts_embedding()
        embedding_option = EmbeddingType.OllamaEmbeddings.name
        st.session_state.embedding_option = embedding_option
        # Dropdown to select vector DB or upload a new document
        vector_db_options = st.session_state.vector_db_options
        
        selected_vector_db = st.selectbox("Selecione Documento para RAG ou Carregue Novo Documento", vector_db_options, index=0)
        st.session_state.selected_vector_db = selected_vector_db
        
        if selected_vector_db != "Carregar Novo Documento":
            check_display_pdf = st.checkbox("Visualizar Documento Selecionado", value=False)
            st.session_state.display_pdf = check_display_pdf

tab_gpt_like, = st.tabs(["🤖 :book: Chat"])

with tab_gpt_like:
    
    if st.session_state.display_pdf:    
        col1, col2 = st.columns([6, 4])
    else: 
        # Create two columns with different ratios
        col1, col2 = st.columns([9, 1])

    # Accept user input
    prompt = st.chat_input("Olá! Como posso ajudar você hoje?")

    with col1:
        with st.container(height=550, border=False, key="chat_container"):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                        
            with st.spinner("Gerando resposta..."):
                if prompt:
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        
                        if not st.session_state.show_rag_opts:
                            
                            model = Chatbot(backend_option, model_option)
                            stream = model.generate_response(prompt)
                            print(stream)
                            response = st.write_stream(parse_stream(stream))
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            selected_vector_db = st.session_state.selected_vector_db
                            embedding_option = st.session_state.embedding_option
                                                            
                            vector_store = process_rag(selected_vector_db, embedding_option)
                            
                            retriever: VectorStoreRetriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5}) # type: ignore

                            # Build and run the RAG chain
                            rag_chain = build_rag_chain(retriever,backend_option, model_option)
                            # Create a placeholder for streaming response
                            response_placeholder = st.empty()  # Create an empty placeholder for the answer

                            # Stream the response as it is generated
                            response = ""
                            for chunk in rag_chain.stream(prompt):
                                response += chunk  # Append each chunk of the response
                                response_placeholder.markdown(response.replace('$', '\\$')
                                                .replace('<think>', '\n:brain:\n\n:green[\\<pensando\\>]\n')
                                                .replace('</think>', '\n\n:green[\\</pensando\\>]\n\n---')
                                                )
                            st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.session_state.show_rag_opts: 
        selected_vector_db = st.session_state.selected_vector_db
        
        with st.sidebar:
            vector_store = process_rag(selected_vector_db, st.session_state.embedding_option)
    
        if st.session_state.display_pdf and selected_vector_db != "Carregar Novo Documento":     
            with col2:
                filename = f"{selected_vector_db}.pdf"
                filepath = f"vector_db/{filename}"

                st.write(f"Documento selecionado: **{filename}**")
                display_pdf_panel(filepath)
