from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
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

def generate_response(input_text):
    
    # Define the system and human message templates
    system_template = """
    You are a helpful AI assistant. 
    
    Your responses should be:
    - Clear and concise
    - Accurate and well-researched
    - Professional in tone
    - Helpful and solution-oriented
    
    Additional notes:
    - Use emojis.
    - You can think in english, but always show your chain of thoughts in brazilian portuguese.
    - Always answer in brazilian portuguese.
    """
    
    human_template = "{input_text}"
    
    # Create the chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    # Format the messages with the input text
    messages = chat_prompt.format_messages(input_text=input_text)
    

    if backend_option == BackendType.OnlineMaritacaAI.name:
        model = ChatOpenAI(
            api_key=os.environ['MARITACA_API_KEY'], # type: ignore
            base_url="https://chat.maritaca.ai/api",
            model = str(ModelType.MaritacaAI.value)
        )
    
    elif backend_option == BackendType.OnlineGroq.name:
        model =ChatGroq(
            #api_key=os.environ['GROQ_API_KEY'],
            model = str(ModelType[model_option].value)
        ) 
    else:
        base_url = "http://localhost:11434/"
        model = ChatOllama(model=ModelType[model_option].value, 
                            base_url=base_url,
                            num_thread=8)
            
    response = model.stream(messages)

    return response
def parse_stream(stream):
    for chunk in stream:
        yield (chunk.content.
                replace('$', '\\$').
                replace('<think>', '\n:brain:\n\n:green[\\<pensando\\>]\n').
                replace('</think>', '\n\n:green[\\</pensando\\>]\n\n---')
                )
        

# ----------------------------------------------------------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_icon="💬", 
                   page_title="Chat...",
                   layout= "centered"
                   )

with st.sidebar:
    st.header("🧠 Exemplos de Aplicações de LLMs")    
    st.markdown("""
    ## Apresentação:
                
    - O app foi desenvolvido para fins didáticos...
    - O app faz ...
    - Os modelos disponíveis são...
    """)
    st.markdown("## Opções:")
    backend_option = model_opts_backend()
    model_option = model_opts_component(backend_option)
    if st.button("Limpar Conversa", icon="🗑️"):
        st.session_state.messages = []
        
    st.divider()

tab_gpt_like, tab_chat_rag = st.tabs(["🤖 Clone GPT","🤖 :book: Chatbot com RAG"])

st.session_state.display_pdf_sidebar = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with tab_gpt_like:
    
    # Create two columns with different ratios
    col1, col2 = st.columns([9, 1])
    
    # Accept user input
    prompt = st.chat_input("Olá! Como posso ajudar você hoje?")

    with col1:
        with st.container(height=500, border=False):
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

                        stream = generate_response(prompt)
                        response = st.write_stream(parse_stream(stream))
                        
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

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