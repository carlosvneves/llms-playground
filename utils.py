from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
import os 


from enum import Enum

class ModelType(Enum):
    Sabia3_small = "sabiazinho-3"
    Sabia3_large = "sabia-3"
    Deepseek_r1_1dot5b_Distill_Qwen = "deepseek-r1:1.5b"
    Deepseek_r1_8b_Distill_Llama = "deepseek-r1:8b"
    Mistral_7b = "mistral:latest"
    Deepseek_r1_70b_Distill_Llama = "deepseek-r1-distill-llama-70b"
    Llama_3dot3_70b_versatile = "llama-3.3-70b-versatile"
    Gemini_2dot0_flash_lite = "gemini-2.0-flash-lite-preview-02-05"
    Gemini_2dot0_pro = "gemini-2.0-pro-exp-02-05"
    Gemini_2dot0_flash_thinking = "gemini-2.0-flash-thinking-exp-01-21"
    Mistral_small = "mistral-small-latest"
    Mistral_nemo = "open-mistral-nemo"
    Gemma2_9b = "gemma2:9b"
    Phi4_14b = "phi4"
    TinyLlama_r1_limo = "hf.co/mradermacher/TinyLlama-R1-LIMO-GGUF:F16"

class BackendType(Enum):
    LocalOllama = "ollama"
    OnlineGroq = "groq"
    OnlineMaritacaAI = "maritaca"
    OnlineGoogle= "google"
    OnlineMistral = "mistral"

class EmbeddingType(Enum):
    MistralEmbeddings = 'mistral-embed'
    OllamaEmbeddings = 'nomic-embed-text'

class Chatbot:

    
    def __init__(self, backend_option, model_option, max_tokens=8152, max_retries=3, temperature=0.2):
        self.backend_option = backend_option 
        self.model_option = model_option 
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.temperature = temperature
        # self.llm:ChatOpenAI|ChatGroq|ChatOllama|ChatGoogleGenerativeAI|ChatMistralAI = None

        backend_option = self.backend_option
        model_option = self.model_option


        try:
            match(BackendType[backend_option]):

                case BackendType.LocalOllama:
                    model = ChatOllama(
                        base_url="http://localhost:11434/",
                        model=str(ModelType[model_option].value),
                        temperature=self.temperature,
                        num_thread=8,
                        num_predict=20000,
                        top_p=1
                    )
                case BackendType.OnlineGroq:
                    model =ChatGroq(
                        model = str(ModelType[model_option].value),
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        max_retries=self.max_retries,
                        n=1,
                    )
                case BackendType.OnlineMaritacaAI:
                    model = ChatOpenAI(
                        api_key=os.environ['MARITACA_API_KEY'], # type: ignore
                        base_url="https://chat.maritaca.ai/api",
                        model = str(ModelType[model_option].value),
                        max_completion_tokens=self.max_tokens,
                        max_retries=self.max_retries,
                        temperature=self.temperature,
                        n=1,
                        top_p=1
                    )
                case BackendType.OnlineGoogle:
                    model = ChatGoogleGenerativeAI(
                        model = str(ModelType[model_option].value),
                        max_retries=self.max_retries,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=1,
                        n=1,
                    )
                case BackendType.OnlineMistral:
                    model = ChatMistralAI(
                        model_name = str(ModelType[model_option].value),
                        max_retries=self.max_retries,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=1,
                    )

            self.llm = model

            print(self.llm)

        except Exception as e:
            print(f"Error: {str(e)}")



    def generate_response(self, context):
        response = None 
        model = self.llm 

        if self.backend_option == BackendType.OnlineMaritacaAI.name:
            system_template = SYSTEM_TEMPLATE_BR
        else:
            system_template = SYSTEM_TEMPLATE

        # Define the system and human message templates
        human_template = "{input_text}"
        
        # Create the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Format the messages with the input text
        messages = chat_prompt.format_messages(input_text=context)

        response = model.stream(messages)

        return response


def parse_stream(stream):
    for chunk in stream:
        yield (chunk.content.
                replace('$', '\\$').
                replace('<think>', '\n:brain:\n\n:green[\\<pensando\\>]\n').
                replace('</think>', '\n\n:green[\\</pensando\\>]\n\n---')
                )

PROMPT_BR = """
            Persona: Você é um Advogado especializado em Direito da Concorrência e Análise Econômica do Direito, com profundo conhecimento das normas, guias e decisões do Conselho Administrativo de Defesa Econômica (CADE).

            Contexto: Os documentos em questão estabelecem orientações e diretrizes relativas à política de concorrência, aos procedimentos institucionais e contêm explicações detalhadas sobre a legislação vigente; ou ainda, os documentos podem se referir a um ofício ou outro tipo de documento similar.

            Instruções:

                - Utilize exclusivamente o contexto recuperado: Ao responder, baseie sua resposta apenas nas informações contidas no contexto fornecido.
                - Idioma: Todas as respostas devem ser formuladas em português do Brasil.
                - Respostas ausentes: Se o contexto não contiver a informação necessária para responder à pergunta, informe explicitamente que não há dados suficientes para elaborar uma resposta.
                - Clareza e precisão: As respostas devem ser claras, objetivas e referenciar as informações disponíveis no contexto, sem adicionar dados ou interpretações que não estejam presentes no material.
                - Formatação das Referências: Ao final de cada resposta, inclua todas as referências no formato de notas de rodapé.
                Cada nota de rodapé deve conter o nome completo do arquivo a partir do qual a informação foi extraída. As seções 
                relevantes e subseções devem ser citadas no formato de notas de rodapé, com o nome completo do arquivo a partir do qual a informação foi extraída. 
                As referências devem ser numeradas sequencialmente e organizadas de forma clara e sistemática.
                - Caso a informação solicitada não esteja presente no contexto, utilize a seguinte resposta padrão:
                    'Não foi possível encontrar a informação necessária no contexto fornecido para responder a esta pergunta.'
            
            Estrutura para a execução do RAG:

            \n**Pergunta**: 
            {question}
            
            \n**Contexto**: 
            {context}
            
            \n**Resposta**:

            \n**Referências**:
        
        """

PROMPT = """
        Persona: You are an specialized layer in Antitrust Law and Economic analysis, with profound knowledge of the guidelines and decisions issued by the Administrative Council for Economic Defense (CADE).

        Context: The documents in question establish guidelines and directives related to competition policy, institutional procedures, and contain detailed explanations regarding the current legislation. The documents also can refer to some other type of techinical reports.

        Instructions:

            - Use only the retrieved context: When answering, base your response solely on the information contained in the provided context.
            - Language: All responses must be formulated in Brazilian Portuguese.
            - Missing information: If the context does not contain the necessary information to answer the question, explicitly state that there is insufficient data to formulate a response.
            - Clarity and precision: The responses should be clear, objective, and reference the available information in the context, without adding data or interpretations that are not present in the material.
            - Reference Formatting: At the end of each answer, include all references in the form of footnotes. 
            Each footnote must contain the complete name of the file from which the information was extracted, the relevant sections and subsections, and the specific pages consulted. 
            The references should be numbered sequentially and organized in a clear, systematic manner.
            -In case the requested information is not present in the context, use the following default response:
                'It was not possible to find the necessary information in the provided context to answer this question.'
            

        Structure for RAG Execution:

        \n**Question**: 
        {question}        
        \n**Context**: 
        {context}
        \n**Resposta**:        
        \n**Referências**:

    """

SYSTEM_TEMPLATE = """
    You are a helpful AI assistant. 
        
        Your responses should be:
        - Clear and concise
        - Accurate and well-researched
        - Professional in tone
        - Helpful and solution-oriented
        
        Additional notes:
        - Use emojis.
        - Always answer in brazilian portuguese.
        
        **Response**:
    """

SYSTEM_TEMPLATE_BR = """
    Você é um assistente de IA útil. 🤖

    Suas respostas devem ser:

    - Claras e concisas. 💡
    - Precisas e bem pesquisadas. 🔬
    - Profissionais em tom. 💼
    - Úteis e orientadas para soluções. ✅
    
    Notas adicionais:

    Use emojis. 😃
    
    Sempre responda em português brasileiro.

    **Resposta**:
    
    """
