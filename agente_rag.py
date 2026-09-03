from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma 
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent


load_dotenv()

documents = PyPDFLoader("arquivos/Flutter_for_Beginners_by_Alessandro_Biessek_(z-lib.org).pdf").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

docs = text_splitter.split_documents(documents)

persist_directory = "./chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if os.path.exists(persist_directory):
    vector_store = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
        )

else:
    vector_store = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=persist_directory
        )


@tool (response_format="content_and_artifact")
def buscar_no_documento(pergunta: str):
    """Essa feramenta busca informações no documento carregado. Use sempre que o ususário perguntar sobre flutter e desenvolvimento mobile."""

    retrieved_docs = vector_store.similarity_search(pergunta, k=2)
    serialized = "\n\n".join(
        f"Fonte: {doc.metadata}\n Conteudo: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

llm = init_chat_model(
    model="gemini-2.0-flash", 
    model_provider="google_genai", 
    temperature=0
)

agent_rag = create_agent(
    model=llm,
    tools=[buscar_no_documento],
    system_prompt="Você é um assistente especializado em Flutter e desenvolvimento mobile. Só responda perguntas sobre esse tema e se encontrar a resposta no documento. Não invente respostas. Responda educadamente como se fosse para uma criança."
)