from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-3.5-flash", model_provider="google_genai")

agente_jady = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo. Responda de forma clara e concisa. Caso não saiba a resposta, solte um palavrão e diga que não sabe. Não invente respostas.",
    tools=[TavilySearch()],
)

# pergunta_usuario = "Qual a temperatura média em São Paulo hoje?"

# resposta_agente = agente_jady.invoke({"messages": [{"role": "user", "content": pergunta_usuario}]})

# print(resposta_agente["messages"][-1].text)