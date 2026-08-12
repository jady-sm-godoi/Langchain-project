# from langgraph.checkpoint.memory import InMemorySaver
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
# checkpoint = InMemorySaver()

model = init_chat_model(
    model="gemini-3.5-flash-lite", model_provider="google_genai"
)

agente_jady = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo. Responda de forma clara e concisa. Caso não saiba a resposta, solte um palavrão e diga que não sabe. Não invente respostas.",
    tools=[TavilySearch()],
    checkpointer=checkpoint,
    # middleware=[
    #     SummarizationMiddleware(
    #         model=model, trigger=("messages", 6), keep=("messages", 2)
    #     )
    # ],# Essa não é a forma mais eficiente de resumir, mas é didática e fácil de testar e visualizar no LangSmith. A forma mais eficiente é resumir com base no tamanho do texto, e não na quantidade de mensagens. Abaixo está a forma mais eficiente de resumir.
    middleware=[
        SummarizationMiddleware(
            model=model, trigger=("tokens", 3000), keep=("messages", 10)
        )
    ],
)

config = {"configurable": {"thread_id": "novo_thread"}}  # TODO: id dinâmico

print("agente em funcionamento")

while True:
    pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        break
    resposta = agente_jady.invoke(
        {"messages": [{"role": "user", "content": pergunta}]}, config=config
    )
    print("Resposta:", resposta["messages"][-1].text)
