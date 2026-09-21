from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import PIIMiddleware
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai:gpt-4.1-mini"
CPF = r"\d{3}\.\d{3}\.\d{3}-\d{2}"

@tool
def abrir_chamado(descricao: str):
    """Abre um chamado para o suporte."""
    return f"Chamado aberto para {descricao}"

agente = create_agent(
    model=MODEL,
    tools=[abrir_chamado],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("cpf", detector=CPF, strategy="mask", apply_to_input=True)
    ]       
)

MENSAGEM = ("Meu email é jady.s.m@example.com e meu CPF é 050.154.759-74. Abre um chamado para o suporte.")

if __name__ == "__main__":
    r = agente.invoke({"messages": [{
            "role": "user",
            "content": MENSAGEM
            }]
        }
    )
    for m in r["messages"]:
        print(f"{m.type}: {m.content}")