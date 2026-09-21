from dotenv import load_dotenv
from langchain import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

client = init_chat_model(model="gpt-3.5-turbo")


@tool
def dobro(n: int) -> int:
    """Retorna o dobro de um número."""
    return n * 2


@tool
def triplo(n: int) -> int:
    """Retorna o triplo de um número."""
    return n * 3


agent = create_agent(model=client, tools=[dobro, triplo])

if __name__ == "__main__":
    r = agent.invoke(
        {"message": [{"role": "user", "content": "Qual é o dobro de 5?"}]}
    )
    print(r["messages"][-1].content)
