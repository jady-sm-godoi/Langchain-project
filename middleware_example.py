from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call

load_dotenv()


principal = init_chat_model("openai:modelo-que-nao-existe")
reserva = init_chat_model("openai:gpt-4.1-mini")

@wrap_model_call
def modelo_reserva(request, handler):
    # Implement the logic for the reserva model
    try:
        return handler(request)
    except Exception as e:
        print(f"Modelo principal falhou, usando modelo de reserva: {e}")
        return handler(request.override(model=reserva))

agent = create_agent(
    model=principal,
    tools=[],
    middleware=[modelo_reserva]
)

if __name__ == "__main__":
    r=agent.invoke(
        {"messages": [
            {"role": "user", "content": "Olá, tudo bem?"}
        ]}
    )
    print(r["messages"][-1].content)